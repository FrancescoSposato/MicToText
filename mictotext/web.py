"""Minimal local web UI for MicToText: same pipeline as the CLI, driven from a browser.

Runs a Flask dev server on 127.0.0.1 only (never exposed on the network) and opens the
default browser automatically. The microphone is still captured server-side by this
process via sounddevice; the browser page is only a remote control + log/result viewer.

Single active session at a time: it's a personal local tool, not a multi-user server.
"""

from __future__ import annotations

import copy
import re
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from mictotext.config import AppConfig, language_name
from mictotext.diagram import generate_diagram
from mictotext.llm import OllamaClient
from mictotext.notes import generate_notes
from mictotext.recorder import MicRecorder, list_input_devices_structured
from mictotext.renderer import build_renderer
from mictotext.transcriber import transcribe

HOST = "127.0.0.1"
PORT = 8765

_ACTIVE_STEPS = ("recording", "transcribing", "notes", "diagram")
_SESSION_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

_STEP_LABELS = {
    "idle": "Pronto.",
    "recording": "Registrazione in corso...",
    "transcribing": "Trascrizione audio...",
    "notes": "Generazione appunti...",
    "diagram": "Generazione schema...",
    "done": "Completato.",
    "error": "Errore.",
}


class _LogTee:
    """Duplicates writes to the real stdout and splits them into lines for the UI log.

    Only one pipeline thread runs at a time in this app, so a process-wide sys.stdout
    swap for the duration of that thread is an acceptable simplification here.
    """

    def __init__(self, original, on_line) -> None:
        self._original = original
        self._on_line = on_line
        self._buffer = ""

    def write(self, text: str) -> int:
        self._original.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip("\r")
            if line:
                self._on_line(line)
        return len(text)

    def flush(self) -> None:
        self._original.flush()


@dataclass
class PipelineState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    step: str = "idle"
    log: list = field(default_factory=list)
    error: str | None = None
    session_dir: Path | None = None
    session_id: str | None = None
    notes_md: str | None = None
    image_name: str | None = None
    html_name: str | None = None
    recorder: MicRecorder | None = None

    def append(self, line: str) -> None:
        with self.lock:
            self.log.append(line)

    def set_step(self, step: str) -> None:
        with self.lock:
            self.step = step

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "step": self.step,
                "label": _STEP_LABELS.get(self.step, self.step),
                "log": list(self.log),
                "error": self.error,
                "session_id": self.session_id,
                "notes_md": self.notes_md,
                "image_name": self.image_name,
                "html_name": self.html_name,
            }


def _config_from_payload(base_cfg: AppConfig, payload: dict) -> AppConfig:
    cfg = copy.deepcopy(base_cfg)
    language = (payload.get("language") or "").strip()
    if language:
        cfg.stt.language = None if language == "auto" else language
    notes_model = (payload.get("notes_model") or "").strip()
    if notes_model:
        cfg.llm.notes_model = notes_model
    diagram_model = (payload.get("diagram_model") or "").strip()
    cfg.llm.diagram_model = diagram_model or None
    return cfg


def create_app(base_cfg: AppConfig) -> Flask:
    app = Flask(__name__)
    state = PipelineState()

    def run_pipeline(cfg: AppConfig, session_dir: Path) -> None:
        """Transcription -> notes -> diagram, run in a background thread."""
        original_stdout = sys.stdout
        sys.stdout = _LogTee(original_stdout, state.append)
        try:
            notes_model = cfg.llm.notes_model
            diagram_model = cfg.llm.diagram_model or notes_model
            client = OllamaClient(cfg.llm)
            client.ensure_ready({notes_model, diagram_model})

            state.set_step("transcribing")
            print("=== Trascrizione ===")
            audio_path = session_dir / "audio.wav"
            transcript = transcribe(audio_path, session_dir, cfg.stt)
            transcript_text = transcript.text
            if not transcript_text.strip():
                raise RuntimeError("Trascrizione vuota: nessun parlato rilevato.")
            (session_dir / "trascrizione.txt").write_text(transcript_text.strip() + "\n", encoding="utf-8")
            language = cfg.stt.language or transcript.language

            state.set_step("notes")
            print("\n=== Appunti strutturati ===")
            notes_md = generate_notes(transcript_text, client, notes_model, cfg.llm, language_name(language))
            (session_dir / "appunti.md").write_text(notes_md, encoding="utf-8")
            with state.lock:
                state.notes_md = notes_md

            state.set_step("diagram")
            print("\n=== Schema Mermaid ===")
            renderer = build_renderer(cfg.render)
            result = generate_diagram(notes_md, client, diagram_model, cfg.llm, cfg.render, renderer,
                                      session_dir, language_name(language))

            with state.lock:
                state.html_name = result.html_path.name
                state.image_name = result.image_path.name if result.image_path else None
                state.step = "done"
            print("\n=== Fatto ===")
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser, not fatal to the server
            with state.lock:
                state.step = "error"
                state.error = str(exc)
            print(f"\nERRORE: {exc}")
        finally:
            sys.stdout = original_stdout

    @app.get("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/devices")
    def api_devices():
        try:
            return jsonify(list_input_devices_structured())
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/defaults")
    def api_defaults():
        return jsonify({
            "language": base_cfg.stt.language or "auto",
            "notes_model": base_cfg.llm.notes_model,
            "diagram_model": base_cfg.llm.diagram_model or "",
        })

    @app.get("/api/status")
    def api_status():
        return jsonify(state.snapshot())

    @app.get("/api/level")
    def api_level():
        with state.lock:
            recorder = state.recorder
            recording = state.step == "recording"
        if recorder is None or not recording:
            return jsonify({"db": -60.0, "elapsed": 0.0})
        return jsonify({"db": recorder.level_db, "elapsed": recorder.elapsed})

    @app.post("/api/start")
    def api_start():
        with state.lock:
            if state.step in _ACTIVE_STEPS:
                return jsonify({"error": "Una sessione e' gia' in corso."}), 409
            state.step = "recording"
            state.log = []
            state.error = None
            state.notes_md = None
            state.image_name = None
            state.html_name = None

        payload = request.get_json(silent=True) or {}
        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir = base_cfg.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        device = payload.get("device")
        device = int(device) if isinstance(device, str) and device.isdigit() else (device or None)

        recorder = MicRecorder(session_dir / "audio.wav", device=device, channels=base_cfg.audio.channels)
        try:
            recorder.start()
        except Exception as exc:  # noqa: BLE001
            with state.lock:
                state.step = "idle"
            return jsonify({"error": f"Impossibile avviare la registrazione: {exc}"}), 500

        with state.lock:
            state.session_dir = session_dir
            state.session_id = session_id
            state.recorder = recorder
        state.append(f"Registrazione avviata (sessione {session_id}).")
        return jsonify({"session_id": session_id})

    @app.post("/api/stop")
    def api_stop():
        with state.lock:
            recorder = state.recorder
            session_dir = state.session_dir
        if recorder is None or session_dir is None:
            return jsonify({"error": "Nessuna registrazione in corso."}), 409

        duration, peak, overflow = recorder.stop()
        with state.lock:
            state.recorder = None
        state.append(f"Registrazione fermata: {duration:.1f}s (picco {peak:.2f}).")
        if overflow:
            state.append(f"Attenzione: {overflow} overflow del buffer audio.")
        if peak < 0.01:
            state.append("Attenzione: segnale quasi assente, controlla il microfono.")
        if duration < 1.0:
            with state.lock:
                state.step = "error"
                state.error = "Registrazione troppo corta (meno di un secondo)."
            return jsonify({"error": state.error}), 400

        payload = request.get_json(silent=True) or {}
        cfg = _config_from_payload(base_cfg, payload)

        state.set_step("transcribing")
        thread = threading.Thread(target=run_pipeline, args=(cfg, session_dir), daemon=True)
        thread.start()
        return jsonify({"ok": True})

    @app.get("/files/<session_id>/<path:filename>")
    def files(session_id: str, filename: str):
        if not _SESSION_ID_RE.match(session_id):
            return "Not found", 404
        return send_from_directory(str(base_cfg.output_root / session_id), filename)

    return app


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MicToText</title>
<style>
  :root {
    --bg: #f4f6f8; --card: #ffffff; --text: #1f2937; --muted: #6b7280;
    --accent: #6e9bd1; --accent-dark: #4d7bb0; --border: #e2e6ea;
    --ok: #79b791; --err: #d98a8a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif; padding: 24px;
  }
  .wrap { max-width: 820px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .subtitle { color: var(--muted); margin: 0 0 20px; font-size: 0.9rem; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 16px;
  }
  .card h2 { font-size: 1rem; margin: 0 0 12px; color: var(--text); }
  .settings-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }
  label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
  select, input[type=text] {
    width: 100%; padding: 7px 9px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.9rem; background: #fff; color: var(--text);
  }
  .record-row { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-top: 6px; }
  button#recordBtn {
    background: var(--accent); color: #fff; border: none; border-radius: 8px;
    padding: 12px 22px; font-size: 1rem; cursor: pointer; font-weight: 600;
  }
  button#recordBtn:hover:not(:disabled) { background: var(--accent-dark); }
  button#recordBtn:disabled { background: #b9c3cc; cursor: default; }
  button#recordBtn.recording { background: var(--err); }
  .meter { flex: 1; min-width: 160px; }
  .meter-bar {
    height: 10px; background: #eef1f4; border-radius: 5px; overflow: hidden; border: 1px solid var(--border);
  }
  .meter-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #79b791, #d6ae55, #d98a8a); }
  .timer { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.85rem; min-width: 42px; }
  .step-label { margin-top: 10px; font-size: 0.9rem; color: var(--muted); }
  #errorBanner {
    display: none; background: #fdf4f4; border: 1px solid var(--err); color: #8a3d3d;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 0.9rem;
  }
  pre#log {
    background: #1f2937; color: #d1e7dd; font-size: 0.78rem; padding: 12px;
    border-radius: 8px; max-height: 220px; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
    margin: 0;
  }
  #results { display: none; }
  pre#notesOutput {
    white-space: pre-wrap; word-break: break-word; font-size: 0.88rem; line-height: 1.5;
    max-height: 420px; overflow-y: auto; background: #fafbfc; border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; margin: 0 0 14px;
  }
  #imageWrap img { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }
  .downloads { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; font-size: 0.85rem; }
  .downloads a { color: var(--accent-dark); text-decoration: none; }
  .downloads a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
  <h1>MicToText</h1>
  <p class="subtitle">Microfono &rarr; trascrizione &rarr; appunti &rarr; schema Mermaid. Tutto in locale.</p>

  <div id="errorBanner"></div>

  <div class="card">
    <h2>Impostazioni</h2>
    <div class="settings-grid">
      <div>
        <label for="device">Microfono</label>
        <select id="device"></select>
      </div>
      <div>
        <label for="language">Lingua parlata</label>
        <select id="language">
          <option value="it">Italiano</option>
          <option value="en">Inglese</option>
          <option value="auto">Rilevamento automatico</option>
        </select>
      </div>
      <div>
        <label for="notesModel">Modello Ollama (appunti)</label>
        <input type="text" id="notesModel">
      </div>
      <div>
        <label for="diagramModel">Modello Ollama (schema, opzionale)</label>
        <input type="text" id="diagramModel" placeholder="uguale al modello appunti">
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Registrazione</h2>
    <div class="record-row">
      <button id="recordBtn">Avvia registrazione</button>
      <div class="meter">
        <div class="meter-bar"><div class="meter-fill" id="meterFill"></div></div>
      </div>
      <div class="timer" id="timer">00:00</div>
    </div>
    <div class="step-label" id="stepLabel">Pronto.</div>
  </div>

  <div class="card">
    <h2>Log</h2>
    <pre id="log"></pre>
  </div>

  <div class="card" id="results">
    <h2>Appunti</h2>
    <pre id="notesOutput"></pre>
    <h2>Schema</h2>
    <div id="imageWrap"></div>
    <div class="downloads">
      <a id="htmlLink" href="#" target="_blank" style="display:none">Apri anteprima HTML</a>
      <a id="downloadNotes" href="#" target="_blank">appunti.md</a>
      <a id="downloadMmd" href="#" target="_blank">schema.mmd</a>
      <a id="downloadTranscript" href="#" target="_blank">trascrizione.txt</a>
    </div>
  </div>
</div>

<script>
let statusPolling = null;
let levelPolling = null;

async function loadDefaults() {
  const res = await fetch('/api/defaults');
  const data = await res.json();
  document.getElementById('language').value = data.language;
  document.getElementById('notesModel').value = data.notes_model;
  document.getElementById('diagramModel').value = data.diagram_model;
}

async function loadDevices() {
  const res = await fetch('/api/devices');
  const devices = await res.json();
  const sel = document.getElementById('device');
  sel.innerHTML = '';
  if (!Array.isArray(devices) || devices.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'Nessun microfono rilevato';
    sel.appendChild(opt);
    return;
  }
  devices.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.index;
    opt.textContent = d.name + (d.is_default ? ' (predefinito)' : '');
    if (d.is_default) opt.selected = true;
    sel.appendChild(opt);
  });
}

function setUIState(mode) {
  const btn = document.getElementById('recordBtn');
  btn.classList.remove('recording');
  if (mode === 'idle') {
    btn.textContent = 'Avvia registrazione';
    btn.disabled = false;
    btn.onclick = startRecording;
  } else if (mode === 'recording') {
    btn.textContent = 'Ferma registrazione';
    btn.disabled = false;
    btn.classList.add('recording');
    btn.onclick = stopRecording;
  } else if (mode === 'processing') {
    btn.textContent = 'Elaborazione in corso...';
    btn.disabled = true;
  }
}

async function startRecording() {
  hideError();
  document.getElementById('results').style.display = 'none';
  const device = document.getElementById('device').value;
  const res = await fetch('/api/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({device})
  });
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore sconosciuto.'); return; }
  setUIState('recording');
  startLevelPolling();
  startStatusPolling();
}

async function stopRecording() {
  stopLevelPolling();
  const language = document.getElementById('language').value;
  const notesModel = document.getElementById('notesModel').value;
  const diagramModel = document.getElementById('diagramModel').value;
  const res = await fetch('/api/stop', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({language, notes_model: notesModel, diagram_model: diagramModel})
  });
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore sconosciuto.'); setUIState('idle'); return; }
  setUIState('processing');
}

function startStatusPolling() {
  if (statusPolling) return;
  statusPolling = setInterval(pollStatus, 1000);
  pollStatus();
}

async function pollStatus() {
  const res = await fetch('/api/status');
  const s = await res.json();
  document.getElementById('stepLabel').textContent = s.label;
  const logEl = document.getElementById('log');
  logEl.textContent = s.log.join('\n');
  logEl.scrollTop = logEl.scrollHeight;

  if (s.step === 'done') {
    clearInterval(statusPolling); statusPolling = null;
    setUIState('idle');
    showResults(s);
  } else if (s.step === 'error') {
    clearInterval(statusPolling); statusPolling = null;
    setUIState('idle');
    showError(s.error || 'Errore sconosciuto.');
  }
}

function startLevelPolling() {
  if (levelPolling) return;
  levelPolling = setInterval(async () => {
    const res = await fetch('/api/level');
    const d = await res.json();
    const pct = Math.max(0, Math.min(100, (d.db + 60) / 60 * 100));
    document.getElementById('meterFill').style.width = pct + '%';
    const m = Math.floor(d.elapsed / 60), sec = Math.floor(d.elapsed % 60);
    document.getElementById('timer').textContent =
      String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
  }, 300);
}
function stopLevelPolling() {
  if (levelPolling) { clearInterval(levelPolling); levelPolling = null; }
  document.getElementById('meterFill').style.width = '0%';
}

function showResults(s) {
  document.getElementById('results').style.display = 'block';
  document.getElementById('notesOutput').textContent = s.notes_md || '(nessun risultato)';
  const imgWrap = document.getElementById('imageWrap');
  const htmlLink = document.getElementById('htmlLink');
  if (s.image_name) {
    imgWrap.innerHTML = '<img src="/files/' + s.session_id + '/' + s.image_name + '?t=' + Date.now() + '" alt="Schema">';
  } else {
    imgWrap.innerHTML = '<p>Immagine non renderizzata: apri l\'anteprima HTML qui sotto.</p>';
  }
  if (s.html_name) {
    htmlLink.href = '/files/' + s.session_id + '/' + s.html_name;
    htmlLink.style.display = 'inline';
  } else {
    htmlLink.style.display = 'none';
  }
  document.getElementById('downloadNotes').href = '/files/' + s.session_id + '/appunti.md';
  document.getElementById('downloadMmd').href = '/files/' + s.session_id + '/schema.mmd';
  document.getElementById('downloadTranscript').href = '/files/' + s.session_id + '/trascrizione.txt';
}

function showError(message) {
  const el = document.getElementById('errorBanner');
  el.textContent = message;
  el.style.display = 'block';
}
function hideError() {
  document.getElementById('errorBanner').style.display = 'none';
}

async function restoreState() {
  const res = await fetch('/api/status');
  const s = await res.json();
  document.getElementById('stepLabel').textContent = s.label;
  document.getElementById('log').textContent = s.log.join('\n');
  if (s.step === 'recording') {
    setUIState('recording');
    startLevelPolling();
    startStatusPolling();
  } else if (s.step === 'transcribing' || s.step === 'notes' || s.step === 'diagram') {
    setUIState('processing');
    startStatusPolling();
  } else if (s.step === 'done') {
    setUIState('idle');
    showResults(s);
  } else if (s.step === 'error') {
    setUIState('idle');
    showError(s.error || 'Errore sconosciuto.');
  } else {
    setUIState('idle');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  loadDefaults();
  loadDevices();
  restoreState();
});
</script>
</body>
</html>
"""


def main() -> int:
    app = create_app(AppConfig())
    url = f"http://{HOST}:{PORT}/"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"MicToText web UI: {url}")
    print("Premi Ctrl+C per fermare il server.")
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
