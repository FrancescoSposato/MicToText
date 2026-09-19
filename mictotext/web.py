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

from mictotext.config import THINKING_LEVELS, AppConfig, language_name
from mictotext.diagram import generate_diagram
from mictotext.llm import OllamaClient, OllamaError
from mictotext.notes import generate_notes
from mictotext.recorder import MicRecorder, list_input_devices_structured
from mictotext.renderer import build_renderer
from mictotext.transcriber import transcribe

HOST = "127.0.0.1"
PORT = 8765

_ACTIVE_STEPS = ("recording", "downloading", "transcribing", "notes", "diagram", "cards")
_SESSION_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

_STEP_LABELS = {
    "idle": "Pronto.",
    "recording": "Registrazione in corso...",
    "downloading": "Download del video...",
    "transcribing": "Trascrizione audio...",
    "notes": "Generazione appunti...",
    "diagram": "Generazione schema...",
    "cards": "Generazione schede concetto...",
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
    card_names: list = field(default_factory=list)  # paths relative to the session folder
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
                "card_names": list(self.card_names),
            }


def _thinking_level(cfg: AppConfig) -> str:
    for name, values in THINKING_LEVELS.items():
        if (bool(cfg.llm.think_notes), bool(cfg.llm.think)) == (values["think_notes"], values["think"]):
            return name
    return "notes"


def _config_from_payload(base_cfg: AppConfig, payload: dict) -> AppConfig:
    cfg = copy.deepcopy(base_cfg)
    cards = payload.get("cards")
    if cards is not None and str(cards).strip().isdigit():
        cfg.llm.concept_cards = max(0, min(8, int(cards)))
    level = THINKING_LEVELS.get((payload.get("thinking") or "").strip())
    if level:
        cfg.llm.think_notes = level["think_notes"]
        cfg.llm.think = level["think"]
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

    def run_pipeline(cfg: AppConfig, session_dir: Path, url: str | None = None) -> None:
        """[download] -> transcription -> notes -> diagram, run in a background thread."""
        original_stdout = sys.stdout
        sys.stdout = _LogTee(original_stdout, state.append)
        try:
            notes_model = cfg.llm.notes_model
            diagram_model = cfg.llm.diagram_model or notes_model
            client = OllamaClient(cfg.llm)
            client.ensure_ready({notes_model, diagram_model})

            audio_path = session_dir / "audio.wav"
            if url:
                from mictotext.fetch import download_audio
                state.set_step("downloading")
                print("=== Download ===")
                media = download_audio(url, session_dir)
                audio_path = media.path
                length = f" ({media.duration / 60:.1f} min)" if media.duration else ""
                print(f"  {media.title}{length}")

            state.set_step("transcribing")
            print("=== Trascrizione ===")
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
                state.html_name = result.html_path.name if result.html_path else None
                state.image_name = result.image_path.name if result.image_path else None

            if cfg.llm.concept_cards > 0:
                from mictotext.diagram import extract_concepts, generate_concept_cards
                state.set_step("cards")
                print("\n=== Schede concetto ===")
                concepts = extract_concepts(notes_md, client, diagram_model, cfg.llm,
                                            language_name(language), cfg.llm.concept_cards)
                if concepts:
                    print(f"  Concetti selezionati: {', '.join(concepts)}")
                    cards = generate_concept_cards(notes_md, concepts, client, diagram_model,
                                                   cfg.llm, cfg.render, renderer, session_dir,
                                                   language_name(language))
                    with state.lock:
                        state.card_names = [p.relative_to(session_dir).as_posix() for p in cards]
                else:
                    print("  Nessun concetto adatto a una scheda.")

            state.set_step("done")
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
            "thinking": _thinking_level(base_cfg),
            "cards": base_cfg.llm.concept_cards,
        })

    @app.get("/api/models")
    def api_models():
        """Models currently pulled in Ollama, to populate the UI dropdowns."""
        try:
            return jsonify(sorted(OllamaClient(base_cfg.llm).list_models()))
        except OllamaError as exc:
            return jsonify({"error": str(exc)}), 503

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
            state.card_names = []

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

    @app.post("/api/url")
    def api_url():
        """Start the pipeline from a video URL instead of the microphone."""
        payload = request.get_json(silent=True) or {}
        url = (payload.get("url") or "").strip()
        if not url:
            return jsonify({"error": "Inserisci un URL."}), 400

        with state.lock:
            if state.step in _ACTIVE_STEPS:
                return jsonify({"error": "Una sessione e' gia' in corso."}), 409
            state.step = "downloading"
            state.log = []
            state.error = None
            state.notes_md = None
            state.image_name = None
            state.html_name = None
            state.card_names = []

        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir = base_cfg.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        with state.lock:
            state.session_dir = session_dir
            state.session_id = session_id
        state.append(f"Sessione {session_id}: {url}")

        cfg = _config_from_payload(base_cfg, payload)
        threading.Thread(target=run_pipeline, args=(cfg, session_dir, url), daemon=True).start()
        return jsonify({"session_id": session_id})

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
  .url-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .url-row input { flex: 1; min-width: 220px; }
  button#urlBtn {
    background: var(--accent); color: #fff; border: none; border-radius: 8px;
    padding: 8px 18px; font-size: 0.9rem; cursor: pointer; font-weight: 600; white-space: nowrap;
  }
  button#urlBtn:hover:not(:disabled) { background: var(--accent-dark); }
  button#urlBtn:disabled { background: #b9c3cc; cursor: default; }
  .hint { font-size: 0.78rem; color: var(--muted); margin-top: 8px; }
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
  /* Cards are tall and narrow, so they tile side by side instead of stacking. */
  .cards-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 14px; align-items: start; margin-top: 10px;
  }
  .cards-grid figure { margin: 0; }
  .cards-grid img {
    width: 100%; border-radius: 8px; border: 1px solid var(--border); cursor: zoom-in;
    background: #fff;
  }
  .cards-grid figcaption {
    font-size: 0.75rem; color: var(--muted); margin-top: 5px; text-align: center;
    word-break: break-word;
  }
  /* Full-screen viewer: diagrams are often far taller than the page, so they need
     zooming and panning rather than a scaled-down thumbnail. */
  #viewer {
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: #ffffff; overflow: hidden; touch-action: none;
  }
  #viewer.open { display: block; }
  #viewerImg {
    position: absolute; top: 0; left: 0; transform-origin: 0 0;
    user-select: none; -webkit-user-drag: none; cursor: grab; background: #fff;
  }
  #viewer.dragging #viewerImg { cursor: grabbing; }
  #viewerBar {
    position: absolute; top: 0; left: 0; right: 0; z-index: 2;
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    background: rgba(255,255,255,0.94); border-bottom: 1px solid var(--border);
  }
  #viewerBar .title {
    flex: 1; font-size: 0.85rem; color: var(--muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  #viewerBar button {
    background: #fff; border: 1px solid var(--border); border-radius: 6px;
    padding: 5px 11px; font-size: 0.85rem; cursor: pointer; color: var(--text);
  }
  #viewerBar button:hover { background: #eef2f6; }
  #zoomLabel { font-size: 0.8rem; color: var(--muted); min-width: 46px; text-align: center; }
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
        <select id="notesModel"></select>
      </div>
      <div>
        <label for="diagramModel">Modello Ollama (schema)</label>
        <select id="diagramModel"></select>
      </div>
      <div>
        <label for="thinking">Ragionamento del modello</label>
        <select id="thinking">
          <option value="none">Nessuno &mdash; veloce, ma puo' inventare</option>
          <option value="notes">Solo appunti &mdash; consigliato</option>
          <option value="full">Completo &mdash; molto lento</option>
        </select>
        <div class="hint" id="thinkingHint"></div>
      </div>
      <div>
        <label for="cards">Schede concetto</label>
        <select id="cards">
          <option value="0">Nessuna</option>
          <option value="2">2</option>
          <option value="3">3</option>
          <option value="4">4</option>
          <option value="6">6</option>
        </select>
        <div class="hint">Spiegazioni discorsive dei concetti chiave, una per scheda.</div>
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
    <h2>Oppure: da un video online</h2>
    <div class="url-row">
      <input type="text" id="urlInput" placeholder="https://www.youtube.com/watch?v=...">
      <button id="urlBtn">Trascrivi da URL</button>
    </div>
    <div class="hint">YouTube e molti altri siti. Viene scaricata solo la traccia audio.</div>
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
    <div id="cardsSection" style="display:none">
      <h2>Schede concetto</h2>
      <div id="cardsWrap" class="cards-grid"></div>
    </div>
    <div class="downloads">
      <a id="htmlLink" href="#" target="_blank" style="display:none">Apri anteprima HTML</a>
      <a id="downloadNotes" href="#" target="_blank">appunti.md</a>
      <a id="downloadMmd" href="#" target="_blank">schema.mmd</a>
      <a id="downloadTranscript" href="#" target="_blank">trascrizione.txt</a>
    </div>
  </div>
</div>

<div id="viewer">
  <div id="viewerBar">
    <span class="title" id="viewerTitle"></span>
    <button id="zoomOut" title="Riduci (-)">&minus;</button>
    <span id="zoomLabel">100%</span>
    <button id="zoomIn" title="Ingrandisci (+)">+</button>
    <button id="zoomFit" title="Adatta alla finestra (0)">Adatta</button>
    <button id="zoomFull" title="Dimensione reale (1)">100%</button>
    <button id="fsToggle" title="Schermo intero (F)">Schermo intero</button>
    <button id="viewerClose" title="Chiudi (Esc)">Chiudi</button>
  </div>
  <img id="viewerImg" alt="">
</div>

<script>
let statusPolling = null;
let levelPolling = null;

async function loadModelsAndDefaults() {
  const [modelsRes, defaultsRes] = await Promise.all([
    fetch('/api/models'), fetch('/api/defaults')
  ]);
  const models = await modelsRes.json();
  const defaults = await defaultsRes.json();

  document.getElementById('language').value = defaults.language;
  if (defaults.thinking) document.getElementById('thinking').value = defaults.thinking;
  if (defaults.cards !== undefined) {
    const sel = document.getElementById('cards');
    // The configured default may not be one of the listed options.
    if (![...sel.options].some(o => o.value == defaults.cards)) {
      sel.appendChild(new Option(defaults.cards, defaults.cards));
    }
    sel.value = defaults.cards;
  }
  updateThinkingHint();

  const notesSel = document.getElementById('notesModel');
  const diagSel = document.getElementById('diagramModel');
  notesSel.innerHTML = '';
  diagSel.innerHTML = '';
  // Empty value = "same as the notes model" (the backend falls back to it).
  diagSel.appendChild(new Option('(uguale al modello appunti)', ''));

  const list = Array.isArray(models) ? models : [];
  if (list.length === 0) {
    notesSel.appendChild(new Option('Nessun modello in Ollama', ''));
    showError(models.error || 'Nessun modello trovato in Ollama. Scaricane uno con: ollama pull <modello>');
    return;
  }
  list.forEach(m => {
    notesSel.appendChild(new Option(m, m));
    diagSel.appendChild(new Option(m, m));
  });

  if (list.includes(defaults.notes_model)) notesSel.value = defaults.notes_model;
  if (defaults.diagram_model && list.includes(defaults.diagram_model)) {
    diagSel.value = defaults.diagram_model;
  }
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
  const urlBtn = document.getElementById('urlBtn');
  btn.classList.remove('recording');
  // The two sources are mutually exclusive: only one session runs at a time.
  urlBtn.disabled = (mode !== 'idle');
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

function currentSettings() {
  return {
    language: document.getElementById('language').value,
    notes_model: document.getElementById('notesModel').value,
    diagram_model: document.getElementById('diagramModel').value,
    thinking: document.getElementById('thinking').value,
    cards: document.getElementById('cards').value
  };
}

// Measured with qwen3.5:9b on the same 30s source, as a rough order of magnitude.
const THINKING_HINTS = {
  none:  'Il ragionamento e\' cio\' che verifica le affermazioni contro la trascrizione: senza, il modello tende a riempire i vuoti con nozioni proprie. Misurato: ~1m20s.',
  notes: 'Ragiona dove il contenuto nasce dalla trascrizione, non dove rimaneggia appunti gia\' scritti. Misurato: ~2m10s.',
  full:  'Ragiona anche su schema e schede, dove serve poco perche\' lavorano su appunti gia\' scritti. Misurato: ~12m40s.'
};

function updateThinkingHint() {
  const level = document.getElementById('thinking').value;
  document.getElementById('thinkingHint').textContent = THINKING_HINTS[level] || '';
}

async function startFromUrl() {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) { showError('Inserisci un URL.'); return; }
  hideError();
  document.getElementById('results').style.display = 'none';
  const res = await fetch('/api/url', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({url: url}, currentSettings()))
  });
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore sconosciuto.'); return; }
  setUIState('processing');
  startStatusPolling();
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
  const res = await fetch('/api/stop', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(currentSettings())
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
    const url = '/files/' + s.session_id + '/' + s.image_name + '?t=' + Date.now();
    imgWrap.innerHTML = '';
    const img = document.createElement('img');
    img.src = url;
    img.alt = 'Schema';
    img.style.cursor = 'zoom-in';
    img.title = 'Clicca per aprire a schermo intero';
    img.onclick = () => openViewer(url, 'Schema');
    imgWrap.appendChild(img);
    const hint = document.createElement('div');
    hint.className = 'hint';
    hint.textContent = 'Clicca lo schema per ingrandirlo: rotella per lo zoom, trascina per spostarti.';
    imgWrap.appendChild(hint);
  } else {
    imgWrap.innerHTML = '<p>Immagine non renderizzata: apri l\'anteprima HTML qui sotto.</p>';
  }
  if (s.html_name) {
    htmlLink.href = '/files/' + s.session_id + '/' + s.html_name;
    htmlLink.style.display = 'inline';
  } else {
    htmlLink.style.display = 'none';
  }
  const cardsSection = document.getElementById('cardsSection');
  const cardsWrap = document.getElementById('cardsWrap');
  cardsWrap.innerHTML = '';
  const cards = s.card_names || [];
  cardsSection.style.display = cards.length ? 'block' : 'none';
  cards.forEach(name => {
    const url = '/files/' + s.session_id + '/' + name + '?t=' + Date.now();
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = url;
    img.alt = name;
    const cap = document.createElement('figcaption');
    // "concetti/01-tempo-relativo.png" -> "tempo relativo"
    cap.textContent = name.split('/').pop().replace(/\.[^.]+$/, '')
                          .replace(/^\d+-/, '').replace(/-/g, ' ');
    img.title = 'Clicca per aprire a schermo intero';
    img.onclick = () => openViewer(url, cap.textContent);
    fig.appendChild(img);
    fig.appendChild(cap);
    cardsWrap.appendChild(fig);
  });

  document.getElementById('downloadNotes').href = '/files/' + s.session_id + '/appunti.md';
  document.getElementById('downloadMmd').href = '/files/' + s.session_id + '/schema.mmd';
  document.getElementById('downloadTranscript').href = '/files/' + s.session_id + '/trascrizione.txt';
}

// --- Visualizzatore a schermo intero (zoom + trascinamento) ---
const viewer = {
  el: null, img: null, scale: 1, tx: 0, ty: 0,
  dragging: false, lastX: 0, lastY: 0, natW: 0, natH: 0
};

function viewerApply() {
  viewer.img.style.transform =
    'translate(' + viewer.tx + 'px,' + viewer.ty + 'px) scale(' + viewer.scale + ')';
  document.getElementById('zoomLabel').textContent = Math.round(viewer.scale * 100) + '%';
}

function viewerBarHeight() {
  return document.getElementById('viewerBar').offsetHeight;
}

function viewerFit() {
  const top = viewerBarHeight();
  const availW = viewer.el.clientWidth;
  const availH = viewer.el.clientHeight - top;
  if (!viewer.natW || !viewer.natH) return;
  viewer.scale = Math.min(availW / viewer.natW, availH / viewer.natH);
  // Centre horizontally; align to the top, since diagrams are usually very tall.
  viewer.tx = (availW - viewer.natW * viewer.scale) / 2;
  viewer.ty = top;
  viewerApply();
}

function viewerZoomAt(factor, cx, cy) {
  const next = Math.min(8, Math.max(0.05, viewer.scale * factor));
  // Keep the point under the cursor anchored while zooming.
  viewer.tx = cx - (cx - viewer.tx) * (next / viewer.scale);
  viewer.ty = cy - (cy - viewer.ty) * (next / viewer.scale);
  viewer.scale = next;
  viewerApply();
}

function viewerZoomCentre(factor) {
  viewerZoomAt(factor, viewer.el.clientWidth / 2, viewer.el.clientHeight / 2);
}

function openViewer(src, title) {
  viewer.el.classList.add('open');
  document.getElementById('viewerTitle').textContent = title || '';
  viewer.img.onload = () => {
    viewer.natW = viewer.img.naturalWidth;
    viewer.natH = viewer.img.naturalHeight;
    viewerFit();
  };
  viewer.img.src = src;
}

function closeViewer() {
  viewer.el.classList.remove('open');
  viewer.img.src = '';
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}

function toggleFullscreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  } else if (viewer.el.requestFullscreen) {
    viewer.el.requestFullscreen().catch(() => {});
  }
}

function initViewer() {
  viewer.el = document.getElementById('viewer');
  viewer.img = document.getElementById('viewerImg');

  document.getElementById('viewerClose').onclick = closeViewer;
  document.getElementById('zoomIn').onclick = () => viewerZoomCentre(1.25);
  document.getElementById('zoomOut').onclick = () => viewerZoomCentre(1 / 1.25);
  document.getElementById('zoomFit').onclick = viewerFit;
  document.getElementById('zoomFull').onclick = () => {
    viewer.scale = 1;
    viewer.tx = (viewer.el.clientWidth - viewer.natW) / 2;
    viewer.ty = viewerBarHeight();
    viewerApply();
  };
  document.getElementById('fsToggle').onclick = toggleFullscreen;

  viewer.el.addEventListener('wheel', e => {
    e.preventDefault();
    viewerZoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX, e.clientY);
  }, {passive: false});

  viewer.img.addEventListener('pointerdown', e => {
    viewer.dragging = true;
    viewer.lastX = e.clientX;
    viewer.lastY = e.clientY;
    viewer.el.classList.add('dragging');
    viewer.img.setPointerCapture(e.pointerId);
  });
  viewer.img.addEventListener('pointermove', e => {
    if (!viewer.dragging) return;
    viewer.tx += e.clientX - viewer.lastX;
    viewer.ty += e.clientY - viewer.lastY;
    viewer.lastX = e.clientX;
    viewer.lastY = e.clientY;
    viewerApply();
  });
  const endDrag = () => {
    viewer.dragging = false;
    viewer.el.classList.remove('dragging');
  };
  viewer.img.addEventListener('pointerup', endDrag);
  viewer.img.addEventListener('pointercancel', endDrag);
  viewer.img.addEventListener('dblclick', () => viewerZoomCentre(1.5));

  window.addEventListener('keydown', e => {
    if (!viewer.el.classList.contains('open')) return;
    if (e.key === 'Escape') closeViewer();
    else if (e.key === '+' || e.key === '=') viewerZoomCentre(1.25);
    else if (e.key === '-') viewerZoomCentre(1 / 1.25);
    else if (e.key === '0') viewerFit();
    else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
  });
  window.addEventListener('resize', () => {
    if (viewer.el.classList.contains('open')) viewerFit();
  });
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
  } else if (['downloading', 'transcribing', 'notes', 'diagram'].includes(s.step)) {
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
  initViewer();
  loadModelsAndDefaults();
  loadDevices();
  restoreState();
  document.getElementById('urlBtn').onclick = startFromUrl;
  document.getElementById('thinking').onchange = updateThinkingHint;
  document.getElementById('urlInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') startFromUrl();
  });
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
