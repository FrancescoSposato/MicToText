"""Minimal local web UI for MicToText: same pipeline as the CLI, driven from a browser.

Runs a Flask dev server on 127.0.0.1 only (never exposed on the network) and opens the
default browser automatically. The microphone is still captured server-side by this
process via sounddevice; the browser page is only a remote control + log/result viewer.

Single active session at a time: it's a personal local tool, not a multi-user server.
"""

from __future__ import annotations

import copy
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from mictotext.config import THINKING_LEVELS, AppConfig, language_name
from mictotext.diagram import generate_diagram
from mictotext.session import is_session_name, rename_session
from mictotext.cancel import Cancelled
from mictotext.cancel import check as check_cancelled
from mictotext.llm import OllamaClient, OllamaError
from mictotext.media import (clean_path, format_time, parse_range_specs, probe_media,
                             resolve_ranges)
from mictotext.notes import generate_notes
from mictotext.recorder import MicRecorder, list_input_devices_structured
from mictotext.renderer import build_renderer
from mictotext.transcriber import transcribe

HOST = "127.0.0.1"
PORT = 8765

_ACTIVE_STEPS = ("recording", "downloading", "transcribing", "notes", "diagram", "cards",
                 "regenerating", "review")
REVIEW_TIMEOUT = 1800  # never block the thread for ever waiting for a confirmation

_STEP_LABELS = {
    "idle": "Pronto.",
    "recording": "Registrazione in corso...",
    "downloading": "Download del video...",
    "transcribing": "Trascrizione audio...",
    "notes": "Generazione appunti...",
    "diagram": "Generazione schema...",
    "cards": "Generazione schede concetto...",
    "regenerating": "Rigenerazione in corso...",
    "review": "In attesa: scegli i blocchi da tenere.",
    "cancelled": "Interrotto.",
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
    topic_names: list = field(default_factory=list)  # per-topic diagrams, same convention
    blocks: list = field(default_factory=list)  # speech blocks awaiting confirmation
    kept_blocks: list = field(default_factory=list)
    review_event: threading.Event = field(default_factory=threading.Event)
    # One event reaches every stage: Ollama streaming, the Whisper child, yt-dlp.
    cancel_event: threading.Event = field(default_factory=threading.Event)
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
                "topic_names": list(self.topic_names),
                "blocks": list(self.blocks),
            }


def _thinking_level(cfg: AppConfig) -> str:
    for name, values in THINKING_LEVELS.items():
        if (bool(cfg.llm.think_notes), bool(cfg.llm.think)) == (values["think_notes"], values["think"]):
            return name
    return "notes"


# Advanced controls: (payload key, config path, kind, min, max). The same table drives
# the UI, the clamping and /api/defaults, so a control can never drift from its limits.
ADVANCED_PARAMS = [
    ("topic", "llm.topic", "text", None, None),
    ("subtopics", "llm.subtopics", "text", None, None),
    ("filter_enabled", "filter.enabled", "bool", None, None),
    ("filter_mode", "filter.mode", "choice", None, ("review", "auto")),
    ("gap_seconds", "filter.gap_seconds", "float", 2, 180),
    ("min_logprob", "filter.min_logprob", "float", -1.5, 0),
    ("max_no_speech", "filter.max_no_speech", "float", 0, 1),
    ("split_chars", "llm.split_topics_over_chars", "int", 500, 10000),
    ("max_topics", "llm.max_topic_diagrams", "int", 1, 20),
    ("min_edges", "llm.min_labeled_edge_ratio", "float", 0, 1),
    ("fix_attempts", "render.max_fix_attempts", "int", 0, 5),
    ("notes_temp", "llm.notes_temperature", "float", 0, 1),
    ("diagram_temp", "llm.diagram_temperature", "float", 0, 1),
    ("num_ctx", "llm.num_ctx", "int", 2048, 65536),
    ("chunk_chars", "llm.chunk_chars", "int", 4000, 32000),
    ("whisper_model", "stt.gpu_model", "text", None, None),
    ("beam_size", "stt.beam_size", "int", 1, 10),
    ("vad_filter", "stt.vad_filter", "bool", None, None),
    ("scale", "render.scale", "int", 1, 4),
    ("image_format", "render.image_format", "choice", None, ("png", "svg")),
]


def _apply_advanced(cfg: AppConfig, payload: dict) -> None:
    """Copy advanced parameters from the payload, clamped. Never raises on bad input."""
    for key, path, kind, low, high in ADVANCED_PARAMS:
        if key not in payload:
            continue
        raw = payload[key]
        try:
            if kind == "bool":
                value = bool(raw) if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "on")
            elif kind == "choice":
                if str(raw) not in high:
                    continue
                value = str(raw)
            elif kind == "text":
                value = str(raw).strip()
            else:
                value = float(raw) if kind == "float" else int(float(raw))
                value = max(low, min(high, value))
        except (TypeError, ValueError):
            continue  # out-of-shape input keeps the default rather than failing the run
        section, field_name = path.split(".")
        setattr(getattr(cfg, section), field_name, value)


def _advanced_defaults(cfg: AppConfig) -> dict:
    out = {}
    for key, path, _kind, _low, _high in ADVANCED_PARAMS:
        section, field_name = path.split(".")
        out[key] = getattr(getattr(cfg, section), field_name)
    return out


def _config_from_payload(base_cfg: AppConfig, payload: dict) -> AppConfig:
    """Merge per-request settings. Raises ValueError on a malformed section time."""
    cfg = copy.deepcopy(base_cfg)
    _apply_advanced(cfg, payload)
    sections = payload.get("sections")
    if isinstance(sections, list):
        specs = [f"{(s.get('start') or '').strip()}-{(s.get('end') or '').strip()}"
                 for s in sections
                 if isinstance(s, dict) and ((s.get("start") or "").strip()
                                             or (s.get("end") or "").strip())]
        cfg.stt.clip_ranges = parse_range_specs(specs)  # syntax only; clamped later
    cards = payload.get("cards")
    if cards is not None and str(cards).strip().isdigit():
        cfg.llm.concept_cards = max(0, min(30, int(cards)))
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

    def _filter_blocks(cfg: AppConfig, transcript, text: str) -> str:
        """Drop pauses and low-confidence speech, asking first in 'review' mode."""
        from mictotext.segments import (apply_selection, auto_selection, describe,
                                        split_into_blocks)
        blocks = split_into_blocks(transcript.segments, cfg.filter.gap_seconds,
                                   cfg.filter.min_logprob, cfg.filter.max_no_speech)
        if len(blocks) <= 1 and not any(b.suspicious for b in blocks):
            print("  Filtro: nessuna pausa o anomalia rilevata, tengo tutto.")
            return text

        if cfg.filter.mode == "review":
            with state.lock:
                state.blocks = [b.as_dict() for b in blocks]
                state.kept_blocks = auto_selection(blocks)
                state.review_event.clear()
                state.step = "review"
            print(f"  {len(blocks)} blocchi rilevati: in attesa della tua conferma.")
            woke = state.review_event.wait(timeout=REVIEW_TIMEOUT)
            check_cancelled(state.cancel_event)  # the stop button also wakes this wait
            if not woke:
                print("  Nessuna conferma entro il tempo limite: tengo tutto.")
                with state.lock:
                    state.kept_blocks = [b.index for b in blocks]
            with state.lock:
                keep = list(state.kept_blocks)
                state.blocks = []
        else:
            keep = auto_selection(blocks)
            if not keep:
                print("  Il filtro scarterebbe tutto: lo ignoro e tengo la trascrizione intera.")
                return text

        for line in describe(blocks, keep):
            print(f"  {line}")
        _, filtered = apply_selection(blocks, keep)
        print(f"  Filtro: {len(keep)}/{len(blocks)} blocchi tenuti, "
              f"{len(text.split())} -> {len(filtered.split())} parole.")
        return filtered

    def run_pipeline(cfg: AppConfig, session_dir: Path, url: str | None = None,
                     source_path: Path | None = None) -> None:
        """[download] -> transcription -> notes -> diagram, run in a background thread.

        `source_path` is a local file, read where it is: nothing is copied into the
        session folder.
        """
        original_stdout = sys.stdout
        sys.stdout = _LogTee(original_stdout, state.append)
        try:
            notes_model = cfg.llm.notes_model
            diagram_model = cfg.llm.diagram_model or notes_model
            client = OllamaClient(cfg.llm, state.cancel_event)
            client.ensure_ready({notes_model, diagram_model})

            audio_path = source_path or (session_dir / "audio.wav")
            if url:
                from mictotext.fetch import download_audio
                state.set_step("downloading")
                print("=== Download ===")
                media = download_audio(url, session_dir, state.cancel_event)
                audio_path = media.path
                length = f" ({media.duration / 60:.1f} min)" if media.duration else ""
                print(f"  {media.title}{length}")

            state.set_step("transcribing")
            print("=== Trascrizione ===")
            # Second phase of range validation: only now does the media certainly exist,
            # so sections can be checked against its real duration. Covers every source.
            if cfg.stt.clip_ranges:
                probed = probe_media(audio_path)
                cfg.stt.clip_ranges, warnings = resolve_ranges(cfg.stt.clip_ranges,
                                                               probed.duration)
                for warning in warnings:
                    print(f"  {warning}")
            transcript = transcribe(audio_path, session_dir, cfg.stt, state.cancel_event)
            transcript_text = transcript.text
            if not transcript_text.strip():
                raise RuntimeError("Trascrizione vuota: nessun parlato rilevato.")
            (session_dir / "trascrizione.txt").write_text(transcript_text.strip() + "\n", encoding="utf-8")
            language = cfg.stt.language or transcript.language

            if cfg.filter.enabled:
                transcript_text = _filter_blocks(cfg, transcript, transcript_text)
                if not transcript_text.strip():
                    raise RuntimeError("Il filtro ha scartato tutto: nessun testo da elaborare.")
                (session_dir / "trascrizione.txt").write_text(transcript_text.strip() + "\n",
                                                              encoding="utf-8")

            state.set_step("notes")
            print("\n=== Appunti strutturati ===")
            notes_md = generate_notes(transcript_text, client, notes_model, cfg.llm, language_name(language))
            (session_dir / "appunti.md").write_text(notes_md, encoding="utf-8")

            # Renamed as soon as the title exists, so the folder is well named even if
            # the diagrams are then cancelled or fail. Everything below writes into the
            # new path, and the page is told the new name for its /files/ URLs.
            session_dir = rename_session(session_dir, notes_md, cfg.llm.topic)
            with state.lock:
                state.session_dir = session_dir
                state.session_id = session_dir.name
            with state.lock:
                state.notes_md = notes_md

            state.set_step("diagram")
            print("\n=== Schema Mermaid ===")
            renderer = build_renderer(cfg.render)
            from mictotext.diagram import generate_topic_diagrams, plan_diagrams
            main_source, topics = plan_diagrams(notes_md, cfg.llm)
            if topics:
                print(f"  Appunti lunghi: lo schema principale mappa {len(topics)} argomenti, "
                      f"ognuno con il proprio schema di dettaglio.")
            result = generate_diagram(main_source, client, diagram_model, cfg.llm, cfg.render,
                                      renderer, session_dir, language_name(language))
            if topics:
                extra = generate_topic_diagrams(topics, client, diagram_model, cfg.llm, cfg.render,
                                                renderer, session_dir, language_name(language))
                with state.lock:
                    state.topic_names = [p.relative_to(session_dir).as_posix() for p in extra]

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
        except Cancelled:
            # Not an error: the user asked for it. Files already written stay on disk.
            with state.lock:
                state.step = "cancelled"
            print(f"\n=== Interrotto === (i file gia' prodotti restano in {session_dir.name})")
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
            **_advanced_defaults(base_cfg),
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
            return jsonify({"db": -60.0, "elapsed": 0.0, "paused": False})
        return jsonify({"db": -60.0 if recorder.paused else recorder.level_db,
                        "elapsed": recorder.elapsed, "paused": recorder.paused})

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
            state.topic_names = []
            state.blocks = []
            state.kept_blocks = []
            state.review_event.clear()
            state.cancel_event.clear()   # a stop from the last run must not kill this one

        payload = request.get_json(silent=True) or {}
        try:
            # Validated here too, so a bad section fails before anything is recorded
            # rather than at /api/stop, when the audio already exists.
            _config_from_payload(base_cfg, payload)
        except ValueError as exc:
            with state.lock:
                state.step = "idle"
            return jsonify({"error": str(exc)}), 400

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

    @app.post("/api/record/pause")
    def api_pause():
        with state.lock:
            recorder = state.recorder
        if recorder is None:
            return jsonify({"error": "Nessuna registrazione in corso."}), 409
        changed = recorder.pause() if not recorder.paused else False
        if changed:
            state.append(f"Registrazione in pausa a {recorder.elapsed:.0f}s.")
        return jsonify({"paused": recorder.paused})

    @app.post("/api/record/resume")
    def api_resume():
        with state.lock:
            recorder = state.recorder
        if recorder is None:
            return jsonify({"error": "Nessuna registrazione in corso."}), 409
        changed = recorder.resume() if recorder.paused else False
        if changed:
            state.append("Registrazione ripresa.")
        return jsonify({"paused": recorder.paused})

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
        try:
            cfg = _config_from_payload(base_cfg, payload)
        except ValueError as exc:
            # The recording is already on disk: say so, so it does not feel lost.
            message = f"{exc} La registrazione e' salva in {session_dir.name}/audio.wav."
            with state.lock:
                state.step = "error"
                state.error = message
            return jsonify({"error": message}), 400

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
            state.topic_names = []
            state.blocks = []
            state.kept_blocks = []
            state.review_event.clear()
            state.cancel_event.clear()   # a stop from the last run must not kill this one

        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir = base_cfg.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        with state.lock:
            state.session_dir = session_dir
            state.session_id = session_id
        state.append(f"Sessione {session_id}: {url}")

        try:
            cfg = _config_from_payload(base_cfg, payload)
        except ValueError as exc:
            state.set_step("idle")
            return jsonify({"error": str(exc)}), 400
        threading.Thread(target=run_pipeline, args=(cfg, session_dir, url), daemon=True).start()
        return jsonify({"session_id": session_id})

    @app.post("/api/cancel")
    def api_cancel():
        """Stop whatever is running. Each stage reacts in its own way."""
        with state.lock:
            step = state.step
            recorder = state.recorder
            if step not in _ACTIVE_STEPS:
                return jsonify({"error": "Nessuna operazione in corso."}), 409

        if step == "recording" and recorder is not None:
            # No pipeline thread exists yet: stop capturing and do not process it.
            duration, _peak, _overflow = recorder.stop()
            with state.lock:
                state.recorder = None
                state.step = "cancelled"
            state.append(f"Registrazione annullata dopo {duration:.0f}s: non verra' elaborata.")
            return jsonify({"cancelled": step})

        state.cancel_event.set()
        state.review_event.set()  # a pipeline waiting for block confirmation must wake up
        state.append("Interruzione richiesta...")
        return jsonify({"cancelling": step})

    @app.post("/api/review")
    def api_review():
        """Confirm which speech blocks to keep, releasing the waiting pipeline."""
        payload = request.get_json(silent=True) or {}
        with state.lock:
            if state.step != "review":
                return jsonify({"error": "Nessuna conferma in attesa."}), 409
            available = {b["index"] for b in state.blocks}
        keep = sorted({int(i) for i in payload.get("keep", []) if str(i).lstrip("-").isdigit()}
                      & available)
        if not keep:
            return jsonify({"error": "Tieni almeno un blocco: senza testo la pipeline "
                                     "non puo' proseguire."}), 400
        with state.lock:
            state.kept_blocks = keep
        state.review_event.set()
        return jsonify({"kept": keep})

    @app.post("/api/probe")
    def api_probe():
        """Inspect a local media file so the user can see its duration before running."""
        payload = request.get_json(silent=True) or {}
        raw = (payload.get("path") or "").strip()
        if not raw:
            return jsonify({"error": "Inserisci un percorso."}), 400
        info = probe_media(clean_path(raw))
        # 200 even on failure: the page renders one info panel either way.
        return jsonify({
            "ok": info.ok,
            "error": info.error,
            "path": str(info.path),
            "duration": info.duration,
            "duration_label": format_time(info.duration) if info.duration else None,
            "has_audio": info.has_audio,
            "has_video": info.has_video,
        })

    @app.post("/api/file")
    def api_file():
        """Start the pipeline from a local file, read in place."""
        payload = request.get_json(silent=True) or {}
        path = clean_path(payload.get("path") or "")
        info = probe_media(path)
        if not info.ok:
            return jsonify({"error": info.error}), 400

        with state.lock:
            if state.step in _ACTIVE_STEPS:
                return jsonify({"error": "Una sessione e' gia' in corso."}), 409
        try:
            cfg = _config_from_payload(base_cfg, payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        with state.lock:
            state.step = "transcribing"
            state.log = []
            state.error = None
            state.notes_md = None
            state.image_name = None
            state.html_name = None
            state.card_names = []
            state.topic_names = []
            state.blocks = []
            state.kept_blocks = []
            state.review_event.clear()
            state.cancel_event.clear()   # a stop from the last run must not kill this one

        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir = base_cfg.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        with state.lock:
            state.session_dir = session_dir
            state.session_id = session_id
        state.append(f"Sessione {session_id}: {path.name} "
                     f"({format_time(info.duration) if info.duration else 'durata ignota'})")

        threading.Thread(target=run_pipeline, args=(cfg, session_dir),
                         kwargs={"source_path": path}, daemon=True).start()
        return jsonify({"session_id": session_id})

    @app.post("/api/regenerate")
    def api_regenerate():
        """Revise one existing diagram or card from structured user feedback."""
        payload = request.get_json(silent=True) or {}
        target = (payload.get("target") or "").strip()  # path relative to the session folder
        with state.lock:
            if state.step in _ACTIVE_STEPS:
                return jsonify({"error": "Una sessione e' gia' in corso."}), 409
            session_dir, notes_md = state.session_dir, state.notes_md
        if not session_dir or not notes_md:
            return jsonify({"error": "Nessuna sessione da rigenerare."}), 409
        if not target or ".." in target or target.startswith("/"):
            return jsonify({"error": "Destinazione non valida."}), 400

        image_path = (session_dir / target).resolve()
        if not image_path.is_file() or session_dir.resolve() not in image_path.parents:
            return jsonify({"error": "File non trovato."}), 404

        try:
            cfg = _config_from_payload(base_cfg, payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        kind = "card" if target.startswith("concetti/") else "diagram"
        # The card prompt needs to know which concept it is explaining.
        concept = (image_path.stem.split("-", 1)[-1].replace("-", " ")
                   if kind == "card" else "")

        def worker() -> None:
            original_stdout = sys.stdout
            sys.stdout = _LogTee(original_stdout, state.append)
            try:
                from mictotext.diagram import regenerate_from_feedback
                model = cfg.llm.diagram_model or cfg.llm.notes_model
                client = OllamaClient(cfg.llm, state.cancel_event)
                print(f"=== Rigenerazione: {target} ===")
                ok, message = regenerate_from_feedback(
                    image_path.with_suffix(".mmd"), image_path, notes_md,
                    payload.get("feedback") or {}, kind, client, model, cfg.llm, cfg.render,
                    build_renderer(cfg.render), language_name(cfg.stt.language), concept)
                print(f"  {message}")
                state.set_step("done" if ok else "error")
                if not ok:
                    with state.lock:
                        state.error = message
            except Cancelled:
                # Every model call happens before anything is written, so the previous
                # diagram is untouched: back to showing the session as it was.
                print("  Rigenerazione interrotta: lo schema precedente e' rimasto invariato.")
                state.set_step("done")
            except Exception as exc:  # noqa: BLE001 - surfaced to the browser
                with state.lock:
                    state.step = "error"
                    state.error = str(exc)
                print(f"\nERRORE: {exc}")
            finally:
                sys.stdout = original_stdout

        state.cancel_event.clear()
        state.set_step("regenerating")
        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True})

    @app.get("/files/<session_id>/<path:filename>")
    def files(session_id: str, filename: str):
        # Named folders ("Titolo_21_09_26") replaced timestamp-only names, so the old
        # regex would now 404 every image. The replacement is an equally strict
        # traversal guard: one plain folder name, directly inside the output root.
        if not is_session_name(base_cfg.output_root, session_id):
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
  select, input[type=text], input[type=number] {
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
  button#pauseBtn {
    background: #fff; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 18px; font-size: 0.95rem; cursor: pointer;
  }
  button#pauseBtn:hover { background: #eef2f6; }
  button#pauseBtn.paused { background: #FFF3D6; border-color: #D6AE55; }
  .meter { flex: 1; min-width: 160px; }
  .meter-bar {
    height: 10px; background: #eef1f4; border-radius: 5px; overflow: hidden; border: 1px solid var(--border);
  }
  .meter-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #79b791, #d6ae55, #d98a8a); }
  .timer { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.85rem; min-width: 42px; }
  .step-label { margin-top: 10px; font-size: 0.9rem; color: var(--muted); }
  .stop-row { display: flex; align-items: center; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
  .stop-row .hint { margin-top: 0; }
  button#stopBtn {
    background: #fff; color: #a04545; border: 1px solid var(--err); border-radius: 8px;
    padding: 8px 16px; font-size: 0.9rem; font-weight: 600; cursor: pointer;
  }
  button#stopBtn:hover:not(:disabled) { background: #fdf4f4; }
  button#stopBtn:disabled { opacity: .55; cursor: default; }
  .url-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .url-row input { flex: 1; min-width: 220px; }
  button#urlBtn {
    background: var(--accent); color: #fff; border: none; border-radius: 8px;
    padding: 8px 18px; font-size: 0.9rem; cursor: pointer; font-weight: 600; white-space: nowrap;
  }
  button#urlBtn:hover:not(:disabled) { background: var(--accent-dark); }
  button#urlBtn:disabled { background: #b9c3cc; cursor: default; }
  .hint { font-size: 0.78rem; color: var(--muted); margin-top: 8px; }
  .hint code { background: #eef2f6; padding: 1px 4px; border-radius: 3px; }
  .full-width { grid-column: 1 / -1; }
  .file-info { font-size: 0.82rem; margin-top: 8px; }
  .file-info.ok { color: #3f7a55; }
  .file-info.err { color: #a04545; }
  .sec-row { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
  .sec-row input { flex: 1; min-width: 80px; }
  .sec-row input.bad { border-color: var(--err); background: #fdf4f4; }
  .sec-row .sep { color: var(--muted); font-size: 0.85rem; }
  .sec-row button, .link-btn {
    background: #fff; border: 1px solid var(--border); border-radius: 6px;
    padding: 5px 10px; font-size: 0.8rem; cursor: pointer; color: var(--text);
  }
  .sec-row button:hover, .link-btn:hover { background: #eef2f6; }
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
  /* Advanced panel: first <details> and first range inputs in the app, so both need
     their own rules (the shared input rule does not cover type=range). */
  details.adv { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 0 20px; margin-bottom: 16px; }
  details.adv > summary {
    cursor: pointer; padding: 18px 0; font-size: 1rem; font-weight: 600; list-style: none;
  }
  details.adv > summary::-webkit-details-marker { display: none; }
  details.adv > summary::before { content: '\\25B8'; display: inline-block; margin-right: 8px;
    transition: transform .15s; color: var(--muted); }
  details.adv[open] > summary::before { transform: rotate(90deg); }
  details.adv[open] { padding-bottom: 18px; }
  .adv-group { margin-bottom: 18px; }
  .adv-group > h3 {
    font-size: 0.82rem; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); margin: 0 0 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border);
  }
  .adv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
  .adv-item .row { display: flex; align-items: center; gap: 10px; }
  .adv-item input[type=range] {
    flex: 1; -webkit-appearance: none; appearance: none; height: 4px; border-radius: 2px;
    background: #dde3ea; outline: none;
  }
  .adv-item input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%;
    background: var(--accent); cursor: pointer; border: 2px solid #fff;
    box-shadow: 0 0 0 1px var(--border);
  }
  .adv-item input[type=range]::-moz-range-thumb {
    width: 15px; height: 15px; border-radius: 50%; background: var(--accent);
    cursor: pointer; border: 2px solid #fff;
  }
  .adv-value {
    font-variant-numeric: tabular-nums; font-size: 0.82rem; color: var(--text);
    min-width: 58px; text-align: right;
  }
  .adv-item.disabled > label, .adv-item.disabled .adv-help,
  .adv-item.disabled .row { opacity: .45; }
  .adv-dep { font-size: 0.72rem; color: #9a6b3f; margin-top: 4px; font-style: italic; }
  .adv-help { font-size: 0.73rem; color: var(--muted); margin-top: 5px; line-height: 1.45; }
  .adv-help b { color: #5a6472; font-weight: 600; }
  .adv-actions { display: flex; justify-content: flex-end; margin-top: 4px; }
  .switch { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; }
  .switch input { width: auto; }
  /* Review card */
  .blk-row {
    display: flex; align-items: flex-start; gap: 10px; padding: 9px 10px;
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 7px; background: #fff;
  }
  .blk-row.susp { background: #fdf7f2; border-color: #e0c3a8; }
  .blk-row input { width: auto; margin-top: 3px; }
  .blk-meta { font-size: 0.78rem; color: var(--muted); }
  .blk-text { font-size: 0.86rem; color: var(--text); }
  .blk-tag { font-size: 0.7rem; color: #9a6b3f; }
  #revise {
    display: none; position: fixed; inset: 0; z-index: 1100;
    background: rgba(31,41,55,0.45); overflow-y: auto; padding: 28px 16px;
  }
  #revise.open { display: block; }
  .revise-box {
    max-width: 1080px; margin: 0 auto; background: #fff; border-radius: 10px;
    border: 1px solid var(--border); padding: 20px 22px;
  }
  /* Form and the thing being changed, side by side: judging "what is wrong" from
     memory is much harder than reading it off the picture. */
  .revise-split { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px; }
  @media (max-width: 860px) { .revise-split { grid-template-columns: 1fr; } }
  .revise-preview { min-width: 0; }
  .revise-preview img {
    width: 100%; max-height: 62vh; object-fit: contain; object-position: top;
    border: 1px solid var(--border); border-radius: 8px; background: #fff; cursor: zoom-in;
  }
  .revise-box h2 { margin: 0 0 14px; font-size: 1rem; }
  .revise-box label { margin-top: 12px; font-weight: 600; color: var(--text); }
  .revise-box textarea {
    width: 100%; padding: 8px 9px; border: 1px solid var(--border); border-radius: 6px;
    font-family: inherit; font-size: 0.88rem; resize: vertical;
  }
  .revise-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 16px; }
  .revise-actions button {
    background: #fff; border: 1px solid var(--border); border-radius: 8px;
    padding: 9px 18px; font-size: 0.9rem; cursor: pointer;
  }
  .revise-actions button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .revise-actions button.primary:hover { background: var(--accent-dark); }
  .img-actions { margin-top: 6px; }
  .img-actions button {
    background: #fff; border: 1px solid var(--border); border-radius: 6px;
    padding: 4px 11px; font-size: 0.78rem; cursor: pointer; color: var(--accent-dark);
  }
  .img-actions button:hover { background: #eef2f6; }
  /* Full-screen viewer: diagrams are often far taller than the page, so they need
     zooming and panning rather than a scaled-down thumbnail. */
  #viewer {
    /* Above the revise modal (1100): the preview there opens into this viewer. */
    display: none; position: fixed; inset: 0; z-index: 1200;
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
      <div class="full-width">
        <label>Sezioni da trascrivere (vuoto = tutto)</label>
        <div id="sections"></div>
        <button id="addSection" type="button" class="link-btn">+ Aggiungi sezione</button>
        <div class="hint">Formati: <code>90</code>, <code>1:30</code>, <code>01:02:03</code>.
          Valgono per microfono, URL e file locale.<br>
          Con le sezioni attive Whisper disattiva il filtro VAD, quindi i silenzi lunghi
          dentro le sezioni vengono comunque elaborati.<br>
          Il file viene letto per intero: le sezioni riducono il tempo di trascrizione,
          non quello di lettura.</div>
      </div>
      <div>
        <label for="cards">Schede concetto (0 = nessuna)</label>
        <input type="number" id="cards" min="0" max="30" step="1">
        <div class="hint">Spiegazioni discorsive dei concetti chiave, una per scheda.
          Su contenuti lunghi alzalo: ogni scheda costa una generazione.</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Registrazione</h2>
    <div class="record-row">
      <button id="recordBtn">Avvia registrazione</button>
      <button id="pauseBtn" style="display:none">Pausa</button>
      <div class="meter">
        <div class="meter-bar"><div class="meter-fill" id="meterFill"></div></div>
      </div>
      <div class="timer" id="timer">00:00</div>
    </div>
    <div class="step-label" id="stepLabel">Pronto.</div>
    <div class="stop-row" id="stopRow" style="display:none">
      <button id="stopBtn" type="button">&#9632; Interrompi</button>
      <span class="hint" id="stopHint">Ferma l'operazione in corso. I file gia' prodotti restano.</span>
    </div>
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
    <h2>Oppure: da un file sul PC</h2>
    <div class="url-row">
      <input type="text" id="fileInput" placeholder="C:\Users\...\lezione.mp4">
      <button id="probeBtn">Verifica file</button>
      <button id="fileBtn">Trascrivi dal file</button>
    </div>
    <div id="fileInfo" class="file-info"></div>
    <div class="hint">Video o audio. Il file resta dov'e': nessuna copia, nessuna estrazione.
      Puoi incollare il percorso con le virgolette di &laquo;Copia come percorso&raquo;.</div>
  </div>

  <div class="card" id="reviewCard" style="display:none">
    <h2>Blocchi rilevati &mdash; scegli cosa tenere</h2>
    <div id="blocksWrap"></div>
    <div class="revise-actions">
      <button id="blocksAll" type="button">Tieni tutto</button>
      <button id="blocksAuto" type="button">Solo i consigliati</button>
      <button id="blocksGo" class="primary" type="button">Continua</button>
    </div>
    <div class="hint">La pipeline e' in attesa. I blocchi evidenziati sono quelli che l'app
      ritiene dubbi: pause lunghe o parlato che il modello ha riconosciuto con poca sicurezza.</div>
  </div>

  <details class="adv">
    <summary>Controllo avanzato</summary>
    <div id="advPanel"></div>
    <div class="adv-actions">
      <button id="advReset" type="button" class="link-btn">Ripristina valori predefiniti</button>
    </div>
  </details>

  <div class="card">
    <h2>Log</h2>
    <pre id="log"></pre>
  </div>

  <div class="card" id="results">
    <h2>Appunti</h2>
    <pre id="notesOutput"></pre>
    <h2>Schema</h2>
    <div id="imageWrap"></div>
    <div id="topicsSection" style="display:none">
      <h2>Schemi per argomento</h2>
      <div id="topicsWrap" class="cards-grid"></div>
    </div>
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

<div id="revise">
  <div class="revise-box">
    <h2>Modifica: <span id="reviseTarget"></span></h2>
    <div class="revise-split">
    <div class="revise-form">
    <label for="rvWrong">1. Cosa non va bene</label>
    <textarea id="rvWrong" rows="2" placeholder="Es: troppi nodi, le relazioni non si capiscono..."></textarea>
    <label for="rvMissing">2. Argomenti mancanti</label>
    <textarea id="rvMissing" rows="2" placeholder="Argomenti da aggiungere (solo se presenti negli appunti)"></textarea>
    <label for="rvCorrect">3. Argomenti corretti (da conservare)</label>
    <textarea id="rvCorrect" rows="2" placeholder="Cosa gia' funziona e non va toccato"></textarea>
    <label for="rvOptions">4. Indicazioni per migliorare</label>
    <textarea id="rvOptions" rows="2" placeholder="Es: raggruppa per fase, evidenzia le cause..."></textarea>
    </div>
    <div class="revise-preview">
      <img id="revisePreview" alt="">
      <div class="hint">Clicca l'immagine per ingrandirla.</div>
    </div>
    </div>
    <div class="revise-actions">
      <button id="reviseCancel">Annulla</button>
      <button id="reviseSubmit" class="primary">Rigenera</button>
    </div>
    <div class="hint">I campi vuoti valgono come "nessuna indicazione". Se la nuova versione non
      si renderizza, viene mantenuta quella attuale.</div>
  </div>
</div>

<div id="viewer">
  <div id="viewerBar">
    <span class="title" id="viewerTitle"></span>
    <button id="viewerPrev" title="Precedente (&larr;)">&larr;</button>
    <button id="viewerNext" title="Successivo (&rarr;)">&rarr;</button>
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
  if (defaults.cards !== undefined) document.getElementById('cards').value = defaults.cards;

  // Server defaults first, then whatever the user last experimented with.
  advDefaults = {};
  ADV_KEYS.forEach(k => { if (defaults[k] !== undefined) advDefaults[k] = defaults[k]; });
  setAdvValues(advDefaults);
  try {
    const saved = JSON.parse(localStorage.getItem(ADV_STORE) || '{}');
    setAdvValues(saved);
  } catch (e) {}
  updateAdvDeps();
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

// --- Interruzione ---
const ACTIVE_STEPS = ['recording', 'downloading', 'transcribing', 'notes', 'diagram',
                      'cards', 'regenerating', 'review'];
let currentStep = 'idle';

function updateStopButton(step) {
  currentStep = step;
  const row = document.getElementById('stopRow');
  const btn = document.getElementById('stopBtn');
  const active = ACTIVE_STEPS.includes(step);
  row.style.display = active ? 'flex' : 'none';
  if (!active) { btn.disabled = false; btn.innerHTML = '&#9632; Interrompi'; }
  // Recording has its own "Ferma" that stops AND processes: say plainly this one discards.
  document.getElementById('stopHint').textContent = step === 'recording'
    ? "Annulla la registrazione senza elaborarla."
    : "Ferma l'operazione in corso. I file gia' prodotti restano.";
}

async function cancelRun() {
  const question = currentStep === 'recording'
    ? "Annullare la registrazione? L'audio non verra' elaborato."
    : currentStep === 'regenerating'
      ? "Interrompere la rigenerazione? Lo schema attuale resta invariato."
      : "Interrompere l'operazione in corso? I file gia' prodotti restano sul disco.";
  if (!confirm(question)) return;

  const btn = document.getElementById('stopBtn');
  btn.disabled = true;
  btn.textContent = 'Interruzione in corso...';
  const res = await fetch('/api/cancel', {method: 'POST'});
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore.'); updateStopButton('idle'); return; }
  if (data.cancelled === 'recording') {
    stopLevelPolling();   // recording is dropped at once: no pipeline to wait for
  }
  // Otherwise the status poll picks up the "cancelled" state when the stage lets go.
}

let isPaused = false;

async function togglePause() {
  const res = await fetch(isPaused ? '/api/record/resume' : '/api/record/pause', {method: 'POST'});
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore.'); return; }
  setPausedUI(!!data.paused);
}

function setPausedUI(paused) {
  isPaused = paused;
  const btn = document.getElementById('pauseBtn');
  btn.textContent = paused ? 'Riprendi' : 'Pausa';
  btn.classList.toggle('paused', paused);
  document.getElementById('stepLabel').textContent =
    paused ? 'In pausa.' : 'Registrazione in corso...';
}

function setUIState(mode) {
  const btn = document.getElementById('recordBtn');
  const urlBtn = document.getElementById('urlBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  pauseBtn.style.display = (mode === 'recording') ? '' : 'none';
  if (mode !== 'recording') { isPaused = false; }
  btn.classList.remove('recording');
  // Sources are mutually exclusive; `busy` stops validateSections re-enabling them.
  ['urlBtn', 'fileBtn', 'probeBtn'].forEach(id => {
    const b = document.getElementById(id);
    if (!b) return;
    if (mode === 'idle') { delete b.dataset.busy; b.disabled = false; }
    else { b.dataset.busy = '1'; b.disabled = true; }
  });
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
    cards: document.getElementById('cards').value,
    sections: currentSections(),
    ...advValues()
  };
}

// --- Controllo avanzato ---
// Rispecchia ADVANCED_PARAMS lato Python: stessa chiave, stessi limiti.
const ADV_GROUPS = [
  {title: 'Contesto della registrazione', items: [
    {k:'topic', t:'text', label:'Argomento', ph:'es. sistemi operativi',
     help:'Dice al modello di cosa si parla. <b>Compilalo</b> per appunti più mirati e meno divagazioni; <b>lascialo vuoto</b> per non influenzarlo.'},
    {k:'subtopics', t:'area', label:'Sottoargomenti', ph:'kernel, gestione memoria, permessi',
     dep:{test:() => advValue('topic').trim() !== '', why:'Senza un argomento non hanno effetto.'},
     help:'Gli aspetti da privilegiare. <b>Elencane di più</b> per guidare la scelta delle schede concetto; <b>lascia vuoto</b> se non vuoi vincoli.'}
  ]},
  {title: 'Filtro pause e rumore', items: [
    {k:'filter_enabled', t:'bool', label:'Filtro attivo',
     help:'<b>Acceso</b>: individua pause e parlato incerto. <b>Spento</b>: trascrive tutto senza toccare nulla.'},
    {k:'filter_mode', t:'choice', label:'Modalità', opts:[['review','Proponi e conferma'],['auto','Applica subito']],
     dep:{on:'filter_enabled', why:'Attiva il filtro per sceglierla.'},
     help:'<b>Proponi</b>: ti mostra i blocchi e attende. <b>Applica</b>: scarta da solo, più rapido ma senza rete di sicurezza.'},
    {k:'gap_seconds', t:'range', label:'Pausa minima', min:2, max:180, step:1, unit:'s',
     dep:{on:'filter_enabled', why:'Attiva il filtro per regolarla.'},
     help:'Silenzio che separa due blocchi. <b>Alza</b> per spezzare solo sulle pause lunghe; <b>abbassa</b> per dividere anche sui silenzi brevi.'},
    {k:'min_logprob', t:'range', label:'Confidenza minima', min:-1.5, max:0, step:0.05,
     dep:{on:'filter_enabled', why:'Attiva il filtro per regolarla.'},
     help:'Quanto Whisper deve essere sicuro. <b>Alza</b> (verso 0) per scartare più parlato dubbio; <b>abbassa</b> per tenere quasi tutto.'},
    {k:'max_no_speech', t:'range', label:'Soglia non-parlato', min:0, max:1, step:0.05,
     dep:{on:'filter_enabled', why:'Attiva il filtro per regolarla.'},
     help:'Tolleranza sul rumore senza voce. <b>Alza</b> per essere permissivo; <b>abbassa</b> per scartare più aggressivamente.'}
  ]},
  {title: 'Qualità degli schemi', items: [
    {k:'split_chars', t:'range', label:'Soglia divisione', min:500, max:10000, step:250, unit:' car.',
     help:'Oltre questa lunghezza gli appunti si dividono in più schemi. <b>Alza</b> per restare su uno schema unico; <b>abbassa</b> per ottenere prima gli schemi per argomento.'},
    {k:'max_topics', t:'range', label:'Max schemi per argomento', min:1, max:20, step:1,
     help:'<b>Alza</b> per coprire ogni sottotema di una lezione lunga; <b>abbassa</b> per limitare tempo e numero di immagini.'},
    {k:'min_edges', t:'range', label:'Archi etichettati minimi', min:0, max:1, step:0.05,
     help:'Quota di frecce che devono avere un\'etichetta. <b>Alza</b> per pretendere schemi più esplicativi (più revisioni, più lento); <b>abbassa</b> per accettare il primo risultato.'},
    {k:'fix_attempts', t:'range', label:'Tentativi di riparazione', min:0, max:5, step:1,
     help:'Quante volte richiedere una correzione se il render fallisce. <b>Alza</b> per insistere; <b>abbassa</b> per fallire subito.'}
  ]},
  {title: 'Generazione del testo', items: [
    {k:'notes_temp', t:'range', label:'Temperatura appunti', min:0, max:1, step:0.05,
     help:'<b>Alza</b> per un testo più vario e creativo; <b>abbassa</b> per appunti più fedeli e ripetibili.'},
    {k:'diagram_temp', t:'range', label:'Temperatura schemi', min:0, max:1, step:0.05,
     help:'<b>Alza</b> per strutture più originali; <b>abbassa</b> per sintassi Mermaid più affidabile.'},
    {k:'num_ctx', t:'choice', label:'Finestra di contesto', opts:[['4096','4k'],['8192','8k'],['16384','16k'],['32768','32k']],
     help:'Quanto testo il modello vede insieme. <b>Alza</b> per trascrizioni lunghe; <b>abbassa</b> per consumare meno VRAM.'},
    {k:'chunk_chars', t:'range', label:'Blocchi trascrizione', min:4000, max:32000, step:1000, unit:' car.',
     help:'Dimensione delle parti su trascrizioni lunghe. <b>Alza</b> per dare più contesto per parte; <b>abbassa</b> se il modello perde pezzi.'}
  ]},
  {title: 'Trascrizione e resa', items: [
    {k:'whisper_model', t:'choice', label:'Modello Whisper',
     opts:[['tiny','tiny'],['base','base'],['small','small'],['medium','medium'],['large-v3','large-v3'],['large-v3-turbo','large-v3-turbo']],
     help:'<b>Piu\' grande</b>: trascrizione più accurata ma più lenta e più VRAM. <b>Piu\' piccolo</b>: veloce, adatto a prove rapide.'},
    {k:'beam_size', t:'range', label:'Ampiezza ricerca', min:1, max:10, step:1,
     help:'<b>Alza</b> per una trascrizione più accurata ma più lenta; <b>abbassa</b> per andare più veloce.'},
    {k:'vad_filter', t:'bool', label:'Filtro VAD',
     dep:{test:() => currentSections().length === 0, why:'Le sezioni lo disattivano comunque.'},
     help:'Scarta i silenzi prima di trascrivere. <b>Acceso</b> riduce le allucinazioni sulle pause. Viene ignorato quando usi le sezioni.'},
    {k:'scale', t:'range', label:'Risoluzione immagini', min:1, max:4, step:1, unit:'x',
     dep:{test:() => advValue('image_format') !== 'svg', why:'Con SVG la risoluzione non conta.'},
     help:'<b>Alza</b> per immagini più nitide da ingrandire; <b>abbassa</b> per file più leggeri.'},
    {k:'image_format', t:'choice', label:'Formato', opts:[['png','PNG'],['svg','SVG']],
     help:'<b>PNG</b>: immagine pronta all\'uso. <b>SVG</b>: vettoriale, ingrandibile senza perdita (la risoluzione non conta).'}
  ]}
];

const ADV_KEYS = ADV_GROUPS.flatMap(g => g.items.map(i => i.k));
const ADV_STORE = 'mictotext.advanced';

function buildAdvancedPanel() {
  const panel = document.getElementById('advPanel');
  panel.innerHTML = '';
  ADV_GROUPS.forEach(group => {
    const box = document.createElement('div');
    box.className = 'adv-group';
    const h = document.createElement('h3');
    h.textContent = group.title;
    box.appendChild(h);
    const grid = document.createElement('div');
    grid.className = 'adv-grid';
    group.items.forEach(item => grid.appendChild(buildAdvControl(item)));
    box.appendChild(grid);
    panel.appendChild(box);
  });
}

function buildAdvControl(item) {
  const cell = document.createElement('div');
  cell.className = 'adv-item';
  const label = document.createElement('label');
  label.textContent = item.label;
  label.setAttribute('for', 'adv_' + item.k);
  cell.appendChild(label);

  const row = document.createElement('div');
  row.className = 'row';
  let input;
  if (item.t === 'range') {
    input = document.createElement('input');
    input.type = 'range';
    input.min = item.min; input.max = item.max; input.step = item.step;
    const out = document.createElement('span');
    out.className = 'adv-value';
    out.id = 'advval_' + item.k;
    input.addEventListener('input', () => {
      out.textContent = input.value + (item.unit || '');
      saveAdvanced();
    });
    row.appendChild(input); row.appendChild(out);
  } else if (item.t === 'choice') {
    input = document.createElement('select');
    item.opts.forEach(([v, t]) => input.appendChild(new Option(t, v)));
    input.addEventListener('change', saveAdvanced);
    row.appendChild(input);
  } else if (item.t === 'bool') {
    const wrap = document.createElement('label');
    wrap.className = 'switch';
    input = document.createElement('input');
    input.type = 'checkbox';
    input.addEventListener('change', saveAdvanced);
    wrap.appendChild(input);
    wrap.appendChild(document.createTextNode('attivo'));
    row.appendChild(wrap);
  } else if (item.t === 'area') {
    input = document.createElement('textarea');
    input.rows = 2; input.placeholder = item.ph || '';
    input.style.width = '100%';
    input.addEventListener('input', saveAdvanced);
    row.appendChild(input);
  } else {
    input = document.createElement('input');
    input.type = 'text'; input.placeholder = item.ph || '';
    input.addEventListener('input', saveAdvanced);
    row.appendChild(input);
  }
  input.id = 'adv_' + item.k;
  cell.appendChild(row);

  const help = document.createElement('div');
  help.className = 'adv-help';
  help.innerHTML = item.help;
  cell.appendChild(help);

  if (item.dep) {
    const note = document.createElement('div');
    note.className = 'adv-dep';
    note.id = 'advdep_' + item.k;
    note.style.display = 'none';
    cell.appendChild(note);
  }
  return cell;
}

function advValue(k) {
  const el = document.getElementById('adv_' + k);
  if (!el) return '';
  return (el.type === 'checkbox') ? el.checked : el.value;
}

function advValues() {
  const out = {};
  ADV_KEYS.forEach(k => { out[k] = advValue(k); });
  return out;
}

// A control whose value the pipeline would ignore is disabled, with the reason shown:
// a greyed-out box with no explanation just looks broken.
function updateAdvDeps() {
  ADV_GROUPS.forEach(g => g.items.forEach(item => {
    if (!item.dep) return;
    const cell = document.getElementById('adv_' + item.k);
    if (!cell) return;
    const active = item.dep.test ? item.dep.test() : !!advValue(item.dep.on);
    cell.disabled = !active;
    const box = cell.closest('.adv-item');
    if (box) box.classList.toggle('disabled', !active);
    const note = document.getElementById('advdep_' + item.k);
    if (note) {
      note.textContent = active ? '' : item.dep.why;
      note.style.display = active ? 'none' : 'block';
    }
  }));
}

function setAdvValues(values) {
  ADV_GROUPS.forEach(g => g.items.forEach(item => {
    const el = document.getElementById('adv_' + item.k);
    if (!el || values[item.k] === undefined || values[item.k] === null) return;
    if (el.type === 'checkbox') el.checked = !!values[item.k];
    else el.value = values[item.k];
    if (item.t === 'range') {
      const out = document.getElementById('advval_' + item.k);
      if (out) out.textContent = el.value + (item.unit || '');
    }
  }));
}

function saveAdvanced() {
  updateAdvDeps();
  // Experimenting means changing one knob at a time without retyping the other twenty.
  try { localStorage.setItem(ADV_STORE, JSON.stringify(advValues())); } catch (e) {}
}

let advDefaults = {};
function resetAdvanced() {
  try { localStorage.removeItem(ADV_STORE); } catch (e) {}
  setAdvValues(advDefaults);
  updateAdvDeps();
}

// --- Conferma dei blocchi ---
function renderBlocks(blocks) {
  const card = document.getElementById('reviewCard');
  const wrap = document.getElementById('blocksWrap');
  if (!blocks || !blocks.length) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  wrap.innerHTML = '';
  blocks.forEach(b => {
    const row = document.createElement('div');
    row.className = 'blk-row' + (b.suspicious ? ' susp' : '');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = !b.suspicious; cb.dataset.index = b.index;
    const body = document.createElement('div');
    const meta = document.createElement('div');
    meta.className = 'blk-meta';
    const pause = b.gap_before ? ` &middot; dopo ${Math.round(b.gap_before)}s di pausa` : '';
    const tag = b.suspicious ? ' <span class="blk-tag">&#9888; dubbio</span>' : '';
    meta.innerHTML = `${fmtTime(b.start)}&ndash;${fmtTime(b.end)} &middot; ${Math.round(b.duration)}s${pause}${tag}`;
    const txt = document.createElement('div');
    txt.className = 'blk-text';
    txt.textContent = b.preview || '(nessun testo)';
    body.appendChild(meta); body.appendChild(txt);
    row.appendChild(cb); row.appendChild(body);
    wrap.appendChild(row);
  });
}

function fmtTime(s) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

function setBlocks(all) {
  document.querySelectorAll('#blocksWrap input[type=checkbox]').forEach((cb, i) => {
    cb.checked = all === null ? !currentBlocks[i].suspicious : all;
  });
}

let currentBlocks = [];

async function submitBlocks() {
  const keep = [...document.querySelectorAll('#blocksWrap input[type=checkbox]')]
    .filter(cb => cb.checked).map(cb => parseInt(cb.dataset.index, 10));
  if (!keep.length) { showError('Tieni almeno un blocco.'); return; }
  const res = await fetch('/api/review', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({keep})
  });
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore.'); return; }
  hideError();
  document.getElementById('reviewCard').style.display = 'none';
}

// --- Sezioni da trascrivere ---
let lastProbedDuration = null;

function currentSections() {
  return [...document.querySelectorAll('#sections .sec-row')]
    .map(r => ({start: r.querySelector('.sec-start').value.trim(),
                end:   r.querySelector('.sec-end').value.trim()}))
    .filter(s => s.start || s.end);
}

// Gemello lato client di parse_time: il server resta comunque l'autorità.
function parseTimeJS(text) {
  const raw = (text || '').trim().replace(',', '.');
  if (!raw) return null;
  const parts = raw.split(':');
  if (parts.length > 3) return NaN;
  const vals = parts.map(Number);
  if (vals.some(v => isNaN(v) || v < 0)) return NaN;
  if (vals.slice(1).some(v => v >= 60)) return NaN;
  return vals.reduce((acc, v) => acc * 60 + v, 0);
}

function validateSections() {
  let valid = true;
  document.querySelectorAll('#sections .sec-row').forEach(row => {
    const si = row.querySelector('.sec-start'), ei = row.querySelector('.sec-end');
    const s = parseTimeJS(si.value), e = parseTimeJS(ei.value);
    const sBad = isNaN(s) || (si.value.trim() && s === null);
    const eBad = isNaN(e) || (s !== null && e !== null && !isNaN(e) && e <= s)
                 || (lastProbedDuration && e > lastProbedDuration + 0.5);
    si.classList.toggle('bad', !!sBad);
    ei.classList.toggle('bad', !!eBad);
    if (sBad || eBad) valid = false;
  });
  ['recordBtn', 'urlBtn', 'fileBtn'].forEach(id => {
    const b = document.getElementById(id);
    if (b && !b.dataset.busy) b.disabled = !valid;
  });
  updateAdvDeps();   // adding a section makes the VAD switch irrelevant
  return valid;
}

function addSectionRow(start, end) {
  const row = document.createElement('div');
  row.className = 'sec-row';
  row.innerHTML = '<input type="text" class="sec-start" placeholder="da es. 2:00">' +
                  '<span class="sep">&rarr;</span>' +
                  '<input type="text" class="sec-end" placeholder="a es. 15:30">' +
                  '<button type="button" title="Rimuovi">&times;</button>';
  row.querySelector('.sec-start').value = start || '';
  row.querySelector('.sec-end').value = end || '';
  row.querySelectorAll('input').forEach(i => i.addEventListener('input', validateSections));
  row.querySelector('button').onclick = () => { row.remove(); validateSections(); };
  document.getElementById('sections').appendChild(row);
  validateSections();
}

// --- File locale ---
async function probeFile() {
  const path = document.getElementById('fileInput').value.trim();
  const box = document.getElementById('fileInfo');
  if (!path) { box.className = 'file-info err'; box.textContent = 'Inserisci un percorso.'; return; }
  box.className = 'file-info';
  box.textContent = 'Verifica in corso...';
  const res = await fetch('/api/probe', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path})
  });
  const d = await res.json();
  if (!res.ok || !d.ok) {
    lastProbedDuration = null;
    box.className = 'file-info err';
    box.textContent = d.error || 'File non utilizzabile.';
  } else {
    lastProbedDuration = d.duration;
    box.className = 'file-info ok';
    box.textContent = `OK — ${d.duration_label || 'durata ignota'} · ` +
                      (d.has_video ? 'video + audio' : 'solo audio');
  }
  validateSections();
}

async function startFromFile() {
  const path = document.getElementById('fileInput').value.trim();
  if (!path) { showError('Inserisci il percorso del file.'); return; }
  if (!validateSections()) { showError('Correggi le sezioni evidenziate in rosso.'); return; }
  hideError();
  document.getElementById('results').style.display = 'none';
  const res = await fetch('/api/file', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({path}, currentSettings()))
  });
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore sconosciuto.'); return; }
  setUIState('processing');
  startStatusPolling();
}

// Measured with qwen3.5:9b on the same 30s source, as a rough order of magnitude.
const THINKING_HINTS = {
  none:  'Il ragionamento è cio\' che verifica le affermazioni contro la trascrizione: senza, il modello tende a riempire i vuoti con nozioni proprie. Misurato: ~1m20s.',
  notes: 'Ragiona dove il contenuto nasce dalla trascrizione, non dove rimaneggia appunti gia\' scritti. Misurato: ~2m10s.',
  full:  'Ragiona anche su schema e schede, dove serve poco perchè lavorano su appunti gia\' scritti. Misurato: ~12m40s.'
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
  if (!validateSections()) { showError('Correggi le sezioni evidenziate in rosso.'); return; }
  const device = document.getElementById('device').value;
  // Settings go out now as well, so a bad section is refused before recording starts.
  const res = await fetch('/api/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({device}, currentSettings()))
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

  if (s.step === 'review') {
    if (JSON.stringify(s.blocks) !== JSON.stringify(currentBlocks)) {
      currentBlocks = s.blocks || [];
      renderBlocks(currentBlocks);
    }
  } else if (currentBlocks.length) {
    currentBlocks = [];
    document.getElementById('reviewCard').style.display = 'none';
  }

  updateStopButton(s.step);
  if (s.step === 'done') {
    clearInterval(statusPolling); statusPolling = null;
    setUIState('idle');
    showResults(s);
  } else if (s.step === 'cancelled') {
    clearInterval(statusPolling); statusPolling = null;
    stopLevelPolling();
    setUIState('idle');
    hideError();   // a deliberate stop is not an error: no red banner
    if (s.image_name || s.notes_md) showResults(s);   // keep whatever was produced
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
    if (d.paused !== undefined && d.paused !== isPaused) setPausedUI(d.paused);
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
  // Rebuilt in page order, so the arrow keys walk the images as they are laid out.
  const stamp = Date.now();
  const entry = (name, title) => ({
    src: '/files/' + s.session_id + '/' + name + '?t=' + stamp, title, target: name
  });
  setGallery([
    ...(s.image_name ? [entry(s.image_name, 'Schema principale')] : []),
    ...(s.topic_names || []).map(n => entry(n, prettyName(n))),
    ...(s.card_names || []).map(n => entry(n, prettyName(n))),
  ]);
  const imgWrap = document.getElementById('imageWrap');
  const htmlLink = document.getElementById('htmlLink');
  if (s.image_name) {
    const url = viewerGallery[0].src;
    imgWrap.innerHTML = '';
    const img = document.createElement('img');
    img.src = url;
    img.alt = 'Schema';
    img.style.cursor = 'zoom-in';
    img.title = 'Clicca per aprire a schermo intero';
    img.onclick = () => openViewer(url, 'Schema principale');
    imgWrap.appendChild(img);
    const hint = document.createElement('div');
    hint.className = 'hint';
    hint.textContent = 'Clicca lo schema per ingrandirlo: rotella per lo zoom, trascina per spostarti, '
                     + 'frecce &larr; &rarr; per passare agli altri.';
    imgWrap.appendChild(hint);
    addReviseButton(imgWrap, s.image_name, 'schema principale');
  } else {
    imgWrap.innerHTML = '<p>Immagine non renderizzata: apri l\'anteprima HTML qui sotto.</p>';
  }
  if (s.html_name) {
    htmlLink.href = '/files/' + s.session_id + '/' + s.html_name;
    htmlLink.style.display = 'inline';
  } else {
    htmlLink.style.display = 'none';
  }
  renderGallery('topicsSection', 'topicsWrap', s.topic_names || [], s.session_id);
  renderGallery('cardsSection', 'cardsWrap', s.card_names || [], s.session_id);

  document.getElementById('downloadNotes').href = '/files/' + s.session_id + '/appunti.md';
  document.getElementById('downloadMmd').href = '/files/' + s.session_id + '/schema.mmd';
  document.getElementById('downloadTranscript').href = '/files/' + s.session_id + '/trascrizione.txt';
}

// "concetti/01-tempo-relativo.png" -> "tempo relativo"
function prettyName(name) {
  return name.split('/').pop().replace(/\.[^.]+$/, '')
             .replace(/^\d+-/, '').replace(/-/g, ' ');
}

// Galleria di immagini con didascalia, zoom e bottone di modifica.
function renderGallery(sectionId, wrapId, names, sessionId) {
  const section = document.getElementById(sectionId);
  const wrap = document.getElementById(wrapId);
  wrap.innerHTML = '';
  section.style.display = names.length ? 'block' : 'none';
  names.forEach(name => {
    const url = '/files/' + sessionId + '/' + name + '?t=' + Date.now();
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = url;
    img.alt = name;
    const cap = document.createElement('figcaption');
    cap.textContent = prettyName(name);
    img.title = 'Clicca per aprire a schermo intero';
    img.onclick = () => openViewer(url, cap.textContent);
    fig.appendChild(img);
    fig.appendChild(cap);
    addReviseButton(fig, name, cap.textContent);
    wrap.appendChild(fig);
  });
}

// --- Modifica mirata di un singolo schema o scheda ---
let reviseTargetPath = null;

function openRevise(targetPath, label) {
  reviseTargetPath = targetPath;
  document.getElementById('reviseTarget').textContent = label;
  ['rvWrong','rvMissing','rvCorrect','rvOptions'].forEach(id => {
    document.getElementById(id).value = '';
  });
  // Show what is being changed next to the form.
  const preview = document.getElementById('revisePreview');
  const entry = viewerGallery.find(e => e.target === targetPath);
  if (entry) {
    preview.src = entry.src;
    preview.style.display = '';
    preview.onclick = () => openViewer(entry.src, entry.title);
  } else {
    preview.removeAttribute('src');
    preview.style.display = 'none';
  }
  document.getElementById('revise').classList.add('open');
  document.getElementById('rvWrong').focus();
}

function closeRevise() {
  document.getElementById('revise').classList.remove('open');
  reviseTargetPath = null;
}

async function submitRevise() {
  if (!reviseTargetPath) return;
  const body = Object.assign(currentSettings(), {
    target: reviseTargetPath,
    feedback: {
      what_is_wrong: document.getElementById('rvWrong').value.trim(),
      missing: document.getElementById('rvMissing').value.trim(),
      correct: document.getElementById('rvCorrect').value.trim(),
      options: document.getElementById('rvOptions').value.trim()
    }
  });
  const res = await fetch('/api/regenerate', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Errore.'); return; }
  closeRevise();
  hideError();
  setUIState('processing');
  startStatusPolling();
}

// Aggiunge sotto un'immagine il bottone che apre il modulo di modifica.
function addReviseButton(container, targetPath, label) {
  const wrap = document.createElement('div');
  wrap.className = 'img-actions';
  const btn = document.createElement('button');
  btn.textContent = 'Modifica questo schema';
  btn.onclick = () => openRevise(targetPath, label);
  wrap.appendChild(btn);
  container.appendChild(wrap);
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

// Tutte le immagini della sessione, nell'ordine in cui compaiono in pagina.
let viewerGallery = [];
let viewerIndex = -1;

function setGallery(entries) {
  viewerGallery = entries || [];
}

function openViewer(src, title) {
  viewerIndex = viewerGallery.findIndex(e => e.src === src);
  viewer.el.classList.add('open');
  showViewerImage(src, title);
}

function showViewerImage(src, title) {
  const counter = (viewerIndex >= 0 && viewerGallery.length > 1)
    ? ` (${viewerIndex + 1}/${viewerGallery.length})` : '';
  document.getElementById('viewerTitle').textContent = (title || '') + counter;
  const canMove = viewerGallery.length > 1;
  document.getElementById('viewerPrev').style.display = canMove ? '' : 'none';
  document.getElementById('viewerNext').style.display = canMove ? '' : 'none';
  viewer.img.onload = () => {
    viewer.natW = viewer.img.naturalWidth;
    viewer.natH = viewer.img.naturalHeight;
    viewerFit();
  };
  viewer.img.src = src;
}

function viewerStep(delta) {
  if (viewerGallery.length < 2 || viewerIndex < 0) return;
  // Wraps around: reaching the end should not feel like a dead stop.
  viewerIndex = (viewerIndex + delta + viewerGallery.length) % viewerGallery.length;
  const entry = viewerGallery[viewerIndex];
  showViewerImage(entry.src, entry.title);
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
  document.getElementById('viewerPrev').onclick = () => viewerStep(-1);
  document.getElementById('viewerNext').onclick = () => viewerStep(1);
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
    else if (e.key === 'ArrowRight' || e.key === 'PageDown') viewerStep(1);
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') viewerStep(-1);
    else if (e.key === '+' || e.key === '=') viewerZoomCentre(1.25);
    else if (e.key === '-') viewerZoomCentre(1 / 1.25);
    else if (e.key === '0') viewerFit();
    else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
    else return;
    e.preventDefault();   // stop the arrows scrolling the page behind the viewer
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
  updateStopButton(s.step);
  if (s.step === 'recording') {
    setUIState('recording');
    startLevelPolling();
    startStatusPolling();
  } else if (s.step === 'cancelled') {
    setUIState('idle');
    if (s.image_name || s.notes_md) showResults(s);
  } else if (s.step === 'review') {
    currentBlocks = s.blocks || [];
    renderBlocks(currentBlocks);
    setUIState('processing');
    startStatusPolling();
  } else if (['downloading', 'transcribing', 'notes', 'diagram', 'cards',
              'regenerating'].includes(s.step)) {
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
  buildAdvancedPanel();   // must exist before defaults are applied to it
  loadModelsAndDefaults();
  document.getElementById('advReset').onclick = resetAdvanced;
  document.getElementById('blocksGo').onclick = submitBlocks;
  document.getElementById('blocksAll').onclick = () => setBlocks(true);
  document.getElementById('blocksAuto').onclick = () => setBlocks(null);
  loadDevices();
  restoreState();
  document.getElementById('urlBtn').onclick = startFromUrl;
  document.getElementById('thinking').onchange = updateThinkingHint;
  document.getElementById('pauseBtn').onclick = togglePause;
  document.getElementById('stopBtn').onclick = cancelRun;
  document.getElementById('addSection').onclick = () => addSectionRow();
  document.getElementById('probeBtn').onclick = probeFile;
  document.getElementById('fileBtn').onclick = startFromFile;
  document.getElementById('fileInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') probeFile();   // safer default than starting a run
  });
  document.getElementById('reviseCancel').onclick = closeRevise;
  document.getElementById('reviseSubmit').onclick = submitRevise;
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
