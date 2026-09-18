"""Command-line entry point: record → transcribe → notes → Mermaid diagram."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from mictotext.config import AppConfig, language_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mictotext",
        description="Local pipeline: microphone -> transcript -> Markdown notes -> Mermaid diagram.",
    )
    source = parser.add_argument_group("input shortcuts (skip earlier steps)")
    source.add_argument("--audio", type=Path, help="use an existing audio file instead of recording")
    source.add_argument("--transcript", type=Path, help="use an existing .txt transcript (skips recording and STT)")
    source.add_argument("--notes", type=Path, help="use an existing Markdown file (only generates the diagram)")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    audio.add_argument("--input-device", help="input device index or name substring")

    stt = parser.add_argument_group("speech-to-text")
    stt.add_argument("--language", default="it", help="spoken language code, or 'auto' (default: it)")
    stt.add_argument("--stt-device", choices=["auto", "cuda", "cpu"], default="auto")
    stt.add_argument("--whisper-model", help="Whisper model used on GPU (default: large-v3-turbo)")
    stt.add_argument("--whisper-cpu-model", help="Whisper model used on CPU (default: small)")

    llm = parser.add_argument_group("LLM (Ollama)")
    llm.add_argument("--llm-model", help="Ollama model for notes (default: qwen2.5:7b)")
    llm.add_argument("--diagram-model", help="Ollama model for Mermaid (default: same as --llm-model)")
    llm.add_argument("--num-ctx", type=int, help="context window in tokens (default: 16384)")
    llm.add_argument("--no-think", action="store_true", help="send think=false (for reasoning models, e.g. qwen3)")
    llm.add_argument("--ollama-url", help="Ollama base URL (default: http://127.0.0.1:11434)")

    render = parser.add_argument_group("diagram rendering")
    render.add_argument("--renderer", choices=["auto", "native", "wsl", "html"], default="auto")
    render.add_argument("--wsl-distro", help="WSL distro name (default: Debian; use '' for the default distro)")
    render.add_argument("--mmdc", help="mmdc command or full path (default: mmdc)")
    render.add_argument("--puppeteer-config", help="puppeteer config JSON path, as seen by mmdc")
    render.add_argument("--format", choices=["png", "svg"], help="image format (default: png)")

    parser.add_argument("--out-dir", type=Path, help="output folder (default: output/<timestamp>)")
    parser.add_argument("--no-open", action="store_true", help="do not open the result when finished")
    return parser


def _parse_device(value: str | None) -> int | str | None:
    if not value:
        return None
    return int(value) if value.isdigit() else value


def build_config(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()
    cfg.audio.device = _parse_device(args.input_device)

    cfg.stt.language = None if args.language.lower() == "auto" else args.language
    cfg.stt.device = args.stt_device
    if args.whisper_model:
        cfg.stt.gpu_model = args.whisper_model
    if args.whisper_cpu_model:
        cfg.stt.cpu_model = args.whisper_cpu_model

    if args.llm_model:
        cfg.llm.notes_model = args.llm_model
    if args.diagram_model:
        cfg.llm.diagram_model = args.diagram_model
    if args.num_ctx:
        cfg.llm.num_ctx = args.num_ctx
    if args.no_think:
        cfg.llm.think = False
    if args.ollama_url:
        cfg.llm.base_url = args.ollama_url

    cfg.render.renderer = args.renderer
    if args.wsl_distro is not None:
        cfg.render.wsl_distro = args.wsl_distro or None
    if args.mmdc:
        cfg.render.mmdc_command = args.mmdc
    if args.puppeteer_config:
        cfg.render.puppeteer_config = args.puppeteer_config
    if args.format:
        cfg.render.image_format = args.format
    return cfg


def _step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _open_file(path: Path) -> None:
    if sys.platform == "win32":
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError:
            pass


def run(args: argparse.Namespace, cfg: AppConfig) -> int:
    from mictotext.diagram import DiagramError, generate_diagram
    from mictotext.llm import OllamaClient, OllamaError
    from mictotext.notes import generate_notes
    from mictotext.renderer import build_renderer

    session_dir = (args.out_dir or cfg.output_root / datetime.now().strftime("%Y-%m-%d_%H%M%S")).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {session_dir}")

    notes_model = cfg.llm.notes_model
    diagram_model = cfg.llm.diagram_model or notes_model

    # Preflight checks run BEFORE recording, so the user never loses a recording.
    _step("Preflight")
    client = OllamaClient(cfg.llm)
    try:
        client.ensure_ready({notes_model, diagram_model})
    except OllamaError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"  Ollama OK - notes model: {notes_model}, diagram model: {diagram_model}")
    renderer = build_renderer(cfg.render)
    print(f"  Mermaid renderer: {renderer.name}" if renderer
          else "  Mermaid renderer: none available -> HTML output only")

    output_language = cfg.stt.language
    timings: dict[str, float] = {}

    try:
        if args.notes:
            notes_md = args.notes.read_text(encoding="utf-8")
        else:
            if args.transcript:
                transcript_text = args.transcript.read_text(encoding="utf-8")
            else:
                audio_path = args.audio.resolve() if args.audio else None
                if audio_path is None:
                    from mictotext.recorder import record_until_enter
                    _step("1/4 Recording")
                    audio_path = session_dir / "audio.wav"
                    duration = record_until_enter(audio_path, cfg.audio.device, cfg.audio.channels)
                    print(f"  Saved {duration:.1f}s of audio to {audio_path.name}")

                from mictotext.transcriber import transcribe
                _step("2/4 Transcription")
                started = time.perf_counter()
                transcript = transcribe(audio_path, session_dir, cfg.stt)
                timings["transcription"] = time.perf_counter() - started
                transcript_text = transcript.text
                output_language = output_language or transcript.language
                print(f"  Done on {transcript.device} with '{transcript.model}' "
                      f"({transcript.duration:.0f}s audio in {transcript.elapsed:.1f}s)")

            if not transcript_text.strip():
                print("ERROR: empty transcript (no speech detected).")
                return 3
            (session_dir / "trascrizione.txt").write_text(transcript_text.strip() + "\n", encoding="utf-8")

            _step("3/4 Structured notes")
            started = time.perf_counter()
            notes_md = generate_notes(transcript_text, client, notes_model, cfg.llm, language_name(output_language))
            timings["notes"] = time.perf_counter() - started
            (session_dir / "appunti.md").write_text(notes_md, encoding="utf-8")

        _step("4/4 Mermaid diagram")
        started = time.perf_counter()
        result = generate_diagram(notes_md, client, diagram_model, cfg.llm, cfg.render, renderer,
                                  session_dir, language_name(output_language))
        timings["diagram"] = time.perf_counter() - started

    except RuntimeError as exc:  # includes OllamaError and DiagramError
        kind = "LLM" if isinstance(exc, OllamaError) else "Diagram" if isinstance(exc, DiagramError) else "Pipeline"
        print(f"\nERROR ({kind}): {exc}")
        print(f"Partial results (if any) are in {session_dir}")
        return 4

    _step("Summary")
    for path in sorted(session_dir.iterdir()):
        print(f"  {path.name}")
    for name, seconds in timings.items():
        print(f"  {name}: {seconds:.1f}s")

    if result.image_path:
        print(f"\nDiagram image: {result.image_path}")
        if not args.no_open:
            _open_file(result.image_path)
    else:
        print(f"\nImage not rendered ({result.error.splitlines()[-1] if result.error else 'unknown reason'}).")
        print(f"Open the HTML fallback instead: {result.html_path}")
        if not args.no_open:
            _open_file(result.html_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        from mictotext.recorder import list_input_devices
        print(list_input_devices())
        return 0
    try:
        return run(args, build_config(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
