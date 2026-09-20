"""Command-line entry point: record → transcribe → notes → Mermaid diagram."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from mictotext.config import THINKING_LEVELS, AppConfig, language_name
from mictotext.media import (clean_path, describe_ranges, format_time, parse_range_specs,
                             probe_media, ranges_duration, resolve_ranges)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mictotext",
        description="Local pipeline: microphone -> transcript -> Markdown notes -> Mermaid diagram.",
    )
    # Mutually exclusive: previously a plain group, so `--url X --audio Y` silently
    # ignored the URL instead of reporting the conflict.
    source = parser.add_argument_group("input shortcuts (skip earlier steps)")
    sources = source.add_mutually_exclusive_group()
    sources.add_argument("--url", help="download audio from a video URL (YouTube and many others)")
    sources.add_argument("--audio", help="use a local audio OR video file instead of recording "
                                         "(quoted Windows paths are accepted)")
    sources.add_argument("--transcript", type=Path, help="use an existing .txt transcript (skips recording and STT)")
    sources.add_argument("--notes", type=Path, help="use an existing Markdown file (only generates the diagram)")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    audio.add_argument("--input-device", help="input device index or name substring")

    stt = parser.add_argument_group("speech-to-text")
    stt.add_argument("--language", default="it", help="spoken language code, or 'auto' (default: it)")
    stt.add_argument("--stt-device", choices=["auto", "cuda", "cpu"], default="auto")
    stt.add_argument("--whisper-model", help="Whisper model used on GPU (default: large-v3-turbo)")
    stt.add_argument("--whisper-cpu-model", help="Whisper model used on CPU (default: small)")
    ctx = parser.add_argument_group("contesto e filtro")
    ctx.add_argument("--topic", help="subject of the recording; steers notes and diagrams")
    ctx.add_argument("--subtopics", help="comma-separated subtopics to focus on")
    ctx.add_argument("--filter-pauses", action="store_true",
                     help="drop pauses and low-confidence speech (applied straight away; "
                          "the web UI can propose them for confirmation instead)")
    ctx.add_argument("--gap-seconds", type=float, help="silence that starts a new block (default: 20)")
    ctx.add_argument("--min-logprob", type=float, help="below this a block is suspect (default: -0.8)")
    ctx.add_argument("--max-no-speech", type=float, help="above this a block is suspect (default: 0.6)")

    stt.add_argument("--keep", action="append", metavar="DA-A",
                     help="transcribe only this section; repeatable. Times as 90, 1:30 or "
                          "01:02:03, e.g. --keep 2:00-15:30 --keep 40:00-55:00. "
                          "Applies to every source. Disables Whisper's VAD filter.")

    llm = parser.add_argument_group("LLM (Ollama)")
    llm.add_argument("--llm-model", help="Ollama model for notes (default: qwen2.5:7b)")
    llm.add_argument("--diagram-model", help="Ollama model for Mermaid (default: same as --llm-model)")
    llm.add_argument("--num-ctx", type=int, help="context window in tokens (default: 16384)")
    llm.add_argument("--thinking", choices=["none", "notes", "full"],
                     help="how much internal reasoning to spend: none (fast, may invent), "
                          "notes (default, reasons only where content is created), full (slow)")
    llm.add_argument("--ollama-url", help="Ollama base URL (default: http://127.0.0.1:11434)")

    render = parser.add_argument_group("diagram rendering")
    render.add_argument("--renderer", choices=["auto", "native", "wsl", "html"], default="auto")
    render.add_argument("--wsl-distro", help="WSL distro name (default: Debian; use '' for the default distro)")
    render.add_argument("--mmdc", help="mmdc command or full path (default: mmdc)")
    render.add_argument("--puppeteer-config", help="puppeteer config JSON path, as seen by mmdc")
    render.add_argument("--format", choices=["png", "svg"], help="image format (default: png)")

    llm.add_argument("--cards", type=int, metavar="N",
                     help="discursive concept cards to generate (default: 4, 0 disables)")
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

    # Syntax only: the real duration is unknown until the media exists, but a typo must
    # fail now rather than after a recording has already been made.
    cfg.stt.clip_ranges = parse_range_specs(args.keep)

    cfg.llm.topic = (args.topic or "").strip()
    cfg.llm.subtopics = (args.subtopics or "").strip()
    cfg.filter.enabled = bool(args.filter_pauses)
    cfg.filter.mode = "auto"  # the terminal has no review step
    for name in ("gap_seconds", "min_logprob", "max_no_speech"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.filter, name, value)

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
    if args.cards is not None:
        cfg.llm.concept_cards = max(0, args.cards)
    if args.thinking:
        level = THINKING_LEVELS[args.thinking]
        cfg.llm.think_notes = level["think_notes"]
        cfg.llm.think = level["think"]
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
                audio_path = clean_path(args.audio) if args.audio else None
                if audio_path is not None:
                    # Fail on a bad path here, with a readable message, rather than deep
                    # inside the child process on an opaque decoder error.
                    info = probe_media(audio_path)
                    if not info.ok:
                        print(f"ERROR: {info.error}")
                        return 2
                    kind = "video" if info.has_video else "audio"
                    print(f"  {audio_path.name} — {format_time(info.duration or 0)}, {kind}")
                if audio_path is None and args.url:
                    from mictotext.fetch import download_audio
                    _step("1/4 Download")
                    media = download_audio(args.url, session_dir)
                    audio_path = media.path
                    length = f", {media.duration / 60:.1f} min" if media.duration else ""
                    print(f"  {media.title}{length}")
                if audio_path is None:
                    from mictotext.recorder import record_until_enter
                    _step("1/4 Recording")
                    audio_path = session_dir / "audio.wav"
                    duration = record_until_enter(audio_path, cfg.audio.device, cfg.audio.channels)
                    print(f"  Saved {duration:.1f}s of audio to {audio_path.name}")

                # Second phase: now the media exists, so the sections can be checked
                # against its real duration and clamped or merged.
                if cfg.stt.clip_ranges:
                    probed = probe_media(audio_path)
                    try:
                        cfg.stt.clip_ranges, warnings = resolve_ranges(cfg.stt.clip_ranges,
                                                                       probed.duration)
                    except ValueError as exc:
                        print(f"ERROR: {exc}")
                        if audio_path.is_relative_to(session_dir):
                            print(f"L'audio registrato resta in {audio_path}")
                        return 2
                    for warning in warnings:
                        print(f"  {warning}")
                    print(f"  Sezioni: {describe_ranges(cfg.stt.clip_ranges)} "
                          f"({format_time(ranges_duration(cfg.stt.clip_ranges))} su "
                          f"{format_time(probed.duration or 0)})")

                from mictotext.transcriber import transcribe
                _step("2/4 Transcription")
                started = time.perf_counter()
                transcript = transcribe(audio_path, session_dir, cfg.stt)
                timings["transcription"] = time.perf_counter() - started
                transcript_text = transcript.text
                if cfg.filter.enabled:
                    from mictotext.segments import (apply_selection, auto_selection,
                                                    describe, split_into_blocks)
                    blocks = split_into_blocks(transcript.segments, cfg.filter.gap_seconds,
                                               cfg.filter.min_logprob, cfg.filter.max_no_speech)
                    keep = auto_selection(blocks)
                    if not keep:
                        print("  Il filtro scarterebbe tutto: lo ignoro e tengo la trascrizione intera.")
                    else:
                        for line in describe(blocks, keep):
                            print(f"  {line}")
                        _, filtered = apply_selection(blocks, keep)
                        print(f"  Filtro: {len(keep)}/{len(blocks)} blocchi tenuti, "
                              f"{len(transcript_text.split())} -> {len(filtered.split())} parole.")
                        transcript_text = filtered
                output_language = output_language or transcript.language
                scope = (f"{transcript.clipped_duration:.0f}s selezionati su "
                         f"{transcript.duration:.0f}s" if transcript.clip_ranges
                         else f"{transcript.duration:.0f}s audio")
                print(f"  Done on {transcript.device} with '{transcript.model}' "
                      f"({scope} in {transcript.elapsed:.1f}s)")

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
        from mictotext.diagram import generate_topic_diagrams, plan_diagrams
        main_source, topics = plan_diagrams(notes_md, cfg.llm)
        if topics:
            print(f"  Appunti lunghi: lo schema principale mappa {len(topics)} argomenti, "
                  f"ognuno con il proprio schema di dettaglio.")
        result = generate_diagram(main_source, client, diagram_model, cfg.llm, cfg.render, renderer,
                                  session_dir, language_name(output_language))
        if topics:
            generate_topic_diagrams(topics, client, diagram_model, cfg.llm, cfg.render, renderer,
                                    session_dir, language_name(output_language))
        timings["diagram"] = time.perf_counter() - started

        if cfg.llm.concept_cards > 0:
            from mictotext.diagram import extract_concepts, generate_concept_cards
            _step("Schede concetto")
            started = time.perf_counter()
            concepts = extract_concepts(notes_md, client, diagram_model, cfg.llm,
                                        language_name(output_language), cfg.llm.concept_cards)
            if concepts:
                print(f"  Concetti selezionati: {', '.join(concepts)}")
                cards = generate_concept_cards(notes_md, concepts, client, diagram_model, cfg.llm,
                                               cfg.render, renderer, session_dir,
                                               language_name(output_language))
                print(f"\n  {len(cards)} scheda/e prodotta/e in concetti/")
            else:
                print("  Nessun concetto adatto a una scheda.")
            timings["cards"] = time.perf_counter() - started

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
        if result.html_path:
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
        cfg = build_config(args)
    except ValueError as exc:  # malformed --keep: fail before recording or preflight
        print(f"ERROR: {exc}")
        return 2
    if args.keep and (args.transcript or args.notes):
        print("Attenzione: --keep non ha effetto con --transcript o --notes.")
    try:
        return run(args, cfg)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
