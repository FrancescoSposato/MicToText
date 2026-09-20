"""Speech-to-text with faster-whisper, executed in a child process.

Running the model in a separate process gives two benefits:
1. If CUDA/cuDNN libraries are missing or broken, a native crash kills only the
   child, and the parent transparently retries on CPU.
2. When the child exits, all VRAM used by Whisper is released before the LLM runs,
   which matters on an 8 GB GPU.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mictotext.config import SttConfig


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    duration: float  # full file, even when only some sections were transcribed
    device: str
    model: str
    elapsed: float
    segments: list[dict]
    # Defaulted so transcripts written before these fields existed still load.
    clip_ranges: list = field(default_factory=list)
    clipped_duration: float = 0.0  # seconds actually sent to the decoder
    source_path: str = ""  # the media is read in place, so record where it came from


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _worker(audio_path: str, result_path: str, device: str, model_name: str,
            compute_type: str, options: dict) -> None:
    """Child-process entry point. Writes the transcript as JSON to result_path."""
    if device == "cuda":
        from mictotext.cuda_setup import register_cuda_dll_dirs
        register_cuda_dll_dirs()

    from faster_whisper import WhisperModel

    started = time.perf_counter()
    print(f"  Loading Whisper '{model_name}' on {device} ({compute_type})...", flush=True)
    model = WhisperModel(model_name, device=device, compute_type=compute_type,
                         download_root=options.get("download_root"))

    # clip_timestamps and vad_filter are mutually exclusive: faster-whisper only runs the
    # VAD when clip_timestamps is left at its default, and disables it silently otherwise.
    # Branching makes the dead parameter impossible to pass by accident.
    clips = options.get("clip_timestamps") or None
    extra: dict = {}
    if clips:
        extra["clip_timestamps"] = clips
        print(f"  Sezioni selezionate: {options.get('clip_label', '')}", flush=True)
        print("  Nota: con le sezioni attive il filtro VAD di Whisper viene disattivato.",
              flush=True)
    else:
        extra["vad_filter"] = options.get("vad_filter", True)
        extra["vad_parameters"] = {"min_silence_duration_ms": 500}

    segments, info = model.transcribe(
        audio_path,
        language=options.get("language"),
        beam_size=options.get("beam_size", 5),
        **extra,
    )
    # info.duration is always the FULL file, so state both figures when clipping.
    clip_ranges = options.get("clip_ranges") or []
    clipped = sum(end - start for start, end in clip_ranges)
    length = (f"{clipped:.1f}s selezionati su {info.duration:.1f}s totali"
              if clip_ranges else f"audio length: {info.duration:.1f}s")
    print(f"  Language: {info.language} (p={info.language_probability:.2f}), {length}", flush=True)

    def _crosses_clip_boundary(gap_start: float, gap_end: float) -> bool:
        """True when the silence between two segments is just the cut between clips.

        With clip_timestamps the skipped audio shows up as a huge gap that has nothing
        to do with a pause in the speech, so it must not be read as one.
        """
        return any(gap_start <= end <= gap_end for _, end in clip_ranges)

    collected: list[dict] = []
    previous_end: float | None = None
    for segment in segments:  # generator: decoding happens while iterating
        text = segment.text.strip()
        if not text:
            continue
        print(f"  [{_format_timestamp(segment.start)} -> {_format_timestamp(segment.end)}] {text}",
              flush=True)
        # Gap from the last KEPT segment: empty ones are skipped above.
        gap = 0.0 if previous_end is None else max(0.0, segment.start - previous_end)
        if gap and _crosses_clip_boundary(previous_end, segment.start):
            gap = 0.0
        collected.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text,
            # Kept for pause/noise detection: cheap, already computed by Whisper.
            "logprob": round(float(segment.avg_logprob), 3),
            "no_speech": round(float(segment.no_speech_prob), 3),
            "gap": round(gap, 2),
        })
        previous_end = segment.end

    transcript = Transcript(
        text=" ".join(item["text"] for item in collected),
        language=info.language,
        language_probability=float(info.language_probability),
        duration=float(info.duration),
        device=device,
        model=model_name,
        elapsed=time.perf_counter() - started,
        segments=collected,
        clip_ranges=[list(r) for r in clip_ranges],
        clipped_duration=round(clipped or float(info.duration), 2),
        source_path=audio_path,
    )
    Path(result_path).write_text(json.dumps(asdict(transcript), ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    sys.stdout.flush()
    # Skip native destructors: CUDA teardown at interpreter exit can crash on Windows,
    # and the result is already safely on disk.
    os._exit(0)


def _cuda_device_count() -> int:
    try:
        from mictotext.cuda_setup import register_cuda_dll_dirs
        register_cuda_dll_dirs()
        import ctranslate2
        return ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001 - any failure simply means "no usable CUDA"
        return 0


def transcribe(audio_path: Path, output_dir: Path, cfg: SttConfig) -> Transcript:
    """Transcribe an audio file, trying CUDA first (if allowed) and then CPU."""
    plan: list[tuple[str, str, str]] = []
    if cfg.device in ("auto", "cuda"):
        if cfg.device == "cuda" or _cuda_device_count() > 0:
            plan.append(("cuda", cfg.gpu_model, cfg.gpu_compute_type))
        else:
            print("  No CUDA device visible to CTranslate2: using CPU.")
    if cfg.device in ("auto", "cpu"):
        plan.append(("cpu", cfg.cpu_model, cfg.cpu_compute_type))

    from mictotext.media import describe_ranges

    clip_ranges = [(float(s), float(e)) for s, e in (cfg.clip_ranges or [])]
    options = {
        "language": cfg.language,
        "beam_size": cfg.beam_size,
        "vad_filter": cfg.vad_filter and not clip_ranges,
        "download_root": cfg.download_root,
        # Flat [start, end, start, end, ...] in seconds, as faster-whisper expects.
        "clip_timestamps": [t for pair in clip_ranges for t in pair],
        "clip_ranges": clip_ranges,
        "clip_label": describe_ranges(clip_ranges),
    }
    result_path = output_dir / "trascrizione.json"
    context = mp.get_context("spawn")

    for device, model_name, compute_type in plan:
        result_path.unlink(missing_ok=True)
        process = context.Process(
            target=_worker,
            args=(str(audio_path.resolve()), str(result_path), device, model_name, compute_type, options),
            name=f"whisper-{device}",
        )
        process.start()
        process.join()

        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            return Transcript(**data)

        print(f"  Transcription on {device} failed (exit code {process.exitcode}).")
        if device == "cuda" and cfg.device == "auto":
            print("  Falling back to CPU...")

    raise RuntimeError("Transcription failed on every configured device.")
