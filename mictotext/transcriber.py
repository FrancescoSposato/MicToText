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
from dataclasses import asdict, dataclass
from pathlib import Path

from mictotext.config import SttConfig


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    duration: float
    device: str
    model: str
    elapsed: float
    segments: list[dict]


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

    segments, info = model.transcribe(
        audio_path,
        language=options.get("language"),
        beam_size=options.get("beam_size", 5),
        vad_filter=options.get("vad_filter", True),
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(f"  Language: {info.language} (p={info.language_probability:.2f}), "
          f"audio length: {info.duration:.1f}s", flush=True)

    collected: list[dict] = []
    for segment in segments:  # generator: decoding happens while iterating
        text = segment.text.strip()
        if not text:
            continue
        print(f"  [{_format_timestamp(segment.start)} -> {_format_timestamp(segment.end)}] {text}",
              flush=True)
        collected.append({"start": round(segment.start, 2), "end": round(segment.end, 2), "text": text})

    transcript = Transcript(
        text=" ".join(item["text"] for item in collected),
        language=info.language,
        language_probability=float(info.language_probability),
        duration=float(info.duration),
        device=device,
        model=model_name,
        elapsed=time.perf_counter() - started,
        segments=collected,
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

    options = {
        "language": cfg.language,
        "beam_size": cfg.beam_size,
        "vad_filter": cfg.vad_filter,
        "download_root": cfg.download_root,
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
