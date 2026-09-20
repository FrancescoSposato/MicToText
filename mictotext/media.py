"""Media probing, path cleaning and time-range parsing.

Shared by both front-ends. Probing reads container headers only, so it is cheap even on
a multi-gigabyte lecture video, and it never raises: a failure is reported in
`MediaInfo.error` so the caller can show it instead of crashing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Cover art is stored as a single-frame image stream; that is not a video track.
_COVER_ART_CODECS = {"mjpeg", "png", "bmp", "gif", "webp"}
_RANGE_SEP_RE = re.compile(r"\s*(?:-|–|—|\.\.|to)\s*", re.IGNORECASE)


@dataclass
class MediaInfo:
    path: Path
    duration: float | None  # seconds; None when the container does not declare one
    has_audio: bool
    has_video: bool
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.has_audio


def clean_path(raw: str) -> Path:
    """Turn user-pasted text into a usable path.

    Windows' "Copy as path" wraps the path in double quotes, which would otherwise be
    taken as part of the filename.
    """
    text = (raw or "").strip().strip('"').strip("'").strip()
    return Path(os.path.expandvars(os.path.expanduser(text)))


def probe_media(path: Path) -> MediaInfo:
    """Read duration and stream layout from a media file. Never raises."""
    if not str(path).strip():
        return MediaInfo(path, None, False, False, "Percorso vuoto.")
    if path.is_dir():
        return MediaInfo(path, None, False, False, "Il percorso e' una cartella, non un file.")
    if not path.is_file():
        return MediaInfo(path, None, False, False, f"File non trovato: {path}")

    try:
        import av
        with av.open(str(path)) as container:
            duration = (container.duration / av.time_base) if container.duration else None
            audio = list(container.streams.audio)
            video = [
                stream for stream in container.streams.video
                if (stream.codec_context.name or "").lower() not in _COVER_ART_CODECS
            ]
            if duration is None and audio:
                # Some containers declare the length only on the stream.
                stream = audio[0]
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
    except Exception as exc:  # noqa: BLE001 - a probe failure must be reported, not raised
        return MediaInfo(path, None, False, False, f"Impossibile leggere il file: {exc}")

    info = MediaInfo(path, duration, bool(audio), bool(video))
    if not info.has_audio:
        info.error = "Il file non contiene una traccia audio."
    return info


def parse_time(text: str) -> float:
    """Parse "90", "1:30" or "01:02:03" into seconds. Raises ValueError on bad input."""
    raw = (text or "").strip().replace(",", ".")
    if not raw:
        raise ValueError("Orario vuoto.")
    parts = raw.split(":")
    if len(parts) > 3:
        raise ValueError(f"Formato orario non valido: {text!r}")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Formato orario non valido: {text!r}") from None
    if any(v < 0 for v in values):
        raise ValueError(f"Orario negativo: {text!r}")
    # Only the leading field may exceed 59 ("90:00" = 90 minutes is legitimate);
    # "1:75" is a typo, not 135 seconds, and silently accepting it hides the mistake.
    if any(v >= 60 for v in values[1:]):
        raise ValueError(f"Orario non valido: {text!r} (minuti e secondi devono stare sotto 60).")

    seconds = 0.0
    for value in values:  # h:m:s, m:s or s, depending on how many parts were given
        seconds = seconds * 60 + value
    return seconds


def format_time(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def parse_range(text: str) -> tuple[float, float]:
    """Parse "02:00-15:30" into (start, end) seconds."""
    parts = _RANGE_SEP_RE.split((text or "").strip(), maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Intervallo non valido: {text!r}. Usa il formato 02:00-15:30.")
    return parse_time(parts[0]), parse_time(parts[1])


def normalize_ranges(ranges, duration: float | None = None) -> list[tuple[float, float]]:
    """Sort, clamp and merge keep-ranges into a clean, non-overlapping list.

    Overlapping keeps are a union, not a conflict, so they are merged rather than
    rejected. When the duration is known, ranges are clamped to the end of the file
    instead of being refused: erring towards "transcribe a bit more" is friendlier than
    blocking the run over a rough guess.
    """
    cleaned: list[tuple[float, float]] = []
    for start, end in ranges or []:
        start, end = float(start), float(end)
        if duration is not None:
            start, end = min(start, duration), min(end, duration)
        if end > start:
            cleaned.append((start, end))

    cleaned.sort()
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def parse_range_specs(specs) -> list[tuple[float, float]]:
    """Parse range strings without knowing the duration yet (syntax check only).

    Called early — before recording starts — so a typo fails immediately instead of
    after the user has already spoken into the microphone.
    """
    parsed = []
    for spec in specs or []:
        for piece in str(spec).split(","):
            if piece.strip():
                parsed.append(parse_range(piece))
    return parsed


def resolve_ranges(ranges, duration: float | None) -> tuple[list[tuple[float, float]], list[str]]:
    """Clamp and merge against the real duration, reporting what was changed.

    Returns (ranges, warnings). Warnings are shown to the user: a silently dropped or
    merged range would otherwise look like the app ignoring the request.
    """
    requested = [(float(s), float(e)) for s, e in ranges or []]
    if not requested:
        return [], []

    warnings: list[str] = []
    if duration is None:
        warnings.append("Durata del file non rilevabile: le sezioni non sono state verificate.")
        return normalize_ranges(requested), warnings

    outside = [r for r in requested if r[0] >= duration]
    for start, end in outside:
        warnings.append(f"Sezione {format_time(start)}-{format_time(end)} oltre la fine del file "
                        f"({format_time(duration)}): ignorata.")
    clamped = [r for r in requested if r[0] < duration]
    for start, end in clamped:
        if end > duration:
            warnings.append(f"Sezione {format_time(start)}-{format_time(end)} troncata a "
                            f"{format_time(duration)}.")

    resolved = normalize_ranges(clamped, duration)
    if len(resolved) < len(clamped):
        warnings.append("Sezioni sovrapposte unite.")
    if not resolved:
        raise ValueError(f"Nessuna sezione ricade dentro la durata reale del file "
                         f"({format_time(duration)}).")
    return resolved, warnings


def ranges_duration(ranges) -> float:
    return sum(end - start for start, end in ranges or [])


def describe_ranges(ranges) -> str:
    return ", ".join(f"{format_time(s)}-{format_time(e)}" for s, e in ranges or [])
