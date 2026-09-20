"""Pause and noise detection over transcribed segments.

Pure functions over the segment dicts produced by `transcriber._worker`: no model, no
I/O, no dependencies. The signals (`gap`, `logprob`, `no_speech`) are computed by Whisper
anyway, so this costs nothing beyond arithmetic.

The filter works on already-transcribed text: nothing is re-transcribed when the user
changes their mind about what to keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Defaults calibrated on a real clean lecture, where confident speech measured
# logprob between -0.03 and -0.06 with no_speech_prob at 0.000. They are deliberately
# permissive: dropping part of a lecture unnoticed is far worse than keeping some noise.
DEFAULT_GAP_SECONDS = 20.0
DEFAULT_MIN_LOGPROB = -0.8
DEFAULT_MAX_NO_SPEECH = 0.6

_PREVIEW_WORDS = 12


@dataclass
class Block:
    """A run of speech with no long pause inside it."""

    index: int
    start: float
    end: float
    segments: list = field(default_factory=list)
    preview: str = ""
    mean_logprob: float = 0.0
    max_no_speech: float = 0.0
    suspicious: bool = False
    gap_before: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "gap_before": round(self.gap_before, 2),
            "preview": self.preview,
            "mean_logprob": round(self.mean_logprob, 3),
            "max_no_speech": round(self.max_no_speech, 3),
            "suspicious": self.suspicious,
            "segments": len(self.segments),
        }


def _preview(segments: list) -> str:
    words = " ".join(s.get("text", "") for s in segments).split()
    text = " ".join(words[:_PREVIEW_WORDS])
    return f"{text}..." if len(words) > _PREVIEW_WORDS else text


def split_into_blocks(segments, gap_threshold: float = DEFAULT_GAP_SECONDS,
                      min_logprob: float = DEFAULT_MIN_LOGPROB,
                      max_no_speech: float = DEFAULT_MAX_NO_SPEECH) -> list[Block]:
    """Split segments wherever a pause longer than `gap_threshold` occurs.

    A long silence is by far the most reliable signal of a break in a lecture, and it
    needs no model at all. Blocks are additionally flagged as suspicious when the
    speech recognition itself was unsure, which is what distant background chatter
    looks like.
    """
    blocks: list[Block] = []
    current: list = []
    gap_before = 0.0

    for segment in segments or []:
        gap = float(segment.get("gap", 0.0) or 0.0)
        if current and gap >= gap_threshold:
            blocks.append(_build_block(len(blocks), current, gap_before, min_logprob,
                                       max_no_speech))
            current, gap_before = [], gap
        current.append(segment)

    if current:
        blocks.append(_build_block(len(blocks), current, gap_before, min_logprob,
                                   max_no_speech))
    return blocks


def _build_block(index: int, segments: list, gap_before: float, min_logprob: float,
                 max_no_speech: float) -> Block:
    # Missing signals (transcripts made before they were recorded) read as neutral,
    # so an old transcript is never flagged as suspicious.
    logprobs = [float(s["logprob"]) for s in segments if s.get("logprob") is not None]
    no_speech = [float(s["no_speech"]) for s in segments if s.get("no_speech") is not None]
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
    peak_no_speech = max(no_speech) if no_speech else 0.0

    return Block(
        index=index,
        start=float(segments[0].get("start", 0.0)),
        end=float(segments[-1].get("end", 0.0)),
        segments=segments,
        preview=_preview(segments),
        mean_logprob=mean_logprob,
        max_no_speech=peak_no_speech,
        suspicious=bool(logprobs) and (mean_logprob < min_logprob
                                       or peak_no_speech > max_no_speech),
        gap_before=gap_before,
    )


def rebuild_text(segments) -> str:
    return " ".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip())


def apply_selection(blocks: list[Block], keep_indices) -> tuple[list, str]:
    """Keep only the chosen blocks. Returns (segments, rebuilt text)."""
    keep = set(keep_indices or [])
    segments = [s for b in blocks if b.index in keep for s in b.segments]
    return segments, rebuild_text(segments)


def auto_selection(blocks: list[Block]) -> list[int]:
    """Indices to keep when filtering automatically: everything not flagged."""
    return [b.index for b in blocks if not b.suspicious]


def describe(blocks: list[Block], kept) -> list[str]:
    """Human-readable lines for the log, so a drop is never silent."""
    keep = set(kept or [])
    lines = []
    for block in blocks:
        if block.index in keep:
            continue
        why = []
        if block.suspicious:
            why.append(f"confidenza {block.mean_logprob:.2f}, "
                       f"non-parlato {block.max_no_speech:.2f}")
        if block.gap_before:
            why.append(f"dopo una pausa di {block.gap_before:.0f}s")
        reason = f" ({'; '.join(why)})" if why else ""
        lines.append(f"Scartato blocco {block.index + 1}: "
                     f"{block.duration:.0f}s{reason} - \"{block.preview[:60]}\"")
    return lines
