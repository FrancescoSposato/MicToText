"""Central configuration for MicToText."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANGUAGE_NAMES = {
    "it": "Italian",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
}


# How much internal reasoning to spend. Reasoning is what checks claims against the
# transcript, so it matters where content is created; it is mostly wasted time where the
# model only re-shapes notes that are already written.
THINKING_LEVELS = {
    "none":  {"think_notes": False, "think": False},
    "notes": {"think_notes": True,  "think": False},
    "full":  {"think_notes": True,  "think": True},
}


def language_name(code: str | None) -> str:
    """Return a human-readable language name for use inside LLM prompts."""
    if not code:
        return "the same language as the input text"
    return LANGUAGE_NAMES.get(code.lower(), f"the language with ISO code '{code}'")


@dataclass
class AudioConfig:
    device: int | str | None = None  # None = system default input device
    channels: int = 1


@dataclass
class SttConfig:
    device: str = "auto"  # auto | cuda | cpu
    gpu_model: str = "large-v3-turbo"
    gpu_compute_type: str = "float16"
    cpu_model: str = "small"
    cpu_compute_type: str = "int8"
    language: str | None = "it"  # None = auto-detect
    beam_size: int = 5
    vad_filter: bool = True
    download_root: str | None = None  # None = Hugging Face cache
    # Sections to transcribe, as [(start, end), ...] in seconds. Empty = whole file.
    # Note: faster-whisper silently disables the VAD filter when clips are used.
    clip_ranges: list = field(default_factory=list)


@dataclass
class LlmConfig:
    base_url: str = "http://127.0.0.1:11434"
    notes_model: str = "qwen3.5:9b"
    diagram_model: str | None = None  # None = same as notes_model
    num_ctx: int = 16384
    notes_temperature: float = 0.3
    diagram_temperature: float = 0.2
    # Reasoning is applied per step, because it buys fidelity but costs a lot of time.
    # Measured on the same 30s source with qwen3.5:9b: notes took 119s with reasoning and
    # 26s without, but the fast version invented content that was not in the transcript
    # (gravitational potential, atomic clocks). Reasoning is what checks claims against
    # the source, so it stays ON where content is created from the transcript, and OFF for
    # the diagram/card steps, which only re-shape notes that are already written.
    # Accepted by non-reasoning models too, so it is safe as a default.
    think_notes: bool | None = True
    think: bool | None = False
    keep_alive: str = "10m"
    request_timeout: int = 600  # seconds between streamed chunks
    chunk_chars: int = 12000  # longer transcripts are processed in parts
    concept_cards: int = 4  # discursive cards to generate after the main diagram (0 = off)
    # Long notes cannot fit one readable diagram, so above this size the main diagram
    # becomes a map of the topics and each topic gets its own detailed diagram.
    split_topics_over_chars: int = 2500
    max_topic_diagrams: int = 12
    min_labeled_edge_ratio: float = 0.5  # below this, the main diagram is sent back for revision
    # Subject hints, prepended to the generation prompts so the model knows the domain.
    # `subtopics` is also stored in the transcript for a future semantic filter.
    topic: str = ""
    subtopics: str = ""


@dataclass
class FilterConfig:
    """Automatic removal of pauses and low-confidence speech."""

    enabled: bool = False
    mode: str = "review"  # review = propose and confirm | auto = apply straight away
    gap_seconds: float = 20.0
    min_logprob: float = -0.8
    max_no_speech: float = 0.6


@dataclass
class RenderConfig:
    renderer: str = "auto"  # auto | native | wsl | html
    wsl_distro: str | None = "Debian"  # None = default WSL distro
    mmdc_command: str = "mmdc"
    puppeteer_config: str | None = None  # path as seen by the renderer (Linux path for WSL)
    image_format: str = "png"  # png | svg
    scale: int = 2
    timeout: int = 180
    max_fix_attempts: int = 2
    local_mermaid_js: Path = PROJECT_ROOT / "assets" / "mermaid.min.js"
    mermaid_cdn_url: str = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"


@dataclass
class AppConfig:
    output_root: Path = PROJECT_ROOT / "output"
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
