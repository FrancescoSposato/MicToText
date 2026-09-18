"""Transcript → structured Markdown notes, with map-reduce for long transcripts."""

from __future__ import annotations

import re

from mictotext import prompts
from mictotext.config import LlmConfig
from mictotext.llm import OllamaClient

_OUTER_FENCE_RE = re.compile(r"^\s*`{3}(?:markdown|md)?\s*\n(.*?)\n`{3}\s*$", re.DOTALL | re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, preferring sentence boundaries."""
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        # Whisper output can lack punctuation: hard-split overly long "sentences" on spaces.
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _clean(markdown: str) -> str:
    match = _OUTER_FENCE_RE.match(markdown)
    return (match.group(1) if match else markdown).strip() + "\n"


def generate_notes(transcript: str, client: OllamaClient, model: str, cfg: LlmConfig, language: str) -> str:
    text = " ".join(transcript.split())
    chunks = split_text(text, cfg.chunk_chars)

    if len(chunks) == 1:
        messages = [
            {"role": "system", "content": prompts.NOTES_SYSTEM.format(language=language)},
            {"role": "user", "content": prompts.NOTES_USER.format(transcript=text)},
        ]
        return _clean(client.chat(model, messages, cfg.notes_temperature))

    print(f"  Long transcript: processing {len(chunks)} parts, then merging.")
    partial_notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n  --- Part {index}/{len(chunks)} ---")
        messages = [
            {"role": "system", "content": prompts.NOTES_CHUNK_SYSTEM.format(
                language=language, index=index, total=len(chunks))},
            {"role": "user", "content": prompts.NOTES_USER.format(transcript=chunk)},
        ]
        partial_notes.append(_clean(client.chat(model, messages, cfg.notes_temperature)))

    print("\n  --- Merging parts ---")
    joined = "\n\n".join(f"<!-- part {i} -->\n{notes}" for i, notes in enumerate(partial_notes, start=1))
    messages = [
        {"role": "system", "content": prompts.NOTES_MERGE_SYSTEM.format(language=language)},
        {"role": "user", "content": prompts.NOTES_MERGE_USER.format(parts=joined)},
    ]
    return _clean(client.chat(model, messages, cfg.notes_temperature))
