"""Cooperative cancellation shared by every pipeline stage.

The stages run in very different places — Whisper in a child process, generation as
streamed HTTP requests to Ollama, downloads inside yt-dlp — so there is no single thing
to kill. Instead one `threading.Event` is handed to each of them, and each checks it at
points where stopping is safe.

`Cancelled` deliberately does NOT subclass OllamaError or DiagramError: the loops that
skip a single failing diagram or card catch those, and must not swallow a cancellation
and carry on with the next item.
"""

from __future__ import annotations

import threading


class Cancelled(Exception):
    """Raised when the user stops the running operation."""

    def __init__(self, message: str = "Interrotto dall'utente.") -> None:
        super().__init__(message)


def check(event: threading.Event | None) -> None:
    """Raise Cancelled if cancellation has been requested. A None event never cancels."""
    if event is not None and event.is_set():
        raise Cancelled()
