"""Minimal client for the local Ollama HTTP API (no cloud calls)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

import requests

from mictotext.config import LlmConfig

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNSET = object()  # distinguishes "not passed" from an explicit None/False


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def _model_matches(requested: str, available: str) -> bool:
    if requested == available:
        return True
    return ":" not in requested and available == f"{requested}:latest"


class OllamaClient:
    def __init__(self, cfg: LlmConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                f"Ollama is not reachable at {self.base_url} ({exc}). "
                "Start the Ollama app or run 'ollama serve'."
            ) from exc
        return [model["name"] for model in response.json().get("models", [])]

    def ensure_ready(self, models: Iterable[str]) -> None:
        """Fail fast (before recording) if Ollama is down or a model is missing."""
        available = self.list_models()
        missing = [m for m in models if not any(_model_matches(m, a) for a in available)]
        if missing:
            commands = "\n".join(f"  ollama pull {m}" for m in missing)
            raise OllamaError(f"Missing model(s) in Ollama. Run:\n{commands}")

    def chat(self, model: str, messages: list[dict], temperature: float, echo: bool = True,
             think: bool | None = _UNSET) -> str:
        """Send a chat request, streaming tokens to the console. Returns the full reply.

        `think` overrides the configured default for this call: reasoning is worth its
        cost when generating content from the transcript, not when reformatting notes.
        """
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": temperature, "num_ctx": self.cfg.num_ctx},
        }
        effective_think = self.cfg.think if think is _UNSET else think
        if effective_think is not None:
            payload["think"] = effective_think

        parts: list[str] = []
        try:
            with requests.post(f"{self.base_url}/api/chat", json=payload, stream=True,
                               timeout=(10, self.cfg.request_timeout)) as response:
                if response.status_code != 200:
                    raise OllamaError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise OllamaError(data["error"])
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        parts.append(chunk)
                        if echo:
                            print(chunk, end="", flush=True)
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaError(f"Request to Ollama failed: {exc}") from exc
        finally:
            if echo and parts:
                print()

        return _THINK_RE.sub("", "".join(parts)).strip()
