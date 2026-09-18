"""Notes → Mermaid code, with render-based validation and LLM repair loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mictotext import prompts
from mictotext.config import LlmConfig, RenderConfig
from mictotext.html_export import write_html
from mictotext.llm import OllamaClient

# Prepended by the app (not by the LLM) so that theme and white background are guaranteed.
INIT_DIRECTIVE = (
    '%%{init: {"theme": "base", "themeVariables": {"background": "#ffffff", '
    '"primaryColor": "#DCEBFA", "primaryTextColor": "#1F2937", "lineColor": "#7B8494", '
    '"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif"}}}%%'
)

_FENCE_RE = re.compile(r"`{3}(?:mermaid)?[ \t]*\n(.*?)`{3}", re.DOTALL | re.IGNORECASE)
_START_RE = re.compile(r"^\s*(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b", re.IGNORECASE)
_SYNTAX_ERROR_MARKERS = ("Parse error", "Lexical error", "Syntax error", "Expecting",
                         "UnknownDiagramError", "No diagram type detected")
_SMART_QUOTES = ("“", "”", "„", "«", "»")


class DiagramError(RuntimeError):
    """Raised when the LLM does not produce usable Mermaid code."""


@dataclass
class DiagramResult:
    mmd_path: Path
    html_path: Path
    image_path: Path | None
    error: str = ""


def clean_mermaid(raw: str) -> str:
    """Extract bare Mermaid flowchart code from an LLM reply."""
    text = raw.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)

    lines = [
        line.rstrip() for line in text.splitlines()
        if not line.lstrip().startswith("%%{") and not line.strip().startswith("`" * 3)
    ]
    start = next((i for i, line in enumerate(lines) if _START_RE.match(line)), None)
    if start is None:
        raise DiagramError("The model did not return Mermaid flowchart code.")

    code = "\n".join(lines[start:]).strip()
    for quote in _SMART_QUOTES:
        code = code.replace(quote, "'")
    return code


def _is_syntax_error(message: str) -> bool:
    return any(marker in message for marker in _SYNTAX_ERROR_MARKERS)


def _ask_for_code(client: OllamaClient, model: str, messages: list[dict], temperature: float) -> str:
    raw = client.chat(model, messages, temperature)
    try:
        code = clean_mermaid(raw)
    except DiagramError:
        print("  Reply was not Mermaid code, asking again...")
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": prompts.DIAGRAM_FORMAT_REMINDER}]
        raw = client.chat(model, messages, temperature)
        code = clean_mermaid(raw)
    messages.append({"role": "assistant", "content": code})
    return code


def _write_sources(code: str, mmd_path: Path, html_path: Path, render_cfg: RenderConfig) -> str:
    full_code = f"{INIT_DIRECTIVE}\n{code}\n"
    mmd_path.write_text(full_code, encoding="utf-8")
    write_html(full_code, html_path, render_cfg)
    return full_code


def generate_diagram(notes_md: str, client: OllamaClient, model: str, llm_cfg: LlmConfig,
                     render_cfg: RenderConfig, renderer, output_dir: Path, language: str) -> DiagramResult:
    mmd_path = output_dir / "schema.mmd"
    html_path = output_dir / "schema.html"
    image_path = output_dir / f"schema.{render_cfg.image_format}"

    messages = [
        {"role": "system", "content": prompts.DIAGRAM_SYSTEM.format(language=language)},
        {"role": "user", "content": prompts.DIAGRAM_USER.format(notes=notes_md)},
    ]
    code = _ask_for_code(client, model, messages, llm_cfg.diagram_temperature)
    _write_sources(code, mmd_path, html_path, render_cfg)

    if renderer is None:
        return DiagramResult(mmd_path, html_path, None, "No mermaid-cli renderer available.")

    last_error = ""
    for attempt in range(render_cfg.max_fix_attempts + 1):
        print(f"  Rendering with {renderer.name} (attempt {attempt + 1})...")
        ok, output = renderer.render(mmd_path, image_path)
        if ok:
            return DiagramResult(mmd_path, html_path, image_path)

        last_error = output
        print(f"  Render failed:\n    " + "\n    ".join(output.splitlines()[-6:]))
        if not _is_syntax_error(output):
            print("  This does not look like a Mermaid syntax error: skipping LLM repair.")
            break
        if attempt == render_cfg.max_fix_attempts:
            break

        print("  Asking the LLM to fix the diagram...")
        messages.append({"role": "user", "content": prompts.DIAGRAM_FIX_USER.format(error=output[-1500:])})
        try:
            code = _ask_for_code(client, model, messages, llm_cfg.diagram_temperature)
        except DiagramError as exc:
            last_error = str(exc)
            break
        _write_sources(code, mmd_path, html_path, render_cfg)

    return DiagramResult(mmd_path, html_path, None, last_error)
