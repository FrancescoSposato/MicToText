"""Notes → Mermaid code, with render-based validation and LLM repair loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mictotext import prompts
from mictotext.config import LlmConfig, RenderConfig
from mictotext.html_export import write_html
from mictotext.llm import OllamaClient

# Prepended by the app (not by the LLM) so that layout, theme and white background are
# guaranteed. The YAML frontmatter form is used because it is the only one that carries
# `layout: elk`: ELK lays out graphs far more compactly than the legacy Dagre engine,
# with orthogonal edge routing and fewer crossings. It applies to graph-based diagram
# types only (flowchart, class, state, ER), which is all this app generates.
_HEADER_TEMPLATE = """---
config:
  layout: {layout}
  theme: base
  themeVariables:
    background: "#ffffff"
    primaryColor: "#DCEBFA"
    primaryTextColor: "#1F2937"
    lineColor: "#7B8494"
    fontFamily: "Segoe UI, Helvetica, Arial, sans-serif"
---"""

# ELK gives much better layouts but is less forgiving than Dagre with odd-but-parseable
# graphs (it can throw instead of rendering), so Dagre stays as an automatic fallback.
PRIMARY_LAYOUT = "elk"
FALLBACK_LAYOUT = "dagre"


def diagram_header(layout: str = PRIMARY_LAYOUT) -> str:
    return _HEADER_TEMPLATE.format(layout=layout)

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
    return _strip_empty_subgraphs(_balance_subgraphs(code))


def _balance_subgraphs(code: str) -> str:
    """Append any missing `end` statements for unclosed subgraphs.

    Small models routinely open several subgraphs and forget to close them, which makes
    the whole diagram unparseable. Fixing it mechanically is far more reliable than asking
    the model to repair its own output.
    """
    lines = code.splitlines()
    opened = sum(1 for line in lines if line.strip().lower().startswith("subgraph "))
    closed = sum(1 for line in lines if line.strip().lower() == "end")
    missing = opened - closed
    if missing <= 0:
        return code

    # Insert before the trailing styling block, so classDef/class/style stay at top level.
    insert_at = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip().lower()
        if stripped.startswith(("classdef ", "class ", "style ", "linkstyle ")):
            insert_at = index
        elif stripped:
            break

    print(f"  Auto-fixed {missing} unclosed subgraph(s).")
    return "\n".join(lines[:insert_at] + ["  end"] * missing + lines[insert_at:])


_SUBGRAPH_RE = re.compile(r"^\s*subgraph\s", re.IGNORECASE)
_DECORATION_RE = re.compile(r"^\s*(classdef|class|style|linkstyle)\s", re.IGNORECASE)


def _strip_empty_subgraphs(code: str) -> str:
    """Drop subgraph blocks that declare no node, which make the ELK layout engine throw."""
    lines = code.splitlines()
    keep = [True] * len(lines)
    index = 0
    removed = 0
    while index < len(lines):
        if not _SUBGRAPH_RE.match(lines[index]):
            index += 1
            continue
        end = next((j for j in range(index + 1, len(lines))
                    if lines[j].strip().lower() == "end"), None)
        if end is None:
            break
        body = lines[index + 1:end]
        has_content = any(
            line.strip() and not _DECORATION_RE.match(line) and not _SUBGRAPH_RE.match(line)
            for line in body
        )
        if not has_content:
            # Keep the decorations (they may define classes used elsewhere), drop the wrapper.
            keep[index] = keep[end] = False
            removed += 1
        index = end + 1

    if not removed:
        return code
    print(f"  Removed {removed} empty subgraph(s).")
    return "\n".join(line for line, k in zip(lines, keep) if k)


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


def _write_sources(code: str, mmd_path: Path, html_path: Path, render_cfg: RenderConfig,
                   layout: str = PRIMARY_LAYOUT) -> str:
    full_code = f"{diagram_header(layout)}\n{code}\n"
    mmd_path.write_text(full_code, encoding="utf-8")
    write_html(full_code, html_path, render_cfg)
    return full_code


def _render_with_layout_fallback(code: str, mmd_path: Path, html_path: Path,
                                 render_cfg: RenderConfig, renderer, image_path: Path) -> tuple[bool, str]:
    """Render with ELK; if ELK itself fails (not a syntax error), retry once with Dagre."""
    _write_sources(code, mmd_path, html_path, render_cfg, PRIMARY_LAYOUT)
    ok, output = renderer.render(mmd_path, image_path)
    if ok or _is_syntax_error(output):
        return ok, output

    print(f"  {PRIMARY_LAYOUT} layout failed, retrying with {FALLBACK_LAYOUT}...")
    _write_sources(code, mmd_path, html_path, render_cfg, FALLBACK_LAYOUT)
    return renderer.render(mmd_path, image_path)


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
        ok, output = _render_with_layout_fallback(code, mmd_path, html_path, render_cfg,
                                                  renderer, image_path)
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
