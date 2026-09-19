"""Notes → Mermaid code, with render-based validation and LLM repair loop."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from mictotext import prompts
from mictotext.config import LlmConfig, RenderConfig
from mictotext.html_export import write_html
from mictotext.llm import OllamaClient, OllamaError

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
---"""
# NOTE: do not set `fontFamily` here. Mermaid measures the label box with one font and
# the headless Chromium renders with another, so a custom family silently clips long
# labels (node titles came out truncated). Its own default keeps the two in sync.

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
    html_path: Path | None  # only written when no image could be rendered
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
                   layout: str = PRIMARY_LAYOUT, with_html: bool = False) -> str:
    """Write the .mmd source, and the browser fallback page only when asked.

    The HTML page is not produced on a successful render: the PNG is the deliverable and
    an extra file per diagram is just clutter. It is still written when no image could be
    made, so a failure does not leave only unrendered Mermaid source behind.
    """
    full_code = f"{diagram_header(layout)}\n{code}\n"
    mmd_path.write_text(full_code, encoding="utf-8")
    if with_html:
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

    # Quality gate: a graph of unlabelled arrows between topic titles explains nothing.
    # Prompting alone does not reliably produce labels, so measure it and send it back once.
    labeled, total = labeled_edge_ratio(code)
    if total >= 4 and labeled / total < llm_cfg.min_labeled_edge_ratio:
        print(f"\n  Solo {labeled}/{total} archi etichettati: chiedo una revisione...")
        messages.append({"role": "user", "content": prompts.DIAGRAM_LABEL_FIX.format(
            labeled=labeled, total=total)})
        try:
            revised = _ask_for_code(client, model, messages, llm_cfg.diagram_temperature)
            new_labeled, new_total = labeled_edge_ratio(revised)
            if new_total and new_labeled / new_total > labeled / max(total, 1):
                code = revised
                print(f"  Revisione accettata: {new_labeled}/{new_total} archi etichettati.")
            else:
                print("  Revisione non migliorativa: tengo la versione originale.")
        except (DiagramError, OllamaError) as exc:
            print(f"  Revisione fallita ({exc}): tengo la versione originale.")

    _write_sources(code, mmd_path, html_path, render_cfg)

    if renderer is None:
        # Without mmdc the browser page is the only viewable output, so it is written.
        _write_sources(code, mmd_path, html_path, render_cfg, with_html=True)
        return DiagramResult(mmd_path, html_path, None, "No mermaid-cli renderer available.")

    last_error = ""
    for attempt in range(render_cfg.max_fix_attempts + 1):
        print(f"  Rendering with {renderer.name} (attempt {attempt + 1})...")
        ok, output = _render_with_layout_fallback(code, mmd_path, html_path, render_cfg,
                                                  renderer, image_path)
        if ok:
            return DiagramResult(mmd_path, None, image_path)

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

    # No image could be produced: fall back to the browser page so something is viewable.
    _write_sources(code, mmd_path, html_path, render_cfg, with_html=True)
    return DiagramResult(mmd_path, html_path, None, last_error)


# --- Edge-label quality gate --------------------------------------------------------

_EDGE_RE = re.compile(r"--[->]")
_LABELED_EDGE_RE = re.compile(r"--[->]\s*\|")


def labeled_edge_ratio(code: str) -> tuple[int, int]:
    """Return (labelled arrows, total arrows). Unlabelled arrows explain nothing."""
    total = len(_EDGE_RE.findall(code))
    labeled = len(_LABELED_EDGE_RE.findall(code))
    return labeled, total


# --- Discursive concept cards -------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    normalised = unicodedata.normalize("NFKD", text.lower())
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    return _SLUG_RE.sub("-", ascii_text).strip("-")[:40] or "concetto"


def extract_concepts(notes_md: str, client: OllamaClient, model: str, llm_cfg: LlmConfig,
                     language: str, max_concepts: int) -> list[str]:
    """Ask the model which concepts deserve their own explanation card."""
    messages = [
        {"role": "system", "content": prompts.CONCEPTS_SYSTEM.format(
            language=language, max_concepts=max_concepts)},
        {"role": "user", "content": prompts.CONCEPTS_USER.format(notes=notes_md)},
    ]
    raw = client.chat(model, messages, llm_cfg.diagram_temperature)
    concepts = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        # Guard against the model echoing the document's scaffolding sections.
        if cleaned and not _is_scaffolding(cleaned) and len(cleaned) <= 60:
            concepts.append(cleaned)
    return concepts[:max_concepts]


_SCAFFOLDING = ("summary", "sintesi", "key terms", "termini chiave", "glossario",
                "open questions", "domande aperte", "next steps", "prossimi passi",
                "introduzione", "conclusione", "conclusioni")


def _is_scaffolding(text: str) -> bool:
    lowered = text.lower().strip(": ")
    return any(lowered.startswith(marker) for marker in _SCAFFOLDING)


def generate_concept_cards(notes_md: str, concepts: list[str], client: OllamaClient, model: str,
                           llm_cfg: LlmConfig, render_cfg: RenderConfig, renderer,
                           output_dir: Path, language: str) -> list[Path]:
    """Generate one discursive card per concept. A failing card is skipped, not fatal."""
    cards_dir = output_dir / "concetti"
    cards_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for index, concept in enumerate(concepts, start=1):
        stem = f"{index:02d}-{_slug(concept)}"
        print(f"\n  --- Scheda {index}/{len(concepts)}: {concept} ---")
        messages = [
            {"role": "system", "content": prompts.CARD_SYSTEM.format(
                concept=concept, language=language)},
            {"role": "user", "content": prompts.CARD_USER.format(concept=concept, notes=notes_md)},
        ]
        try:
            code = _ask_for_code(client, model, messages, llm_cfg.diagram_temperature)
        except (DiagramError, OllamaError) as exc:
            print(f"  Scheda saltata ({exc}).")
            continue

        mmd_path = cards_dir / f"{stem}.mmd"
        html_path = cards_dir / f"{stem}.html"
        image_path = cards_dir / f"{stem}.{render_cfg.image_format}"
        if renderer is None:
            _write_sources(code, mmd_path, html_path, render_cfg, with_html=True)
            continue

        ok, output = _render_with_layout_fallback(code, mmd_path, html_path, render_cfg,
                                                  renderer, image_path)
        if ok:
            produced.append(image_path)
        else:
            print(f"  Render della scheda fallito: {output.splitlines()[-1][:120] if output else '?'}")
            _write_sources(code, mmd_path, html_path, render_cfg, with_html=True)

    return produced
