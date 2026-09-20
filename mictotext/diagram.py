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


def plan_diagrams(notes_md: str, llm_cfg: LlmConfig) -> tuple[str, list[tuple[str, str]]]:
    """Decide what the main diagram covers, and which topics get their own.

    Returns (source for the main diagram, topics to detail). For short notes the main
    diagram is built from everything and there are no topic diagrams, exactly as before.
    """
    if len(notes_md) <= llm_cfg.split_topics_over_chars:
        return notes_md, []
    topics = split_notes_by_topic(notes_md)
    if len(topics) < 2:
        return notes_md, []
    topics = topics[:llm_cfg.max_topic_diagrams]
    return notes_outline(notes_md, topics), topics


def generate_diagram(notes_md: str, client: OllamaClient, model: str, llm_cfg: LlmConfig,
                     render_cfg: RenderConfig, renderer, output_dir: Path, language: str) -> DiagramResult:
    mmd_path = output_dir / "schema.mmd"
    html_path = output_dir / "schema.html"
    image_path = output_dir / f"schema.{render_cfg.image_format}"

    messages = [
        {"role": "system", "content": prompts.with_topic(
            prompts.DIAGRAM_SYSTEM.format(language=language),
            llm_cfg.topic, llm_cfg.subtopics)},
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
        {"role": "system", "content": prompts.with_topic(
            prompts.CONCEPTS_SYSTEM.format(language=language, max_concepts=max_concepts),
            llm_cfg.topic, llm_cfg.subtopics)},
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
            {"role": "system", "content": prompts.with_topic(
                prompts.CARD_SYSTEM.format(concept=concept, language=language),
                llm_cfg.topic, llm_cfg.subtopics)},
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


# --- Splitting long notes into one diagram per topic ---------------------------------

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_MIN_TOPIC_CHARS = 120  # below this a section has nothing worth diagramming


def split_notes_by_topic(notes_md: str) -> list[tuple[str, str]]:
    """Split notes at '## ' headings into (title, markdown) pairs.

    The notes already carry the author's own topic structure, so there is no need to ask
    a model where to cut. Scaffolding sections are skipped: they are not topics.
    """
    matches = list(_HEADING_RE.finditer(notes_md))
    topics: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(notes_md)
        body = notes_md[match.start():end].strip()
        if _is_scaffolding(title) or len(body) < _MIN_TOPIC_CHARS:
            continue
        topics.append((title, body))
    return topics


def notes_outline(notes_md: str, topics: list[tuple[str, str]]) -> str:
    """A condensed view of the notes: the title plus each topic's first bullets.

    Feeding the overview diagram the whole document would just reproduce the problem it
    exists to solve, so it only sees the skeleton.
    """
    first_line = notes_md.strip().splitlines()[0] if notes_md.strip() else ""
    parts = [first_line if first_line.startswith("# ") else "# Contenuto"]
    for title, body in topics:
        parts.append(f"\n## {title}")
        bullets = [ln.strip() for ln in body.splitlines() if ln.strip().startswith(("-", "*"))]
        parts.extend(f"  {b}" for b in bullets[:2])
    return "\n".join(parts)


def generate_topic_diagrams(topics: list[tuple[str, str]], client: OllamaClient, model: str,
                            llm_cfg: LlmConfig, render_cfg: RenderConfig, renderer,
                            output_dir: Path, language: str) -> list[Path]:
    """One logical diagram per topic. A failing topic is skipped, not fatal."""
    topics_dir = output_dir / "schemi"
    topics_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for index, (title, body) in enumerate(topics, start=1):
        stem = f"{index:02d}-{_slug(title)}"
        print(f"\n  --- Schema {index}/{len(topics)}: {title} ---")
        messages = [
            {"role": "system", "content": prompts.with_topic(
            prompts.DIAGRAM_SYSTEM.format(language=language),
            llm_cfg.topic, llm_cfg.subtopics)},
            {"role": "user", "content": prompts.DIAGRAM_USER.format(notes=body)},
        ]
        try:
            code = _ask_for_code(client, model, messages, llm_cfg.diagram_temperature)
        except (DiagramError, OllamaError) as exc:
            print(f"  Schema saltato ({exc}).")
            continue

        mmd_path = topics_dir / f"{stem}.mmd"
        html_path = topics_dir / f"{stem}.html"
        image_path = topics_dir / f"{stem}.{render_cfg.image_format}"
        if renderer is None:
            _write_sources(code, mmd_path, html_path, render_cfg, with_html=True)
            continue

        ok, output = _render_with_layout_fallback(code, mmd_path, html_path, render_cfg,
                                                  renderer, image_path)
        if ok:
            produced.append(image_path)
        else:
            print(f"  Render fallito: {output.splitlines()[-1][:120] if output else '?'}")
            _write_sources(code, mmd_path, html_path, render_cfg, with_html=True)

    return produced


# --- Targeted revision of a single diagram or card -----------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_EMPTY_FEEDBACK = "(nessuna indicazione)"


def _strip_header(full_code: str) -> str:
    return _FRONTMATTER_RE.sub("", full_code).strip()


_STOPWORDS = {"della", "delle", "degli", "dello", "questo", "questa", "queste", "questi",
              "conservare", "conservata", "conservati", "mantenere", "mantenuta", "distinzione",
              "parte", "punto", "sezione", "schema", "diagramma", "nodo", "nodi", "vanno",
              "essere", "sono", "come", "anche", "molto", "tutto", "tutti", "tutte"}


def _kept_terms(correct_text: str, previous_code: str) -> list[str]:
    """Words the user asked to keep that were actually in the previous diagram.

    Only those can be checked: a term that was never there cannot have been dropped.
    """
    words = {w.lower() for w in re.findall(r"[^\W\d_]{5,}", correct_text, re.UNICODE)}
    lowered = previous_code.lower()
    return sorted(w for w in words - _STOPWORDS if w in lowered)


def _dropped_terms(terms: list[str], new_code: str) -> list[str]:
    lowered = new_code.lower()
    return [t for t in terms if t not in lowered]


def regenerate_from_feedback(mmd_path: Path, image_path: Path, notes_md: str, feedback: dict,
                             kind: str, client: OllamaClient, model: str, llm_cfg: LlmConfig,
                             render_cfg: RenderConfig, renderer, language: str,
                             concept: str = "") -> tuple[bool, str]:
    """Revise one existing diagram according to user feedback, in place.

    `kind` picks which set of rules applies: "card" for the discursive cards, anything
    else for the relational diagrams. The previous code is replayed as the assistant's
    turn so the model revises it rather than starting over.
    """
    if not mmd_path.exists():
        return False, "Il file di origine non esiste piu'."

    previous = _strip_header(mmd_path.read_text(encoding="utf-8"))
    if kind == "card":
        system = prompts.with_topic(
            prompts.CARD_SYSTEM.format(concept=concept or "il concetto", language=language),
            llm_cfg.topic, llm_cfg.subtopics)
        first_user = prompts.CARD_USER.format(concept=concept or "", notes=notes_md)
    else:
        system = prompts.with_topic(prompts.DIAGRAM_SYSTEM.format(language=language),
                                    llm_cfg.topic, llm_cfg.subtopics)
        first_user = prompts.DIAGRAM_USER.format(notes=notes_md)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": first_user},
        {"role": "assistant", "content": previous},
        {"role": "user", "content": prompts.REVISE_USER.format(
            what_is_wrong=feedback.get("what_is_wrong") or _EMPTY_FEEDBACK,
            missing=feedback.get("missing") or _EMPTY_FEEDBACK,
            correct=feedback.get("correct") or _EMPTY_FEEDBACK,
            options=feedback.get("options") or _EMPTY_FEEDBACK,
        )},
    ]

    try:
        code = _ask_for_code(client, model, messages, llm_cfg.diagram_temperature)
    except (DiagramError, OllamaError) as exc:
        return False, str(exc)

    # Models reliably honour "what is wrong" but quietly drop items from "keep these",
    # especially when another part of the feedback asks to shrink the diagram. Checking
    # is cheap; trusting is not.
    warning = ""
    protected = _kept_terms(feedback.get("correct") or "", previous)
    dropped = _dropped_terms(protected, code)
    if dropped:
        print(f"  Elementi da conservare spariti ({', '.join(dropped)}): richiedo una correzione...")
        messages.append({"role": "user", "content": prompts.REVISE_KEEP_FIX.format(
            dropped=", ".join(dropped))})
        try:
            retry = _ask_for_code(client, model, messages, llm_cfg.diagram_temperature)
            still = _dropped_terms(protected, retry)
            if len(still) < len(dropped):
                code, dropped = retry, still
        except (DiagramError, OllamaError):
            pass
        if dropped:
            warning = (f" Attenzione: non ha conservato {', '.join(dropped)} nonostante la "
                       f"richiesta.")

    html_path = mmd_path.with_suffix(".html")
    if renderer is None:
        _write_sources(code, mmd_path, html_path, render_cfg, with_html=True)
        return True, "Rigenerato (nessun renderer: solo HTML)."

    backup = mmd_path.read_text(encoding="utf-8")
    ok, output = _render_with_layout_fallback(code, mmd_path, html_path, render_cfg,
                                              renderer, image_path)
    if ok:
        return True, "Rigenerato." + warning

    # Keep the working version rather than leaving a broken one behind.
    mmd_path.write_text(backup, encoding="utf-8")
    return False, f"La revisione non si renderizza, versione precedente mantenuta. {output[-200:]}"
