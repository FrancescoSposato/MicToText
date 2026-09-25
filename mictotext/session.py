"""Naming of session folders: "Titolo_DD_MM_YY", from the title the AI gave the notes.

The folder has to exist before the title does — the recording and the transcript are
written into it first — so it is created with a timestamp name and renamed once the
notes exist. The title costs nothing extra: the notes prompt already makes the model
open with "# <Short descriptive title>".
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9]+")
_MAX_TITLE_CHARS = 50
# Windows refuses these as file or folder names, with any extension.
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def extract_title(notes_md: str, fallback: str = "") -> str:
    """The first "# " heading of the notes, else the fallback, else "Sessione"."""
    match = _HEADING_RE.search(notes_md or "")
    title = match.group(1).strip() if match else ""
    return title or (fallback or "").strip() or "Sessione"


def safe_title(title: str) -> str:
    """Make a title safe as a folder name on Windows and inside a URL.

    Accents are transliterated to plain ASCII: the name ends up in /files/<name>/ URLs
    built by the page, and plain ASCII avoids any encoding surprise there.
    """
    ascii_text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    words = [w for w in _UNSAFE_RE.split(ascii_text) if w]
    name = ""
    for word in words:  # cut at a word boundary, never mid-word
        candidate = f"{name}_{word}" if name else word
        if len(candidate) > _MAX_TITLE_CHARS:
            break
        name = candidate
    if not name:
        name = (words[0][:_MAX_TITLE_CHARS] if words else "Sessione")
    if name.upper() in _RESERVED:
        name = f"Sessione_{name}"
    return name


def folder_name(title: str, when: datetime | None = None) -> str:
    """"Titolo_DD_MM_YY" — the date format requested, day first as usual in Italian."""
    when = when or datetime.now()
    return f"{safe_title(title)}_{when:%d_%m_%y}"


def unique_dir(root: Path, name: str) -> Path:
    """`root/name`, or `name_2`, `name_3`… if taken: the same lecture twice a day is normal."""
    candidate = root / name
    counter = 2
    while candidate.exists():
        candidate = root / f"{name}_{counter}"
        counter += 1
    return candidate


def rename_session(session_dir: Path, notes_md: str, fallback: str = "") -> Path:
    """Rename a session folder after its notes' title. Returns the path to use from now on.

    Never fails the run: if the rename is refused (a file held open, antivirus scanning
    the folder) the session simply keeps its timestamp name.
    """
    title = extract_title(notes_md, fallback)
    started = _started_at(session_dir.name)
    target = unique_dir(session_dir.parent, folder_name(title, started))
    try:
        session_dir.rename(target)
    except OSError as exc:
        print(f"  Cartella non rinominata ({exc}): resta {session_dir.name}")
        return session_dir
    _fix_source_path(target, session_dir)
    print(f"  Cartella della sessione: {target.name}")
    return target


def _fix_source_path(new_dir: Path, old_dir: Path) -> None:
    """Keep trascrizione.json truthful after the move.

    For microphone and URL sessions the source audio lives inside the folder, so the
    recorded path would otherwise point at a folder that no longer exists. A local file
    read in place is outside it and is left alone.
    """
    import json
    meta = new_dir / "trascrizione.json"
    if not meta.is_file():
        return
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        source = Path(data.get("source_path") or "")
        if source.is_absolute() and old_dir.resolve() in source.resolve().parents:
            data["source_path"] = str(new_dir / source.resolve().relative_to(old_dir.resolve()))
            meta.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass  # metadata only: never worth failing the run over


def _started_at(name: str) -> datetime:
    """The session's own start date, read from its timestamp name when possible."""
    try:
        return datetime.strptime(name[:17], "%Y-%m-%d_%H%M%S")
    except ValueError:
        return datetime.now()


def is_session_name(root: Path, name: str) -> bool:
    """Path-traversal guard for URLs that name a session folder.

    Replaces a check that only accepted timestamp names: now any single, plain folder
    name directly inside the output root is accepted, and nothing that climbs out of it.
    """
    if not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
        return False
    try:
        resolved = (root / name).resolve()
        return resolved.parent == root.resolve() and resolved.is_dir()
    except (OSError, ValueError):
        return False
