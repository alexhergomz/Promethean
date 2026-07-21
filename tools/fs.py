"""tools_fs.py — File-system tool implementations: Read, Write, Edit, Glob."""
from __future__ import annotations

import difflib
import os
import re
from pathlib import Path


def _resolve(file_path: str) -> Path:
    """Normalize a model-supplied path: expand ~ and $ENV before use.

    Models often emit literal '~/Desktop/...' paths; without expansion the
    tilde is taken literally and a bogus '~' directory is created under cwd.
    """
    return Path(os.path.expandvars(os.path.expanduser(file_path)))


def _read_preserving_newlines(p: Path) -> str:
    """Read a text file without newline translation.

    Path.read_text gained a `newline=` parameter only in Python 3.14; the
    project supports 3.10+, so we use open() which has accepted `newline=`
    since the pathlib API was introduced.
    """
    with p.open(encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


# ── Diff helpers ──────────────────────────────────────────────────────────

def generate_unified_diff(old: str, new: str, filename: str,
                           context_lines: int = 3) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        n=context_lines,
    )
    return "".join(diff)


def maybe_truncate_diff(diff_text: str, max_lines: int = 80) -> str:
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    shown     = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown) + f"\n\n[... {remaining} more lines ...]"


# ── Read ─────────────────────────────────────────────────────────────────

def _read(file_path: str, limit: int = None, offset: int = None) -> str:
    p = _resolve(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if p.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        lines = _read_preserving_newlines(p).splitlines(keepends=True)
        start = offset or 0
        chunk = lines[start:start + limit] if limit else lines[start:]
        if not chunk:
            return "(empty file)"
        return "".join(f"{start + i + 1:6}\t{l}" for i, l in enumerate(chunk))
    except Exception as e:
        return f"Error: {e}"


# ── Write ─────────────────────────────────────────────────────────────────

def _write(file_path: str, content: str) -> str:
    p = _resolve(file_path)
    try:
        is_new      = not p.exists()
        old_content = "" if is_new else _read_preserving_newlines(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")
        if is_new:
            lc = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            # Synthetic all-additions diff so print_tool_end's renderer
            # gives a content preview instead of just a one-line summary.
            # Capped tighter (40 lines) than edit diffs because new files
            # can be very long.
            preview = generate_unified_diff("", content, p.name)
            return (f"Created {file_path} ({lc} lines):\n\n"
                    + maybe_truncate_diff(preview, max_lines=40))
        diff = generate_unified_diff(old_content, content, p.name)
        if not diff:
            return f"No changes in {file_path}"
        return f"File updated — {file_path}:\n\n{maybe_truncate_diff(diff)}"
    except Exception as e:
        return f"Error: {e}"


# ── Edit ──────────────────────────────────────────────────────────────────
#
# Local models are much worse than frontier models at reproducing an exact
# byte-for-byte span. The three common failure modes are: (1) copying the
# "  123\t" line-number prefix out of a Read result into old_string, (2)
# getting the indentation slightly wrong, and (3) a small transcription
# drift (a renamed variable, a dropped comment). A strict exact-match Edit
# rejects all three and the model loops re-emitting near-identical mistakes.
#
# _locate_inexact recovers those cases when the verbatim search misses. It
# only ever applies a *uniquely* located span; anything ambiguous is
# reported back so the model adds context rather than editing blind. Every
# recovered edit is annotated and the full diff is returned, so a reviewer
# can see exactly what changed.

_AMBIGUOUS = object()          # sentinel: matched, but in more than one place

_LINE_NO_PREFIX = re.compile(r"(?m)^[ \t]*\d+\t")   # Read's "   123\t" gutter
_FUZZY_MIN_RATIO = 0.90        # similarity floor for the drift-tolerant pass
_FUZZY_MIN_MARGIN = 0.05       # winner must beat the runner-up by this much


def _line_offsets(lines: list[str]) -> list[int]:
    """Prefix character offsets of each line in "\n".join(lines).

    offsets[i] is the start of line i; offsets[len(lines)] is one past the
    end. Reconstructing with "\n" means each line contributes len+1 chars.
    """
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _block_span(content_lines: list[str], start_line: int, n: int) -> tuple[int, int]:
    """Character span of content_lines[start_line:start_line+n] within the
    newline-joined text, excluding the trailing newline after the block."""
    offsets = _line_offsets(content_lines)
    start = offsets[start_line]
    end   = offsets[start_line + n] - 1   # drop the separator after the block
    return start, max(start, end)


def _locate_inexact(content_norm: str, old_norm: str):
    """Find old_norm in content_norm when a verbatim search failed.

    Returns (start, end, strategy) for a unique match, _AMBIGUOUS if a
    match exists but is not unique, or None if nothing plausible was found.
    """
    ambiguous = False

    # 1. Strip Read-style line-number prefixes and retry verbatim.
    destripped = _LINE_NO_PREFIX.sub("", old_norm)
    if destripped and destripped != old_norm:
        c = content_norm.count(destripped)
        if c == 1:
            i = content_norm.index(destripped)
            return i, i + len(destripped), "line-number-stripped match"
        if c > 1:
            ambiguous = True

    old_lines = old_norm.split("\n")
    if old_lines and old_lines[-1] == "":
        old_lines = old_lines[:-1]      # a trailing newline is not its own line
    if not old_lines:
        return _AMBIGUOUS if ambiguous else None

    content_lines = content_norm.split("\n")
    n = len(old_lines)

    # 2. Indentation-insensitive block match (compare lines stripped).
    target = [ln.strip() for ln in old_lines]
    hits = [
        i for i in range(len(content_lines) - n + 1)
        if [content_lines[j].strip() for j in range(i, i + n)] == target
    ]
    if len(hits) == 1:
        start, end = _block_span(content_lines, hits[0], n)
        return start, end, "indentation-insensitive match"
    if len(hits) > 1:
        ambiguous = True

    # 3. Drift-tolerant match: the most-similar same-height window, but only
    #    if it clears a high similarity floor and clearly beats the next
    #    candidate. Guards against silently editing the wrong span.
    old_block = "\n".join(old_lines)
    scored = []
    for i in range(len(content_lines) - n + 1):
        window = "\n".join(content_lines[i:i + n])
        scored.append((difflib.SequenceMatcher(None, old_block, window).ratio(), i))
    if scored:
        scored.sort(reverse=True)
        best_ratio, best_i = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        if best_ratio >= _FUZZY_MIN_RATIO and best_ratio - runner >= _FUZZY_MIN_MARGIN:
            start, end = _block_span(content_lines, best_i, n)
            return start, end, f"closest match ({best_ratio * 100:.0f}% similar)"

    return _AMBIGUOUS if ambiguous else None


def _edit(file_path: str, old_string: str, new_string: str,
          replace_all: bool = False, fuzzy: bool = True) -> str:
    p = _resolve(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    try:
        content = _read_preserving_newlines(p)

        crlf_count = content.count("\r\n")
        lf_count   = content.count("\n")
        is_pure_crlf = crlf_count > 0 and crlf_count == lf_count

        content_norm = content.replace("\r\n", "\n")
        old_norm     = old_string.replace("\r\n", "\n")
        new_norm     = new_string.replace("\r\n", "\n")

        note  = ""
        count = content_norm.count(old_norm)
        if count == 0:
            located = _locate_inexact(content_norm, old_norm) if fuzzy else None
            if located is None:
                return ("Error: old_string not found in file. Match it exactly, "
                        "including indentation and surrounding lines, and drop any "
                        "line-number prefixes from Read output. Re-read the file if "
                        "it may have changed.")
            if located is _AMBIGUOUS:
                return ("Error: old_string has no exact match, and an approximate "
                        "match appears in more than one place. Add surrounding lines "
                        "to make it unique.")
            start, end, strategy = located
            new_content_norm = content_norm[:start] + new_norm + content_norm[end:]
            note = (f"\n\n(no verbatim match — applied via {strategy}. "
                    "Confirm the diff above is the change you intended.)")
        elif count > 1 and not replace_all:
            return (f"Error: old_string appears {count} times. "
                    "Provide more context to make it unique, or use replace_all=true.")
        elif replace_all:
            new_content_norm = content_norm.replace(old_norm, new_norm)
        else:
            new_content_norm = content_norm.replace(old_norm, new_norm, 1)

        if is_pure_crlf:
            final_content    = new_content_norm.replace("\n", "\r\n")
            old_content_final = content
        else:
            final_content    = new_content_norm
            old_content_final = content_norm

        p.write_text(final_content, encoding="utf-8", newline="")
        diff = generate_unified_diff(old_content_final, final_content, p.name)
        return f"Changes applied to {p.name}:\n\n{diff}{note}"
    except Exception as e:
        return f"Error: {e}"


# ── Glob ──────────────────────────────────────────────────────────────────

def _glob(pattern: str, path: str = None, cwd: str = None) -> str:
    base = Path(path) if path else (Path(cwd) if cwd else Path.cwd())
    try:
        matches = sorted(base.glob(pattern))
        if not matches:
            return "No files matched"
        return "\n".join(str(m) for m in matches[:500])
    except Exception as e:
        return f"Error: {e}"
