"""Auto-inject callers/refs info into Edit and Write tool results.

When the model edits a function or class definition, append a short
footer listing other places that reference the symbol so the next turn
can spot-check whether they need follow-up changes. M2 (and most non-
Anthropic models) forget to do this manually; this closes the gap with
zero prompt overhead until an edit actually happens.

Triggers
    Edit       — extract `def NAME` / `class NAME` from old_string OR
                 new_string (catches both "edit the signature" and "add
                 a new def"); look up callers; emit footer.
    Write      — same parse but on the whole content; useful for "I
                 just wrote a new function — show me what already
                 references that name elsewhere".

Bounds (keep tool output tight)
    - Cap symbols inspected per edit at 6.
    - Cap callers shown per symbol at 8. If more, summarize count only.
    - Skip entirely if symbol name is < 3 chars (avoids `i`, `x`, `_t`).
    - Skip Python dunders, single-letter ids, and lowercase one-word
      keywords (return, async, etc.) — handled by the def/class anchor.
"""
from __future__ import annotations

import os
import re
from typing import Optional

# A def or class line — anchored on the keyword so we don't catch
# the name string from comments or docstrings.
_DEF_OR_CLASS = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z_0-9]*)",
                            re.MULTILINE)

_MAX_SYMBOLS_PER_EDIT  = 6
_MAX_CALLERS_PER_SYM   = 8
_MIN_NAME_LEN          = 3


def _extract_symbols(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _DEF_OR_CLASS.finditer(text):
        name = m.group(1)
        if len(name) < _MIN_NAME_LEN or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= _MAX_SYMBOLS_PER_EDIT:
            break
    return out


def for_edit(inputs: dict, cwd: Optional[str] = None) -> str:
    """Build the auto-inject footer for a successful Edit/Write tool result.

    Returns "" when there's nothing to add (no definitions touched, or
    the symbol-graph index is unreachable). Never raises — callers wrap
    this in a try/except either way.
    """
    # Symbol-graph indexing depends on agent_tools/repomap.py + cached
    # tags; loading it is non-trivial. We import lazily so the import
    # cost is paid only on edits, not on every tool call.
    try:
        from agent_tools.helpers import get_symbol_index
    except Exception:
        return ""

    edited_path = inputs.get("file_path", "")
    text = (inputs.get("new_string", "") + "\n"
            + inputs.get("old_string", "") + "\n"
            + inputs.get("content", ""))   # Write uses 'content'
    symbols = _extract_symbols(text)
    if not symbols:
        return ""

    try:
        idx = get_symbol_index(cwd or os.getcwd())
    except Exception:
        return ""

    parts: list[str] = []
    for sym in symbols:
        # `refs` are all references to this name across the repo. Filter
        # out refs in the file we just edited — they're either the
        # def-site itself (which we already touched) or in-file calls
        # already updated by the same Edit.
        refs = idx.refs.get(sym, [])
        cross_file = [
            r for r in refs
            if getattr(r, "rel_fname", "") != edited_path
            and not (edited_path and edited_path.endswith(getattr(r, "rel_fname", "")))
        ]
        if not cross_file:
            continue
        if len(cross_file) > _MAX_CALLERS_PER_SYM:
            tail = (f"+{len(cross_file) - _MAX_CALLERS_PER_SYM} more")
            slice_ = cross_file[:_MAX_CALLERS_PER_SYM]
        else:
            tail = ""
            slice_ = cross_file
        locs = ", ".join(f"{r.rel_fname}:{r.line}" for r in slice_)
        parts.append(
            f"  `{sym}` is referenced in {len(cross_file)} other file"
            f"{'s' if len(cross_file) != 1 else ''}: {locs}"
            + (f" ({tail})" if tail else "")
        )
    if not parts:
        return ""
    return (
        "\n\n[symbol-graph] verify these callers don't need follow-up updates:\n"
        + "\n".join(parts)
    )
