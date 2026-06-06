"""Per-session reversible tool log.

Snapshots file state before any mutating tool call (Write / Edit /
NotebookEdit) so the user can undo recent edits with `/undo` or
`/undo N`. Bash commands are out of scope — they can do arbitrary things
and a generic undo is not possible.

Storage layout:
    ~/.promethean/undo/<session_id>/log.jsonl
    ~/.promethean/undo/<session_id>/<seq>.snap   # raw bytes, or absent
                                                   # if file didn't exist
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional


_MUTATING_TOOLS = {"Write", "Edit", "NotebookEdit"}


def _root(session_id: str) -> Path:
    p = Path.home() / ".promethean" / "undo" / (session_id or "default")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log_path(session_id: str) -> Path:
    return _root(session_id) / "log.jsonl"


def _next_seq(session_id: str) -> int:
    """Strictly monotonic sequence per session, persisted to disk."""
    f = _root(session_id) / "seq"
    cur = 0
    if f.exists():
        try:
            cur = int(f.read_text().strip() or "0")
        except ValueError:
            cur = 0
    cur += 1
    f.write_text(str(cur))
    return cur


def is_mutating(tool_name: str) -> bool:
    return tool_name in _MUTATING_TOOLS


def snapshot_before(session_id: str, tool_name: str,
                    inputs: dict) -> Optional[int]:
    """Take a pre-state snapshot of the target file. Returns the seq
    number to be paired with `record_after` on success, or None if the
    tool isn't snapshotable or the target path is missing.
    """
    if tool_name not in _MUTATING_TOOLS:
        return None
    path = inputs.get("file_path") or inputs.get("notebook_path")
    if not path:
        return None
    seq = _next_seq(session_id)
    snap = _root(session_id) / f"{seq}.snap"
    src = Path(path)
    if src.exists():
        # Best-effort copy; symlinks are followed so we capture content
        # rather than a stale link target. shutil.copy handles binary too.
        try:
            shutil.copy(src, snap)
        except Exception:
            # If we can't snapshot, we can't undo — record None and skip.
            return None
    else:
        # File didn't exist → undo restores to "absent". Marker file.
        snap.with_suffix(".absent").touch()
    return seq


def record_after(session_id: str, seq: int, tool_name: str,
                 inputs: dict, result_excerpt: str = "") -> None:
    """Append a log entry once the tool execution has actually completed.

    `result_excerpt` is a short string used only for /undo's preview
    output — never replayed.
    """
    path = inputs.get("file_path") or inputs.get("notebook_path") or ""
    entry = {
        "seq":            seq,
        "ts":             time.time(),
        "tool":           tool_name,
        "file_path":      str(path),
        "had_before":     (_root(session_id) / f"{seq}.snap").exists()
                          or (_root(session_id) / f"{seq}.absent").exists(),
        "preview":        result_excerpt[:160],
        "undone":         False,
    }
    with _log_path(session_id).open("a") as f:
        f.write(json.dumps(entry) + "\n")


def last_n(session_id: str, n: int = 1) -> list[dict]:
    """Return the last N *un-undone* entries, newest first."""
    p = _log_path(session_id)
    if not p.exists():
        return []
    entries: list[dict] = []
    for line in p.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    pending = [e for e in entries if not e.get("undone")]
    return pending[-n:][::-1]


def revert(session_id: str, entry: dict) -> tuple[bool, str]:
    """Undo a single entry. Restores the snapshot to the original path.

    Returns (ok, message).
    """
    seq  = entry["seq"]
    path = Path(entry["file_path"])
    snap_present = _root(session_id) / f"{seq}.snap"
    snap_absent  = _root(session_id) / f"{seq}.absent"

    if snap_present.exists():
        try:
            shutil.copy(snap_present, path)
            msg = f"restored {path} from snap"
        except Exception as e:
            return False, f"failed to restore {path}: {e}"
    elif snap_absent.exists():
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                return False, f"failed to delete {path}: {e}"
        msg = f"removed {path} (was absent before tool)"
    else:
        return False, f"no snapshot for seq {seq}"

    # Mark this entry undone in the log (rewrite the file).
    log_p = _log_path(session_id)
    new_lines = []
    for line in log_p.read_text().splitlines():
        try:
            e = json.loads(line)
            if e.get("seq") == seq:
                e["undone"] = True
            new_lines.append(json.dumps(e))
        except json.JSONDecodeError:
            new_lines.append(line)
    log_p.write_text("\n".join(new_lines) + "\n")
    return True, msg


def clear_session(session_id: str) -> None:
    """Wipe a session's undo data (called by /clear)."""
    root = _root(session_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
