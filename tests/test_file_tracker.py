"""File-context tracker (Cline pattern).

When the model Reads a file, the tracker stores (mtime, size, turn).
A subsequent whole-file Read of the same unchanged file returns a stub
instead of the content — saving a substantial number of tokens on long
sessions where the model otherwise re-reads files it already saw.

Edge cases covered:
  - first read is a real read
  - second read of unchanged file is stubbed
  - second read after external mtime change is real with banner
  - slice reads (limit/offset) always pass through (different region)
  - Write/Edit invalidates the tracker so subsequent Read returns real content
  - failure modes (missing file) bypass tracking
  - tracking is disabled when no agent_state is set up
"""
from __future__ import annotations

import os
import sys
import time
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from tools import execute_tool


@pytest.fixture
def tracking_session(tmp_path):
    """Set up a real RuntimeContext with an agent_state (turn_count=0)
    so the tracker is active. Yields (config, sctx)."""
    import runtime
    from agent import AgentState
    sid = "tracker-test-session"
    sctx = runtime.get_session_ctx(sid)
    sctx.agent_state = AgentState()
    sctx.file_tracker.clear()
    config = {"_session_id": sid}
    yield config, sctx
    # Cleanup
    runtime.release_session_ctx(sid)


@pytest.fixture
def tmp_file(tmp_path):
    """A file with predictable content."""
    f = tmp_path / "sample.txt"
    f.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
    return f


# ── First read is a real read ──────────────────────────────────────────────

def test_first_read_returns_real_content(tracking_session, tmp_file):
    config, _ = tracking_session
    out = execute_tool("Read", {"file_path": str(tmp_file)},
                       config=config, permission_mode="accept-all")
    assert "line 1" in out
    assert "line 5" in out


def test_first_read_records_in_tracker(tracking_session, tmp_file):
    config, sctx = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    abs_path = os.path.abspath(str(tmp_file))
    assert abs_path in sctx.file_tracker
    mtime, size, turn = sctx.file_tracker[abs_path]
    assert mtime == os.path.getmtime(abs_path)
    assert size == os.path.getsize(abs_path)


# ── Second read of unchanged file: stub ───────────────────────────────────

def test_second_read_of_unchanged_file_is_stubbed(tracking_session, tmp_file):
    config, _ = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    out2 = execute_tool("Read", {"file_path": str(tmp_file)},
                        config=config, permission_mode="accept-all")
    assert "line 1" not in out2, f"expected stub, got real content: {out2!r}"
    assert "unchanged" in out2.lower()


def test_stub_mentions_turn_seen(tracking_session, tmp_file):
    config, sctx = tracking_session
    sctx.agent_state.turn_count = 7
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    sctx.agent_state.turn_count = 9
    out = execute_tool("Read", {"file_path": str(tmp_file)},
                       config=config, permission_mode="accept-all")
    assert "turn 7" in out


# ── External change: real read with banner ────────────────────────────────

def test_external_change_triggers_banner(tracking_session, tmp_file):
    config, _ = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    # Simulate external modification with a forward mtime jump (sub-second
    # mtime granularity may collapse otherwise — sleep 1.1s).
    time.sleep(1.1)
    tmp_file.write_text("brand new content\nyay\n")

    out = execute_tool("Read", {"file_path": str(tmp_file)},
                       config=config, permission_mode="accept-all")
    assert "FILE CHANGED" in out
    assert "brand new content" in out


# ── Slice reads always pass through ───────────────────────────────────────

def test_slice_read_passes_through_after_full_read(tracking_session, tmp_file):
    config, _ = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    # Now read just lines 2-3 with a slice — should NOT be stubbed
    out = execute_tool("Read",
                       {"file_path": str(tmp_file), "offset": 1, "limit": 2},
                       config=config, permission_mode="accept-all")
    assert "line 2" in out
    assert "line 3" in out
    assert "unchanged" not in out.lower()


def test_slice_read_does_not_overwrite_full_read_record(tracking_session, tmp_file):
    """A slice read should not corrupt the tracker entry from the full read."""
    config, sctx = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    abs_path = os.path.abspath(str(tmp_file))
    before = sctx.file_tracker[abs_path]
    execute_tool("Read",
                 {"file_path": str(tmp_file), "offset": 1, "limit": 2},
                 config=config, permission_mode="accept-all")
    assert sctx.file_tracker[abs_path] == before


# ── Write/Edit invalidates tracker ────────────────────────────────────────

def test_write_invalidates_tracker(tracking_session, tmp_file):
    config, sctx = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    abs_path = os.path.abspath(str(tmp_file))
    assert abs_path in sctx.file_tracker
    execute_tool("Write",
                 {"file_path": str(tmp_file), "content": "fresh\n"},
                 config=config, permission_mode="accept-all")
    assert abs_path not in sctx.file_tracker, \
        "Write should invalidate the tracker entry"


def test_read_after_write_is_real(tracking_session, tmp_file):
    config, _ = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    execute_tool("Write",
                 {"file_path": str(tmp_file), "content": "rewritten\n"},
                 config=config, permission_mode="accept-all")
    out = execute_tool("Read", {"file_path": str(tmp_file)},
                       config=config, permission_mode="accept-all")
    # Should be real content (no stub, no banner — clean re-read)
    assert "rewritten" in out
    assert "unchanged" not in out.lower()
    assert "FILE CHANGED" not in out


def test_edit_invalidates_tracker(tracking_session, tmp_file):
    config, sctx = tracking_session
    execute_tool("Read", {"file_path": str(tmp_file)},
                 config=config, permission_mode="accept-all")
    abs_path = os.path.abspath(str(tmp_file))
    assert abs_path in sctx.file_tracker
    execute_tool("Edit",
                 {"file_path": str(tmp_file),
                  "old_string": "line 1", "new_string": "LINE-1"},
                 config=config, permission_mode="accept-all")
    assert abs_path not in sctx.file_tracker


# ── Tracking disabled when no agent_state ─────────────────────────────────

def test_tracking_disabled_without_agent_state(tmp_file):
    """If the harness hasn't set up an AgentState, reads pass through
    unchanged and the tracker is not consulted (avoids breaking init-time
    or test invocations)."""
    out1 = execute_tool("Read", {"file_path": str(tmp_file)},
                        config={}, permission_mode="accept-all")
    out2 = execute_tool("Read", {"file_path": str(tmp_file)},
                        config={}, permission_mode="accept-all")
    assert "line 1" in out1
    assert "line 1" in out2  # second read still real, no tracking


def test_failed_read_does_not_record(tracking_session, tmp_path):
    """Reading a missing file should NOT record a tracker entry."""
    config, sctx = tracking_session
    missing = str(tmp_path / "does_not_exist.txt")
    out = execute_tool("Read", {"file_path": missing},
                       config=config, permission_mode="accept-all")
    assert "Error" in out or "not found" in out.lower()
    assert os.path.abspath(missing) not in sctx.file_tracker
