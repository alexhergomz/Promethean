"""Snapshot tests for the graph-tool CLI visualizations.

We force `_enabled()` to True (patches `sys.stdout.isatty` and re-creates
the Rich console with `force_terminal=True`), then capture stdout while
calling each `viz_*` function on the fixture repo. Substring asserts only
— covers presence of box-drawing chars and ANSI green sequences without
locking in exact byte layouts (which Rich's renderer is free to tweak).
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

import re

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)

import bootstrap as _bs  # noqa: F401

# The repo-map / symbol-graph feature is optional (diskcache, grep_ast,
# pygments, tree_sitter). Skip cleanly when those extras aren't installed —
# the runtime itself degrades gracefully via context._agent_tools_available().
pytest.importorskip("agent_tools.helpers", reason="graph extra not installed")

import agent_tools.visualize as v
from agent_tools.helpers import (
    clear_caches,
    imports,
    neighborhood,
    path_between,
    search_files,
)

FIXTURE = str(Path(_HERE) / "fixtures" / "symbol_graph_repo")
GREEN = "\x1b[32m"
BRIGHT_GREEN = "\x1b[92m"


@contextmanager
def _force_viz_capture():
    """Patch viz to write rendered ANSI into a captured buffer.

    The viz module suppresses output when stdout isn't a TTY (correct
    behavior in production / pytest pipes). For these tests we want to
    EXERCISE the rendering path, so we monkey-patch `sys.stdout.isatty`
    to True and replace the module Console with one wired to a StringIO
    AND `force_terminal=True` so colors are emitted as ANSI escapes.
    """
    from rich.console import Console
    buf = io.StringIO()
    original_isatty = sys.stdout.isatty
    sys.stdout.isatty = lambda: True   # type: ignore[assignment]
    original_console = v._console
    v._console = Console(file=buf, force_terminal=True, color_system="256",
                         width=120)
    try:
        yield buf
    finally:
        v._console = original_console
        sys.stdout.isatty = original_isatty   # type: ignore[assignment]


# ── viz_neighborhood ────────────────────────────────────────────────────────

def test_viz_neighborhood_renders_tree():
    nb = neighborhood("validate_token", root=FIXTURE)
    with _force_viz_capture() as buf:
        v.viz_neighborhood("validate_token", nb)
    out = buf.getvalue()
    # Title, branch labels, tree connectors
    assert "validate_token" in out
    assert "called by" in out
    assert "calls" in out
    assert "├" in out or "└" in out, "expected tree connectors"
    # Caller files mentioned
    assert "api/handler.py" in out


def test_viz_neighborhood_missing_symbol_shows_red_x():
    nb = neighborhood("nonexistent_xyz", root=FIXTURE)
    with _force_viz_capture() as buf:
        v.viz_neighborhood("nonexistent_xyz", nb)
    out = buf.getvalue()
    assert "✗" in out
    assert "no symbol" in out


# ── viz_path_between ────────────────────────────────────────────────────────

def test_viz_path_between_renders_boxed_chain():
    chain = path_between(
        "handle_request", "query_db", root=FIXTURE, max_hops=4
    )
    with _force_viz_capture() as buf:
        v.viz_path_between("handle_request", "query_db", chain)
    out = buf.getvalue()
    plain = _strip_ansi(out)
    # Box-drawing chars for the lit-up boxes
    assert "╭" in plain
    assert "╰" in plain
    assert "│" in plain
    # Arrow between hops
    assert "▼" in plain
    # Endpoint markers
    assert "start" in plain
    assert "end" in plain
    # Bright-green ANSI for the path nodes (Rich emits "1;92" for bold + 92)
    assert "92m" in out, "expected bright-green ANSI somewhere in output"
    # All hop symbols are mentioned
    for sym in ("handle_request", "fetch_user", "query_db"):
        assert sym in plain


def test_viz_path_between_no_path_shows_red_x():
    with _force_viz_capture() as buf:
        v.viz_path_between("a", "b", [])
    out = buf.getvalue()
    assert "✗" in out
    assert "no path" in out


def test_viz_path_between_with_siblings_shows_branch():
    """When sibling_callees are passed, the dim ├── line should appear."""
    chain = path_between(
        "handle_request", "query_db", root=FIXTURE, max_hops=4
    )
    siblings = [
        ["validate_token", "log_info"],   # other callees of handle_request
        ["User"],                           # other callees of fetch_user
        [],
    ]
    with _force_viz_capture() as buf:
        v.viz_path_between(
            "handle_request", "query_db", chain, siblings,
        )
    out = buf.getvalue()
    assert "├──" in out
    assert "siblings" in out
    assert "validate_token" in out


# ── viz_imports ─────────────────────────────────────────────────────────────

def test_viz_imports_two_panel_layout():
    deps = imports("api/handler.py", root=FIXTURE)
    with _force_viz_capture() as buf:
        v.viz_imports("api/handler.py", deps)
    out = buf.getvalue()
    # Header file path
    assert "api/handler.py" in out
    # Both panel titles
    assert "uses" in out
    assert "used_by" in out
    # Box border for the two panels
    assert "╭" in out
    assert "╮" in out


def test_viz_imports_depth_marker_when_depth_gt_1():
    deps = imports("api/handler.py", root=FIXTURE, depth=2)
    with _force_viz_capture() as buf:
        v.viz_imports("api/handler.py", deps, depth=2)
    plain = _strip_ansi(buf.getvalue())
    assert "depth=2" in plain


# ── viz_search_files ────────────────────────────────────────────────────────

def test_viz_search_files_score_bars():
    results = search_files(
        "validate_token refresh_token", root=FIXTURE, top_k=3
    )
    with _force_viz_capture() as buf:
        v.viz_search_files("validate_token refresh_token", results)
    out = buf.getvalue()
    # Header + result count
    assert "search:" in out
    assert "result" in out
    # Score-bar fill char is U+2587 (▇)
    assert "▇" in out
    # Best-match file present
    assert "api/auth.py" in out
    # Green ANSI for the bars
    assert GREEN in out or BRIGHT_GREEN in out


def test_viz_search_files_no_matches():
    with _force_viz_capture() as buf:
        v.viz_search_files("zzzz_no_match_xyz", [])
    out = buf.getvalue()
    assert "no matches" in out


# ── Disable toggle ──────────────────────────────────────────────────────────

def test_viz_disabled_via_env_emits_nothing(monkeypatch):
    """CC_GRAPH_VIEW=0 must silence the viz even if isatty=True."""
    monkeypatch.setenv("CC_GRAPH_VIEW", "0")
    nb = neighborhood("validate_token", root=FIXTURE)
    with _force_viz_capture() as buf:
        v.viz_neighborhood("validate_token", nb)
    assert buf.getvalue() == "", "viz should produce no output when disabled"


def test_viz_off_when_stdout_not_tty():
    """When isatty() is False (the default in pipes / pytest), viz is a
    no-op regardless of CC_GRAPH_VIEW. This is the production-default
    state and protects normal test runs from rendering noise."""
    from rich.console import Console
    buf = io.StringIO()
    original_console = v._console
    v._console = Console(file=buf, force_terminal=True, color_system="256")
    # Don't patch isatty — leave the real one (False under pytest)
    try:
        nb = neighborhood("validate_token", root=FIXTURE)
        v.viz_neighborhood("validate_token", nb)
        assert buf.getvalue() == ""
    finally:
        v._console = original_console


# ── set_enabled / is_enabled / /graph-view command ──────────────────────────

def test_set_enabled_explicit_true_overrides_no_tty():
    """set_enabled(True) forces viz on even when stdout isn't a TTY —
    the user might be deliberately piping/redirecting and wants color."""
    v.set_enabled(True)
    try:
        assert v.is_enabled() is True
    finally:
        v.set_enabled(None)


def test_set_enabled_explicit_false_overrides_env(monkeypatch):
    monkeypatch.setenv("CC_GRAPH_VIEW", "1")
    v.set_enabled(False)
    try:
        assert v.is_enabled() is False
    finally:
        v.set_enabled(None)


def test_set_enabled_none_reverts_to_auto(monkeypatch):
    monkeypatch.setenv("CC_GRAPH_VIEW", "0")
    v.set_enabled(True)
    assert v.is_enabled() is True
    v.set_enabled(None)
    # With env=0, auto path returns False
    assert v.is_enabled() is False


def test_graph_view_slash_command_toggles():
    from commands.core import cmd_graph_view

    # Reset to known state
    v.set_enabled(False)

    class _S:
        pass

    cmd_graph_view("on", _S(), {})
    assert v.is_enabled() is True

    cmd_graph_view("off", _S(), {})
    assert v.is_enabled() is False

    cmd_graph_view("toggle", _S(), {})
    assert v.is_enabled() is True

    cmd_graph_view("toggle", _S(), {})
    assert v.is_enabled() is False

    cmd_graph_view("auto", _S(), {})
    # After auto, should match _enabled() default behavior
    assert v._explicit is None

    # Cleanup
    v.set_enabled(None)


def test_graph_view_unknown_arg_does_not_change_state():
    from commands.core import cmd_graph_view

    v.set_enabled(True)
    cmd_graph_view("garbage", None, {})
    assert v.is_enabled() is True
    v.set_enabled(None)
