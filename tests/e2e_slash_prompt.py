"""PTY smoke test for the prompt_toolkit input layer.

Exercises `ui/input.read_line` end-to-end through a pseudo-terminal without
spinning up the full Promethean REPL. Gated on prompt_toolkit availability.
"""

from __future__ import annotations

import os
import pty
import select
import sys
import time

import pytest

from ui.input import HAS_PROMPT_TOOLKIT

if not HAS_PROMPT_TOOLKIT:
    pytest.skip("prompt_toolkit not installed", allow_module_level=True)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_CHILD_SCRIPT = r"""
import sys
sys.path.insert(0, {repo_root!r})
import ui.input as _ui

_COMMANDS = {{"help": True, "clear": True, "checkpoint": True, "cwd": True,
              "compact": True, "config": True, "cost": True, "copy": True,
              "context": True, "cloudsave": True}}
_META = {{
    "help":       ("Show help", []),
    "clear":      ("Clear", []),
    "checkpoint": ("Checkpoints", ["clear"]),
    "cwd":        ("Working directory", []),
    "compact":    ("Compact", []),
    "config":     ("Config", []),
    "cost":       ("Cost", []),
    "copy":       ("Copy", []),
    "context":    ("Context", []),
    "cloudsave":  ("Cloud save", ["setup", "auto", "list"]),
}}
_ui._commands_provider = lambda: _COMMANDS
_ui._meta_provider = lambda: _META

result = _ui.read_line("[test] > ")
sys.stdout.write("RESULT=" + repr(result) + chr(10))
sys.stdout.flush()
"""


def _run_child(keystrokes: list[tuple[float, bytes]], timeout: float = 4.0,
               script_tmpl: str = _CHILD_SCRIPT) -> bytes:
    """Spawn the child under a PTY and play a sequence of (delay, bytes) writes.

    Reads until 'RESULT=' lands on stdout or the timeout expires.
    """
    script = script_tmpl.format(repo_root=_REPO_ROOT)
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-c", script])

    collected = bytearray()
    start = time.monotonic()
    write_queue = list(keystrokes)

    try:
        # Give prompt_toolkit time to render its initial screen.
        time.sleep(0.3)

        while time.monotonic() - start < timeout:
            # Send the next keystroke if its scheduled delay has elapsed.
            if write_queue:
                delay, payload = write_queue[0]
                if time.monotonic() - start >= delay:
                    try:
                        os.write(fd, payload)
                    except OSError:
                        pass
                    write_queue.pop(0)

            rlist, _, _ = select.select([fd], [], [], 0.1)
            if fd in rlist:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                collected.extend(chunk)
                if b"RESULT=" in bytes(collected):
                    break
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    return bytes(collected)


def test_enter_dispatches_typed_text():
    """Typing a command and pressing Enter returns that command verbatim."""
    output = _run_child([(0.3, b"/help"), (0.6, b"\r")])
    assert b"RESULT='/help'" in output, output[-500:]


def test_typing_slash_c_renders_menu_with_matches():
    """Typing `/c` produces completion output containing at least one /c-prefixed command."""
    output = _run_child([(0.3, b"/c"), (0.8, b"\r")])
    # Some /c-prefixed command name must appear in the rendered buffer.
    assert any(
        fragment in output for fragment in (
            b"/checkpoint", b"/clear", b"/cwd", b"/compact",
            b"/config", b"/copy", b"/cost", b"/context", b"/cloudsave",
        )
    ), output[-500:]


_CYCLE_SCRIPT = r"""
import sys, os
sys.path.insert(0, {repo_root!r})
import ui.input as _ui

_marker = os.environ.get("PROM_MODE_MARKER", "")
_state = {{"mode": "auto"}}
_CYCLE = ["auto", "accept-all", "plan"]
def _cycle():
    _state["mode"] = _CYCLE[(_CYCLE.index(_state["mode"]) + 1) % len(_CYCLE)]
    if _marker:
        with open(_marker, "a") as f:
            f.write(_state["mode"] + chr(10))
    return _state["mode"]

_ui.setup(lambda: {{}}, lambda: {{}}, mode_cycler=_cycle)
result = _ui.read_line("[test] > ")
sys.stdout.write("RESULT=" + repr(result) + chr(10))
sys.stdout.flush()
"""


def test_shift_tab_cycles_permission_mode(tmp_path):
    """Shift+Tab invokes the registered mode_cycler and cycles in order.

    Sends two Shift+Tab (BackTab = ESC [ Z) then Enter. The cycler records
    each new mode to a marker file (the bottom toolbar itself does not render
    reliably under a headless PTY), so we assert auto -> accept-all -> plan.
    """
    marker = tmp_path / "modes.txt"
    os.environ["PROM_MODE_MARKER"] = str(marker)
    try:
        _run_child(
            [(0.3, b"\x1b[Z"), (0.8, b"\x1b[Z"), (1.2, b"\r")],
            script_tmpl=_CYCLE_SCRIPT,
        )
    finally:
        os.environ.pop("PROM_MODE_MARKER", None)
    modes = marker.read_text().split() if marker.exists() else []
    assert modes[:2] == ["accept-all", "plan"], modes


def test_arrow_down_then_enter_picks_first_menu_entry():
    """Down-arrow selects a completion, Enter accepts it.

    Completions are fuzzy-ranked (not alphabetical), so the exact first entry
    depends on scoring; assert Enter accepts some `/c` menu command.
    """
    output = _run_child([
        (0.3, b"/c"),
        (0.8, b"\x1b[B"),   # down arrow
        (1.1, b"\r"),
    ])
    candidates = (
        b"/checkpoint", b"/clear", b"/cwd", b"/compact",
        b"/config", b"/copy", b"/cost", b"/context", b"/cloudsave",
    )
    assert any(b"RESULT='" + c + b"'" in output for c in candidates), output[-500:]
