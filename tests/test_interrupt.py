"""Tests for the ESC single-key interrupt, driven through a real pty."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

if os.name != "posix":
    pytest.skip("pty-based interrupt is POSIX-only", allow_module_level=True)

import pty  # noqa: E402

from interrupt import EscInterrupt  # noqa: E402


class _FakeTTY:
    """Minimal stdin/stdout stand-in over a pty slave fd."""
    def __init__(self, fd, is_tty=True):
        self._fd = fd
        self._is_tty = is_tty

    def fileno(self):
        return self._fd

    def isatty(self):
        return self._is_tty


def _with_slave(fn, is_tty=True):
    master, slave = pty.openpty()
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin = _FakeTTY(slave, is_tty)
    sys.stdout = _FakeTTY(slave, is_tty)
    try:
        return fn(master, slave)
    finally:
        sys.stdin, sys.stdout = old_in, old_out
        os.close(master)
        os.close(slave)


def test_bare_esc_triggers():
    def body(master, slave):
        esc = EscInterrupt()
        esc.start()
        assert esc.enabled
        os.write(master, b"\x1b")
        # Poll for the flag.
        for _ in range(30):
            if esc.triggered():
                break
            time.sleep(0.02)
        triggered = esc.triggered()
        esc.stop()
        assert triggered, "bare ESC should trigger the interrupt"
    _with_slave(body)


def test_arrow_key_does_not_trigger():
    def body(master, slave):
        esc = EscInterrupt()
        esc.start()
        os.write(master, b"\x1b[A")   # up-arrow escape sequence
        time.sleep(0.3)
        triggered = esc.triggered()
        esc.stop()
        assert not triggered, "an escape sequence must not false-trigger"
    _with_slave(body)


def test_normal_key_does_not_trigger():
    def body(master, slave):
        esc = EscInterrupt()
        esc.start()
        os.write(master, b"a")
        time.sleep(0.2)
        triggered = esc.triggered()
        esc.stop()
        assert not triggered
    _with_slave(body)


def test_non_tty_is_noop():
    def body(master, slave):
        esc = EscInterrupt()
        esc.start()
        try:
            assert not esc.enabled, "must be a no-op when stdin isn't a TTY"
        finally:
            esc.stop()
    _with_slave(body, is_tty=False)


def test_stop_restores_and_is_idempotent():
    def body(master, slave):
        import termios
        before = termios.tcgetattr(slave)
        esc = EscInterrupt()
        esc.start()
        esc.stop()
        esc.stop()   # second stop must be safe
        after = termios.tcgetattr(slave)
        assert after == before, "terminal attributes must be restored"
    _with_slave(body)


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
