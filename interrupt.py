"""Single-key (ESC) interrupt for an in-flight model turn.

While a turn streams, the main thread is blocked iterating the response, so
prompt_toolkit's event loop is not running and cannot catch keypresses. This
module puts the terminal into cbreak mode for the duration of a turn and runs a
tiny daemon thread that watches stdin for a bare ESC (0x1b), setting a flag the
REPL polls between stream events to abort the turn cleanly while keeping the
session intact.

It is deliberately conservative:
  * POSIX + interactive TTY only — a no-op everywhere else (Windows, pipes, CI).
  * cbreak keeps ISIG, so Ctrl+C still raises KeyboardInterrupt as before.
  * A lone ESC triggers; an ESC that begins a longer sequence (arrow / function
    keys) is drained and ignored, so navigation keys don't false-trigger.
  * Always restores the original terminal attributes in stop().

Usage:
    esc = EscInterrupt()
    esc.start()
    try:
        for event in stream(...):
            if esc.triggered():
                break            # abort cleanly, session preserved
            ...
    finally:
        esc.stop()
Pause/resume around a blocking stdin read (e.g. a permission prompt):
    esc.stop(); ans = input(...); esc.start()
"""

from __future__ import annotations

import os
import sys
import threading

_ESC = b"\x1b"


class EscInterrupt:
    def __init__(self):
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._saved = None
        self._active = False

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        """True when the watcher is actually installed (TTY + POSIX)."""
        return self._active

    def triggered(self) -> bool:
        """True once a bare ESC has been seen since the last start()."""
        return self._event.is_set()

    def start(self) -> None:
        """Install the watcher and enter cbreak mode. No-op if unsupported."""
        self._event.clear()
        if self._active or not self._can_use():
            return
        try:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)   # keeps ISIG → Ctrl+C still works
        except Exception:
            self._saved = None
            self._fd = None
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._active = True

    def stop(self) -> None:
        """Stop the watcher and restore the terminal. Safe to call repeatedly."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
            self._thread = None
        self._restore()
        self._active = False

    # ── internals ───────────────────────────────────────────────────────────
    def _can_use(self) -> bool:
        if os.name != "posix":
            return False
        try:
            return sys.stdin.isatty() and sys.stdout.isatty()
        except Exception:
            return False

    def _loop(self) -> None:
        import select
        fd = self._fd
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.1)
            except Exception:
                break
            if not r:
                continue
            try:
                b = os.read(fd, 1)
            except Exception:
                break
            if not b:
                continue
            if b == _ESC:
                # Disambiguate a bare ESC from an escape sequence (arrow keys,
                # etc.): if more bytes arrive within a short window, it's a
                # sequence — drain and ignore. Otherwise treat as an interrupt.
                try:
                    more, _, _ = select.select([fd], [], [], 0.05)
                except Exception:
                    more = [fd]
                if more:
                    try:
                        os.read(fd, 32)   # drain the rest of the sequence
                    except Exception:
                        pass
                    continue
                self._event.set()
                return
            # Any other keystroke during generation is swallowed so it doesn't
            # leak into the next prompt.

    def _restore(self) -> None:
        if self._saved is not None and self._fd is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
        self._saved = None
        self._fd = None
