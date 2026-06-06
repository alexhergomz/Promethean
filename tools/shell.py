"""tools_shell.py — Shell tool implementations: Bash, Grep."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional


# ── Process tree kill ─────────────────────────────────────────────────────

def _kill_proc_tree(pid: int) -> None:
    """Kill a process and all its children."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


# ── Bash ──────────────────────────────────────────────────────────────────

def _bash(command: str, timeout: int = 30, cwd: str = None,
          shell_policy: str = "allow", session_id: str = "default",
          on_output: Optional[Callable[[str], None]] = None) -> str:
    """Run `command` under bash, capture stdout+stderr, return combined output.

    If `on_output` is provided, it is called with each line of output as it
    arrives (best-effort, for live UI display). The final return value is
    still the full captured output regardless of callback usage.
    """
    if shell_policy == "deny":
        return "Error: Bash execution is disabled (shell_policy=deny)."
    if shell_policy == "log":
        print(
            f"[bash][session={session_id}] {command[:300]}",
            file=sys.stderr, flush=True,
        )
    # Merge stderr into stdout so the live tail shows both. The final
    # return joins them with no separator (already merged).
    kwargs = dict(
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', cwd=cwd or os.getcwd(),
        bufsize=1,  # line-buffered
    )
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(command, **kwargs)
    except Exception as e:
        return f"Error: {e}"

    lines: list[str] = []
    pump_done = threading.Event()

    def _pump():
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                lines.append(line)
                if on_output:
                    try:
                        on_output(line)
                    except Exception:
                        pass
        finally:
            pump_done.set()

    t = threading.Thread(target=_pump, daemon=True)
    t.start()

    deadline = time.time() + timeout
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if time.time() > deadline:
            _kill_proc_tree(proc.pid)
            proc.wait(timeout=2)
            pump_done.wait(timeout=1)
            tail = "\n".join(lines[-20:])
            return (f"Error: timed out after {timeout}s (process killed)"
                    + (f"\n[partial output]\n{tail}" if tail else ""))
        time.sleep(0.05)

    pump_done.wait(timeout=2)
    out = "\n".join(lines).strip()
    return out or "(no output)"


# ── Grep ──────────────────────────────────────────────────────────────────

def _has_rg() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True, encoding='utf-8', errors='replace')
        return True
    except Exception:
        return False


def _grep(
    pattern: str,
    path: str = None,
    glob: str = None,
    output_mode: str = "files_with_matches",
    case_insensitive: bool = False,
    context: int = 0,
    cwd: str = None,
) -> str:
    use_rg = _has_rg()
    cmd = ["rg" if use_rg else "grep", "--no-heading"]
    if case_insensitive:
        cmd.append("-i")
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if context:
            cmd += ["-C", str(context)]
    if glob:
        cmd += (["--glob", glob] if use_rg else ["--include", glob])
    cmd.append(pattern)
    cmd.append(path or cwd or str(Path.cwd()))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
        out = r.stdout.strip()
        return out[:20000] if out else "No matches found"
    except Exception as e:
        return f"Error: {e}"
