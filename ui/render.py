"""
ui/render.py — All terminal rendering for Promethean.

Provides:
  - ANSI color helpers (C, clr, info, ok, warn, err)
  - Rich Markdown streaming (stream_text, flush_response)
  - Spinner management
  - Tool call display (print_tool_start, print_tool_end)
  - Diff rendering (render_diff)
"""
from __future__ import annotations

import sys
import json
import threading

# ── Optional rich for markdown rendering ──────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None
    Live = None
    Markdown = None

# ── ANSI helpers ───────────────────────────────────────────────────────────
C = {
    "cyan":    "\033[36m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "red":     "\033[31m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "white":   "\033[37m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "reset":   "\033[0m",
    # ── Promethean brand palette (24-bit truecolor) ──
    "flame":   "\033[38;2;255;95;31m",    # #FF5F1F — primary accent
    "ember":   "\033[38;2;255;140;66m",   # #FF8C42
    "warmth":  "\033[38;2;255;176;104m",  # #FFB068
    "glow":    "\033[38;2;255;207;157m",  # #FFCF9D
    "coal":    "\033[38;2;184;61;0m",     # #B83D00
    "violet":  "\033[38;2;139;127;212m",  # #8B7FD4 — secondary accent
    "dust":    "\033[38;2;123;115;148m",  # #7B7394 — muted text
}

def clr(text: str, *keys: str) -> str:
    return "".join(C[k] for k in keys) + str(text) + C["reset"]

def info(msg: str):   print(clr(msg, "cyan"))
def ok(msg: str):     print(clr(msg, "green"))
def warn(msg: str):   print(clr(f"Warning: {msg}", "yellow"))
def err(msg: str):    print(clr(f"Error: {msg}", "red"), file=sys.stderr)

def _truncate_err_global(s: str, max_len: int = 200) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


# ── Diff rendering ─────────────────────────────────────────────────────────

def render_diff(text: str):
    """Print diff text with ANSI colors: red for removals, green for additions."""
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            print(C["bold"] + line + C["reset"])
        elif line.startswith("+"):
            print(C["green"] + line + C["reset"])
        elif line.startswith("-"):
            print(C["red"] + line + C["reset"])
        elif line.startswith("@@"):
            print(C["cyan"] + line + C["reset"])
        else:
            print(line)

def _has_diff(text: str) -> bool:
    """Check if text contains a unified diff."""
    return "--- a/" in text and "+++ b/" in text


# ── Conversation rendering ─────────────────────────────────────────────────

_accumulated_text: list[str] = []   # buffer text during streaming
_current_live = None                # active Rich Live instance (one at a time)
_RICH_LIVE = True                   # set False (via config rich_live=false) to disable

def set_rich_live(enabled: bool) -> None:
    """Called from repl.py to apply the rich_live config setting."""
    global _RICH_LIVE
    _RICH_LIVE = _RICH and enabled

def _make_renderable(text: str):
    """Return a Rich renderable: Markdown if text contains markup, else plain.

    Rich's Markdown parser can raise IndexError on partially-streamed text
    that contains unclosed inline constructs (links, code spans, formatting).
    Wrap in try/except so a transient parse error during streaming doesn't
    bubble up and kill the whole turn.
    """
    if any(c in text for c in ("#", "*", "`", "_", "[")):
        try:
            return Markdown(text)
        except (IndexError, ValueError):
            # Partial markdown couldn't be parsed yet; render as plain
            # for this update and let the next chunk retry.
            return text
    return text

def _start_live() -> None:
    """Start a Rich Live block for in-place Markdown streaming (no-op if not Rich)."""
    global _current_live
    if _RICH and _RICH_LIVE and _current_live is None:
        _current_live = Live(console=console, auto_refresh=False,
                             vertical_overflow="visible")
        _current_live.start()

def stream_text(chunk: str) -> None:
    """Buffer chunk; update Live in-place when Rich available, else print directly.

    Older versions auto-switched from Live to plain streaming past 80 lines
    on the theory that terminals can't handle large Live regions. In practice
    that path called Live.stop() (which leaves the frozen partial render
    visible because Live is non-transient) and then console.print(full),
    producing a visible duplicate. Removed: just let Live render the whole
    response. Modern terminals handle long Live regions fine.
    """
    global _current_live
    _accumulated_text.append(chunk)

    if _RICH and _RICH_LIVE:
        full = "".join(_accumulated_text)
        if _current_live is None:
            _start_live()
        # Rich's render can raise on partial markdown (unclosed links,
        # code spans, etc.). Fall back to plain text for this update on
        # error and let the next chunk retry.
        try:
            _current_live.update(_make_renderable(full), refresh=True)
        except Exception:
            _current_live.update(full, refresh=True)
    else:
        print(chunk, end="", flush=True)

def stream_thinking(chunk: str, verbose: bool):
    if verbose:
        clean_chunk = chunk.replace("\n", " ")
        if clean_chunk:
            print(f"{C['dim']}{clean_chunk}", end="", flush=True)

def flush_response() -> None:
    """Commit buffered text to screen: stop Live (freezes rendered Markdown in place)."""
    global _current_live
    full = "".join(_accumulated_text)
    _accumulated_text.clear()
    if _current_live is not None:
        _current_live.stop()
        _current_live = None
    elif _RICH and _RICH_LIVE and full.strip():
        console.print(_make_renderable(full))
    else:
        print()  # ensure newline after plain-text stream


# ── Spinner ────────────────────────────────────────────────────────────────

_TOOL_SPINNER_PHRASES = [
    "⚡ Rewriting light speed...",
    "🏁 Winning a race against light...",
    "🤔 Who is Barry Allen?...",
    "🐆 Outrunning the compiler...",
    "💨 Leaving electrons behind...",
    "🌍 Orbiting the codebase...",
    "⏱️ Breaking the sound barrier...",
    "🔥 Faster than a hot reload...",
    "🚀 Terminal velocity reached...",
    "🐾 Claw marks on the stack...",
    "🏎️ Shifting to 6th gear...",
    "⚡ Speed force activated...",
    "🌪️ Blitzing through the AST...",
    "💫 Bending spacetime...",
    "🐆 Cheetah mode engaged...",
]

_DEBATE_SPINNER_PHRASES = [
    "⚔️  Experts taking their positions...",
    "🧠  Experts formulating arguments...",
    "🗣️  Debate in progress...",
    "⚖️  Weighing the evidence...",
    "💡  Building counter-arguments...",
    "🔥  Debate heating up...",
    "📜  Drafting the consensus...",
    "🎯  Finding common ground...",
]

_tool_spinner_thread = None
_tool_spinner_stop = threading.Event()
_spinner_phrase = ""
_spinner_tail = ""           # live tail line (e.g. last Bash stdout line)
_spinner_lock = threading.Lock()


def _run_tool_spinner():
    """Background spinner on a single line using carriage return.

    If `_spinner_tail` is set, appends it to the line, dimmed and truncated
    so total line stays under ~120 chars. Used to surface live Bash stdout.
    """
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not _tool_spinner_stop.is_set():
        with _spinner_lock:
            phrase = _spinner_phrase
            tail = _spinner_tail
        frame = chars[i % len(chars)]
        line = f"\r  {frame} {clr(phrase, 'dim')}"
        if tail:
            t = tail.replace("\n", " ").replace("\r", " ").strip()
            if len(t) > 80:
                t = t[:77] + "..."
            line += f"  {clr('│ ' + t, 'dim')}"
        # Pad to overwrite any leftover chars from a previous longer line.
        sys.stdout.write(line + " " * 4)
        sys.stdout.flush()
        i += 1
        _tool_spinner_stop.wait(0.1)


def set_spinner_tail(line: str) -> None:
    """Update the live tail shown next to the spinner. Empty string clears."""
    global _spinner_tail
    with _spinner_lock:
        _spinner_tail = line or ""


def clear_spinner_tail() -> None:
    set_spinner_tail("")

def _start_tool_spinner():
    global _tool_spinner_thread
    if _tool_spinner_thread and _tool_spinner_thread.is_alive():
        return
    import random
    with _spinner_lock:
        global _spinner_phrase
        _spinner_phrase = random.choice(_TOOL_SPINNER_PHRASES)
    _tool_spinner_stop.clear()
    _tool_spinner_thread = threading.Thread(target=_run_tool_spinner, daemon=True)
    _tool_spinner_thread.start()

def _change_spinner_phrase():
    """Change the spinner phrase without stopping it."""
    import random
    with _spinner_lock:
        global _spinner_phrase
        _spinner_phrase = random.choice(_TOOL_SPINNER_PHRASES)

def set_spinner_phrase(phrase: str) -> None:
    """Set a specific spinner phrase (used by SSJ debate mode)."""
    global _spinner_phrase
    with _spinner_lock:
        _spinner_phrase = phrase

def _stop_tool_spinner():
    global _tool_spinner_thread
    if not _tool_spinner_thread:
        return
    _tool_spinner_stop.set()
    _tool_spinner_thread.join(timeout=1)
    _tool_spinner_thread = None
    # Wide enough to overwrite "spinner phrase | tail …" lines.
    sys.stdout.write(f"\r{' ' * 160}\r")
    sys.stdout.flush()
    clear_spinner_tail()


# ── Tool call display ──────────────────────────────────────────────────────

def _tool_desc(name: str, inputs: dict) -> str:
    if name == "Read":   return f"Read({inputs.get('file_path','')})"
    if name == "Write":  return f"Write({inputs.get('file_path','')})"
    if name == "Edit":   return f"Edit({inputs.get('file_path','')})"
    if name == "Bash":   return f"Bash({inputs.get('command','')[:80]})"
    if name == "Glob":   return f"Glob({inputs.get('pattern','')})"
    if name == "Grep":   return f"Grep({inputs.get('pattern','')})"
    if name == "WebFetch":    return f"WebFetch({inputs.get('url','')[:60]})"
    if name == "WebSearch":   return f"WebSearch({inputs.get('query','')})"
    if name == "Agent":
        atype = inputs.get("subagent_type", "")
        aname = inputs.get("name", "")
        iso   = inputs.get("isolation", "")
        bg    = not inputs.get("wait", True)
        parts = []
        if atype:  parts.append(atype)
        if aname:  parts.append(f"name={aname}")
        if iso:    parts.append(f"isolation={iso}")
        if bg:     parts.append("background")
        suffix = f"({', '.join(parts)})" if parts else ""
        prompt_short = inputs.get("prompt", "")[:60]
        return f"Agent{suffix}: {prompt_short}"
    if name == "SendMessage":
        return f"SendMessage(to={inputs.get('to','')}: {inputs.get('message','')[:50]})"
    if name == "CheckAgentResult": return f"CheckAgentResult({inputs.get('task_id','')})"
    if name == "ListAgentTasks":   return "ListAgentTasks()"
    if name == "ListAgentTypes":   return "ListAgentTypes()"
    return f"{name}({list(inputs.values())[:1]})"


def print_tool_start(name: str, inputs: dict, verbose: bool):
    """Show tool invocation."""
    desc = _tool_desc(name, inputs)
    print(clr(f"  ⚙  {desc}", "dim", "cyan"), flush=True)
    if verbose:
        print(clr(f"     inputs: {json.dumps(inputs, ensure_ascii=False)[:200]}", "dim"))

def print_tool_end(name: str, result: str, verbose: bool):
    lines = result.count("\n") + 1
    size = len(result)
    summary = f"→ {lines} lines ({size} chars)"
    if not result.startswith("Error") and not result.startswith("Denied"):
        print(clr(f"  ✓ {summary}", "dim", "green"), flush=True)
        if name in ("Edit", "Write") and _has_diff(result):
            parts = result.split("\n\n", 1)
            if len(parts) == 2:
                print(clr(f"  {parts[0]}", "dim"))
                render_diff(parts[1])
    else:
        print(clr(f"  ✗ {result[:120]}", "dim", "red"), flush=True)
    if verbose and not result.startswith("Denied"):
        preview = result[:500] + ("…" if len(result) > 500 else "")
        print(clr(f"     {preview.replace(chr(10), chr(10)+'     ')}", "dim"))
