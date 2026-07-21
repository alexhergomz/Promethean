"""prompt_toolkit-based REPL input with typing-time slash-command autosuggest.

Optional dependency: when prompt_toolkit is not installed, HAS_PROMPT_TOOLKIT
is False and callers should fall through to readline-based input.

Dependency-injected: callers register command/meta providers via setup()
before calling read_line(). This module never imports promethean — keeping
the dependency one-way and eliminating any circular-import risk.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Callable, List, Optional

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


# ── Injected providers ───────────────────────────────────────────────────────
# Callers (promethean.repl) must call setup() before read_line().
_commands_provider: Optional[Callable[[], dict]] = None
_meta_provider: Optional[Callable[[], dict]] = None
# Optional provider that returns the status-footer text (ANSI ok) shown
# at the bottom of the input box. Set by promethean.repl. Wrapped in a
# function so the toolbar refreshes on every redraw.
_status_provider: Optional[Callable[[], str]] = None
# Optional callback bound to Shift+Tab: advances the permission mode and
# returns the new mode label (or None). Set by promethean.repl so the
# input layer stays free of any config/permission knowledge.
_mode_cycler: Optional[Callable[[], Optional[str]]] = None


def setup(
    commands_provider: Callable[[], dict],
    meta_provider: Callable[[], dict],
    status_provider: Optional[Callable[[], str]] = None,
    mode_cycler: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    """Register providers for the live command registry and metadata.

    `commands_provider` returns the dispatcher's COMMANDS dict.
    `meta_provider` returns the _CMD_META dict (descriptions + subcommands).
    `status_provider` (optional) returns the bottom-toolbar text for the
    input box — typically a 1-line summary of model/ctx/cost/failover.
    `mode_cycler` (optional) advances the permission mode (Shift+Tab) and
    returns the new label.
    """
    global _commands_provider, _meta_provider, _status_provider, _mode_cycler
    _commands_provider = commands_provider
    _meta_provider = meta_provider
    _status_provider = status_provider
    _mode_cycler = mode_cycler


# ── Fuzzy matching ───────────────────────────────────────────────────────────
# Subsequence matching so "ranch" still surfaces "branch": every character of
# the query must appear in the candidate, in order (case-insensitive). The
# strict subsequence rule (vs. a looser edit-distance) keeps absent-character
# guarantees intact — typing "/c" never surfaces a command without a 'c'.

def _subseq_score(q: str, cand: str) -> Optional[float]:
    """Score how well `cand` matches `q` as a subsequence; None if no match.

    Higher is better. Ordering preference: prefix > contiguous substring >
    compact subsequence, with a difflib-ratio nudge.  Both args lowercased.
    """
    if cand.startswith(q):
        return 1000.0 - len(cand)            # prefix: strongest, prefer shorter
    idx = cand.find(q)
    if idx != -1:
        return 500.0 - idx - len(cand) * 0.01  # contiguous substring
    first = last = -1
    qi = 0
    for ci, ch in enumerate(cand):
        if qi < len(q) and ch == q[qi]:
            if first == -1:
                first = ci
            last = ci
            qi += 1
    if qi < len(q):
        return None                          # not a subsequence
    span = last - first + 1
    return 100.0 - span + difflib.SequenceMatcher(None, q, cand).ratio()


def _fuzzy_rank(query: str, candidates: List[str]) -> List[str]:
    """Return candidates that fuzzily match `query`, best-first.

    Empty query returns everything (alphabetical). Matching is the
    subsequence rule in `_subseq_score`; ties break on shorter name then
    alphabetically for stable, predictable ordering.
    """
    q = query.lower()
    if not q:
        return sorted(candidates)
    scored: List[tuple] = []
    for name in candidates:
        s = _subseq_score(q, name.lower())
        if s is not None:
            scored.append((s, name))
    scored.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    return [name for _, name in scored]


# ── Path completion ────────────────────────────────────────────────────────

def _path_token(text: str) -> str:
    """The whitespace-delimited token under the cursor (the maximal non-space
    suffix of the text left of the cursor). Empty if the cursor sits on a space."""
    i = len(text)
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return text[i:]


def _looks_like_path(token: str) -> bool:
    """Path-like enough to complete while typing (vs. only on an explicit Tab)."""
    return ("/" in token) or token.startswith(("~", "."))


def _filesystem_completions(token: str, limit: int = 50) -> list[tuple[str, bool]]:
    """Return [(replacement_token, is_dir)] for filesystem entries matching the
    token's basename, relative to its dir part (cwd when there is none).

    The replacement keeps whatever directory prefix the user typed (including a
    leading ``~/``) and appends a trailing ``/`` for directories.
    """
    expanded = os.path.expanduser(token)
    target = os.path.dirname(expanded) or "."
    prefix = os.path.basename(expanded)
    try:
        names = os.listdir(target)
    except OSError:
        return []
    typed_dir = token[:len(token) - len(os.path.basename(token))]  # keeps ~/, ./, etc.
    hide_hidden = not prefix.startswith(".")
    out: list[tuple[str, bool]] = []
    for name in names:
        if hide_hidden and name.startswith("."):
            continue
        if not name.startswith(prefix):
            continue
        is_dir = os.path.isdir(os.path.join(target, name))
        out.append((typed_dir + name + ("/" if is_dir else ""), is_dir))
    out.sort(key=lambda t: (not t[1], t[0].lower()))   # directories first
    return out[:limit]


# ── Completer ────────────────────────────────────────────────────────────────
if HAS_PROMPT_TOOLKIT:

    class SlashCompleter(Completer):
        """Two-level completer for slash commands.

        Level 1: /partial  (no space)  → command names.
        Level 2: /cmd partial           → subcommands listed in the meta dict.

        Providers default to the module-level ones registered via setup(),
        but can be injected via the constructor for testing.
        """

        def __init__(
            self,
            commands_provider: Optional[Callable[[], dict]] = None,
            meta_provider: Optional[Callable[[], dict]] = None,
        ):
            self._commands_override = commands_provider
            self._meta_override = meta_provider
            self._cache_key: Optional[tuple] = None
            self._cache_names: list[str] = []

        def _get_commands(self) -> dict:
            provider = self._commands_override or _commands_provider
            return (provider() if provider else {}) or {}

        def _get_meta(self) -> dict:
            provider = self._meta_override or _meta_provider
            return (provider() if provider else {}) or {}

        def _live_command_names(self) -> list[str]:
            keys = sorted(set(self._get_commands().keys()) | set(self._get_meta().keys()))
            sig = tuple(keys)
            if self._cache_key == sig:
                return self._cache_names
            self._cache_key = sig
            self._cache_names = keys
            return keys

        def get_completions(self, document, complete_event):  # type: ignore[override]
            text = document.text_before_cursor
            if not text.startswith("/"):
                yield from self._path_completions(text, complete_event)
                return

            meta = self._get_meta()

            if " " not in text:
                word = text[1:]
                for name in _fuzzy_rank(word, self._live_command_names()):
                    desc, subs = meta.get(name, ("", []))
                    hint = ""
                    if subs:
                        head = ", ".join(subs[:3])
                        more = "…" if len(subs) > 3 else ""
                        hint = f"  [{head}{more}]"
                    yield Completion(
                        "/" + name,
                        start_position=-len(text),
                        display=ANSI(f"\x1b[36m/{name}\x1b[0m"),
                        display_meta=(desc + hint) if desc else hint.strip(),
                    )
                return

            head, _, tail = text.partition(" ")
            cmd = head[1:]
            meta_entry = meta.get(cmd)
            if not meta_entry:
                return
            subs = meta_entry[1]
            if not subs:
                return
            partial = tail.rsplit(" ", 1)[-1]
            for sub in _fuzzy_rank(partial, list(subs)):
                yield Completion(
                    sub,
                    start_position=-len(partial),
                    display_meta=f"{cmd} subcommand",
                )

        def _path_completions(self, text, complete_event):
            """Complete the token under the cursor as a filesystem path.

            While typing, this only fires for path-like tokens (containing a
            slash, or starting with ~ / .), so ordinary prose doesn't trigger a
            menu. Pressing Tab (an explicit completion request) completes any
            token as a path.
            """
            token = _path_token(text)
            if not token:
                return
            requested = getattr(complete_event, "completion_requested", False)
            if not (requested or _looks_like_path(token)):
                return
            for new_token, is_dir in _filesystem_completions(token):
                yield Completion(
                    new_token,
                    start_position=-len(token),
                    display=os.path.basename(new_token.rstrip("/")) + ("/" if is_dir else ""),
                    display_meta="dir" if is_dir else "file",
                )


else:  # pragma: no cover — unreachable when prompt_toolkit is installed
    class SlashCompleter:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("prompt_toolkit is not installed")


# ── Session cache ────────────────────────────────────────────────────────────
_SESSION = None
_SESSION_HISTORY_PATH: Optional[Path] = None


def reset_session() -> None:
    """Drop the cached session so the next read_line() rebuilds from scratch."""
    global _SESSION, _SESSION_HISTORY_PATH
    _SESSION = None
    _SESSION_HISTORY_PATH = None


def _build_session(history_path: Optional[Path]):
    if not HAS_PROMPT_TOOLKIT:
        raise RuntimeError("prompt_toolkit is not installed")
    completer = SlashCompleter()
    history = FileHistory(str(history_path)) if history_path else InMemoryHistory()
    style = Style.from_dict({
        # Promethean palette: flame highlight on the selected completion.
        "completion-menu.completion":              "bg:#191624 #d6cfc0",
        "completion-menu.completion.current":      "bg:#ff5f1f #05040a bold",
        "completion-menu.meta.completion":         "bg:#191624 #7b7394",
        "completion-menu.meta.completion.current": "bg:#ff8c42 #05040a",
        "auto-suggestion":                         "#7b7394 italic",
        # Without this, prompt_toolkit's default bottom-toolbar style is
        # `reverse`, which paints the padding after our text as a bright
        # (white) block. Pin it to the app's ink background instead.
        "bottom-toolbar":                          "bg:#05040a #7b7394 noreverse",
    })
    def _toolbar():
        if _status_provider is None:
            return None
        try:
            txt = _status_provider()
        except Exception:
            return None
        if not txt:
            return None
        return ANSI(txt)

    # Shift+Tab cycles the permission mode (Tab itself is completion). The
    # handler just delegates to the registered cycler and repaints so the
    # bottom-toolbar indicator updates immediately.
    kb = KeyBindings()

    @kb.add("s-tab")
    def _cycle_mode(event):  # noqa: ANN001
        if _mode_cycler is not None:
            try:
                _mode_cycler()
            except Exception:
                pass
            event.app.invalidate()

    return PromptSession(
        history=history,
        completer=completer,
        auto_suggest=AutoSuggestFromHistory(),
        complete_while_typing=True,
        enable_history_search=False,
        mouse_support=False,
        style=style,
        bottom_toolbar=_toolbar,
        key_bindings=kb,
        refresh_interval=0.5,
    )


def read_line(prompt_ansi: str, history_path: Optional[Path] = None) -> str:
    """Read one line of input via prompt_toolkit; caches the session across calls.

    The history file passed here MUST NOT be the readline history file — the
    two line-editors use incompatible formats. See promethean.repl for the
    dedicated PT_HISTORY_FILE.
    """
    global _SESSION, _SESSION_HISTORY_PATH
    if _SESSION is not None and _SESSION_HISTORY_PATH != history_path:
        _SESSION = None
    if _SESSION is None:
        _SESSION = _build_session(history_path)
        _SESSION_HISTORY_PATH = history_path
    with patch_stdout(raw=True):
        return _SESSION.prompt(ANSI(prompt_ansi))
