"""
commands/core.py — Core utility commands for Promethean.

Commands: /help, /clear, /context, /cost, /compact, /init, /export,
          /copy, /status, /doctor, /proactive, /image, /circuit
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union

from ui.render import clr, info, ok, warn, err

# VERSION is imported lazily from promethean to avoid circular imports
_VERSION_STR = ""

def _get_version() -> str:
    global _VERSION_STR
    if not _VERSION_STR:
        try:
            import importlib
            cc = importlib.import_module("promethean")
            _VERSION_STR = getattr(cc, "VERSION", "?")
        except Exception:
            _VERSION_STR = "?"
    return _VERSION_STR


def cmd_help(_args: str, _state, config) -> bool:
    try:
        import promethean
    except Exception:
        info("Promethean — type /model, /save, /load, /history, /context, /exit for commands.")
        return True

    doc = promethean.__doc__ or ""
    print(doc)

    # Safety net: surface any registered command that the curated docstring
    # forgot to mention (e.g. modular/plugin additions, or newly added commands
    # whose author didn't update the docstring). Walks COMMANDS, groups by
    # handler so aliases share a row, skips anything already referenced.
    commands = getattr(promethean, "COMMANDS", {})
    meta     = getattr(promethean, "_CMD_META", {})

    aliases_by_func: dict[object, list[str]] = {}
    for name, func in commands.items():
        aliases_by_func.setdefault(func, []).append(name)

    missing: list[tuple[str, str]] = []
    seen: set[object] = set()
    for func, names in aliases_by_func.items():
        if func in seen:
            continue
        seen.add(func)
        if any(f"/{n}" in doc for n in names):
            continue
        primary = min(names, key=len)
        extra = [n for n in names if n != primary]
        label = f"/{primary}" + (f" (/{', /'.join(extra)})" if extra else "")
        desc = next((meta[n][0] for n in names if n in meta), "(no description)")
        missing.append((label, desc))

    if missing:
        print()
        print("Also available (auto-detected — not in curated list above):")
        w = max(len(m[0]) for m in missing)
        for label, desc in missing:
            print(f"  {label:<{w}}  {desc}")

    return True


def cmd_clear(_args: str, state, config) -> bool:
    state.messages.clear()
    state.turn_count = 0
    state.total_input_tokens = 0
    state.total_output_tokens = 0
    state.__dict__.pop("_usage_estimated", None)
    try:
        import undo_log
        undo_log.clear_session(config.get("_session_id", "default"))
    except Exception:
        pass
    ok("Conversation cleared.")
    return True


def cmd_graph_view(args: str, _state, _config) -> bool:
    """Toggle the live graph-tool CLI visualization (Neighborhood /
    PathBetween / Imports / SearchFiles).

    Usage:
        /graph-view             show current state
        /graph-view on|off      force on / off for this session
        /graph-view toggle      flip current state
        /graph-view auto        revert to auto (env + TTY detect)
    """
    try:
        from agent_tools.visualize import is_enabled, set_enabled
    except ImportError:
        warn("graph-view unavailable (agent_tools missing)")
        return True

    arg = args.strip().lower()
    if not arg:
        info(f"graph-view: {'on' if is_enabled() else 'off'}")
        return True
    if arg in ("on", "true", "1", "yes", "enable"):
        set_enabled(True)
        ok("graph-view: on")
    elif arg in ("off", "false", "0", "no", "disable"):
        set_enabled(False)
        ok("graph-view: off")
    elif arg in ("toggle", "t"):
        set_enabled(not is_enabled())
        ok(f"graph-view: {'on' if is_enabled() else 'off'}")
    elif arg in ("auto", "default", "reset"):
        set_enabled(None)
        info(f"graph-view: auto ({'on' if is_enabled() else 'off'} now)")
    else:
        warn(f"unknown arg: {arg!r}. Use: /graph-view [on|off|toggle|auto]")
    return True


def cmd_context(_args: str, state, config) -> bool:
    msg_chars = sum(len(str(m.get("content", ""))) for m in state.messages)
    est_tokens = msg_chars // 4
    info(f"Messages:         {len(state.messages)}")
    info(f"Estimated tokens: ~{est_tokens:,}")
    info(f"Model:            {config['model']}")
    info(f"Max tokens:       {config['max_tokens']:,}")
    return True


def cmd_undo(args: str, _state, config) -> bool:
    """Revert the last N file mutations from this session.

    Usage:
        /undo            preview the last mutation and ask to revert
        /undo N          preview the last N mutations
        /undo confirm N  actually revert (no prompt)
        /undo list       show the full history for this session
    """
    import undo_log

    session_id = config.get("_session_id", "default")
    parts = args.strip().split()
    confirm = False
    n = 1
    if parts and parts[0] == "list":
        entries = undo_log.last_n(session_id, 999)
        if not entries:
            info("(no reversible operations in this session)")
            return True
        info(f"Undo history for session {session_id} ({len(entries)} pending):")
        for e in entries:
            ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            info(f"  [{e['seq']:3d}] {ts}  {e['tool']:13s} {e['file_path']}")
        return True
    if parts and parts[0] == "confirm":
        confirm = True
        parts = parts[1:]
    if parts and parts[0].isdigit():
        n = max(1, int(parts[0]))

    entries = undo_log.last_n(session_id, n)
    if not entries:
        info("(nothing to undo — no file mutations recorded this session)")
        return True

    info(f"Will revert the last {len(entries)} file mutation(s):")
    for e in entries:
        info(f"  - {e['tool']} on {e['file_path']}")
    if not confirm:
        info(f"\nRun  /undo confirm {n}  to actually apply.")
        return True

    for e in entries:
        ok_, msg = undo_log.revert(session_id, e)
        (ok if ok_ else err)(msg)
    return True


_OPTILLM_VALID = {
    "moa", "mcts", "bon", "plansearch", "cot_reflection", "re2",
    "self_consistency", "mars", "cepo", "leap", "rstar", "rto",
    "pvg", "z3", "autothink",
}

def cmd_optillm(args: str, _state, config) -> bool:
    """Pick (or clear) the OptiLLM inference-time technique for upcoming turns.

    Usage:
        /optillm                  show current approach
        /optillm <slug>           set approach (moa | mcts | bon | mars | ...)
        /optillm off              clear (disable)

    Requires the optillm proxy to be running and MINIMAX_BASE_URL (or
    equivalent) pointed at it. See TODO_NEXT.md for install instructions.
    """
    arg = args.strip().lower()
    if not arg:
        cur = config.get("optillm_approach") or "off"
        info(f"optillm_approach: {cur}")
        info(f"valid slugs: {', '.join(sorted(_OPTILLM_VALID))}")
        return True
    if arg in ("off", "none", "clear", "disable"):
        config["optillm_approach"] = None
        ok("optillm: off")
        return True
    if arg not in _OPTILLM_VALID:
        warn(f"unknown approach: {arg!r}. Valid: {', '.join(sorted(_OPTILLM_VALID))}")
        return True
    config["optillm_approach"] = arg
    ok(f"optillm: {arg} (next request will use this technique)")
    return True


def cmd_cost(_args: str, state, config) -> bool:
    from cc_config import calc_cost
    cost = calc_cost(config["model"],
                     state.total_input_tokens,
                     state.total_output_tokens)
    # "~" prefix means at least one turn arrived without real usage and was
    # estimated (typically because an OptiLLM-style proxy stripped the
    # usage chunk). Figures are approximate, not authoritative.
    approx = "~" if getattr(state, "_usage_estimated", False) else ""
    info(f"Input tokens:  {approx}{state.total_input_tokens:,}")
    info(f"Output tokens: {approx}{state.total_output_tokens:,}")
    info(f"Est. cost:     {approx}${cost:.4f} USD")
    if approx:
        info("  (~ = some turns arrived via a proxy that stripped usage; estimated from emitted bytes)")
    return True


def cmd_compact(args: str, state, config) -> bool:
    """Manually compact conversation history."""
    from compaction import manual_compact
    focus = args.strip()
    if focus:
        info(f"Compacting with focus: {focus}")
    else:
        info("Compacting conversation...")
    success, msg = manual_compact(state, config, focus=focus)
    if success:
        info(msg)
    else:
        err(msg)
    return True


# File extensions → language name, for the /init tech-stack scan.
_LANG_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin",
    ".rb": "Ruby", ".php": "PHP", ".c": "C", ".h": "C", ".cpp": "C++",
    ".cc": "C++", ".hpp": "C++", ".cs": "C#", ".swift": "Swift",
    ".scala": "Scala", ".sh": "Shell", ".lua": "Lua",
}

# Directories skipped while scanning, and root manifests worth naming.
_SCAN_SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv",
              "dist", "build", ".nano_claude", ".mypy_cache", ".pytest_cache",
              "target", ".next", "vendor"}
_MANIFESTS = ["pyproject.toml", "requirements.txt", "setup.py", "package.json",
              "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "Gemfile",
              "composer.json", "Makefile", "CMakeLists.txt", "Dockerfile"]
_ENTRY_CANDIDATES = ["main.py", "__main__.py", "app.py", "manage.py", "cli.py",
                     "index.js", "index.ts", "server.js", "main.go", "main.rs",
                     "src/main.rs", "src/index.ts", "src/index.js"]


def _scan_project(root: Path) -> dict:
    """Best-effort deterministic survey of a repo for /init. Bounded so it
    stays fast on large trees; no network, no model call."""
    from collections import Counter
    langs: Counter = Counter()
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP
                       and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            lang = _LANG_BY_EXT.get(ext)
            if lang:
                langs[lang] += 1
            seen += 1
            if seen > 20000:            # cap on very large repos
                break
        if seen > 20000:
            break

    manifests = [m for m in _MANIFESTS if (root / m).exists()]
    entries = [e for e in _ENTRY_CANDIDATES if (root / e).exists()]
    has_tests = (root / "tests").is_dir() or (root / "test").is_dir()
    return {
        "languages": [lang for lang, _ in langs.most_common(5)],
        "manifests": manifests,
        "entries": entries,
        "has_tests": has_tests,
        "test_command": _infer_test_command(root, manifests, has_tests),
    }


def _infer_test_command(root: Path, manifests: list[str], has_tests: bool) -> str:
    """Guess how tests are run from the manifests present."""
    if "package.json" in manifests:
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
            if isinstance(pkg.get("scripts"), dict) and "test" in pkg["scripts"]:
                return "npm test"
        except Exception:
            pass
    if "Cargo.toml" in manifests:
        return "cargo test"
    if "go.mod" in manifests:
        return "go test ./..."
    if "Makefile" in manifests:
        try:
            mk = (root / "Makefile").read_text(encoding="utf-8")
            if any(line.startswith("test:") for line in mk.splitlines()):
                return "make test"
        except Exception:
            pass
    if has_tests or "pyproject.toml" in manifests or "setup.py" in manifests:
        return "pytest"
    return ""


def _render_claude_md(project_name: str, scan: dict) -> str:
    """Render a starter CLAUDE.md from a project scan. Sections the scan
    can't fill keep an HTML-comment prompt so the user knows what to add."""
    def bullets(items):
        return "".join(f"- {i}\n" for i in items)

    tech = scan["languages"] + scan["manifests"]
    lines = [f"# {project_name}\n"]

    lines.append("## Project Overview")
    lines.append("<!-- One or two sentences on what this project does. -->\n")

    lines.append("## Tech Stack")
    if tech:
        lines.append(bullets(tech).rstrip())
    else:
        lines.append("<!-- Languages, frameworks, key dependencies. -->")
    lines.append("")

    lines.append("## Conventions")
    lines.append("<!-- Coding style, naming, patterns to follow. -->\n")

    lines.append("## Important Files")
    if scan["entries"]:
        lines.append(bullets(f"`{e}`" for e in scan["entries"]).rstrip())
    else:
        lines.append("<!-- Key entry points, config files. -->")
    lines.append("")

    lines.append("## Testing")
    if scan["test_command"]:
        lines.append(f"Run the test suite with `{scan['test_command']}`.")
    else:
        lines.append("<!-- How to run tests, testing conventions. -->")
    lines.append("")

    return "\n".join(lines) + "\n"


def cmd_init(args: str, state, config) -> bool:
    """Create a starter CLAUDE.md, pre-filled from a scan of the repo."""
    root = Path.cwd()
    target = root / "CLAUDE.md"
    if target.exists():
        err(f"CLAUDE.md already exists at {target}")
        info("Edit it directly or delete it first.")
        return True

    scan = _scan_project(root)
    target.write_text(_render_claude_md(root.name, scan), encoding="utf-8")
    info(f"Created {target}")
    detected = scan["languages"] or scan["manifests"]
    if detected:
        info(f"Detected: {', '.join(detected)}")
    info("Edit it to describe the project for the agent.")
    return True


def cmd_export(args: str, state, config) -> bool:
    """Export conversation history to a file."""
    if not state.messages:
        err("No conversation to export.")
        return True

    arg = args.strip()
    if arg:
        out_path = Path(arg)
    else:
        export_dir = Path.cwd() / ".nano_claude" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / f"conversation_{ts}.md"

    is_json = out_path.suffix.lower() == ".json"

    if is_json:
        out_path.write_text(
            json.dumps(state.messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        lines = []
        for m in state.messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "(structured content)"
            if role == "user":
                lines.append(f"## User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"## Assistant\n\n{content}\n")
            elif role == "tool":
                name = m.get("name", "tool")
                lines.append(f"### Tool: {name}\n\n```\n{content[:2000]}\n```\n")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")

    info(f"Exported {len(state.messages)} messages to {out_path}")
    return True


def cmd_copy(args: str, state, config) -> bool:
    """Copy the last assistant response to clipboard."""
    last_reply = None
    for m in reversed(state.messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                last_reply = content
                break

    if not last_reply:
        err("No assistant response to copy.")
        return True

    try:
        import subprocess as _sp
        if sys.platform == "win32":
            proc = _sp.Popen(["clip"], stdin=_sp.PIPE)
            proc.communicate(last_reply.encode("utf-16le"))
        elif sys.platform == "darwin":
            proc = _sp.Popen(["pbcopy"], stdin=_sp.PIPE)
            proc.communicate(last_reply.encode("utf-8"))
        else:
            for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
                try:
                    proc = _sp.Popen(cmd, stdin=_sp.PIPE)
                    proc.communicate(last_reply.encode("utf-8"))
                    break
                except FileNotFoundError:
                    continue
            else:
                err("No clipboard tool found. Install xclip or xsel.")
                return True
        info(f"Copied {len(last_reply)} chars to clipboard.")
    except Exception as e:
        err(f"Failed to copy: {e}")
    return True


def cmd_status(args: str, state, config) -> bool:
    """Show current session status."""
    from providers import detect_provider
    from compaction import estimate_tokens, get_context_limit

    model = config.get("model", "unknown")
    provider = detect_provider(model)
    perm_mode = config.get("permission_mode", "auto")
    session_id = config.get("_session_id", "N/A")
    turn_count = getattr(state, "turn_count", 0)
    msg_count = len(getattr(state, "messages", []))
    tokens_in = getattr(state, "total_input_tokens", 0)
    tokens_out = getattr(state, "total_output_tokens", 0)
    est_ctx = estimate_tokens(getattr(state, "messages", []))
    ctx_limit = get_context_limit(model, config)
    ctx_pct = (est_ctx / ctx_limit * 100) if ctx_limit else 0
    plan_mode = config.get("permission_mode") == "plan"

    print(f"  Version:     {_get_version()}")
    print(f"  Model:       {model} ({provider})")
    print(f"  Permissions: {perm_mode}" + (" [PLAN MODE]" if plan_mode else ""))
    print(f"  Session:     {session_id}")
    print(f"  Turns:       {turn_count}")
    print(f"  Messages:    {msg_count}")
    print(f"  Tokens:      ~{tokens_in} in / ~{tokens_out} out")
    print(f"  Context:     ~{est_ctx} / {ctx_limit} ({ctx_pct:.0f}%)")
    return True


def cmd_doctor(args: str, state, config) -> bool:
    """Diagnose installation health and connectivity."""
    import subprocess as _sp
    from providers import PROVIDERS, detect_provider, get_api_key

    ok_n = warn_n = fail_n = 0

    def _print_safe(s):
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", errors="replace").decode())

    def _ok(msg):
        nonlocal ok_n; ok_n += 1
        _print_safe(clr("  [PASS] ", "green") + msg)

    def _warn(msg):
        nonlocal warn_n; warn_n += 1
        _print_safe(clr("  [WARN] ", "yellow") + msg)

    def _fail(msg):
        nonlocal fail_n; fail_n += 1
        _print_safe(clr("  [FAIL] ", "red") + msg)

    info("Running diagnostics...")
    print()

    v = sys.version_info
    if v >= (3, 10):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor}.{v.micro} (need ≥3.10)")

    try:
        r = _sp.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok(f"Git: {r.stdout.strip()}")
        else:
            _fail("Git: not working")
    except Exception:
        _fail("Git: not found")

    try:
        r = _sp.run(["git", "rev-parse", "--is-inside-work-tree"],
                    capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok("Inside a git repository")
        else:
            _warn("Not inside a git repository")
    except Exception:
        _warn("Could not check git repo status")

    model = config.get("model", "")
    base = config.get("custom_base_url") or os.environ.get("CUSTOM_BASE_URL", "")
    key  = get_api_key(detect_provider(model), config)

    if not base:
        _fail("No backend configured — set custom_base_url "
              "(llama-server, e.g. http://127.0.0.1:8080/v1)")
    else:
        print(f"  ... testing backend at {base} ...")
        try:
            import urllib.request, urllib.error
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            req = urllib.request.Request(base.rstrip("/") + "/models", headers=headers)
            try:
                urllib.request.urlopen(req, timeout=10)
                _ok(f"Backend reachable at {base}")
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    _fail("Backend: unauthorized (401) — check custom_api_key")
                else:
                    _warn(f"Backend: HTTP {e.code}")
            except Exception as e:
                _fail(f"Backend: cannot reach {base} — is llama-server running? ({e})")
        except Exception as e:
            _warn(f"Backend test skipped: {e}")

    # ── General network connectivity ──
    print()
    try:
        import urllib.request
        urllib.request.urlopen("https://httpbin.org/status/200", timeout=5)
        _ok("Internet connectivity: OK")
    except Exception:
        _fail("Internet connectivity: cannot reach external hosts")

    # ── Dependencies ──
    print()
    for mod, desc, required in [
        ("rich", "Rich (live markdown rendering)", True),
        ("pyte", "pyte (terminal emulator for bridges)", True),
        ("PIL", "Pillow (clipboard image /image)", False),
        ("sounddevice", "sounddevice (voice recording)", False),
        ("faster_whisper", "faster-whisper (local STT)", False),
    ]:
        try:
            __import__(mod)
            _ok(desc)
        except ImportError:
            if required:
                _fail(f"{desc}: not installed (required)")
            else:
                _warn(f"{desc}: not installed (optional)")

    print()
    claude_md = Path.cwd() / "CLAUDE.md"
    global_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        _ok(f"Project CLAUDE.md: {claude_md}")
    else:
        _warn("No project CLAUDE.md (run /init to create)")
    if global_md.exists():
        _ok(f"Global CLAUDE.md: {global_md}")

    ckpt_root = Path.home() / ".nano_claude" / "checkpoints"
    if ckpt_root.exists():
        total = sum(f.stat().st_size for f in ckpt_root.rglob("*") if f.is_file())
        mb = total / (1024 * 1024)
        sessions = sum(1 for d in ckpt_root.iterdir() if d.is_dir())
        if mb > 100:
            _warn(f"Checkpoints: {mb:.1f} MB ({sessions} sessions)")
        else:
            _ok(f"Checkpoints: {mb:.1f} MB ({sessions} sessions)")

    perm = config.get("permission_mode", "auto")
    if perm == "accept-all":
        _warn(f"Permission mode: {perm} (all operations auto-approved)")
    else:
        _ok(f"Permission mode: {perm}")

    print()
    total = ok_n + warn_n + fail_n
    summary = f"  {ok_n} passed, {warn_n} warnings, {fail_n} failures ({total} checks)"
    if fail_n:
        _print_safe(clr(summary, "red"))
    elif warn_n:
        _print_safe(clr(summary, "yellow"))
    else:
        _print_safe(clr(summary, "green"))

    return True


# ── Setup wizard ──────────────────────────────────────────────────────────

def _first_served_model(base_url: str) -> str | None:
    """Return the first model id a server advertises on /v1/models, or None."""
    try:
        import urllib.request
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as resp:
            data = json.loads(resp.read())
        for entry in data.get("data", []):
            if entry.get("id"):
                return entry["id"]
    except Exception:
        pass
    return None


def run_setup_wizard(config: dict) -> None:
    """Interactive first-run setup for the local llama.cpp backend."""
    from cc_config import save_config

    print()
    info("Welcome to Promethean! Let's point it at your local model.\n")
    info("Promethean runs against llama.cpp (llama-server) over the "
         "OpenAI-compatible\nAPI — or any other OpenAI-compatible server.\n")

    # ── Base URL ──
    default_url = config.get("custom_base_url") or "http://127.0.0.1:8080/v1"
    try:
        base = input(clr(f"  Server base URL [{default_url}]: ", "cyan")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    base = base or default_url
    config["custom_base_url"] = base

    # ── Model ── prefer the model the server already has loaded.
    default_model = config.get("model") or "custom/qwen3.5-9b"
    if not default_model.startswith("custom/"):
        default_model = "custom/qwen3.5-9b"
    detected = _first_served_model(base)
    if detected:
        default_model = f"custom/{detected}"
        ok(f"Detected loaded model: {detected}")
    try:
        m = input(clr(f"  Model [{default_model}]: ", "cyan")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    model = m or default_model
    if not model.startswith("custom/"):
        model = "custom/" + model
    config["model"] = model

    # ── Optional API key (only for authenticated proxies) ──
    # llama-server needs none; a fronting proxy might. Leave blank to skip.
    if not (os.environ.get("CUSTOM_API_KEY") or config.get("custom_api_key")):
        try:
            key = input(clr("  API key (blank for llama-server): ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if key:
            config["custom_api_key"] = key

    # ── Verify ──
    print()
    info("Verifying connection...")
    if detected or _first_served_model(base):
        ok("Connected to the server.")
    else:
        warn(f"Could not reach {base}.")
        info("Start llama-server first, for example:")
        info("  llama-server -m model.gguf -c 8192 --host 127.0.0.1 --port 8080")

    save_config(config)
    print()
    ok(f"Setup complete! Model: {config['model']}")
    info("Type a message to start, or /help for available commands.\n")


def cmd_proactive(args: str, state, config) -> bool:
    """Manage proactive background polling.

    /proactive            — show current status
    /proactive 5m         — enable, trigger after 5 min of inactivity
    /proactive 30s / 1h   — enable with custom interval
    /proactive off        — disable
    """
    args = args.strip().lower()

    import runtime
    sctx = runtime.get_ctx(config)

    if not args:
        if sctx.proactive_enabled:
            interval = sctx.proactive_interval
            info(f"Proactive background polling: ON  (triggering every {interval}s of inactivity)")
        else:
            info("Proactive background polling: OFF  (use /proactive 5m to enable)")
        return True

    if args == "off":
        sctx.proactive_enabled = False
        info("Proactive background polling: OFF")
        return True

    multiplier = 1
    val_str = args
    if args.endswith("m"):
        multiplier = 60
        val_str = args[:-1]
    elif args.endswith("h"):
        multiplier = 3600
        val_str = args[:-1]
    elif args.endswith("s"):
        val_str = args[:-1]

    try:
        val = int(val_str)
        sctx.proactive_interval = val * multiplier
    except ValueError:
        err(f"Invalid duration: '{args}'. Use '5m', '30s', '1h', or 'off'.")
        return True

    sctx.proactive_enabled = True
    sctx.last_interaction_time = time.time()
    info(f"Proactive background polling: ON  (triggering every {sctx.proactive_interval}s of inactivity)")
    return True


def cmd_image(args: str, state, config) -> Union[bool, tuple]:
    """Grab image from clipboard and send to vision model with optional prompt."""
    try:
        from PIL import ImageGrab
        import io, base64
    except ImportError:
        err("Pillow is required for /image. Install with: pip install promethean[vision]")
        if sys.platform == "linux":
            err("On Linux, clipboard support also requires xclip: sudo apt install xclip")
        return True

    img = ImageGrab.grabclipboard()
    if img is None:
        if sys.platform == "linux":
            err("No image found in clipboard. On Linux, xclip is required (sudo apt install xclip). "
                "Copy an image with Flameshot, GNOME Screenshot, or: xclip -selection clipboard -t image/png -i file.png")
        elif sys.platform == "darwin":
            err("No image found in clipboard. Copy an image first "
                "(Cmd+Ctrl+Shift+4 captures a screenshot region to clipboard).")
        else:
            err("No image found in clipboard. Copy an image first "
                "(Win+Shift+S captures a screenshot region to clipboard).")
        return True

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    size_kb = len(buf.getvalue()) / 1024

    info(f"📷 Clipboard image captured ({size_kb:.0f} KB, {img.size[0]}x{img.size[1]})")
    import runtime
    runtime.get_ctx(config).pending_image = b64

    prompt = args.strip() if args.strip() else "What do you see in this image? Describe it in detail."
    return ("__image__", prompt)


_web_thread = None  # daemon thread running start_web_server(), if any


def cmd_web(args: str, state, config) -> bool:
    """Start the web terminal / chat UI in a background thread.

    /web                          — start on 127.0.0.1:8080 (auto-picks free port)
    /web 9000                     — use port 9000
    /web --host 0.0.0.0           — bind to network
    /web --no-auth                — disable terminal password (local only)
    /web status                   — show whether it's running
    """
    global _web_thread
    import threading

    tokens = (args or "").strip().split()
    sub = tokens[0].lower() if tokens else ""

    if sub == "status":
        if _web_thread and _web_thread.is_alive():
            info("Web server: running (started via /web this session).")
        else:
            info("Web server: not running.")
        return True

    if _web_thread and _web_thread.is_alive():
        info("Web server already running in this session. Use /web status to check.")
        return True

    if os.environ.get("PROMETHEAN_WEB_SERVER") == "1":
        warn("You're already inside a web-terminal session. Nested web launch refused.")
        return True

    port: int | None = None
    host = "127.0.0.1"
    no_auth = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            port = int(t)
        elif t == "--no-auth":
            no_auth = True
        elif t == "--host" and i + 1 < len(tokens):
            host = tokens[i + 1]; i += 1
        elif t.startswith("--host="):
            host = t.split("=", 1)[1]
        elif t.startswith("--port="):
            try: port = int(t.split("=", 1)[1])
            except ValueError: pass
        else:
            warn(f"Unknown /web arg: {t}  (try: [port] [--host H] [--no-auth])")
            return True
        i += 1

    try:
        from web.server import start_web_server
    except ImportError as e:
        err(f"Web module unavailable: {e}")
        return True

    def _run():
        try:
            start_web_server(port=port, host=host, no_auth=no_auth)
        except SystemExit:
            pass
        except Exception as e:
            import logging_utils as _log
            _log.error("web_server_crashed", error=str(e)[:200])

    _web_thread = threading.Thread(target=_run, daemon=True, name="web-server")
    _web_thread.start()
    time.sleep(0.3)  # let the banner print before the REPL redraws its prompt
    info("Web server started in background. Continue typing — REPL is still live.")
    return True


def cmd_circuit(args: str, state, config) -> bool:
    """Inspect and manage per-provider circuit breakers.

    /circuit                    — list all breakers and their state
    /circuit status [provider]  — same as above, optionally filtered
    /circuit reset <provider>   — force-close a breaker (or 'all')
    """
    import circuit_breaker as _cb

    parts = args.strip().split()
    sub = parts[0].lower() if parts else "status"
    target = parts[1] if len(parts) > 1 else ""

    if sub in ("reset", "close", "clear"):
        if not target:
            err("Usage: /circuit reset <provider>  (or 'all')")
            return True
        if target.lower() == "all":
            names = list(_cb._registry.keys())
            if not names:
                info("No circuit breakers to reset.")
                return True
            for name in names:
                _cb.reset_breaker(name)
            ok(f"Reset {len(names)} circuit breaker(s): {', '.join(names)}")
            return True
        if target not in _cb._registry:
            warn(f"No circuit breaker registered for '{target}'. Nothing to reset.")
            return True
        _cb.reset_breaker(target)
        ok(f"Circuit breaker for '{target}' reset (force-closed).")
        return True

    if sub not in ("status", ""):
        err(f"Unknown /circuit subcommand: {sub}. Use: status | reset")
        return True

    breakers = _cb._registry
    if target:
        breakers = {k: v for k, v in breakers.items() if k == target}

    if not breakers:
        info("No circuit breakers active yet (none have been exercised this session).")
        return True

    for name, b in breakers.items():
        st = b.state.value
        color = {"closed": "green", "half_open": "yellow", "open": "red"}.get(st, "dim")
        line = f"  {name:<12} state={clr(st, color)}  failures={len(b._failure_times)}/{b.threshold}"
        if b._opened_at is not None and b.state.value == "open":
            remaining = max(0.0, b.cooldown - (time.monotonic() - b._opened_at))
            line += f"  cooldown_remaining={remaining:.0f}s"
        print(line)
    return True
