"""Tool definitions and implementations for Promethean.

Implementations live in focused sub-modules:
  tools.security     _check_path_allowed, _is_safe_bash
  tools.fs           Read / Write / Edit / Glob + diff helpers
  tools.shell        Bash / Grep
  tools.web          WebFetch / WebSearch
  tools.notebook     NotebookEdit
  tools.diagnostics  GetDiagnostics
  tools.interaction  AskUserQuestion / SleepTimer / bridge routing

This module re-exports every public symbol for backward compatibility,
holds the TOOL_SCHEMAS list, the execute_tool dispatcher, and calls
_register_builtins() which wires all built-ins into the tool registry.
"""
from __future__ import annotations

from typing import Callable, Optional

# ── Re-exports (backward compat) ──────────────────────────────────────────

from tools.security import (  # noqa: F401
    _check_path_allowed, _is_safe_bash, _is_dangerous_bash,
)

from tools.fs import (  # noqa: F401
    _read, _write, _edit, _glob,
    generate_unified_diff, maybe_truncate_diff,
)

from tools.shell import _bash, _grep, _kill_proc_tree, _has_rg  # noqa: F401

from tools.web import _webfetch, _websearch  # noqa: F401

from tools.research import _research  # noqa: F401

from tools.notebook import _notebook_edit, _parse_cell_id  # noqa: F401

from tools.diagnostics import (  # noqa: F401
    _get_diagnostics, _detect_language, _run_quietly,
)

from tools.interaction import (  # noqa: F401
    _tg_thread_local, _wx_thread_local, _slack_thread_local,
    _is_in_tg_turn, _is_in_wx_turn, _is_in_slack_turn, _is_in_web_turn,
    _ask_user_question, ask_input_interactive, drain_pending_questions,
    _sleeptimer, _pending_questions, _ask_lock, _INPUT_WAIT_TIMEOUT,
)

from tool_registry import ToolDef, register_tool
from tool_registry import execute_tool as _registry_execute


def _bash_live_callback():
    """Return an on_output callback that pushes each line to the spinner
    tail, or None if the UI module isn't importable / the spinner isn't
    running. Soft-import keeps tools/ free of a hard UI dependency.
    """
    try:
        from ui.render import set_spinner_tail
    except Exception:
        return None
    return set_spinner_tail


def _think(thought: str) -> str:
    """No-op scratchpad tool. The thought lands in the transcript so the
    model can refer back to it on later turns; this function does nothing
    else. Pattern lifted from Anthropic's `think` tool — empirically
    +54% on tau-bench airline domain vs baseline.
    """
    if not thought.strip():
        return "(empty thought — provide a non-empty 'thought' argument)"
    return "Thought recorded."


# ── File-context tracker (Cline pattern) ─────────────────────────────────

def _get_file_tracker(config: dict):
    """Return (tracker_dict, current_turn) or (None, None) if the session
    isn't tracking yet (no agent_state — e.g. early init or tests).
    """
    if not config:
        return None, None
    try:
        import runtime as _runtime
        sctx = _runtime.get_ctx(config)
    except Exception:
        return None, None
    if sctx is None or sctx.agent_state is None:
        return None, None
    return sctx.file_tracker, sctx.agent_state.turn_count


def _read_with_tracker(file_path: str, limit, offset, config: dict) -> str:
    """Wrap _read with Cline-style file-context tracking.

    Stops the model from re-reading files it already saw that haven't
    changed. Stores (mtime, size, turn) per absolute path; whole-file
    reads of an unchanged previously-seen file return a stub. Slice
    reads (limit or offset set) always pass through — the slice may
    target a region the model didn't see before. External changes are
    detected by mtime+size diff and the real content is returned with
    a banner.
    """
    import os
    try:
        abs_path = os.path.abspath(file_path)
        st = os.stat(abs_path)
    except OSError:
        return _read(file_path, limit=limit, offset=offset)

    tracker, current_turn = _get_file_tracker(config)
    if tracker is None:
        return _read(file_path, limit=limit, offset=offset)

    prior = tracker.get(abs_path)
    is_slice = bool(limit) or bool(offset)

    if prior is not None and not is_slice:
        prior_mtime, prior_size, prior_turn = prior
        if prior_mtime == st.st_mtime and prior_size == st.st_size:
            return (
                f"(file unchanged since you last read it on turn {prior_turn} — "
                f"the content is already in your earlier context. Use "
                f"offset/limit to fetch a slice you didn't see before, or "
                f"if you genuinely need the full content again, call Grep "
                f"or use a different tool.)"
            )
        # External change — show real content with a banner.
        out = _read(file_path, limit=limit, offset=offset)
        if not out.startswith("Error:"):
            tracker[abs_path] = (st.st_mtime, st.st_size, current_turn)
            return (
                f"[FILE CHANGED since your last read on turn {prior_turn}]\n\n"
                + out
            )
        return out

    # First whole-file read OR a slice read.
    out = _read(file_path, limit=limit, offset=offset)
    if not out.startswith("Error:") and not is_slice:
        tracker[abs_path] = (st.st_mtime, st.st_size, current_turn)
    return out


def _invalidate_file_tracker(file_path: str, config: dict) -> None:
    """Remove a file from the tracker. Called after Write/Edit so the
    next Read fetches actual content with no stale banner. The model
    already saw the diff in the Edit/Write response, so this isn't
    required for correctness — but the alternative (claiming "you read
    this on turn N", where turn N was before the edit) is confusing.
    """
    import os
    tracker, _ = _get_file_tracker(config)
    if tracker is None:
        return
    tracker.pop(os.path.abspath(file_path), None)


def _write_and_invalidate(p: dict, c: dict) -> str:
    """Wrap _write to invalidate the file-context tracker on success."""
    result = _write(**p)
    if not result.startswith("Error:"):
        _invalidate_file_tracker(p["file_path"], c)
    return result


def _edit_and_invalidate(p: dict, c: dict) -> str:
    """Wrap _edit to invalidate the file-context tracker on success."""
    result = _edit(**p)
    if not result.startswith("Error:"):
        _invalidate_file_tracker(p["file_path"], c)
    return result


# ── Tool JSON schemas (sent to the LLM API) ───────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "Read",
        "description": (
            "Read a file's contents. Returns content with line numbers "
            "(format: 'N\\tline'). Use limit/offset to read large files in chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path"},
                "limit":     {"type": "integer", "description": "Max lines to read"},
                "offset":    {"type": "integer", "description": "Start line (0-indexed)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content":   {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace exact text in a file. old_string must match exactly (including whitespace). "
            "If old_string appears multiple times, use replace_all=true or add more context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path":   {"type": "string"},
                "old_string":  {"type": "string", "description": "Exact text to replace"},
                "new_string":  {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a shell command. Returns stdout+stderr. Stateless (no cd persistence).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds before timeout (default 30). Use 120-300 for package installs (npm, pip, npx), builds, and long-running commands."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern. Returns sorted list of matching paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. **/*.py"},
                "path":    {"type": "string", "description": "Base directory (default: cwd)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex using ripgrep (falls back to grep).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":          {"type": "string", "description": "Regex pattern"},
                "path":             {"type": "string", "description": "File or directory to search"},
                "glob":             {"type": "string", "description": "File filter e.g. *.py"},
                "output_mode":      {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "content=matching lines, files_with_matches=file paths, count=match counts",
                },
                "case_insensitive": {"type": "boolean"},
                "context":          {"type": "integer", "description": "Lines of context around matches"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text content (HTML stripped).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":    {"type": "string"},
                "prompt": {"type": "string", "description": "Hint for what to extract"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "WebSearch",
        "description": "Search the web via DuckDuckGo and return top results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "Research",
        "description": (
            "Research a topic across up to 11 sources in parallel "
            "(arXiv, Semantic Scholar, OpenAlex, HackerNews, GitHub, Reddit, "
            "StackOverflow, Google News, Polymarket, SEC EDGAR, Tavily, Brave). "
            "Returns a synthesized markdown brief with TL;DR, per-domain "
            "findings, minority views, open questions, and numbered citations. "
            "Use this instead of WebSearch when the agent needs current, "
            "engagement-ranked information — academic papers with citation "
            "counts, GitHub repos with star counts, HN threads with points, "
            "SEC filings, prediction market odds, etc. "
            "Domains are auto-classified from the topic; override with "
            "`domains` or pick explicit `sources`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Natural-language query"},
                "domains": {
                    "type": "array",
                    "items": {"type": "string",
                              "enum": ["academic", "tech", "finance",
                                       "news", "social", "web"]},
                    "description": "Restrict to these domains. Omit for auto-classification.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit source names (overrides domains).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results per source (default 15).",
                },
                "synthesize": {
                    "type": "boolean",
                    "description": "Run model synthesis (default true).",
                },
                "use_cache": {
                    "type": "boolean",
                    "description": "Use 24h cache (default true).",
                },
                "time_range": {
                    "type": "string",
                    "description": (
                        "Preset time window: 1d, 3d, 7d, 30d, 90d, 6m, 1y, "
                        "2y, 5y, all, or natural forms like '30days', "
                        "'6months', '2years'. Affects arXiv submittedDate, "
                        "HN created_at, GitHub pushed, Reddit t, Tavily "
                        "start_published_date, etc."
                    ),
                },
                "since": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) lower bound. Overrides time_range.",
                },
                "until": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) upper bound. Overrides time_range.",
                },
                "analyze_citations": {
                    "type": "boolean",
                    "description": (
                        "If true, run secondary Semantic Scholar queries on "
                        "top academic results to surface notable citing "
                        "authors (default 10k-citation threshold). Adds "
                        "2-5 extra API calls per run."
                    ),
                },
                "citation_threshold": {
                    "type": "integer",
                    "description": "Min total citations for a citer to be notable (default 10000).",
                },
                "expand": {
                    "type": "integer",
                    "description": (
                        "If > 0, ask the active model to propose N related "
                        "subqueries (2-6) and merge their results for broader "
                        "coverage. Adds ~1 LLM call and N×source_count HTTP "
                        "calls. Default 0 (disabled)."
                    ),
                },
                "save_as": {
                    "type": "string",
                    "description": "Also copy the rendered brief to this path (absolute or ~-relative).",
                },
                "auto_save": {
                    "type": "boolean",
                    "description": (
                        "Auto-save to ~/.promethean/research_reports/ "
                        "(default true)."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "TaskCreate",
        "description": (
            "Create a new task in the task list. "
            "Use this to track work items, to-dos, and multi-step plans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject":     {"type": "string", "description": "Brief title"},
                "description": {"type": "string", "description": "What needs to be done"},
                "active_form": {"type": "string", "description": "Present-continuous label while in_progress"},
                "metadata":    {"type": "object", "description": "Arbitrary metadata"},
            },
            "required": ["subject", "description"],
        },
    },
    {
        "name": "TaskUpdate",
        "description": (
            "Update a task: change status, subject, description, owner, "
            "dependency edges, or metadata. "
            "Set status='deleted' to remove. "
            "Statuses: pending, in_progress, completed, cancelled, deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":        {"type": "string"},
                "subject":        {"type": "string"},
                "description":    {"type": "string"},
                "status":         {"type": "string", "enum": ["pending","in_progress","completed","cancelled","deleted"]},
                "active_form":    {"type": "string"},
                "owner":          {"type": "string"},
                "add_blocks":     {"type": "array", "items": {"type": "string"}},
                "add_blocked_by": {"type": "array", "items": {"type": "string"}},
                "metadata":       {"type": "object"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskGet",
        "description": "Retrieve full details of a single task by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to retrieve"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskList",
        "description": "List all tasks with their status, owner, and pending blockers.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "NotebookEdit",
        "description": (
            "Edit a Jupyter notebook (.ipynb) cell. "
            "Supports replace (modify existing cell), insert (add new cell after cell_id), "
            "and delete (remove cell) operations. "
            "Read the notebook with the Read tool first to see cell IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_path": {
                    "type": "string",
                    "description": "Absolute path to the .ipynb notebook file",
                },
                "new_source": {
                    "type": "string",
                    "description": "New source code/text for the cell",
                },
                "cell_id": {
                    "type": "string",
                    "description": (
                        "ID of the cell to edit. For insert, the new cell is inserted after this cell "
                        "(or at the beginning if omitted). Use 'cell-N' (0-indexed) if no IDs are set."
                    ),
                },
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown"],
                    "description": "Cell type. Required for insert; defaults to current type for replace.",
                },
                "edit_mode": {
                    "type": "string",
                    "enum": ["replace", "insert", "delete"],
                    "description": "replace (default) / insert / delete",
                },
            },
            "required": ["notebook_path", "new_source"],
        },
    },
    {
        "name": "GetDiagnostics",
        "description": (
            "Get LSP-style diagnostics (errors, warnings, hints) for a source file. "
            "Uses pyright/mypy/flake8 for Python, tsc for TypeScript/JavaScript, "
            "and shellcheck for shell scripts. Returns structured diagnostic output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to diagnose",
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Override auto-detected language: python, javascript, typescript, "
                        "shellscript. Omit to auto-detect from file extension."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "AskUserQuestion",
        "description": (
            "Pause execution and ask the user a clarifying question. "
            "Use this when you need a decision from the user before proceeding. "
            "Returns the user's answer as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "description": "Optional list of choices. Each item: {label, description}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label":       {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label"],
                    },
                },
                "allow_freetext": {
                    "type": "boolean",
                    "description": "If true (default), user may type a free-text answer instead of selecting an option.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "SleepTimer",
        "description": (
            "Schedule a silent background timer. When the timer finishes, it injects an automated "
            "prompt: '(System Automated Event): The timer has finished...' so you can wake up and "
            "execute deferred monitoring tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Number of seconds to sleep before waking up."},
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "Think",
        "description": (
            "Use this tool to write down a thought, hypothesis, intermediate finding, "
            "or note to yourself. It performs no action and changes nothing in the world — "
            "the only effect is that the thought stays in the conversation transcript so "
            "you can refer back to it on later turns. "
            "\n\n"
            "Use it BETWEEN tool calls when you need to reason about what you just learned, "
            "compare options, plan next steps, or jot down a finding before it slips out of "
            "context. Especially useful for: (1) consolidating information after a series of "
            "Read/Grep/Bash calls, (2) deciding which of several approaches to try, "
            "(3) recording a hypothesis you want to test in a later step, (4) listing the "
            "files / symbols you've already examined so you don't re-examine them."
            "\n\n"
            "This tool is your own scratchpad — keep entries short and concrete (one to "
            "three sentences). Do NOT use Think to communicate with the user; use it for "
            "private deliberation that helps you, the agent, stay coherent in long sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The thought, hypothesis, or note to record.",
                },
            },
            "required": ["thought"],
        },
    },
]


# ── Dispatcher (backward-compatible wrapper) ──────────────────────────────

def execute_tool(
    name: str,
    inputs: dict,
    permission_mode: str = "auto",
    ask_permission: Optional[Callable[[str], bool]] = None,
    config: dict = None,
) -> str:
    """Dispatch tool execution; ask permission for write/destructive ops."""
    cfg = config or {}

    # Whitelist enforcement — defense-in-depth alongside the schema-filter
    # in agent.py. If a model somehow generates a tool call for something
    # outside the whitelist (e.g. via prefix matching against tool names
    # learned during pretraining), reject it cleanly here. The schema
    # filter in agent.py prevents the model from being TOLD about
    # forbidden tools; this prevents EXECUTION even if it tries.
    _whitelist = cfg.get("_agent_tools_whitelist")
    if _whitelist and name not in _whitelist:
        return (f"Denied: tool {name!r} is not in this agent's whitelist. "
                f"Allowed: {sorted(_whitelist)}")

    def _check(desc: str) -> bool:
        if permission_mode == "accept-all":
            return True
        if ask_permission:
            return ask_permission(desc)
        return False  # deny by default when no permission handler is set

    if name == "Write":
        if not _check(f"Write to {inputs['file_path']}"):
            return "Denied: user rejected write operation"
    elif name == "Edit":
        if not _check(f"Edit {inputs['file_path']}"):
            return "Denied: user rejected edit operation"
    elif name == "Bash":
        cmd = inputs["command"]
        # Hard deny — fires even in accept-all. Catastrophic patterns only.
        danger = _is_dangerous_bash(cmd)
        if danger:
            return (f"Blocked: catastrophic command pattern detected "
                    f"({danger}). Refusing to execute.")
        if permission_mode != "accept-all" and not _is_safe_bash(cmd):
            if not _check(f"Bash: {cmd}"):
                return "Denied: user rejected bash command"
    elif name == "NotebookEdit":
        if not _check(f"Edit notebook {inputs['notebook_path']}"):
            return "Denied: user rejected notebook edit operation"

    # pre_tool hooks. Non-blocking by default; if a hook has
    # block_on_error=True and exits non-zero, we abort the tool call and
    # return its stderr as the tool result (visible to the model).
    try:
        import hooks as _hooks
        ok_, msg = _hooks.fire("pre_tool", cfg,
                               {"tool": name, "inputs": inputs}, match_key=name)
        if not ok_:
            return f"Hook blocked {name}: {msg}"
    except Exception:
        pass

    # Reversible-tool-log snapshot. Take a pre-state copy of the target
    # file (or mark absent) BEFORE executing the mutating tool. On success
    # we log the entry; on failure or denial we leave the orphaned snap in
    # place — harmless, only consumes a few KB until /clear wipes the
    # session's undo dir.
    _undo_seq = None
    if cfg.get("undo_log", True):
        try:
            import undo_log as _undo
            if _undo.is_mutating(name):
                _undo_seq = _undo.snapshot_before(
                    cfg.get("_session_id", "default"), name, inputs,
                )
        except Exception:
            _undo_seq = None

    result = _registry_execute(name, inputs, cfg,
                               max_output=cfg.get("max_tool_output", 32000))

    if _undo_seq is not None:
        try:
            import undo_log as _undo
            looks_failed = result.startswith(("Error", "Denied", "Blocked"))
            if not looks_failed:
                _undo.record_after(
                    cfg.get("_session_id", "default"),
                    _undo_seq, name, inputs, result,
                )
        except Exception:
            pass

    # post_tool hooks (handy for formatters: "after Write, run ruff format").
    try:
        import hooks as _hooks
        _hooks.fire("post_tool", cfg,
                    {"tool": name, "inputs": inputs, "result": result[:4000]},
                    match_key=name)
    except Exception:
        pass

    # Symbol-graph injection: when the model edits or writes a function/
    # class definition, append a small footer to the tool result listing
    # other files that reference the symbol. The model sees it on the
    # next turn and can decide whether the callers need follow-up. Skipped
    # for failed tool calls, controlled by config["symbol_context"]
    # (default True).
    if (cfg.get("symbol_context", True)
            and name in ("Edit", "Write")
            and not result.startswith(("Error", "Denied", "Blocked"))):
        try:
            import symbol_context as _sc
            footer = _sc.for_edit(inputs, cwd=cfg.get("_cwd"))
            if footer:
                result = result + footer
        except Exception:
            pass

    return result


# ── Register built-in tools with the plugin registry ─────────────────────

def _register_builtins() -> None:
    """Register all built-in tools into the central registry."""
    _schemas = {s["name"]: s for s in TOOL_SCHEMAS}

    _tool_defs = [
        ToolDef(
            name="Read",
            schema=_schemas["Read"],
            func=lambda p, c: (
                "Error: missing required parameter 'file_path'" if not p.get("file_path")
                else _check_path_allowed(p["file_path"], c)
                  or _read_with_tracker(
                      p["file_path"], p.get("limit"), p.get("offset"), c)
            ),
            read_only=True, concurrent_safe=True,
            # FileContextTracker handles its own caching; opt out of the
            # registry's naive (name, params) memoization which would
            # short-circuit the tracker on a second call.
            cacheable=False,
        ),
        ToolDef(
            name="Write",
            schema=_schemas["Write"],
            func=lambda p, c: (
                "Error: missing required parameter 'file_path'" if not p.get("file_path")
                else _check_path_allowed(p["file_path"], c)
                  or _write_and_invalidate(p, c)
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="Edit",
            schema=_schemas["Edit"],
            func=lambda p, c: (
                "Error: missing required parameter 'file_path'" if not p.get("file_path")
                else _check_path_allowed(p["file_path"], c)
                  or _edit_and_invalidate(p, c)
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="Bash",
            schema=_schemas["Bash"],
            func=lambda p, c: _bash(
                p["command"], p.get("timeout", 30),
                c.get("_worktree_cwd"),
                c.get("shell_policy", "allow"),
                c.get("_session_id", "default"),
                on_output=_bash_live_callback(),
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="Glob",
            schema=_schemas["Glob"],
            func=lambda p, c: _glob(p["pattern"], p.get("path"), c.get("_worktree_cwd")),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="Grep",
            schema=_schemas["Grep"],
            func=lambda p, c: _grep(
                p["pattern"], p.get("path"), p.get("glob"),
                p.get("output_mode", "files_with_matches"),
                p.get("case_insensitive", False),
                p.get("context", 0),
                c.get("_worktree_cwd"),
            ),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="WebFetch",
            schema=_schemas["WebFetch"],
            func=lambda p, c: _webfetch(p["url"], p.get("prompt")),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="WebSearch",
            schema=_schemas["WebSearch"],
            func=lambda p, c: _websearch(p["query"]),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="Research",
            schema=_schemas["Research"],
            func=lambda p, c: _research(
                topic=p["topic"],
                domains=p.get("domains"),
                sources=p.get("sources"),
                limit=p.get("limit", 15),
                synthesize=p.get("synthesize", True),
                use_cache=p.get("use_cache", True),
                time_range=p.get("time_range"),
                since=p.get("since"),
                until=p.get("until"),
                analyze_citations=p.get("analyze_citations", False),
                citation_threshold=p.get("citation_threshold", 10000),
                expand=p.get("expand", 0),
                save_as=p.get("save_as"),
                auto_save=p.get("auto_save", True),
                config=c,
            ),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="NotebookEdit",
            schema=_schemas["NotebookEdit"],
            func=lambda p, c: _notebook_edit(
                p["notebook_path"], p["new_source"],
                p.get("cell_id"), p.get("cell_type"),
                p.get("edit_mode", "replace"),
            ),
            read_only=False, concurrent_safe=False,
        ),
        ToolDef(
            name="GetDiagnostics",
            schema=_schemas["GetDiagnostics"],
            func=lambda p, c: _get_diagnostics(p["file_path"], p.get("language")),
            read_only=True, concurrent_safe=True,
        ),
        ToolDef(
            name="AskUserQuestion",
            schema=_schemas["AskUserQuestion"],
            func=lambda p, c: _ask_user_question(
                p["question"], p.get("options"), p.get("allow_freetext", True),
            ),
            read_only=True, concurrent_safe=False,
        ),
        ToolDef(
            name="SleepTimer",
            schema=_schemas["SleepTimer"],
            func=lambda p, c: _sleeptimer(p["seconds"], c),
            read_only=False, concurrent_safe=True,
        ),
        ToolDef(
            name="Think",
            schema=_schemas["Think"],
            func=lambda p, c: _think(p.get("thought", "")),
            read_only=True, concurrent_safe=True,
        ),
    ]
    for td in _tool_defs:
        register_tool(td)


_register_builtins()


# ── Extension tools (auto-discovery) ─────────────────────────────────────
# Each module self-registers its tools on import. Failures are best-effort.

_EXTENSION_MODULES = [
    "memory.tools",
    "multi_agent.tools",
    "rabbit_hole.tools_register",
    "skill.tools",
    "cc_mcp.tools",
    "task.tools",
]

for _mod_name in _EXTENSION_MODULES:
    try:
        __import__(_mod_name)
    except Exception:
        pass  # Extension loading is best-effort; never crash startup

from multi_agent.tools import get_agent_manager as _get_agent_manager  # noqa: F401

try:
    from plugin.loader import register_plugin_tools as _reg_plugin_tools
    _reg_plugin_tools()
except Exception:
    pass   # Plugin loading is best-effort; never crash startup

try:
    from checkpoint.hooks import install_hooks as _install_checkpoint_hooks
    _install_checkpoint_hooks()
except Exception:
    pass

# Sub-modules within tools/ package (self-registering on import)
import importlib as _il
for _sub in ("browser", "email", "files"):
    try:
        _il.import_module(f"tools.{_sub}")
    except Exception:
        pass

# ── Plan mode tools (EnterPlanMode / ExitPlanMode) ────────────────────────

from pathlib import Path as _Path


def _enter_plan_mode(params: dict, config: dict) -> str:
    if config.get("permission_mode") == "plan":
        return "Already in plan mode. Write your plan to the plan file, then call ExitPlanMode."

    session_id = config.get("_session_id", "default")
    plans_dir  = _Path(config.get("_worktree_cwd") or _Path.cwd()) / ".nano_claude" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path  = plans_dir / f"{session_id}.md"

    task_desc = params.get("task_description", "")
    if not plan_path.exists() or plan_path.stat().st_size == 0:
        header = f"# Plan: {task_desc}\n\n" if task_desc else "# Plan\n\n"
        plan_path.write_text(header, encoding="utf-8")

    import runtime
    sctx = runtime.get_ctx(config)
    sctx.prev_permission_mode = config.get("permission_mode", "auto")
    config["permission_mode"]  = "plan"
    sctx.plan_file             = str(plan_path)
    return (
        f"Plan mode activated. Plan file: {plan_path}\n"
        "Write your step-by-step plan to the plan file, then call ExitPlanMode when ready to implement."
    )


def _exit_plan_mode(params: dict, config: dict) -> str:
    if config.get("permission_mode") != "plan":
        return "Not in plan mode."
    import runtime
    sctx = runtime.get_ctx(config)
    plan_file = sctx.plan_file or ""
    plan_content = ""
    if plan_file:
        try:
            plan_content = _Path(plan_file).read_text(encoding="utf-8").strip()
        except Exception:
            plan_content = ""

    # Reject if plan file is effectively empty (only whitespace / top-level title)
    # A top-level title is exactly "# ..." (single #).  ## sections count as content.
    non_trivial_lines = [
        l for l in plan_content.splitlines()
        if l.strip() and not (l.strip().startswith("# ") and not l.strip().startswith("## "))
    ]
    if not non_trivial_lines:
        return (
            "Plan is empty — please write your step-by-step plan to the plan file "
            f"({plan_file}) before exiting plan mode."
        )

    config["permission_mode"] = sctx.prev_permission_mode or "auto"
    sctx.prev_permission_mode = None
    sctx.plan_file = None
    return (
        f"Plan mode exited. Resuming normal permissions.\n\n"
        f"Plan content:\n{plan_content}\n\n"
        "Wait for the user to approve the plan before executing any steps."
    )


_plan_schema_enter = {
    "name": "EnterPlanMode",
    "description": (
        "Switch to plan mode: read-only except for writing the plan file. "
        "Use this to analyze a task and write a step-by-step plan before executing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "Brief description of what you plan to do",
            },
        },
        "required": [],
    },
}
_plan_schema_exit = {
    "name": "ExitPlanMode",
    "description": "Exit plan mode and return to normal permissions to begin executing the plan.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

register_tool(ToolDef("EnterPlanMode", _plan_schema_enter, _enter_plan_mode,
                       read_only=True, concurrent_safe=False))
register_tool(ToolDef("ExitPlanMode",  _plan_schema_exit,  _exit_plan_mode,
                       read_only=False, concurrent_safe=False))
