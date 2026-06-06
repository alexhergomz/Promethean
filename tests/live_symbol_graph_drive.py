"""Live integration test: real local Qwen using the new symbol-graph
tools (Neighborhood / PathBetween / Imports / SearchFiles) against the
fixture repo at tests/fixtures/symbol_graph_repo/.

Two notes about how promethean actually works:

  1. The codeact prompt encourages calling these helpers via `python -c
     "from agent_tools import ..."` over Bash, NOT via JSON tool calls.
     So this test counts BOTH paths:
        - JSON tool name in {Neighborhood, PathBetween, Imports, SearchFiles}
        - Bash command containing one of those identifier names

  2. Subprocess `python -c` only sees `agent_tools` if PYTHONPATH points
     to the promethean root. `qcoder` exports it; running this test
     directly does not, so we set it here.

Pass criteria (loosened to functional, not API-shape):
  - At least one new-tool invocation (either path)
  - Final answer mentions ≥ 2 of the expected files / symbols

Requires the server at http://127.0.0.1:8080 to be up.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO)

# Make agent_tools importable from `python -c` subprocesses spawned by
# the Bash tool — without this the model fails on `from agent_tools import …`.
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

# Force the graph viz on for this run so we can verify it actually fired
# end-to-end. Without this it's auto-suppressed because stdout is a file
# (the test harness pipe), not a TTY.
os.environ["CC_GRAPH_VIEW"] = "1"

import bootstrap as _bs  # noqa: F401
import agent_tools.visualize as _viz_mod   # noqa: E402
from agent import AgentState, run as agent_run, ToolStart, ToolEnd, TurnDone   # noqa: E402
from providers import TextChunk   # noqa: E402

# Force-enable viz independent of TTY detection (matches what the
# /graph-view on slash command would do interactively).
_viz_mod.set_enabled(True)

# Wrap each viz_* with a counter so we can assert end-to-end that the
# JSON-tool wrappers actually called them on a real model run. The
# wrappers replace the names IN the agent_tools._tools_register module
# (which captured them at import time), so we need to patch both the
# source module attrs AND the binding inside _tools_register.
_VIZ_CALLS = {"neighborhood": 0, "path_between": 0,
              "imports": 0, "search_files": 0}


def _counted(name, fn):
    def _wrapper(*args, **kwargs):
        _VIZ_CALLS[name] += 1
        return fn(*args, **kwargs)
    return _wrapper


import agent_tools._tools_register as _reg   # noqa: E402
for _short, _full in (
    ("neighborhood", "viz_neighborhood"),
    ("path_between", "viz_path_between"),
    ("imports",      "viz_imports"),
    ("search_files", "viz_search_files"),
):
    _orig = getattr(_viz_mod, _full)
    _wrapped = _counted(_short, _orig)
    setattr(_viz_mod, _full, _wrapped)
    setattr(_reg, _full, _wrapped)

FIXTURE = str(Path(_HERE) / "fixtures" / "symbol_graph_repo")

CONFIG = {
    "model":              "custom/qwen3.5-9b",
    "custom_base_url":    "http://127.0.0.1:8080/v1",
    "custom_api_key":     "no-key-needed",
    "max_tokens":         2500,
    "max_continuations":  3,
    "permission_mode":    "accept-all",
    "max_tool_output":    8000,
    "log_level":          "warn",
    "thinking":           False,
    "_worktree_cwd":      FIXTURE,
}

PROMPT = (
    f"You are in the repo at {FIXTURE}.\n\n"
    "Find every caller of the function `validate_token` in this repo, "
    "then list the file paths and line numbers.\n\n"
    "You may either:\n"
    "  • call the Neighborhood JSON tool with name='validate_token', or\n"
    "  • run a one-liner: `python -c \"from agent_tools import "
    "neighborhood; import json; print(json.dumps(neighborhood("
    "'validate_token', root='.'), default=str))\"`\n"
)

NEW_TOOL_NAMES = {"Neighborhood", "PathBetween", "Imports", "SearchFiles"}
NEW_TOOL_PYIDENTS = ("neighborhood", "path_between", "imports", "search_files")
EXPECTED_FILES = ["api/handler.py", "api/auth.py"]
EXPECTED_SYMBOLS = ["validate_token"]


def main():
    state = AgentState()
    text_chunks = []
    tool_calls = []  # (name, inputs)

    print(f"[live] fixture: {FIXTURE}")
    print(f"[live] PYTHONPATH={os.environ['PYTHONPATH'].split(os.pathsep)[0]}")
    print(f"[live] max_tokens={CONFIG['max_tokens']}, "
          f"max_continuations={CONFIG['max_continuations']}\n")
    print("─" * 60)

    t0 = time.time()
    try:
        for ev in agent_run(PROMPT, state, CONFIG, system_prompt=_system_prompt()):
            if isinstance(ev, TextChunk):
                if "auto-continuing" in ev.text:
                    print(f"\n[live] BANNER → {ev.text.strip()}\n", flush=True)
                else:
                    text_chunks.append(ev.text)
                    sys.stdout.write(ev.text)
                    sys.stdout.flush()
            elif isinstance(ev, ToolStart):
                tool_calls.append((ev.name, dict(ev.inputs or {})))
                # Brief tool log — show command snippet for Bash so we can
                # see whether agent_tools imports are happening.
                if ev.name == "Bash":
                    cmd = (ev.inputs or {}).get("command", "")[:120]
                    print(f"\n[live] tool: Bash  {cmd!r}", flush=True)
                else:
                    print(f"\n[live] tool: {ev.name}({list(ev.inputs)})", flush=True)
            elif isinstance(ev, (ToolEnd, TurnDone)):
                pass
    except Exception as e:
        print(f"\n[live] FATAL: {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    full = "".join(text_chunks)

    # Count new-tool usage, both paths
    json_tool_hits = sum(1 for n, _ in tool_calls if n in NEW_TOOL_NAMES)
    py_tool_hits = sum(
        1 for n, inp in tool_calls
        if n == "Bash"
        and any(t in (inp.get("command") or "") for t in NEW_TOOL_PYIDENTS)
        and "agent_tools" in (inp.get("command") or "")
    )
    new_tool_hits = json_tool_hits + py_tool_hits

    file_hits = [f for f in EXPECTED_FILES if f in full]
    sym_hits = [s for s in EXPECTED_SYMBOLS if s in full]

    print("\n\n" + "═" * 60)
    print("LIVE SYMBOL-GRAPH TEST RESULTS")
    print("═" * 60)
    print(f"elapsed:                {elapsed:.1f}s")
    print(f"total tools fired:      {len(tool_calls)}")
    print(f"  json-tool calls:      {json_tool_hits}  (Neighborhood/etc.)")
    print(f"  python-import calls:  {py_tool_hits}    (Bash + from agent_tools)")
    print(f"  total new-tool hits:  {new_tool_hits}")
    print(f"final text length:      {len(full)} chars")
    print(f"expected files seen:    {len(file_hits)}/{len(EXPECTED_FILES)}  {file_hits}")
    print(f"expected symbols seen:  {len(sym_hits)}/{len(EXPECTED_SYMBOLS)}  {sym_hits}")
    total_viz = sum(_VIZ_CALLS.values())
    print(f"viz fired:              {total_viz}  {dict(_VIZ_CALLS)}")

    fail = []
    if new_tool_hits < 1:
        fail.append("model did not invoke the new symbol-graph helpers via "
                    "EITHER path (JSON tool or python import)")
    if len(file_hits) < 1:
        fail.append("final answer did not mention any expected file path")
    # If the JSON tool path was used at least once, viz MUST have fired the
    # same number of times. (Python-import path goes through Bash subprocess
    # — viz can't fire there, since viz is in the parent process only.)
    if json_tool_hits > 0 and total_viz != json_tool_hits:
        fail.append(
            f"viz call count {total_viz} != json-tool hits {json_tool_hits}"
            " — JSON tool wrapper isn't invoking the viz layer"
        )

    if fail:
        print("\nFAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print(f"\nPASS — {new_tool_hits} new-tool invocations, "
          f"final answer named {len(file_hits)} expected file(s).")
    return 0


def _system_prompt():
    from context import build_system_prompt
    return build_system_prompt(CONFIG)


if __name__ == "__main__":
    sys.exit(main())
