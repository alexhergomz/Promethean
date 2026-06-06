"""Hooks system — run shell commands on agent lifecycle events.

Configured in config.json under the `hooks` key:
    "hooks": [
        {
            "event": "pre_tool",          # one of EVENTS below
            "match": "Write|Edit",         # optional regex on the dispatch key
            "run":   "ruff format $FILE",  # shell command, env-substituted
            "timeout": 10,                  # seconds, default 30
            "block_on_error": false        # default false — log and continue
        }
    ]

Events
    pre_tool   — fired BEFORE a tool executes.   Match: tool name.
                  Stdin payload: {event, tool, inputs}.
                  If block_on_error and the hook exits non-zero, the tool
                  call is aborted with the hook's stderr as the result.
    post_tool  — fired AFTER a tool returns. Match: tool name.
                  Stdin: {event, tool, inputs, result}.
    user_message — fired when the user submits a turn. Match: ignored.
                  Stdin: {event, message}.
    turn_done  — fired after the assistant turn completes. Match: ignored.
                  Stdin: {event, in_tokens, out_tokens, model}.

Env vars passed to every hook:  $CC_EVENT, $CC_TOOL, $CC_SESSION,
                                 $CC_CWD, $CC_MODEL.
$FILE is set when inputs has a `file_path` (handy for formatters/linters).

Best-effort: errors are logged but don't crash the harness unless the
hook itself asked for block_on_error.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

EVENTS = ("pre_tool", "post_tool", "user_message", "turn_done")
_DEFAULT_TIMEOUT = 30


def _select(config: dict, event: str, match_key: str) -> list[dict]:
    hooks = config.get("hooks") or []
    out = []
    for h in hooks:
        if h.get("event") != event:
            continue
        m = h.get("match")
        if m and not re.search(m, match_key or ""):
            continue
        out.append(h)
    return out


def _build_env(config: dict, event: str, payload: dict) -> dict:
    e = dict(os.environ)
    e["CC_EVENT"] = event
    e["CC_TOOL"] = payload.get("tool", "")
    e["CC_SESSION"] = str(config.get("_session_id", ""))
    e["CC_CWD"] = os.getcwd()
    e["CC_MODEL"] = config.get("model", "")
    inputs = payload.get("inputs") or {}
    fp = inputs.get("file_path") or inputs.get("notebook_path") or ""
    if fp:
        e["FILE"] = fp
    return e


def fire(event: str, config: dict, payload: dict,
         match_key: str = "") -> tuple[bool, str]:
    """Run all matching hooks for `event` in order.

    Returns (ok, message). `ok` is False only if some hook had
    block_on_error=True and exited non-zero. `message` is the stderr of
    the first failing blocking hook (used by callers to surface aborts).
    """
    if event not in EVENTS:
        return True, ""
    hooks = _select(config, event, match_key)
    if not hooks:
        return True, ""

    env = _build_env(config, event, payload)
    stdin_blob = json.dumps({"event": event, **payload}, default=str)

    for h in hooks:
        cmd = h.get("run") or ""
        if not cmd:
            continue
        timeout = int(h.get("timeout") or _DEFAULT_TIMEOUT)
        block = bool(h.get("block_on_error"))
        try:
            r = subprocess.run(
                cmd, shell=True, env=env,
                input=stdin_blob,
                capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            if block:
                return False, f"hook timed out after {timeout}s: {cmd}"
            continue
        except Exception as ex:
            if block:
                return False, f"hook crashed: {ex}"
            continue
        if r.returncode != 0 and block:
            return False, (r.stderr or r.stdout or "(no output)").strip()[:500]
    return True, ""
