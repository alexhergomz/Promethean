"""Live integration test: drive the local llama-server with max_tokens forced
low so finish_reason='length' fires, and verify the auto-continue path recovers.

Not part of the pytest suite — invoked manually:
    python tests/live_truncation_drive.py

Requires the server at http://127.0.0.1:8080 to be up.
Writes a probe file at /tmp/qcoder_truncation_probe.md and reports what
happened at the harness level.
"""
from __future__ import annotations

import os
import sys
import time

# Project root on sys.path (mirror what promethean.py does)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

# Bootstrap registers built-in tools + agent_tools.
import bootstrap as _bs  # noqa: F401

from agent import AgentState, run as agent_run
from providers import TextChunk, ThinkingChunk
from agent import ToolStart, ToolEnd, TurnDone


PROBE = "/tmp/qcoder_truncation_probe.md"

# Pick max_tokens that's:
#   - large enough for the model to plan a couple of sentences and START a
#     Write tool call (otherwise it never emits a tool call and we just see
#     pure text-truncation, which doesn't exercise the malformed-tool-call
#     recovery path), AND
#   - small enough that a 200-line file body can't fit, so the JSON args of
#     the Write call get cut mid-string.
# 1200 is empirically in this sweet spot for Qwen3.5-9B Q4_K_M.
SMALL_MAX_TOKENS = 2500

CONFIG = {
    "model":              "custom/qwen3.5-9b",
    "custom_base_url":    "http://127.0.0.1:8080/v1",
    "custom_api_key":     "no-key-needed",
    "max_tokens":         SMALL_MAX_TOKENS,
    "max_continuations":  3,
    "permission_mode":    "accept-all",
    "max_tool_output":    8000,
    "log_level":          "warn",
    "thinking":           False,
    "thinking_budget":    0,
}

PROMPT = (
    f"Use the Write tool RIGHT NOW (no planning, no preamble) to create "
    f"{PROBE}. Content: a study guide on the Mersenne Twister algorithm "
    "with sections for history, mathematical structure, the recurrence "
    "relation, the tempering step, period analysis, security weaknesses, "
    "and three example use cases. Aim for 200+ lines, with code blocks. "
    "Do NOT use bash heredocs. Just call Write."
)


def main():
    if os.path.exists(PROBE):
        os.remove(PROBE)

    state = AgentState()
    events = {
        "text_chunks":    0,
        "tool_start":     [],
        "tool_end":       [],
        "turn_done":      [],
        "auto_banners":   0,
        "errors":         [],
    }

    print(f"[drive] max_tokens={SMALL_MAX_TOKENS}, max_continuations="
          f"{CONFIG['max_continuations']}")
    print(f"[drive] target file: {PROBE}")
    print(f"[drive] prompt: {PROMPT[:100]}…\n")

    t0 = time.time()
    text_buf = []
    try:
        for ev in agent_run(PROMPT, state, CONFIG, system_prompt=_system_prompt()):
            if isinstance(ev, TextChunk):
                events["text_chunks"] += 1
                text_buf.append(ev.text)
                if "auto-continuing" in ev.text:
                    events["auto_banners"] += 1
                    # Flush any preceding model text on its own line
                    pre = "".join(text_buf[:-1]).strip()
                    if pre:
                        print(f"[drive] MODEL  → {pre[:300]!r}")
                    text_buf.clear()
                    print(f"[drive] BANNER → {ev.text.strip()}")
                if "[Failed" in ev.text or "[Retry" in ev.text:
                    events["errors"].append(ev.text.strip())
                    print(f"[drive] ERR    → {ev.text.strip()}")
            elif isinstance(ev, ToolStart):
                events["tool_start"].append((ev.name, dict(ev.inputs)))
                print(f"[drive] TOOL   → {ev.name}({_short_inputs(ev.inputs)})")
            elif isinstance(ev, ToolEnd):
                events["tool_end"].append((ev.name, ev.permitted, len(ev.result)))
                head = ev.result[:120].replace("\n", " ")
                print(f"[drive] RESULT ← {ev.name} permitted={ev.permitted} "
                      f"len={len(ev.result)}  {head}")
            elif isinstance(ev, TurnDone):
                events["turn_done"].append((ev.input_tokens, ev.output_tokens))
                print(f"[drive] TURN done  in={ev.input_tokens} "
                      f"out={ev.output_tokens}")
    except Exception as e:
        print(f"[drive] FATAL: {type(e).__name__}: {e}")
        events["errors"].append(f"{type(e).__name__}: {e}")

    elapsed = time.time() - t0
    print(f"\n[drive] elapsed: {elapsed:.1f}s")

    print("\n" + "─" * 60)
    print("RESULTS")
    print("─" * 60)
    print(f"turns:           {len(events['turn_done'])}")
    print(f"tools started:   {len(events['tool_start'])}  "
          f"({[n for n, _ in events['tool_start']]})")
    print(f"tools completed: {len(events['tool_end'])}")
    print(f"auto banners:    {events['auto_banners']}")
    print(f"errors:          {len(events['errors'])}")
    print(f"file exists:     {os.path.exists(PROBE)}")
    if os.path.exists(PROBE):
        size = os.path.getsize(PROBE)
        with open(PROBE) as f:
            content = f.read()
        lines = content.count("\n") + 1
        print(f"file size:       {size} bytes")
        print(f"file lines:      {lines}")
        print(f"first 200 chars: {content[:200]!r}")
        # Sanity: file should have at least a heading and not be raw JSON
        looks_ok = "#" in content[:500] and not content.startswith('{"')
        print(f"looks like md:   {looks_ok}")

    # Pass/fail criteria
    print("\n" + "─" * 60)
    print("VERDICT")
    print("─" * 60)
    fail = []
    if events["auto_banners"] == 0:
        fail.append("no auto-continue banner — truncation never triggered "
                    "(max_tokens too generous?) or detection broken")
    if not os.path.exists(PROBE):
        fail.append("file was never written")
    if events["errors"]:
        fail.append(f"errors during run: {events['errors']}")

    # Always dump a brief history transcript so we can see what the model did
    print("\n" + "─" * 60)
    print(f"HISTORY ({len(state.messages)} messages)")
    print("─" * 60)
    for i, m in enumerate(state.messages):
        role = m["role"]
        content = (m.get("content") or "")
        tcs = m.get("tool_calls") or []
        head = content.replace("\n", " ")[:140]
        if role == "tool":
            head = f"[{m.get('name')}] {head}"
        print(f"  [{i}] {role:<10} tool_calls={len(tcs)}  {head}")
        for tc in tcs:
            inp = tc.get("input") or {}
            if "_raw" in inp:
                raw = inp["_raw"]
                print(f"        ↳ MALFORMED {tc.get('name')}({len(raw)} bytes raw)")
            else:
                summ = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in inp.items())
                print(f"        ↳ {tc.get('name')}({summ[:120]})")

    if fail:
        print("FAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("PASS — truncation triggered, recovery banner fired, file written.")
    return 0


def _system_prompt() -> str:
    from context import build_system_prompt
    return build_system_prompt(CONFIG)


def _short_inputs(inp: dict) -> str:
    parts = []
    for k, v in inp.items():
        s = str(v)
        if len(s) > 60:
            s = s[:60] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
