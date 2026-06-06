"""Live integration test: text-only continuation path.

Asks the model a question that warrants a long pure-text answer (no tools)
with max_tokens forced low. The harness should detect length truncation,
inject a 'continue from where you stopped' hint, and stitch the response
across multiple turns.

This complements live_truncation_drive.py: that one tests the malformed
tool-call recovery (which depends on the model heeding a chunk-split
instruction); this one tests the text-continuation path (which works on
any model that can simply keep talking).

Requires the server at http://127.0.0.1:8080 to be up.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import bootstrap as _bs  # noqa: F401
from agent import AgentState, run as agent_run, ToolStart, ToolEnd, TurnDone
from providers import TextChunk, ThinkingChunk

# Tiny budget to force at least one length truncation on a chatty answer
SMALL_MAX_TOKENS = 250

CONFIG = {
    "model":              "custom/qwen3.5-9b",
    "custom_base_url":    "http://127.0.0.1:8080/v1",
    "custom_api_key":     "no-key-needed",
    "max_tokens":         SMALL_MAX_TOKENS,
    "max_continuations":  3,
    "permission_mode":    "accept-all",
    "log_level":          "warn",
    "thinking":           False,
    "no_tools":           True,   # plain text only
}

PROMPT = (
    "Count from 1 to 150, one number per line, in this exact format:\n"
    "1\n2\n3\n... (continue) ...\n150\n"
    "Do not number with markdown, do not skip numbers, do not stop early. "
    "Just emit the integers. After 150 say DONE."
)


def main():
    state = AgentState()
    text_total = []
    banners    = 0
    turns      = 0

    print(f"[txt-drive] max_tokens={SMALL_MAX_TOKENS}, "
          f"max_continuations={CONFIG['max_continuations']}")
    print(f"[txt-drive] prompt: {PROMPT[:100]}…\n")

    t0 = time.time()
    try:
        for ev in agent_run(PROMPT, state, CONFIG, system_prompt=_system_prompt()):
            if isinstance(ev, TextChunk):
                if "auto-continuing" in ev.text:
                    banners += 1
                    print(f"\n[txt-drive] BANNER → {ev.text.strip()}\n",
                          flush=True)
                else:
                    text_total.append(ev.text)
                    sys.stdout.write(ev.text)
                    sys.stdout.flush()
            elif isinstance(ev, TurnDone):
                turns += 1
            elif isinstance(ev, (ToolStart, ToolEnd)):
                # Should never fire — no_tools=True
                print(f"\n[txt-drive] UNEXPECTED tool event: {ev}")
    except Exception as e:
        print(f"\n[txt-drive] FATAL: {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    full = "".join(text_total)
    print("\n\n" + "─" * 60)
    print(f"turns:        {turns}")
    print(f"banners:      {banners}")
    print(f"text length:  {len(full)} chars / {full.count(chr(10))} newlines")
    print(f"elapsed:      {elapsed:.1f}s")

    # Pass criteria for the text-continuation path:
    #   - banner fired at least once (truncation actually happened)
    #   - the stitched text continued past the first truncation
    #   - final text is meaningfully longer than a single max_tokens chunk
    fail = []
    if banners == 0:
        fail.append("no banner — truncation never triggered or detection broken")
    if turns < 2:
        fail.append(f"only {turns} turn(s) — auto-continue did not loop")

    # Did the model actually count past the truncation point?
    # Per-turn the chunk is ~70 numbers (250-token cap). If recovery worked,
    # the highest integer seen should be well above 70. (Note: the flat
    # text_total can have digits from adjacent lines fused at chunk
    # boundaries, e.g. "74" + "75" → "7475". We tolerate that — any int
    # above the per-chunk ceiling proves continuation.)
    nums = []
    for line in full.split("\n"):
        s = line.strip()
        if s.isdigit() and 1 <= int(s) <= 200:
            nums.append(int(s))
    highest = max(nums) if nums else 0
    print(f"highest int (in valid range): {highest}")
    # The first chunk maxes out around 70-75 with max_tokens=250. Anything
    # past 80 proves the second turn ran AND the model continued the
    # sequence rather than restarting from 1.
    if highest < 80:
        fail.append(f"model never counted past {highest} — continuation "
                    "didn't actually progress the sequence")

    if fail:
        print("\nFAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nPASS — text truncation auto-continued and stitched cleanly.")
    return 0


def _system_prompt():
    from context import build_system_prompt
    return build_system_prompt(CONFIG)


if __name__ == "__main__":
    sys.exit(main())
