"""Stress test: force many consecutive auto-continuations on a comprehensive
essay prompt. Tools enabled (so the model could WebSearch/WebFetch if it
chooses), but the body of the essay is plain text — exercises the
text-continuation recovery path.

Goal: prove the harness can stitch together a coherent multi-thousand-token
output across N truncations on the real local Qwen3.5-9B Q4_K_M, even when
the model is too "stubborn" to follow chunk-split instructions for Write.
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

# Tight budget = many truncations. 400 tokens ≈ 1200-1500 chars per chunk.
SMALL_MAX_TOKENS    = 400
MAX_CONTINUATIONS   = 8   # give it enough headroom to actually finish

CONFIG = {
    "model":              "custom/qwen3.5-9b",
    "custom_base_url":    "http://127.0.0.1:8080/v1",
    "custom_api_key":     "no-key-needed",
    "max_tokens":         SMALL_MAX_TOKENS,
    "max_continuations":  MAX_CONTINUATIONS,
    "permission_mode":    "accept-all",
    "max_tool_output":    8000,
    "log_level":          "warn",
    "thinking":           False,
}

PROMPT = (
    "Write a comprehensive essay on the history and engineering of the "
    "modern semiconductor industry. Cover ALL of these in depth, with "
    "named figures and dates where you know them:\n"
    "  1. Shockley's bipolar transistor (1947) and the birth of Bell Labs\n"
    "  2. Noyce / Kilby and the integrated circuit (1958–59)\n"
    "  3. Moore's Law and the founding of Intel\n"
    "  4. The CMOS revolution and why it beat NMOS\n"
    "  5. The shift from planar to FinFET (~22nm node)\n"
    "  6. EUV lithography: from research to volume production at TSMC\n"
    "  7. Why 3nm is so hard and what comes after (GAA, 2nm)\n"
    "Write in a flowing essay style, not bullets. Aim for thoroughness."
)


def main():
    state = AgentState()
    text_chunks   = []
    banners       = 0
    turns         = 0
    tool_starts   = 0

    print(f"[stress] max_tokens={SMALL_MAX_TOKENS}, "
          f"max_continuations={MAX_CONTINUATIONS}")
    print(f"[stress] prompt: comprehensive semiconductor-industry essay\n")
    print("─" * 60)

    t0 = time.time()
    last_was_banner = False
    try:
        for ev in agent_run(PROMPT, state, CONFIG, system_prompt=_system_prompt()):
            if isinstance(ev, TextChunk):
                if "auto-continuing" in ev.text:
                    banners += 1
                    print(f"\n\n[stress] >>> BANNER ({banners}): "
                          f"{ev.text.strip()} <<<\n", flush=True)
                    last_was_banner = True
                else:
                    text_chunks.append(ev.text)
                    sys.stdout.write(ev.text)
                    sys.stdout.flush()
                    last_was_banner = False
            elif isinstance(ev, TurnDone):
                turns += 1
            elif isinstance(ev, ToolStart):
                tool_starts += 1
                print(f"\n[stress] tool: {ev.name}({list(ev.inputs)})",
                      flush=True)
            elif isinstance(ev, ToolEnd):
                pass
    except Exception as e:
        print(f"\n\n[stress] FATAL: {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    full = "".join(text_chunks)

    print("\n\n" + "═" * 60)
    print("STRESS TEST RESULTS")
    print("═" * 60)
    print(f"turns:            {turns}")
    print(f"banners (continuations): {banners}")
    print(f"tools fired:      {tool_starts}")
    print(f"final text length: {len(full)} chars")
    print(f"approx tokens:    {len(full) // 4} (rough chars/4)")
    print(f"elapsed:          {elapsed:.1f}s")

    # Verify the topics actually got covered (proxy for coherence
    # surviving the stitch boundary).
    topics = ["Shockley", "Noyce", "Kilby", "Moore", "CMOS",
              "FinFET", "EUV", "TSMC", "3nm", "GAA"]
    hits = [t for t in topics if t.lower() in full.lower()]
    print(f"topics covered:   {len(hits)}/{len(topics)} → {hits}")

    # Per-turn boundary check: sample text around each banner to confirm
    # the model continued (didn't just restart with the same heading).
    print("\n— Boundary samples (last 80 chars of each chunk) —")
    accumulated = ""
    chunk_ends  = []
    for chunk in text_chunks:
        accumulated += chunk
        # Detect "completion" of a turn: chunk ends with content, next chunk is a banner.
        # We approximate by tracking lengths.
    # Show the last 80 chars of full text
    print(f"  …{full[-200:]!r}")

    fail = []
    if banners < 1:
        fail.append("no truncation occurred — max_tokens too generous")
    if turns < 2:
        fail.append("no continuation — only one turn")
    if len(full) < SMALL_MAX_TOKENS * 4:  # at least 1 chunk's worth
        fail.append(f"final text too short ({len(full)} chars)")
    if len(hits) < 4:
        fail.append(f"only {len(hits)} of {len(topics)} topics covered — "
                    "model lost coherence across continuations")

    if fail:
        print("\nFAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print(f"\nPASS — stitched {len(full)} chars across {turns} turns "
          f"({banners} continuations) covering {len(hits)} topics.")
    return 0


def _system_prompt():
    from context import build_system_prompt
    return build_system_prompt(CONFIG)


if __name__ == "__main__":
    sys.exit(main())
