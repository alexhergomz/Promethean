"""Anti-stuck detection for the main agent loop.

Generic counterpart to rabbit_hole.store.stuck_for() — that one is
research-specific (counts new sources/findings/closed questions); this
one is fully general: it watches the recent ring of tool invocations
and trips when the agent has called the same tool with identical
arguments too many times in a row. Used by agent.run() to inject a
single steering user message.

Threshold logic is intentionally conservative: weaker API models
(MiniMax-M2, Qwen3.5-9B) sometimes need 2-3 retries on the same Read
or Grep before they spot the relevant file — the threshold of 3
identical-arg invocations in a 6-call ring catches genuine loops
without firing on legitimate exploration.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable

_RING_SIZE = 6
_REPEAT_THRESHOLD = 3   # ≥3 identical (name+args) in the last 6 → stuck
_COOLDOWN_TURNS = 2     # don't re-fire within this many turns


def _tool_call_key(tc: dict) -> str:
    """Stable key for `(tool_name, normalized_args)`.

    Truncates serialized args to 2k chars to bound the hash cost on
    pathological inputs (e.g. an Edit with a 100k-char replacement).
    """
    name = tc.get("name", "")
    inp = tc.get("input", {})
    try:
        body = json.dumps(inp, sort_keys=True, default=str)[:2000]
    except Exception:
        body = repr(inp)[:2000]
    h = hashlib.sha1(body.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{name}::{h}"


def detect_stuck(state, new_tool_calls: Iterable[dict]) -> tuple[bool, str]:
    """Update the ring buffer with `new_tool_calls`, then check for stuckness.

    Args:
        state: AgentState — uses `_tool_ring` (created lazily).
                            uses `_anti_stuck_fired_turn` for cooldown.
        new_tool_calls: the tool_calls list from the latest assistant turn.

    Returns:
        (is_stuck, tool_name).
        is_stuck=False if not enough history, no repetition, or still in
        cooldown from a recent fire.
    """
    if not hasattr(state, "_tool_ring"):
        state._tool_ring = []
    for tc in new_tool_calls:
        state._tool_ring.append(_tool_call_key(tc))
    if len(state._tool_ring) > _RING_SIZE:
        state._tool_ring = state._tool_ring[-_RING_SIZE:]

    if len(state._tool_ring) < _REPEAT_THRESHOLD:
        return False, ""

    counts = Counter(state._tool_ring)
    top_key, top_n = counts.most_common(1)[0]
    if top_n < _REPEAT_THRESHOLD:
        return False, ""

    # Cooldown: if we just fired in the last _COOLDOWN_TURNS turns, hold off.
    last_fired = getattr(state, "_anti_stuck_fired_turn", -10_000)
    if state.turn_count - last_fired < _COOLDOWN_TURNS:
        return False, ""

    tool_name = top_key.split("::", 1)[0]
    return True, tool_name


def mark_fired(state) -> None:
    """Record that the nudge fired this turn (drives the cooldown)."""
    state._anti_stuck_fired_turn = state.turn_count


def steering_message(tool_name: str) -> str:
    """One-shot user-message nudge. The model can ignore it; it's a hint."""
    return (
        f"[anti-stuck] You've called `{tool_name}` with identical "
        f"arguments three times now. The result isn't changing. "
        f"Step back: what assumption might be wrong? Try a different "
        f"approach — a different tool, different arguments, or ask the "
        f"user for clarification before retrying."
    )
