"""Cross-provider failover ladder tests.

Verifies that on terminal API failure, the agent loop advances to the next
model in config["failover_models"] and retries, rather than giving up. The
switch sticks for the rest of the session (config["model"] mutated in place).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent
from agent import AgentState, TextChunk, AssistantTurn


def _make_stream_fn(plan):
    """Return a stream() stand-in driven by `plan`.

    Each entry of `plan` is keyed by model and contains a list of attempt
    behaviors: 'fail' (raises) or AssistantTurn(...) (yields it and stops).
    Calls consume entries in order per model. Raises StopIteration-style
    RuntimeError if a model is asked beyond its plan.
    """
    state = {m: 0 for m in plan}

    def _stream(*, model, system, messages, tool_schemas, config):
        idx = state[model]
        state[model] += 1
        items = plan[model]
        if idx >= len(items):
            raise RuntimeError(f"plan exhausted for {model}")
        step = items[idx]
        if step == "fail":
            raise Exception("simulated rate_limit: 429 too many requests")
        # Else assume it's an AssistantTurn
        yield step
    return _stream, state


def _bare_config(model="primary-model"):
    return {
        "model":       model,
        "_session_id": "test",
        # Stop the loop after one assistant text turn — no tools.
        "no_tools":    True,
    }


def _make_turn(text="done"):
    return AssistantTurn(
        text=text, tool_calls=[], in_tokens=10, out_tokens=2,
        cache_read_tokens=0, cache_write_tokens=0,
        reasoning_content="", finish_reason="stop",
    )


class TestFailoverLadder:
    def test_no_failover_models_old_behavior(self, monkeypatch):
        """With no failover_models, agent gives up after retries (existing behavior)."""
        plan = {"primary-model": ["fail", "fail", "fail", "fail"]}
        fn, calls = _make_stream_fn(plan)
        monkeypatch.setattr(agent, "stream", fn)
        monkeypatch.setattr(agent, "maybe_compact", lambda *a, **kw: False)
        monkeypatch.setattr(agent, "sanitize_history", lambda m: m)
        monkeypatch.setattr(agent.time, "sleep", lambda *a, **kw: None)

        config = _bare_config()
        events = list(agent.run("hi", AgentState(), config, "sys"))
        # The give-up emits "[Failed — ..." text.
        assert any("Failed —" in getattr(e, "text", "") for e in events)
        # Primary was attempted exactly max_retries+1 times.
        assert calls["primary-model"] == 4

    def test_failover_advances_to_fallback(self, monkeypatch):
        """Primary fails terminally, fallback succeeds within the turn cycle.

        (agent.run shallow-copies config on entry, so the switch is sticky
        within this run() invocation but not visible to the caller's config.
        That's the intended scope — each new run() retries the primary.)
        """
        plan = {
            "primary-model":  ["fail", "fail", "fail", "fail"],
            "fallback-model": [_make_turn("from fallback")],
        }
        fn, calls = _make_stream_fn(plan)
        monkeypatch.setattr(agent, "stream", fn)
        monkeypatch.setattr(agent, "maybe_compact", lambda *a, **kw: False)
        monkeypatch.setattr(agent, "sanitize_history", lambda m: m)
        monkeypatch.setattr(agent.time, "sleep", lambda *a, **kw: None)

        config = _bare_config()
        config["failover_models"] = ["fallback-model"]
        events = list(agent.run("hi", AgentState(), config, "sys"))

        # Failover yield surfaced to the UI.
        assert any("Failover" in getattr(e, "text", "") for e in events)
        # Primary was hit 4× before giving up; fallback succeeded first try.
        assert calls["primary-model"] == 4
        assert calls["fallback-model"] == 1
        # No "Failed —" giving-up message.
        assert not any("Failed —" in getattr(e, "text", "") for e in events)

    def test_failover_skips_duplicate_in_ladder(self, monkeypatch):
        """Duplicate of primary in failover_models is filtered."""
        plan = {"primary-model": [_make_turn("ok")]}
        fn, _ = _make_stream_fn(plan)
        monkeypatch.setattr(agent, "stream", fn)
        monkeypatch.setattr(agent, "maybe_compact", lambda *a, **kw: False)
        monkeypatch.setattr(agent, "sanitize_history", lambda m: m)

        config = _bare_config()
        # Same model listed in failover — should be ignored.
        config["failover_models"] = ["primary-model", "primary-model"]
        list(agent.run("hi", AgentState(), config, "sys"))
        # Nothing crashed, model unchanged.
        assert config["model"] == "primary-model"

    def test_two_step_failover(self, monkeypatch):
        """Primary fails, fallback-A fails, fallback-B succeeds."""
        plan = {
            "primary-model":  ["fail"] * 4,
            "fallback-a":     ["fail"] * 4,
            "fallback-b":     [_make_turn("rescued")],
        }
        fn, calls = _make_stream_fn(plan)
        monkeypatch.setattr(agent, "stream", fn)
        monkeypatch.setattr(agent, "maybe_compact", lambda *a, **kw: False)
        monkeypatch.setattr(agent, "sanitize_history", lambda m: m)
        monkeypatch.setattr(agent.time, "sleep", lambda *a, **kw: None)

        config = _bare_config()
        config["failover_models"] = ["fallback-a", "fallback-b"]
        events = list(agent.run("hi", AgentState(), config, "sys"))

        assert calls["primary-model"] == 4
        assert calls["fallback-a"] == 4
        assert calls["fallback-b"] == 1
        # Two failover events surfaced.
        failovers = [e for e in events
                     if "Failover" in getattr(e, "text", "")]
        assert len(failovers) == 2
        # And we did NOT bottom out.
        assert not any("Failed —" in getattr(e, "text", "") for e in events)
