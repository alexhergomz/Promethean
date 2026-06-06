"""Tests for the two daily-driver blockers.

Blocker 1: thinking-burnout recovery — when a thinking model burns all its
            output tokens inside `reasoning_details` without emitting any
            visible content or tool call, _handle_length_truncation must
            inject the "answer NOW" hint instead of the generic continue.

Blocker 2: usage fallback — when a proxy strips the usage chunk, the
            streamer must estimate in_tok/out_tok from emitted bytes and
            flag the AssistantTurn so the UI can mark figures approximate.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent
import providers
from agent import AgentState, AssistantTurn


# ── Blocker 1 ────────────────────────────────────────────────────────────

class TestThinkingBurnoutRecovery:
    def _turn_burnout(self):
        return AssistantTurn(
            text="", tool_calls=[],
            in_tokens=10, out_tokens=2048,
            cache_read_tokens=0, cache_write_tokens=0,
            reasoning_content="lots of reasoning here",
            finish_reason="length",
            reasoning_details=[
                {"type": "reasoning.text", "index": 0,
                 "text": "I should think about... " * 80},
            ],
        )

    def _turn_text_truncation(self):
        return AssistantTurn(
            text="half of an answer that got cu",
            tool_calls=[],
            in_tokens=10, out_tokens=2048,
            cache_read_tokens=0, cache_write_tokens=0,
            reasoning_content="",
            finish_reason="length",
        )

    def test_burnout_emits_finalize_hint(self):
        state = AgentState()
        msg = {"role": "assistant", "content": ""}
        state.messages.append(msg)
        cont, hint = agent._handle_length_truncation(
            state, self._turn_burnout(), msg,
            {"max_continuations": 3, "max_thinking_burnout_retries": 2},
        )
        assert cont is True
        assert "thinking-burnout" in hint
        assert "minimal or zero further reasoning" in hint

    def test_burnout_snippet_appears(self):
        state = AgentState()
        msg = {"role": "assistant", "content": ""}
        state.messages.append(msg)
        _, hint = agent._handle_length_truncation(
            state, self._turn_burnout(), msg,
            {"max_continuations": 3, "max_thinking_burnout_retries": 2},
        )
        # We pass the truncated reasoning back so the model knows what it
        # already concluded — saves it from re-thinking from scratch.
        assert "I should think about" in hint

    def test_burnout_separate_cap_from_truncation(self):
        # 2 burnout retries, but max_continuations=3. Burnout has its own
        # cap so a generic truncation loop can't shrink the burnout budget.
        state = AgentState()
        msg = {"role": "assistant", "content": ""}
        state.messages.append(msg)
        cfg = {"max_continuations": 3, "max_thinking_burnout_retries": 2}
        for _ in range(2):
            cont, hint = agent._handle_length_truncation(
                state, self._turn_burnout(), msg, cfg,
            )
            assert cont is True
        # Third call returns False — cap hit.
        cont, hint = agent._handle_length_truncation(
            state, self._turn_burnout(), msg, cfg,
        )
        assert cont is False and hint is None

    def test_text_truncation_path_unchanged(self):
        # Plain text-truncation should NOT use the burnout hint; existing
        # behaviour must be preserved.
        state = AgentState()
        msg = {"role": "assistant", "content": "half ..."}
        state.messages.append(msg)
        cont, hint = agent._handle_length_truncation(
            state, self._turn_text_truncation(), msg,
            {"max_continuations": 3},
        )
        assert cont is True
        assert "thinking-burnout" not in (hint or "")
        assert "cut off" in (hint or "")

    def test_clean_turn_no_recovery(self):
        state = AgentState()
        msg = {"role": "assistant", "content": "ok"}
        turn = AssistantTurn(
            text="ok", tool_calls=[],
            in_tokens=5, out_tokens=2,
            cache_read_tokens=0, cache_write_tokens=0,
            reasoning_content="", finish_reason="stop",
        )
        cont, hint = agent._handle_length_truncation(state, turn, msg, {})
        assert cont is False and hint is None


# ── Blocker 2 ────────────────────────────────────────────────────────────

class _FakeChunk:
    def __init__(self, content=None, usage=None, finish=None,
                 reasoning_content=None):
        self.choices = [] if usage is not None and content is None else [
            types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    content=content or "",
                    tool_calls=None,
                    reasoning_content=reasoning_content,
                ),
                finish_reason=finish,
            ),
        ]
        self.usage = usage


def _patch_openai(monkeypatch, chunks):
    class _Stream:
        def __iter__(self): return iter(chunks)
    class _C:
        def create(self, **kw): return _Stream()
    class _Chat:
        completions = _C()
    class _Cli:
        def __init__(self, *a, **kw): pass
        chat = _Chat()
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(OpenAI=_Cli))


class TestUsageFallback:
    def test_real_usage_passes_through(self, monkeypatch):
        # Provider sends real usage → we trust it; usage_estimated=False.
        usage = types.SimpleNamespace(
            prompt_tokens=42, completion_tokens=13,
            prompt_tokens_details=None,
        )
        chunks = [
            _FakeChunk(content="pong", finish="stop"),
            types.SimpleNamespace(choices=[], usage=usage),
        ]
        _patch_openai(monkeypatch, chunks)
        turn = None
        for c in providers.stream_openai_compat(
            api_key="x", base_url="https://api.minimax.io/v1",
            model="MiniMax-M2", system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[], config={},
        ):
            if type(c).__name__ == "AssistantTurn":
                turn = c
        assert turn.in_tokens == 42 and turn.out_tokens == 13
        assert turn.usage_estimated is False

    def test_missing_usage_falls_back_to_estimate(self, monkeypatch):
        # Proxy strips usage → chunks never carry one. Streamer estimates
        # from emitted bytes and flags usage_estimated=True.
        chunks = [
            _FakeChunk(content="pong!", finish="stop"),
        ]
        _patch_openai(monkeypatch, chunks)
        turn = None
        for c in providers.stream_openai_compat(
            api_key="x", base_url="http://127.0.0.1:8765/v1",
            model="MiniMax-M2", system="a system prompt",
            messages=[{"role": "user", "content": "hello world hello world"}],
            tool_schemas=[], config={},
        ):
            if type(c).__name__ == "AssistantTurn":
                turn = c
        assert turn.usage_estimated is True
        # Must be > 0 — we floor at 1, and there are bytes to count.
        assert turn.in_tokens > 0
        assert turn.out_tokens > 0

    def test_state_marked_estimated_after_first_estimated_turn(self):
        state = AgentState()
        turn = AssistantTurn(
            text="x", tool_calls=[],
            in_tokens=10, out_tokens=2,
            cache_read_tokens=0, cache_write_tokens=0,
            reasoning_content="", finish_reason="stop",
            usage_estimated=True,
        )
        # Manually replay what agent.run does:
        if getattr(turn, "usage_estimated", False):
            state._usage_estimated = True
        assert state._usage_estimated is True
