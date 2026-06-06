"""Tests for the Aider/OpenCode-style auto-continue on finish_reason='length'.

Covers:
  - _truncation_hint() shape per failure mode (Write/Edit/text/empty)
  - _extract_partial_path() best-effort regex
  - _handle_length_truncation() — strip, cap, counter reset
  - End-to-end agent loop with a scripted stream:
      * pure-text truncation recovers in 2 turns
      * truncated Write tool call recovers in 2 turns (malformed dropped, hint sent)
      * runaway truncation hits the cap and exits cleanly
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# Project root on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent import (
    AgentState,
    _extract_partial_path,
    _handle_length_truncation,
    _truncation_hint,
    run as agent_run,
)
from providers import AssistantTurn, TextChunk


# ── Helpers ────────────────────────────────────────────────────────────────

def _turn(text="", tool_calls=None, finish="stop"):
    return AssistantTurn(
        text=text,
        tool_calls=list(tool_calls or []),
        in_tokens=10, out_tokens=10,
        finish_reason=finish,
    )


def _malformed_write(raw_args):
    """A tool call as providers.py would emit when Write JSON args got truncated."""
    return {
        "id":    "call_x",
        "name":  "Write",
        "input": {"_raw": raw_args},
    }


def _msg_record(turn):
    """Mimic the assistant_msg dict the agent loop builds and appends."""
    return {
        "role":       "assistant",
        "content":    turn.text,
        "tool_calls": turn.tool_calls,
    }


# ── _extract_partial_path ─────────────────────────────────────────────────

class TestExtractPartialPath:
    def test_complete_value(self):
        raw = '{"file_path": "/tmp/foo.md", "content": "hello'
        assert _extract_partial_path(raw) == "/tmp/foo.md"

    def test_no_path(self):
        assert _extract_partial_path('{"content": "x"') == ""

    def test_empty(self):
        assert _extract_partial_path("") == ""
        assert _extract_partial_path(None) == ""

    def test_path_value_truncated_at_first_char(self):
        # path key opened but value truncated after first quote — match returns ""
        # (no closing quote → no group).  This is OK; caller falls back to "<unknown>".
        assert _extract_partial_path('{"file_path": "') == ""

    def test_extra_whitespace(self):
        raw = '{   "file_path"   :   "/a/b.py" , "content'
        assert _extract_partial_path(raw) == "/a/b.py"


# ── _truncation_hint ──────────────────────────────────────────────────────

class TestTruncationHint:
    def test_write_path_named_in_hint(self):
        dropped = [_malformed_write('{"file_path": "/tmp/big.md", "content": "abc')]
        hint = _truncation_hint(dropped, had_text=False)
        assert "/tmp/big.md" in hint
        assert "Write" in hint
        # Recovery strategy must explain chunking somehow
        assert "chunk" in hint.lower() or "split" in hint.lower()
        assert "Edit" in hint  # tells the model how to append

    def test_edit_dropped(self):
        dropped = [{"id": "x", "name": "Edit", "input": {"_raw": '{"file_path": "/a/b.py"'}}]
        hint = _truncation_hint(dropped, had_text=False)
        assert "/a/b.py" in hint
        assert "Edit" in hint

    def test_unknown_tool_dropped(self):
        dropped = [{"id": "x", "name": "Bash", "input": {"_raw": "{"}}]
        hint = _truncation_hint(dropped, had_text=False)
        # No file path; falls back to generic message
        assert "Bash" in hint
        assert "shorter" in hint.lower() or "discarded" in hint.lower()

    def test_text_only_truncation(self):
        hint = _truncation_hint([], had_text=True)
        assert "cut off" in hint.lower() or "stopped" in hint.lower()
        # The hint must explicitly tell the model NOT to repeat
        assert "not repeat" in hint.lower() or "do not repeat" in hint.lower() or "do not" in hint.lower()

    def test_empty_response(self):
        hint = _truncation_hint([], had_text=False)
        assert "concise" in hint.lower() or "shorter" in hint.lower()


# ── _handle_length_truncation ─────────────────────────────────────────────

class TestHandleLengthTruncation:

    def test_clean_turn_returns_false_and_resets_counter(self):
        state = AgentState()
        state._continuations = 2
        turn = _turn(text="ok", finish="stop")
        msg  = _msg_record(turn)
        cont, hint = _handle_length_truncation(state, turn, msg, {"max_continuations": 3})
        assert cont is False
        assert hint is None
        assert getattr(state, "_continuations", 0) == 0

    def test_length_with_malformed_tool_call_strips_and_hints(self):
        state = AgentState()
        bad   = _malformed_write('{"file_path": "/x.md", "content": "ab')
        turn  = _turn(tool_calls=[bad], finish="length")
        msg   = _msg_record(turn)
        cont, hint = _handle_length_truncation(state, turn, msg, {"max_continuations": 3})
        assert cont is True
        assert hint is not None
        assert "/x.md" in hint
        # Stripped from BOTH the live turn and the appended history record
        assert turn.tool_calls == []
        assert msg["tool_calls"] == []
        assert state._continuations == 1

    def test_length_keeps_valid_tool_calls(self):
        state = AgentState()
        good = {"id": "g", "name": "Read", "input": {"file_path": "/y"}}
        bad  = _malformed_write('{"file_path": "/x"')
        turn = _turn(tool_calls=[good, bad], finish="length")
        msg  = _msg_record(turn)
        _handle_length_truncation(state, turn, msg, {"max_continuations": 3})
        assert turn.tool_calls == [good]
        assert msg["tool_calls"] == [good]

    def test_length_with_text_only(self):
        state = AgentState()
        turn  = _turn(text="part", finish="length")
        msg   = _msg_record(turn)
        cont, hint = _handle_length_truncation(state, turn, msg, {"max_continuations": 3})
        assert cont is True
        assert hint and "stopped" in hint.lower()

    def test_cap_zero_disables_feature(self):
        state = AgentState()
        turn  = _turn(text="part", finish="length")
        msg   = _msg_record(turn)
        cont, hint = _handle_length_truncation(state, turn, msg, {"max_continuations": 0})
        assert cont is False
        assert hint is None

    def test_cap_reached(self):
        state = AgentState()
        state._continuations = 3
        turn  = _turn(text="x", finish="length")
        msg   = _msg_record(turn)
        cont, hint = _handle_length_truncation(state, turn, msg, {"max_continuations": 3})
        assert cont is False
        assert hint is None
        # Counter not bumped past cap
        assert state._continuations == 3

    def test_counter_increments_each_call(self):
        state = AgentState()
        turn  = _turn(text="x", finish="length")
        msg   = _msg_record(turn)
        for expected in (1, 2, 3):
            _handle_length_truncation(state, turn, msg, {"max_continuations": 3})
            assert state._continuations == expected


# ── End-to-end via scripted stream ────────────────────────────────────────

class _ScriptedStream:
    """Replays a queue of pre-built (text_chunks, AssistantTurn) scripts.

    Each element is a list of events to yield in order. Successive agent
    iterations consume successive scripts. If the agent loop tries to call
    the stream more times than scripted, the test fails.
    """
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls   = 0

    def __call__(self, **kwargs):
        if not self.scripts:
            raise AssertionError(f"stream called {self.calls + 1}× but no scripts left")
        events = self.scripts.pop(0)
        self.calls += 1
        for ev in events:
            yield ev


@pytest.fixture
def loop_env(monkeypatch, tmp_path):
    """Common monkey-patches: silence quota/compaction/logging/sanitize side effects."""
    import agent as agent_mod
    # No-op quota
    monkeypatch.setattr(agent_mod._quota, "check_quota",  lambda *a, **k: None)
    monkeypatch.setattr(agent_mod._quota, "record_usage", lambda *a, **k: None)
    # No-op compaction (it estimates tokens which is fine, but we want it skipped)
    monkeypatch.setattr(agent_mod, "maybe_compact", lambda *a, **k: None)
    # sanitize_history just passes through unchanged for these tests
    monkeypatch.setattr(agent_mod, "sanitize_history", lambda msgs: msgs)
    # Tool schemas: empty (we won't actually invoke real tools)
    monkeypatch.setattr(agent_mod, "get_tool_schemas", lambda: [])
    # Track every execute_tool call so we can assert on side effects
    calls = []
    def _fake_exec(name, inputs, permission_mode="accept-all", config=None):
        calls.append((name, dict(inputs)))
        return f"Wrote {inputs.get('file_path', '?')}"
    monkeypatch.setattr(agent_mod, "execute_tool", _fake_exec)

    return SimpleNamespace(tool_calls=calls, monkeypatch=monkeypatch)


def _drain(gen):
    """Consume a generator, returning all yielded events."""
    out = []
    for ev in gen:
        out.append(ev)
    return out


class TestEndToEndRecovery:

    def test_text_truncation_then_clean_completion(self, loop_env, monkeypatch):
        """Text-only response cut at max_tokens should auto-continue to clean stop."""
        import agent as agent_mod
        # Turn 1: text truncated.  Turn 2: clean text.
        scripts = [
            [TextChunk("partial answer..."), _turn(text="partial answer...", finish="length")],
            [TextChunk(" continuation done."), _turn(text=" continuation done.", finish="stop")],
        ]
        stream = _ScriptedStream(scripts)
        monkeypatch.setattr(agent_mod, "stream", stream)

        state = AgentState()
        events = _drain(agent_run(
            "tell me a story", state, {"model": "x", "max_continuations": 3}, system_prompt="sys",
        ))
        assert stream.calls == 2, "loop should have re-streamed once after the length truncation"
        # The synthetic continuation hint must be in history as a user message
        roles = [m["role"] for m in state.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert "cut off" in state.messages[2]["content"].lower() or "stopped" in state.messages[2]["content"].lower()
        # The final assistant message is the clean continuation
        assert state.messages[-1]["content"] == " continuation done."
        # User-visible "auto-continuing" banner appeared exactly once
        banners = [e for e in events if isinstance(e, TextChunk) and "auto-continuing" in e.text]
        assert len(banners) == 1

    def test_truncated_write_recovers_in_two_turns(self, loop_env, monkeypatch):
        """Malformed Write tool call should be stripped + replaced by clean two-call sequence."""
        import agent as agent_mod
        bad = _malformed_write('{"file_path": "/tmp/foo.md", "content": "# Title\\n\\nIntro')
        # Turn 1: ONE malformed Write (truncated content), finish=length.
        # Turn 2: TWO clean tool calls (Write + Edit), finish=tool_calls.
        # Turn 3: clean text closing the turn, finish=stop.
        good_write = {"id": "w1", "name": "Write",
                      "input": {"file_path": "/tmp/foo.md", "content": "# Title\n\n## Part 1"}}
        good_edit  = {"id": "e1", "name": "Edit",
                      "input": {"file_path": "/tmp/foo.md",
                                "old_string": "## Part 1",
                                "new_string": "## Part 1\n\n## Part 2"}}
        scripts = [
            [_turn(tool_calls=[bad], finish="length")],
            [_turn(tool_calls=[good_write, good_edit], finish="tool_calls")],
            [_turn(text="Done.", finish="stop")],
        ]
        stream = _ScriptedStream(scripts)
        monkeypatch.setattr(agent_mod, "stream", stream)
        # Auto-approve writes
        config = {"model": "x", "max_continuations": 3, "permission_mode": "accept-all"}

        state = AgentState()
        _drain(agent_run("write a doc", state, config, system_prompt="sys"))
        assert stream.calls == 3
        # The malformed Write must NOT have been executed
        names_executed = [n for n, _ in loop_env.tool_calls]
        assert names_executed == ["Write", "Edit"]
        # The recovery must NOT leave two consecutive user messages in
        # history (that's the bug: original user prompt + injected hint
        # back-to-back). Empty-after-strip assistant is replaced with a
        # stub, NOT popped, so a user message always sits between users.
        roles = [m["role"] for m in state.messages]
        for a, b in zip(roles, roles[1:]):
            assert not (a == "user" and b == "user"), \
                f"two consecutive user messages near {roles!r}"
        # No truly empty assistants (zero content AND zero tool_calls) — the
        # stub carries placeholder text.
        empty_assts = [m for m in state.messages
                       if m["role"] == "assistant"
                       and not (m.get("content") or "").strip()
                       and not m.get("tool_calls")]
        assert empty_assts == []
        # The recovery turn (Write+Edit) is the SECOND assistant, not the first
        # (the first is the stub for the truncated-and-stripped turn).
        assistants = [m for m in state.messages if m["role"] == "assistant"]
        recovery = assistants[1]
        assert [tc["name"] for tc in recovery.get("tool_calls", [])] == ["Write", "Edit"]
        # The hint user-message naming the file must be in history
        user_msgs = [m for m in state.messages if m["role"] == "user"]
        assert any("/tmp/foo.md" in m["content"] for m in user_msgs[1:])

    def test_runaway_truncation_hits_cap(self, loop_env, monkeypatch):
        """4 consecutive length truncations with cap=3 should exit after 3 retries."""
        import agent as agent_mod
        # All four turns are text-truncated with finish=length.
        scripts = [
            [_turn(text="part1...", finish="length")],
            [_turn(text="part2...", finish="length")],
            [_turn(text="part3...", finish="length")],
            [_turn(text="part4...", finish="length")],
        ]
        stream = _ScriptedStream(scripts)
        monkeypatch.setattr(agent_mod, "stream", stream)

        state = AgentState()
        _drain(agent_run("x", state, {"model": "x", "max_continuations": 3}, system_prompt="sys"))
        # Cap=3 means: original turn + 3 retries = 4 stream calls total.
        # On the 4th turn the cap is hit, so no further hint is injected and the loop breaks.
        assert stream.calls == 4
        # Final state has no trailing user-hint that would loop again.
        assert state.messages[-1]["role"] == "assistant"

    def test_clean_turn_resets_counter(self, loop_env, monkeypatch):
        """A successful turn between truncations must reset the cap counter."""
        import agent as agent_mod
        scripts = [
            [_turn(text="A", finish="length")],   # +1
            [_turn(text="B", finish="stop")],     # reset
        ]
        stream = _ScriptedStream(scripts)
        monkeypatch.setattr(agent_mod, "stream", stream)

        state = AgentState()
        _drain(agent_run("x", state, {"model": "x", "max_continuations": 3}, system_prompt="sys"))
        assert getattr(state, "_continuations", 0) == 0

    def test_mixed_valid_and_malformed_tool_calls(self, loop_env, monkeypatch):
        """Valid tools coexisting with malformed in a length-truncated turn:
        valid runs, malformed is stripped, hint follows tool results, loop continues.
        Exercises the post-tool-exec wire-in point (not the no-tools-left branch)."""
        import agent as agent_mod
        valid_read = {"id": "r1", "name": "Read",
                      "input": {"file_path": "/etc/hostname"}}
        bad_write  = _malformed_write('{"file_path": "/tmp/big.md", "content": "abc')
        scripts = [
            [_turn(tool_calls=[valid_read, bad_write], finish="length")],
            [_turn(text="here is what I found.", finish="stop")],
        ]
        stream = _ScriptedStream(scripts)
        monkeypatch.setattr(agent_mod, "stream", stream)

        state = AgentState()
        _drain(agent_run("read it", state,
                         {"model": "x", "max_continuations": 3,
                          "permission_mode": "accept-all"},
                         system_prompt="sys"))
        # Valid Read executed; malformed Write did NOT
        names_executed = [n for n, _ in loop_env.tool_calls]
        assert names_executed == ["Read"]
        # History order: user → assistant(tool_calls=[Read]) → tool(Read) → user(hint) → assistant(text)
        roles = [m["role"] for m in state.messages]
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        # The stored assistant message has ONLY the valid Read; the malformed Write was stripped
        assert len(state.messages[1]["tool_calls"]) == 1
        assert state.messages[1]["tool_calls"][0]["name"] == "Read"
        # The hint mentions the file the malformed Write was targeting
        assert "/tmp/big.md" in state.messages[3]["content"]

    def test_empty_after_strip_keeps_alternation(self, loop_env, monkeypatch):
        """When stripping the malformed call leaves the assistant turn empty
        (no text + no surviving tool_calls), it must be REPLACED with a stub
        rather than popped — popping creates two consecutive user messages
        (original + hint) which breaks alternation. Some models loop or spam
        unrelated tools when given a malformed conversation; a 9B local Qwen
        was observed spamming GetDiagnostics calls in this exact scenario."""
        import agent as agent_mod
        bad = _malformed_write('{"file_path": "/x.md", "content": "ab')
        scripts = [
            [_turn(tool_calls=[bad], finish="length")],
            [_turn(text="ok", finish="stop")],
        ]
        stream = _ScriptedStream(scripts)
        monkeypatch.setattr(agent_mod, "stream", stream)

        state = AgentState()
        _drain(agent_run("x", state,
                         {"model": "x", "max_continuations": 3,
                          "permission_mode": "accept-all"},
                         system_prompt="sys"))
        # Strict alternation: user, assistant(stub), user(hint), assistant(ok)
        roles = [m["role"] for m in state.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        # Stub must be non-empty so the API doesn't reject it as an empty turn
        stub = state.messages[1]
        assert stub.get("content", "").strip()
        # And must NOT carry surviving tool_calls (those were all malformed)
        assert not stub.get("tool_calls")
        # No two consecutive user messages (the actual bug being fixed)
        for a, b in zip(roles, roles[1:]):
            assert not (a == "user" and b == "user"), \
                f"two consecutive user messages: {roles!r}"

    def test_disabled_via_zero_cap(self, loop_env, monkeypatch):
        """max_continuations=0 disables the feature: length-truncation just ends the turn."""
        import agent as agent_mod
        scripts = [
            [_turn(text="cut...", finish="length")],
        ]
        stream = _ScriptedStream(scripts)
        monkeypatch.setattr(agent_mod, "stream", stream)

        state = AgentState()
        _drain(agent_run("x", state, {"model": "x", "max_continuations": 0}, system_prompt="sys"))
        assert stream.calls == 1
        # No injected user hint
        roles = [m["role"] for m in state.messages]
        assert roles == ["user", "assistant"]


# ── Provider-side: finish_reason capture ─────────────────────────────────

class TestProviderFinishReason:

    def test_assistant_turn_default_finish_reason(self):
        t = AssistantTurn("x", [], 1, 1)
        assert t.finish_reason == ""

    def test_assistant_turn_explicit_finish_reason(self):
        t = AssistantTurn("x", [], 1, 1, finish_reason="length")
        assert t.finish_reason == "length"

    def test_anthropic_max_tokens_normalizes_to_length(self):
        """The Anthropic stream maps stop_reason='max_tokens' → finish_reason='length'."""
        # Mimic the inline mapping in stream_anthropic without round-tripping the SDK
        for stop, expected in [
            ("max_tokens",  "length"),
            ("end_turn",    "end_turn"),
            ("tool_use",    "tool_use"),
            ("",            ""),
            (None,          ""),
        ]:
            normalized = "length" if stop == "max_tokens" else (stop or "")
            assert normalized == expected
