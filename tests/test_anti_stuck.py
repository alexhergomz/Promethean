"""Anti-stuck heuristic tests — pure unit tests on the ring buffer logic.

The integration with agent.run() is exercised by a single end-to-end test
that mocks providers.stream() to emit the same Read three times.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anti_stuck
import agent
from agent import AgentState, AssistantTurn


def _state(turn=1):
    s = SimpleNamespace(turn_count=turn)
    return s


def _read_tc(path):
    return {"id": f"c_{path}", "name": "Read", "input": {"file_path": path}}


def _bash_tc(cmd):
    return {"id": f"c_{cmd[:8]}", "name": "Bash", "input": {"command": cmd}}


class TestRingBufferStuck:
    def test_two_calls_below_threshold(self):
        s = _state()
        stuck, _ = anti_stuck.detect_stuck(s, [_read_tc("/a.py"), _read_tc("/a.py")])
        assert stuck is False

    def test_three_identical_trips(self):
        s = _state(turn=5)
        anti_stuck.detect_stuck(s, [_read_tc("/a.py")])
        anti_stuck.detect_stuck(s, [_read_tc("/a.py")])
        stuck, tool = anti_stuck.detect_stuck(s, [_read_tc("/a.py")])
        assert stuck is True and tool == "Read"

    def test_three_different_args_does_not_trip(self):
        s = _state()
        for p in ("/a.py", "/b.py", "/c.py"):
            stuck, _ = anti_stuck.detect_stuck(s, [_read_tc(p)])
            assert stuck is False

    def test_different_tools_dont_trip(self):
        s = _state()
        stuck, _ = anti_stuck.detect_stuck(
            s, [_read_tc("/a"), _bash_tc("ls"), _read_tc("/a"), _bash_tc("ls")])
        assert stuck is False

    def test_cooldown_prevents_immediate_refire(self):
        s = _state(turn=10)
        for _ in range(3):
            anti_stuck.detect_stuck(s, [_read_tc("/a")])
        # First call (turn 10) tripped — mark it.
        anti_stuck.mark_fired(s)
        # Turn 11 — still within cooldown of 2 turns.
        s.turn_count = 11
        stuck, _ = anti_stuck.detect_stuck(s, [_read_tc("/a")])
        assert stuck is False
        # Turn 13 — past cooldown.
        s.turn_count = 13
        stuck, _ = anti_stuck.detect_stuck(s, [_read_tc("/a")])
        assert stuck is True

    def test_ring_size_bounded(self):
        s = _state()
        for i in range(20):
            anti_stuck.detect_stuck(s, [_read_tc(f"/f{i}.py")])
        # Ring should hold at most _RING_SIZE entries.
        assert len(s._tool_ring) == anti_stuck._RING_SIZE

    def test_steering_message_mentions_tool(self):
        msg = anti_stuck.steering_message("Grep")
        assert "Grep" in msg and "anti-stuck" in msg


class TestIntegrationWithAgent:
    """End-to-end: agent.run() with a mock stream that emits three
    identical Read tool calls. After the third, the nudge should fire."""

    def _make_stream(self, paths):
        """Each call yields one AssistantTurn that asks for Read of paths[i]."""
        idx = {"i": 0}

        def _stream(*, model, system, messages, tool_schemas, config):
            i = idx["i"]
            idx["i"] += 1
            if i >= len(paths):
                # Final turn: no tool calls → end loop
                yield AssistantTurn(
                    text="done", tool_calls=[], in_tokens=1, out_tokens=1,
                    cache_read_tokens=0, cache_write_tokens=0,
                    reasoning_content="", finish_reason="stop",
                )
                return
            yield AssistantTurn(
                text="", tool_calls=[_read_tc(paths[i])],
                in_tokens=1, out_tokens=1,
                cache_read_tokens=0, cache_write_tokens=0,
                reasoning_content="", finish_reason="tool_calls",
            )
        return _stream

    def test_three_identical_reads_trips_nudge(self, monkeypatch):
        # Mock stream, the tool execution path, and compaction/sanitize.
        monkeypatch.setattr(
            agent, "stream",
            self._make_stream(["/a.py", "/a.py", "/a.py"]),
        )
        monkeypatch.setattr(agent, "maybe_compact", lambda *a, **kw: False)
        monkeypatch.setattr(agent, "sanitize_history", lambda m: m)

        # Stub tool execution: pretend the Read returns the same content.
        def fake_exec(tc, config, session_id, runtime_ctx, depth):
            return tc, "fake content", True
        monkeypatch.setattr(agent, "_exec_one_tool", fake_exec, raising=False)

        # The actual code uses an inner _exec_one — patch the path used in
        # agent.run by intercepting _check_permission to allow all and
        # patching subprocess via tools.execute. Simpler approach: just
        # patch ToolEnd emission by stubbing the tools subsystem.
        # For this test we accept that the easiest path is to bypass the
        # real tool runner; we patch the dispatch via execute_tool.
        import tools as _tools
        monkeypatch.setattr(
            _tools, "execute",
            lambda tc, **kw: "fake content",
            raising=False,
        )

        config = {
            "model": "test-model", "_session_id": "t", "no_tools": False,
            "permission_mode": "accept-all", "anti_stuck": True,
        }
        events = list(agent.run("hi", AgentState(), config, "sys"))
        text = "".join(getattr(e, "text", "") for e in events)
        assert "anti-stuck" in text, f"nudge never fired. Captured: {text[:400]}"
