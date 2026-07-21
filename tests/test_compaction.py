"""Tests for compaction.py — token estimation, context limits, snipping, split point."""
from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compaction import (
    estimate_tokens,
    get_context_limit,
    _build_workspace_state_block,
    compact_for_research_continuity,
    elide_old_tool_observations,
    snip_old_tool_results,
    find_split_point,
    sanitize_history,
    _respect_tool_pairs,
)


# ── estimate_tokens ───────────────────────────────────────────────────────

def _expected(chars: int, msgs: int) -> int:
    """Mirror the estimate_tokens formula: int((int(chars/2.8) + msgs*4) * 1.1)"""
    return int((int(chars / 2.8) + msgs * 4) * 1.1)


class TestEstimateTokens:
    def test_simple_messages(self):
        msgs = [
            {"role": "user", "content": "Hello world"},          # 11 chars
            {"role": "assistant", "content": "Hi there!"},       # 9 chars
        ]
        result = estimate_tokens(msgs)
        # 20 chars, 2 messages → _expected(20, 2)
        assert result == _expected(20, 2)

    def test_empty_messages(self):
        assert estimate_tokens([]) == 0

    def test_empty_content(self):
        msgs = [{"role": "user", "content": ""}]
        # 0 chars + 1 msg framing overhead
        assert estimate_tokens(msgs) == _expected(0, 1)

    def test_tool_result_messages(self):
        msgs = [
            {"role": "tool", "tool_call_id": "abc", "name": "Read", "content": "x" * 350},
        ]
        result = estimate_tokens(msgs)
        assert result == _expected(350, 1)

    def test_structured_content(self):
        """Content that is a list of dicts (e.g. Anthropic tool_result blocks)."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "id1", "content": "A" * 70},
                ],
            },
        ]
        result = estimate_tokens(msgs)
        # "tool_result" (11) + "id1" (3) + "A"*70 (70) = 84 chars, 1 message
        assert result == _expected(84, 1)

    def test_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        ]
        result = estimate_tokens(msgs)
        # content "ok" (2) + tool_calls recursively: "c1"(2) + "Bash"(4) + "ls"(2) = 10 chars
        # (nested input dict values ARE now counted)
        assert result == _expected(10, 1)

    def test_deeply_nested_tool_input(self):
        """Deeply nested tool input values are all counted."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "x", "name": "Write", "input": {"file_path": "/a/b", "content": "hello"}},
                ],
            },
        ]
        result = estimate_tokens(msgs)
        # "x"(1) + "Write"(5) + "/a/b"(4) + "hello"(5) = 15 chars, 1 msg
        assert result == _expected(15, 1)


# ── get_context_limit ─────────────────────────────────────────────────────

class TestGetContextLimit:
    def test_custom_backend_default(self):
        # The llama.cpp/custom backend defaults to 128000 without an override.
        assert get_context_limit("custom/qwen3.5-9b") == 128000

    def test_config_override_wins(self):
        # A llama-server run with -np > 1 gives each slot n_ctx/n_parallel;
        # the explicit override must take precedence over the provider default.
        assert get_context_limit("custom/qwen3.5-9b", {"context_limit": 57344}) == 57344

    def test_unknown_model_fallback(self):
        assert get_context_limit("some-random-model-xyz") == 128000

    def test_bare_name_resolves_to_custom(self):
        assert get_context_limit("qwen3.5-9b") == 128000


# ── snip_old_tool_results ─────────────────────────────────────────────────

class TestSnipOldToolResults:
    def test_old_tool_results_get_truncated(self):
        long_content = "A" * 5000
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "let me check", "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1", "name": "Read", "content": long_content},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "you're welcome"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
            {"role": "user", "content": "wait"},
            {"role": "assistant", "content": "yes?"},
            {"role": "user", "content": "never mind"},
        ]
        result = snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert result is msgs  # mutated in place
        tool_msg = msgs[2]
        assert len(tool_msg["content"]) < 5000
        assert "snipped" in tool_msg["content"]

    def test_recent_tool_results_preserved(self):
        long_content = "B" * 5000
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok", "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1", "name": "Read", "content": long_content},
        ]
        # All 3 messages are within preserve_last_n_turns=6
        result = snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert msgs[2]["content"] == long_content  # not truncated

    def test_short_tool_results_not_touched(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "name": "Bash", "content": "short"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
            {"role": "user", "content": "f"},
        ]
        snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert msgs[0]["content"] == "short"

    def test_non_tool_messages_untouched(self):
        msgs = [
            {"role": "user", "content": "X" * 5000},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
            {"role": "user", "content": "f"},
        ]
        snip_old_tool_results(msgs, max_chars=2000, preserve_last_n_turns=6)
        assert msgs[0]["content"] == "X" * 5000


# ── elide_old_tool_observations (SWE-agent ACI pattern) ───────────────────

def _make_tool_msg(content: str, tool_call_id: str = "t1", name: str = "Read"):
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


class TestElideOldToolObservations:
    def test_old_tool_messages_get_elided_to_one_line(self):
        long = "x" * 5000 + "\n" + "y" * 5000
        msgs = [
            _make_tool_msg(long, "t1"),
            {"role": "user", "content": "ok"},
            _make_tool_msg(long, "t2"),
            {"role": "user", "content": "ok"},
            _make_tool_msg(long, "t3"),
            {"role": "user", "content": "ok"},
            _make_tool_msg(long, "t4"),
            {"role": "user", "content": "ok"},
            _make_tool_msg(long, "t5"),
            {"role": "user", "content": "ok"},
            _make_tool_msg(long, "t6"),
            {"role": "user", "content": "ok"},
            _make_tool_msg(long, "t7"),  # this is the most recent
        ]
        elide_old_tool_observations(msgs, preserve_last_n_tool_messages=5)
        # The 5 most recent tool messages (t3, t4, t5, t6, t7) preserved.
        # t1, t2 elided.
        assert msgs[0]["content"].startswith("[elided:"), \
            f"t1 should be elided, got: {msgs[0]['content'][:80]}"
        assert msgs[2]["content"].startswith("[elided:"), \
            f"t2 should be elided, got: {msgs[2]['content'][:80]}"
        assert msgs[4]["content"] == long  # t3 preserved
        assert msgs[12]["content"] == long  # t7 preserved

    def test_elision_includes_size_and_preview(self):
        msgs = [
            _make_tool_msg("alpha\nbeta\ngamma\ndelta\n" + "x" * 5000, "t1"),
            _make_tool_msg("y", "t2"),
            _make_tool_msg("y", "t3"),
            _make_tool_msg("y", "t4"),
            _make_tool_msg("y", "t5"),
            _make_tool_msg("y", "t6"),
        ]
        elide_old_tool_observations(msgs, preserve_last_n_tool_messages=5)
        elided = msgs[0]["content"]
        assert "lines" in elided
        assert "chars" in elided
        # Preview should contain at least the start of the original content
        assert "alpha" in elided

    def test_short_tool_messages_not_elided(self):
        """Below min_chars_to_elide, leave alone — there's nothing to gain."""
        short = "ok"
        msgs = [
            _make_tool_msg(short, "t1"),
            _make_tool_msg(short, "t2"),
            _make_tool_msg(short, "t3"),
            _make_tool_msg(short, "t4"),
            _make_tool_msg(short, "t5"),
            _make_tool_msg(short, "t6"),
            _make_tool_msg(short, "t7"),
        ]
        elide_old_tool_observations(msgs, preserve_last_n_tool_messages=3)
        for m in msgs:
            assert m["content"] == short

    def test_idempotent_on_double_call(self):
        long = "x" * 5000
        msgs = [
            _make_tool_msg(long, "t1"),
            _make_tool_msg(long, "t2"),
            _make_tool_msg(long, "t3"),  # preserved
            _make_tool_msg(long, "t4"),  # preserved
        ]
        elide_old_tool_observations(msgs, preserve_last_n_tool_messages=2)
        first_pass = msgs[0]["content"]
        elide_old_tool_observations(msgs, preserve_last_n_tool_messages=2)
        assert msgs[0]["content"] == first_pass, \
            "second call shouldn't re-elide an already-elided message"

    def test_non_tool_messages_unaffected(self):
        long = "x" * 5000
        msgs = [
            {"role": "user", "content": long},
            _make_tool_msg(long, "t1"),
            _make_tool_msg(long, "t2"),
            _make_tool_msg(long, "t3"),
            _make_tool_msg(long, "t4"),
            _make_tool_msg(long, "t5"),
            _make_tool_msg(long, "t6"),
        ]
        elide_old_tool_observations(msgs, preserve_last_n_tool_messages=3)
        # User message at index 0 should be unchanged
        assert msgs[0]["content"] == long

    def test_no_op_when_few_tool_messages(self):
        long = "x" * 5000
        msgs = [
            _make_tool_msg(long, "t1"),
            _make_tool_msg(long, "t2"),
        ]
        elide_old_tool_observations(msgs, preserve_last_n_tool_messages=5)
        # Only 2 tool messages, preserve_last=5 → both kept intact.
        assert msgs[0]["content"] == long
        assert msgs[1]["content"] == long


# ── find_split_point ──────────────────────────────────────────────────────

class TestFindSplitPoint:
    def test_returns_reasonable_index(self):
        msgs = [
            {"role": "user", "content": "A" * 1000},
            {"role": "assistant", "content": "B" * 1000},
            {"role": "user", "content": "C" * 1000},
            {"role": "assistant", "content": "D" * 1000},
            {"role": "user", "content": "E" * 1000},
        ]
        idx = find_split_point(msgs, keep_ratio=0.3)
        # With equal-size messages and keep_ratio=0.3, split should be around index 3-4
        assert 2 <= idx <= 4

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        idx = find_split_point(msgs, keep_ratio=0.3)
        assert idx == 0

    def test_empty_messages(self):
        idx = find_split_point([], keep_ratio=0.3)
        assert idx == 0

    def test_split_preserves_recent(self):
        # Recent portion should contain ~30% of tokens
        msgs = [{"role": "user", "content": "X" * 100} for _ in range(10)]
        idx = find_split_point(msgs, keep_ratio=0.3)
        total = estimate_tokens(msgs)
        recent = estimate_tokens(msgs[idx:])
        # Recent should be roughly 30% of total (allow some tolerance)
        assert recent >= total * 0.2
        assert recent <= total * 0.5

    def test_split_never_splits_tool_pair(self):
        # Large old portion, then one tool-call pair right where the natural
        # split would fall, followed by a small recent portion.
        msgs = (
            [{"role": "user", "content": "U" * 2000}]
            + [{"role": "assistant", "content": "A" * 2000} for _ in range(3)]
            + [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
                ]},
                {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "R" * 500},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        for ratio in (0.2, 0.3, 0.4, 0.5):
            idx = find_split_point(msgs, keep_ratio=ratio)
            # The split must not leave an orphan 'tool' as messages[idx].
            if 0 < idx < len(msgs):
                assert msgs[idx].get("role") != "tool", (
                    f"split at {idx} orphans a tool message (ratio={ratio})")
                prev = msgs[idx - 1]
                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                    # Must have kept the tool responses with it.
                    assert msgs[idx].get("role") != "tool"


class TestRespectToolPairs:
    def test_noop_when_not_on_tool_pair(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        assert _respect_tool_pairs(msgs, 2) == 2

    def test_advances_past_tool_response(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        # Natural split at 2 (between assistant and its tool response) → push to 3.
        assert _respect_tool_pairs(msgs, 2) == 3

    def test_advances_past_multiple_tool_responses(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
                {"id": "c2", "name": "Bash", "input": {"command": "b"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "1"},
            {"role": "tool", "tool_call_id": "c2", "name": "Bash", "content": "2"},
            {"role": "user", "content": "next"},
        ]
        assert _respect_tool_pairs(msgs, 1) == 3

    def test_orphan_tool_at_split_gets_skipped(self):
        # messages[split]=='tool' but messages[split-1] is not an assistant
        # with tool_calls → still push forward past the tool.
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "tool", "tool_call_id": "cX", "name": "Bash", "content": "x"},
            {"role": "user", "content": "b"},
        ]
        assert _respect_tool_pairs(msgs, 1) == 2


# ── sanitize_history ──────────────────────────────────────────────────────

class TestSanitizeHistory:
    def test_well_formed_history_unchanged(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        out = sanitize_history(msgs)
        assert out == msgs

    def test_drops_orphan_tool_message(self):
        # tool with no preceding assistant(tool_calls) — simulates post-compaction orphan.
        msgs = [
            {"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "understood"},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "orphan"},
            {"role": "user", "content": "thanks"},
        ]
        out = sanitize_history(msgs)
        assert all(m.get("role") != "tool" for m in out)
        assert len(out) == 3
        assert [m["role"] for m in out] == ["user", "assistant", "user"]

    def test_strips_unanswered_tool_calls(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
                {"id": "c2", "name": "Bash", "input": {"command": "b"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "A"},
            {"role": "user", "content": "next"},
        ]
        out = sanitize_history(msgs)
        asst = out[1]
        # c2 had no response → stripped; c1 remains.
        assert [tc["id"] for tc in asst["tool_calls"]] == ["c1"]
        assert out[2]["role"] == "tool" and out[2]["tool_call_id"] == "c1"

    def test_strips_all_tool_calls_when_none_answered(self):
        msgs = [
            {"role": "assistant", "content": "thinking", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
            {"role": "user", "content": "never mind"},
        ]
        out = sanitize_history(msgs)
        assert "tool_calls" not in out[0]
        assert out[0]["content"] == "thinking"

    def test_unanswered_at_end_also_stripped(self):
        # No trailing user/assistant; unanswered tool_calls at the very end.
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
        ]
        out = sanitize_history(msgs)
        assert "tool_calls" not in out[1]

    def test_empty_assistant_after_strip_gets_stub_content(self):
        """When sanitize strips ALL tool_calls AND content was empty,
        the assistant must end up with non-empty stub content. Empty
        assistants confuse models (some loop on a random tool, others
        re-emit prior text — the 'harness repeated text' bug)."""
        msgs = [
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Write", "input": {"_raw": "trunc"}},
            ]},
            {"role": "user", "content": "continue"},
        ]
        out = sanitize_history(msgs)
        assistant = out[1]
        assert "tool_calls" not in assistant
        assert assistant.get("content", "").strip(), (
            f"empty assistant after strip — content={assistant.get('content')!r}"
        )

    def test_existing_stub_assistant_is_preserved(self):
        """A history that already has the truncation stub must round-trip
        through sanitize_history unchanged in that respect — no
        re-emptying, no popping, alternation kept."""
        stub_text = "[output cut off at max_tokens]"
        msgs = [
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": stub_text},
            {"role": "user", "content": "continue from where you stopped"},
            {"role": "assistant", "content": "ok continuing"},
        ]
        out = sanitize_history(msgs)
        assert out[1]["content"] == stub_text
        roles = [m["role"] for m in out]
        for a, b in zip(roles, roles[1:]):
            assert not (a == "user" and b == "user"), (
                f"alternation broke: {roles!r}"
            )

    def test_input_not_mutated(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
            {"role": "user", "content": "x"},
        ]
        import copy
        snapshot = copy.deepcopy(msgs)
        sanitize_history(msgs)
        assert msgs == snapshot

    def test_tool_with_wrong_id_dropped(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "Bash", "input": {"command": "a"}},
            ]},
            {"role": "tool", "tool_call_id": "WRONG", "name": "Bash", "content": "bad"},
            {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "good"},
            {"role": "user", "content": "ok"},
        ]
        out = sanitize_history(msgs)
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "c1"


# ── compact_for_research_continuity (rabbit-hole-aware) ────────────────────

class TestBuildWorkspaceStateBlock:
    def test_renders_open_questions(self, tmp_path):
        from rabbit_hole.store import RabbitHoleWorkspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "investigate X")
        q1 = ws.add_question("how does X work?")
        ws.add_question("alternatives to X?")
        ws.add_finding("X uses Y", q1, ["https://example.com"])
        block = _build_workspace_state_block(str(tmp_path / "wk"))
        assert "investigate X" in block
        assert "Open sub-questions (2)" in block
        assert "how does X work?" in block

    def test_includes_recent_findings(self, tmp_path):
        from rabbit_hole.store import RabbitHoleWorkspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        q = ws.add_question("question")
        ws.add_finding("Claim A", q, ["https://a"])
        ws.add_finding("Claim B", q, ["https://b"])
        block = _build_workspace_state_block(str(tmp_path / "wk"))
        assert "Claim A" in block
        assert "Claim B" in block

    def test_includes_stuck_counter(self, tmp_path):
        from rabbit_hole.store import RabbitHoleWorkspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        for _ in range(7):
            ws.advance_turn()
        block = _build_workspace_state_block(str(tmp_path / "wk"))
        assert "stuck_for=7" in block

    def test_caps_question_list_at_15(self, tmp_path):
        from rabbit_hole.store import RabbitHoleWorkspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        for i in range(20):
            ws.add_question(f"question {i}")
        block = _build_workspace_state_block(str(tmp_path / "wk"))
        assert "and 5 more" in block


class TestCompactForResearchContinuity:
    def _make_history(self, n_user_assistant_pairs: int = 6) -> list:
        msgs = [{"role": "system", "content": "You are a research agent."}]
        msgs.append({"role": "user", "content": "Investigate X."})
        for i in range(n_user_assistant_pairs):
            msgs.append({"role": "assistant",
                         "content": f"Working on it, step {i}…"})
            msgs.append({"role": "user",
                         "content": f"continue, step {i} response"})
        return msgs

    def test_no_op_without_workspace(self):
        msgs = self._make_history()
        out = compact_for_research_continuity(msgs, config={})
        assert out is msgs  # returned unchanged

    def test_no_op_for_short_history(self, tmp_path):
        from rabbit_hole.store import RabbitHoleWorkspace
        RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        msgs = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Q"},
        ]
        config = {"_rabbit_hole_workspace_dir": str(tmp_path / "wk")}
        out = compact_for_research_continuity(msgs, config)
        assert out == msgs  # nothing to compact

    def test_preserves_system_and_first_user(self, tmp_path, monkeypatch):
        from rabbit_hole.store import RabbitHoleWorkspace
        RabbitHoleWorkspace(str(tmp_path / "wk"), "root question")
        # Stub out the auxiliary LLM call so we don't make a network request
        monkeypatch.setattr("auxiliary.stream_auxiliary",
                            lambda **kw: "stubbed summary text")
        msgs = self._make_history(8)
        config = {
            "_rabbit_hole_workspace_dir": str(tmp_path / "wk"),
            "model": "claude-haiku-4-5-20251001",
        }
        out = compact_for_research_continuity(msgs, config)
        # System prompt + first user message untouched at head
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "You are a research agent."
        assert out[1]["role"] == "user"
        assert out[1]["content"] == "Investigate X."

    def test_inserts_workspace_state_block(self, tmp_path, monkeypatch):
        from rabbit_hole.store import RabbitHoleWorkspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root question")
        ws.add_question("the open question")
        monkeypatch.setattr("auxiliary.stream_auxiliary",
                            lambda **kw: "stubbed summary")
        msgs = self._make_history(8)
        config = {
            "_rabbit_hole_workspace_dir": str(tmp_path / "wk"),
            "model": "claude-haiku-4-5-20251001",
        }
        out = compact_for_research_continuity(msgs, config)
        # The workspace state block lands as the third message (after
        # system + first user) as a user-role message.
        assert out[2]["role"] == "user"
        assert "Workspace state" in out[2]["content"]
        assert "the open question" in out[2]["content"]

    def test_includes_summary_message(self, tmp_path, monkeypatch):
        from rabbit_hole.store import RabbitHoleWorkspace
        RabbitHoleWorkspace(str(tmp_path / "wk"), "root question")
        monkeypatch.setattr("auxiliary.stream_auxiliary",
                            lambda **kw: "stubbed summary content X")
        msgs = self._make_history(8)
        config = {
            "_rabbit_hole_workspace_dir": str(tmp_path / "wk"),
            "model": "claude-haiku-4-5-20251001",
        }
        out = compact_for_research_continuity(msgs, config)
        # Find the summary message — should be a user-role message that
        # mentions "Mid-investigation summary" and contains the stubbed text
        summary_msgs = [m for m in out
                        if m.get("role") == "user"
                        and "Mid-investigation summary" in (m.get("content") or "")]
        assert len(summary_msgs) == 1
        assert "stubbed summary content X" in summary_msgs[0]["content"]

    def test_summary_prompt_targets_continuity_signals(self, tmp_path, monkeypatch):
        from rabbit_hole.store import RabbitHoleWorkspace
        RabbitHoleWorkspace(str(tmp_path / "wk"), "root question")

        captured_prompt = {}

        def fake_aux(**kw):
            captured_prompt["msg"] = kw["messages"][0]["content"]
            return "stubbed"

        monkeypatch.setattr("auxiliary.stream_auxiliary", fake_aux)
        msgs = self._make_history(8)
        config = {
            "_rabbit_hole_workspace_dir": str(tmp_path / "wk"),
            "model": "claude-haiku-4-5-20251001",
        }
        compact_for_research_continuity(msgs, config)
        # Prompt should call out the priority signals we want preserved
        prompt = captured_prompt["msg"]
        assert "CURRENT line of investigation" in prompt
        assert "research-continuity" in prompt or "research continuity" in prompt or "deep-research agent" in prompt
        # And explicitly say to drop already-saved findings
        assert "Already-saved findings" in prompt
