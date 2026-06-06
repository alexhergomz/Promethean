"""Rabbit-hole long-running deep-research subagent tests.

Three layers covered:
  1. RabbitHoleWorkspace — disk-backed store, dedup, persistence
  2. Tools — the agent's vocabulary (rabbit_fetch, save_finding, …)
  3. Synthesis — final report generation from workspace state

Live network is NOT used; rabbit_fetch is exercised both for the cache-hit
path (no fetch) and the cache-miss path (with WebFetch monkeypatched).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from rabbit_hole.store import (
    Finding, Question, RabbitHoleWorkspace, _normalize_url, _url_key,
)


# ── URL normalization ─────────────────────────────────────────────────────

class TestNormalizeUrl:
    def test_lowercases_host(self):
        assert _normalize_url("HTTP://Example.COM/Path") == \
               "http://example.com/Path"

    def test_drops_fragment(self):
        a = _normalize_url("https://x.com/page#section1")
        b = _normalize_url("https://x.com/page#section2")
        assert a == b
        assert "#" not in a

    def test_sorts_query_params(self):
        a = _normalize_url("https://x.com/?b=2&a=1")
        b = _normalize_url("https://x.com/?a=1&b=2")
        assert a == b

    def test_strips_trailing_slash(self):
        assert _normalize_url("https://x.com/page/") == \
               _normalize_url("https://x.com/page")

    def test_url_key_is_stable(self):
        assert _url_key("https://x.com/foo") == _url_key("HTTPS://X.COM/foo/")


# ── Workspace persistence ─────────────────────────────────────────────────

class TestWorkspaceCRUD:
    def test_creates_layout(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "What is X?")
        assert ws.sources_dir.exists()
        assert ws.findings_dir.exists()
        assert ws.notes_dir.exists()
        assert ws.reports_dir.exists()
        assert ws.manifest_path.exists()

    def test_manifest_persists(self, tmp_path):
        root = str(tmp_path / "wk")
        RabbitHoleWorkspace(root, "Original question")
        # Reload — manifest should not be overwritten with a new timestamp
        ws2 = RabbitHoleWorkspace(root, "Different question")
        assert ws2.manifest["root_question"] == "Original question"

    def test_questions_persist(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        qid = ws.add_question("first sub-q")
        ws.add_question("second sub-q", parent_id=qid)
        # Reload — both should still be there
        ws2 = RabbitHoleWorkspace(str(tmp_path / "wk"))
        qs = ws2.list_open_questions()
        assert len(qs) == 2
        assert any(q.parent_id == qid for q in qs)

    def test_source_dedup_by_url(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        sha1 = ws.add_source("https://example.com/page", "content v1", "Title")
        # Same URL, normalized differently — should produce the same key
        # but add_source overwrites the file, which is fine
        assert ws.has_source("HTTPS://EXAMPLE.COM/page")
        assert ws.has_source("https://example.com/page#fragment")
        sources = ws.list_sources()
        assert len(sources) == 1

    def test_findings_link_to_question(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        qid = ws.add_question("about X")
        fid = ws.add_finding("X is true", qid, ["https://example.com/proof"])
        # Backref recorded
        assert fid in ws.questions[qid].finding_ids
        # Listing scoped by question
        scoped = ws.list_findings(sub_question_id=qid)
        assert len(scoped) == 1
        assert scoped[0].claim == "X is true"

    def test_mark_done_updates_state(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        qid = ws.add_question("Q")
        ws.advance_turn()
        ws.advance_turn()
        ok = ws.mark_question_done(qid, summary="learned X")
        assert ok is True
        assert ws.questions[qid].status == "closed"
        assert ws.questions[qid].summary == "learned X"
        # Marker reflects the turn at which it closed
        assert ws.state.last_question_close_turn == ws.state.turn


# ── Anti-stuck ─────────────────────────────────────────────────────────────

class TestStuckDetection:
    def test_stuck_zero_at_start(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        assert ws.stuck_for() == 0

    def test_stuck_grows_with_idle_turns(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        for _ in range(5):
            ws.advance_turn()
        assert ws.stuck_for() == 5

    def test_progress_resets_stuck(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        for _ in range(5):
            ws.advance_turn()
        assert ws.stuck_for() == 5
        ws.add_source("https://x.com", "data", "Title")
        # add_source sets last_new_source_turn = current turn → stuck_for=0
        assert ws.stuck_for() == 0


# ── Tools (rabbit_fetch dedup, save_finding, etc.) ─────────────────────────

@pytest.fixture
def workspace_config(tmp_path):
    """Return (workspace, config) ready for tool calls."""
    ws_dir = str(tmp_path / "wk")
    RabbitHoleWorkspace(ws_dir, "root question")
    config = {"_rabbit_hole_workspace_dir": ws_dir}
    return ws_dir, config


class TestRabbitFetchDedup:
    def test_cache_miss_calls_web_fetch(self, workspace_config, monkeypatch):
        ws_dir, config = workspace_config
        from rabbit_hole import tools
        called = []

        def fake_fetch(url):
            called.append(url)
            return "<html><title>Hi</title>some content</html>"

        monkeypatch.setattr("tools.web._webfetch", fake_fetch)
        out = tools.rabbit_fetch("https://example.com/page", config)
        assert called == ["https://example.com/page"]
        assert "some content" in out
        assert "[CACHED" not in out

    def test_cache_hit_skips_web_fetch(self, workspace_config, monkeypatch):
        ws_dir, config = workspace_config
        from rabbit_hole import tools

        # Pre-populate cache
        ws = RabbitHoleWorkspace(ws_dir)
        ws.add_source("https://example.com/page", "cached body", "Title")

        called = []
        monkeypatch.setattr("tools.web._webfetch",
                            lambda url: called.append(url) or "")
        out = tools.rabbit_fetch("https://example.com/page", config)
        assert "[CACHED" in out
        assert "cached body" in out
        assert called == []

    def test_cache_hit_via_url_normalization(self, workspace_config, monkeypatch):
        """Fetching the same URL with a fragment should still hit the cache."""
        ws_dir, config = workspace_config
        from rabbit_hole import tools
        ws = RabbitHoleWorkspace(ws_dir)
        ws.add_source("https://example.com/page", "body", "T")
        monkeypatch.setattr(
            "tools.web._webfetch",
            lambda url: pytest.fail("should not be called — cache hit"),
        )
        out = tools.rabbit_fetch("https://example.com/page#anchor", config)
        assert "[CACHED" in out


class TestRabbitHoleTools:
    def test_add_sub_question_returns_id(self, workspace_config):
        from rabbit_hole import tools
        _, config = workspace_config
        out = tools.add_sub_question("what is X?", config)
        assert "Sub-question added" in out
        assert "q-" in out

    def test_list_open_questions_after_close(self, workspace_config):
        from rabbit_hole import tools
        ws_dir, config = workspace_config
        out1 = tools.add_sub_question("q1", config)
        # Parse qid from output
        qid = [w for w in out1.split() if w.startswith("q-")][0]
        listed = tools.list_open_questions(config)
        assert qid in listed
        tools.mark_question_done(qid, "summary", config)
        listed_after = tools.list_open_questions(config)
        assert qid not in listed_after

    def test_save_finding_rejects_unknown_question(self, workspace_config):
        from rabbit_hole import tools
        _, config = workspace_config
        out = tools.save_finding("a claim", "q-bogus", ["https://x"], config)
        assert "Error" in out

    def test_save_finding_links_to_question(self, workspace_config):
        from rabbit_hole import tools
        ws_dir, config = workspace_config
        qid_out = tools.add_sub_question("q1", config)
        qid = [w for w in qid_out.split() if w.startswith("q-")][0]
        out = tools.save_finding("A is true", qid, ["https://a"], config)
        assert "saved" in out
        # Verify on disk
        ws = RabbitHoleWorkspace(ws_dir)
        findings = ws.list_findings(sub_question_id=qid)
        assert len(findings) == 1
        assert findings[0].claim == "A is true"

    def test_search_findings_uses_bm25(self, workspace_config):
        from rabbit_hole import tools
        ws_dir, config = workspace_config
        qid = [w for w in tools.add_sub_question("q1", config).split()
               if w.startswith("q-")][0]
        tools.save_finding("TurboQuant compresses KV cache 4x", qid,
                           ["https://x"], config)
        tools.save_finding("HQQ is a 4-bit weight quantization method", qid,
                           ["https://y"], config)
        out = tools.search_findings("TurboQuant", config, top_k=2)
        assert "TurboQuant" in out
        # The HQQ finding should not rank first
        first_line = [l for l in out.splitlines() if "TurboQuant" in l][0]
        assert "TurboQuant" in first_line

    def test_finish_marks_workspace(self, workspace_config):
        from rabbit_hole import tools
        ws_dir, config = workspace_config
        out = tools.finish("exhausted research", config)
        assert "Finish requested" in out
        ws = RabbitHoleWorkspace(ws_dir)
        assert ws.manifest["status"] == "finished"

    def test_tools_outside_workspace_error_clearly(self):
        """When called without _rabbit_hole_workspace_dir in config, tools
        return a clear error rather than crashing."""
        from rabbit_hole import tools
        config = {}
        for fn, args in [
            (tools.list_open_questions, (config,)),
            (lambda c: tools.add_sub_question("q", c), (config,)),
            (lambda c: tools.save_finding("c", "q", [], c), (config,)),
        ]:
            out = fn(*args)
            assert "Error" in out and "rabbit-hole" in out


# ── Progress event log (live activity feed) ───────────────────────────────

class TestProgressEvents:
    def test_add_question_records_event(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        qid = ws.add_question("the question")
        events = ws.list_events()
        assert any(e["kind"] == "question_added"
                   and e["payload"]["question_id"] == qid for e in events)

    def test_add_source_records_event(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        ws.add_source("https://x.com/foo", "page body", "Page Title")
        events = ws.list_events()
        fetched = [e for e in events if e["kind"] == "source_fetched"]
        assert len(fetched) == 1
        assert fetched[0]["payload"]["url"] == "https://x.com/foo"
        assert fetched[0]["payload"]["title"] == "Page Title"

    def test_add_finding_records_event(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        qid = ws.add_question("Q")
        fid = ws.add_finding("a claim", qid, ["https://e1", "https://e2"])
        events = ws.list_events()
        finding_events = [e for e in events if e["kind"] == "finding_saved"]
        assert len(finding_events) == 1
        assert finding_events[0]["payload"]["finding_id"] == fid
        assert finding_events[0]["payload"]["n_evidence"] == 2

    def test_mark_question_done_records_event(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        qid = ws.add_question("Q")
        ws.mark_question_done(qid, summary="learned X")
        events = ws.list_events()
        closed = [e for e in events if e["kind"] == "question_closed"]
        assert len(closed) == 1
        assert closed[0]["payload"]["summary"] == "learned X"

    def test_events_are_chronological(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        qid = ws.add_question("Q1")
        ws.add_source("https://a.com", "body", "A")
        ws.add_finding("claim", qid, ["https://a.com"])
        ws.mark_question_done(qid, "done")
        events = ws.list_events()
        # Order should be: question_added, source_fetched, finding_saved, question_closed
        kinds = [e["kind"] for e in events]
        assert kinds == ["question_added", "source_fetched",
                         "finding_saved", "question_closed"]

    def test_list_events_last_n(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        for i in range(10):
            ws.add_question(f"Q{i}")
        events = ws.list_events(last_n=3)
        assert len(events) == 3
        # Last 3 should be the highest-numbered questions
        for ev, expected in zip(events, ["Q7", "Q8", "Q9"]):
            assert expected in ev["payload"]["text"]

    def test_events_persist_to_jsonl(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        ws.add_question("Q")
        # File on disk
        progress = tmp_path / "wk" / "progress.jsonl"
        assert progress.exists()
        lines = progress.read_text().splitlines()
        assert len(lines) == 1
        # Each line is valid JSON
        ev = json.loads(lines[0])
        assert ev["kind"] == "question_added"

    def test_event_log_survives_workspace_reload(self, tmp_path):
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        ws.add_question("Q1")
        ws.add_source("https://a", "body", "A")
        # Reload from disk
        ws2 = RabbitHoleWorkspace(str(tmp_path / "wk"))
        events = ws2.list_events()
        assert len(events) == 2


# ── Note tool (free-form progress) ─────────────────────────────────────────

class TestNoteTool:
    def test_note_records_event(self, workspace_config):
        from rabbit_hole import tools
        ws_dir, config = workspace_config
        out = tools.note("switching focus to inference angle", config)
        assert "Note recorded" in out
        ws = RabbitHoleWorkspace(ws_dir)
        events = ws.list_events()
        notes = [e for e in events if e["kind"] == "note"]
        assert len(notes) == 1
        assert notes[0]["payload"]["text"] == "switching focus to inference angle"

    def test_note_caps_long_text(self, workspace_config):
        from rabbit_hole import tools
        ws_dir, config = workspace_config
        long_text = "x" * 1000
        tools.note(long_text, config)
        ws = RabbitHoleWorkspace(ws_dir)
        notes = [e for e in ws.list_events() if e["kind"] == "note"]
        assert len(notes[0]["payload"]["text"]) <= 510  # 500 + …

    def test_empty_note_returns_error(self, workspace_config):
        from rabbit_hole import tools
        _, config = workspace_config
        out = tools.note("   ", config)
        assert "empty" in out.lower()

    def test_note_outside_workspace(self):
        from rabbit_hole import tools
        out = tools.note("hi", {})
        assert "Error" in out


# ── Agent definition has Note in whitelist ─────────────────────────────────

class TestNoteInRabbitHoleAgent:
    def test_note_in_tool_whitelist(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research-rabbit-hole")
        assert "Note" in ad.tools


# ── Synthesis ──────────────────────────────────────────────────────────────

class TestSynthesis:
    def test_empty_workspace_produces_minimal_report(self, tmp_path):
        from rabbit_hole.synthesis import synthesize_workspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        report = synthesize_workspace(ws, final=True)
        assert "TL;DR" in report
        assert "(no findings recorded)" in report

    def test_report_has_sections_per_question(self, tmp_path):
        from rabbit_hole.synthesis import synthesize_workspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        q1 = ws.add_question("about A")
        q2 = ws.add_question("about B")
        ws.add_finding("A claim", q1, ["https://a.com"])
        ws.add_finding("B claim", q2, ["https://b.com"])
        ws.mark_question_done(q1, "A summary")
        report = synthesize_workspace(ws, final=True)
        assert "about A" in report
        assert "about B" in report
        assert "A claim" in report
        assert "B claim" in report

    def test_report_lists_open_questions_explicitly(self, tmp_path):
        from rabbit_hole.synthesis import synthesize_workspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        ws.add_question("unanswered Q")
        report = synthesize_workspace(ws, final=True)
        assert "Open questions" in report
        assert "unanswered Q" in report

    def test_report_writes_to_disk(self, tmp_path):
        from rabbit_hole.synthesis import synthesize_workspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        ws.add_question("Q")
        synthesize_workspace(ws, final=True)
        assert (ws.reports_dir / "final.md").exists()

    def test_intermediate_reports_are_numbered(self, tmp_path):
        from rabbit_hole.synthesis import synthesize_workspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        synthesize_workspace(ws, final=False)
        synthesize_workspace(ws, final=False)
        files = sorted(ws.reports_dir.glob("intermediate-*.md"))
        assert len(files) == 2
        assert files[0].name == "intermediate-001.md"
        assert files[1].name == "intermediate-002.md"

    def test_cancelled_run_marked_in_report(self, tmp_path):
        from rabbit_hole.synthesis import synthesize_workspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        ws.add_question("Q")
        report = synthesize_workspace(ws, final=True, cancelled=True)
        assert "cancelled" in report.lower()

    def test_sources_section_includes_citation_count(self, tmp_path):
        from rabbit_hole.synthesis import synthesize_workspace
        ws = RabbitHoleWorkspace(str(tmp_path / "wk"), "root")
        ws.add_source("https://a.com", "body", "Title A")
        ws.add_source("https://b.com", "body", "Title B")
        qid = ws.add_question("Q")
        ws.add_finding("c1", qid, ["https://a.com"])
        ws.add_finding("c2", qid, ["https://a.com"])  # cite a twice
        ws.add_finding("c3", qid, ["https://b.com"])
        report = synthesize_workspace(ws, final=True)
        # Most-cited source should appear first in the sources section
        sources_section = report.split("## Sources")[1].split("##")[0]
        a_pos = sources_section.find("Title A")
        b_pos = sources_section.find("Title B")
        assert 0 <= a_pos < b_pos


# ── Agent registration ─────────────────────────────────────────────────────

class TestRabbitHoleAgentRegistered:
    def test_agent_in_builtin_registry(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research-rabbit-hole")
        assert ad is not None
        assert ad.source == "built-in"

    def test_agent_has_rabbit_hole_tools(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research-rabbit-hole")
        for required in ("RabbitFetch", "SaveFinding", "AddSubQuestion",
                         "MarkQuestionDone", "Think", "Note", "WebSearch"):
            assert required in ad.tools, f"missing {required}"

    def test_agent_cannot_self_finish(self):
        """The rabbit-hole agent intentionally lacks Finish so the model
        can't terminate its own run — only the user can stop it. This is
        the test-time-scaling 'force continue' pattern; without it,
        confused models call Finish in tight loops to escape research."""
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research-rabbit-hole")
        assert "Finish" not in ad.tools

    def test_agent_does_not_have_dangerous_tools(self):
        """Sandboxing-via-allowlist: no Bash, Write, Edit, NotebookEdit
        in the tool whitelist."""
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research-rabbit-hole")
        for forbidden in ("Bash", "Write", "Edit", "NotebookEdit"):
            assert forbidden not in ad.tools, f"unexpected {forbidden}"
