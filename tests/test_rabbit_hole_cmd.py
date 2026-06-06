"""Tests for the /rabbit-hole slash command.

We patch get_agent_manager() to use a fresh SubAgentManager with a
mocked _agent_run, so the spawn path is exercised end-to-end without
hitting a live model. UI output (info/ok/err/warn) is captured via
capsys and asserted on visible content (not exact formatting — colors
include ANSI escapes).
"""
from __future__ import annotations

import os
import re
import sys
from unittest.mock import patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from commands.rabbit_hole_cmd import cmd_rabbit_hole
from multi_agent.subagent import SubAgentManager


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI.sub("", s)


def _all(captured) -> str:
    """Combined stdout + stderr, ANSI-stripped. err() goes to stderr;
    info/ok/warn go to stdout."""
    return _strip(captured.out + captured.err)


@pytest.fixture
def fresh_manager(monkeypatch):
    """Patch get_agent_manager() to return a fresh manager with a fast
    mock for _agent_run, plus shutdown teardown."""
    import time

    def mock_agent_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
        for _ in range(2):
            if cancel_check and cancel_check():
                return
            time.sleep(0.02)
        state.messages.append({
            "role": "assistant",
            "content": f"Result for: {prompt[:60]}",
            "tool_calls": [],
        })
        yield None

    monkeypatch.setattr("multi_agent.subagent._agent_run", mock_agent_run)
    mgr = SubAgentManager(max_concurrent=3, max_depth=3)
    monkeypatch.setattr("multi_agent.tools.get_agent_manager", lambda: mgr)
    yield mgr
    mgr.shutdown()


# ── Help / no-args path ───────────────────────────────────────────────────

def test_no_args_shows_help(fresh_manager, capsys):
    cmd_rabbit_hole("", None, {})
    out = _all(capsys.readouterr())
    assert "Usage" in out
    assert "spawn a new rabbit-hole" in out
    assert "list" in out
    assert "stop" in out


def test_help_subcommand(fresh_manager, capsys):
    cmd_rabbit_hole("help", None, {})
    out = _all(capsys.readouterr())
    assert "Usage" in out


# ── Spawn path ────────────────────────────────────────────────────────────

def test_spawn_creates_rabbit_hole_task(fresh_manager, capsys):
    cmd_rabbit_hole("Investigate every variant of speculative decoding",
                    None, {})
    out = _all(capsys.readouterr())
    assert "Rabbit-hole spawned" in out
    assert "Workspace:" in out
    # Manager should have one rabbit-hole task
    rh_tasks = [t for t in fresh_manager.list_tasks()
                if getattr(t, "rabbit_hole_dir", "")]
    assert len(rh_tasks) == 1
    # Cleanup
    fresh_manager.cancel(rh_tasks[0].id)
    fresh_manager.wait(rh_tasks[0].id, timeout=3)


def test_spawn_reports_task_id_and_workspace(fresh_manager, capsys):
    cmd_rabbit_hole("the question", None, {})
    out = _all(capsys.readouterr())
    rh_tasks = [t for t in fresh_manager.list_tasks()
                if getattr(t, "rabbit_hole_dir", "")]
    task = rh_tasks[0]
    # Task id appears in output
    assert task.id in out
    assert task.rabbit_hole_dir in out
    # Cleanup
    fresh_manager.cancel(task.id)
    fresh_manager.wait(task.id, timeout=3)


# ── List ──────────────────────────────────────────────────────────────────

def test_list_empty(fresh_manager, capsys):
    cmd_rabbit_hole("list", None, {})
    out = _all(capsys.readouterr())
    assert "No rabbit-hole" in out


def test_list_shows_active_tasks(fresh_manager, capsys):
    cmd_rabbit_hole("first question", None, {})
    cmd_rabbit_hole("second question", None, {})
    capsys.readouterr()  # discard spawn output
    cmd_rabbit_hole("list", None, {})
    out = _all(capsys.readouterr())
    assert "rabbit-hole task(s)" in out
    # Two tasks listed by name
    rh_tasks = [t for t in fresh_manager.list_tasks()
                if getattr(t, "rabbit_hole_dir", "")]
    assert len(rh_tasks) == 2
    for t in rh_tasks:
        assert t.name in out
    # Cleanup
    for t in rh_tasks:
        fresh_manager.cancel(t.id)


def test_list_alias_ls(fresh_manager, capsys):
    cmd_rabbit_hole("ls", None, {})
    out = _all(capsys.readouterr())
    assert "No rabbit-hole" in out


# ── Status ────────────────────────────────────────────────────────────────

def test_status_unknown_task(fresh_manager, capsys):
    cmd_rabbit_hole("status nonexistent-name", None, {})
    out = _all(capsys.readouterr())
    assert "No rabbit-hole task" in out


def test_status_for_real_task(fresh_manager, capsys):
    import time
    cmd_rabbit_hole("the question", None, {})
    capsys.readouterr()
    rh = [t for t in fresh_manager.list_tasks()
          if getattr(t, "rabbit_hole_dir", "")][0]
    # Wait for main run to settle
    for _ in range(40):
        if rh.status in ("idle", "completed"):
            break
        time.sleep(0.05)
    cmd_rabbit_hole(f"status {rh.name}", None, {})
    out = _all(capsys.readouterr())
    assert rh.name in out
    assert "Workspace:" in out
    assert "Sub-questions" in out
    assert "Findings" in out
    fresh_manager.cancel(rh.id)


def test_status_resolves_by_id_prefix(fresh_manager, capsys):
    import time
    cmd_rabbit_hole("q", None, {})
    capsys.readouterr()
    rh = [t for t in fresh_manager.list_tasks()
          if getattr(t, "rabbit_hole_dir", "")][0]
    for _ in range(20):
        if rh.status in ("idle", "completed"):
            break
        time.sleep(0.05)
    # Use first 4 chars of id as the lookup key
    cmd_rabbit_hole(f"status {rh.id[:4]}", None, {})
    out = _all(capsys.readouterr())
    assert "Workspace:" in out
    fresh_manager.cancel(rh.id)


# ── Send message ──────────────────────────────────────────────────────────

def test_msg_requires_two_args(fresh_manager, capsys):
    cmd_rabbit_hole("msg", None, {})
    out = _all(capsys.readouterr())
    assert "Usage" in out


def test_msg_to_unknown_task(fresh_manager, capsys):
    cmd_rabbit_hole("msg nonexistent hello", None, {})
    out = _all(capsys.readouterr())
    assert "No rabbit-hole task" in out


def test_msg_to_idle_task_queues(fresh_manager, capsys):
    import time
    cmd_rabbit_hole("research X", None, {})
    capsys.readouterr()
    rh = [t for t in fresh_manager.list_tasks()
          if getattr(t, "rabbit_hole_dir", "")][0]
    # Wait for idle (background mode reaches idle after main run)
    for _ in range(40):
        if rh.status == "idle":
            break
        time.sleep(0.1)
    cmd_rabbit_hole(f"msg {rh.name} also look at Y", None, {})
    out = _all(capsys.readouterr())
    assert "queued" in out.lower()
    # Cleanup
    fresh_manager.cancel(rh.id)


# ── Stop ──────────────────────────────────────────────────────────────────

def test_stop_unknown(fresh_manager, capsys):
    cmd_rabbit_hole("stop nonexistent", None, {})
    out = _all(capsys.readouterr())
    assert "No rabbit-hole task" in out


def test_stop_cancels_alive_task(fresh_manager, capsys):
    import time
    cmd_rabbit_hole("q", None, {})
    capsys.readouterr()
    rh = [t for t in fresh_manager.list_tasks()
          if getattr(t, "rabbit_hole_dir", "")][0]
    for _ in range(20):
        if rh.status == "idle":
            break
        time.sleep(0.1)
    cmd_rabbit_hole(f"stop {rh.name}", None, {})
    out = _all(capsys.readouterr())
    assert "Cancelled" in out
    # Wait for cancellation to complete
    fresh_manager.wait(rh.id, timeout=3)
    assert rh.status == "cancelled"


def test_stop_all_cancels_every_task(fresh_manager, capsys):
    import time
    cmd_rabbit_hole("q1", None, {})
    cmd_rabbit_hole("q2", None, {})
    capsys.readouterr()
    rhs = [t for t in fresh_manager.list_tasks()
           if getattr(t, "rabbit_hole_dir", "")]
    for _ in range(20):
        if all(t.status == "idle" for t in rhs):
            break
        time.sleep(0.1)
    cmd_rabbit_hole("stop all", None, {})
    out = _all(capsys.readouterr())
    assert "Cancelled" in out
    assert "2" in out  # number of cancelled tasks
    for t in rhs:
        fresh_manager.wait(t.id, timeout=3)


# ── Report ────────────────────────────────────────────────────────────────

def test_report_no_reports_yet(fresh_manager, capsys):
    cmd_rabbit_hole("question", None, {})
    capsys.readouterr()
    rh = [t for t in fresh_manager.list_tasks()
          if getattr(t, "rabbit_hole_dir", "")][0]
    # Don't wait for synthesis — just call report immediately
    cmd_rabbit_hole(f"report {rh.name}", None, {})
    out = _all(capsys.readouterr())
    # Either "No reports yet" or it picked up a final report (in case
    # the run finished). Either is correct behavior.
    assert "report" in out.lower() or "No reports" in out
    fresh_manager.cancel(rh.id)
    fresh_manager.wait(rh.id, timeout=3)


def test_report_finds_final_after_cancel(fresh_manager, capsys):
    """After cancel + synthesis runs, report should point to final.md."""
    import time
    cmd_rabbit_hole("q", None, {})
    capsys.readouterr()
    rh = [t for t in fresh_manager.list_tasks()
          if getattr(t, "rabbit_hole_dir", "")][0]
    for _ in range(20):
        if rh.status == "idle":
            break
        time.sleep(0.1)
    fresh_manager.cancel(rh.id)
    fresh_manager.wait(rh.id, timeout=3)
    # Synthesis ran in the finally block
    cmd_rabbit_hole(f"report {rh.name}", None, {})
    out = _all(capsys.readouterr())
    assert "final.md" in out


# ── Aliases ───────────────────────────────────────────────────────────────

def test_unknown_subcommand_is_treated_as_question(fresh_manager, capsys):
    """Anything that isn't a known subcommand → treated as the research
    question. So `/rabbit-hole list-something-weird` would spawn a task
    with that as the question. We don't try to be clever with parsing."""
    cmd_rabbit_hole("how does X work", None, {})
    out = _all(capsys.readouterr())
    assert "Rabbit-hole spawned" in out
    rh = [t for t in fresh_manager.list_tasks()
          if getattr(t, "rabbit_hole_dir", "")][0]
    assert rh.prompt == "how does X work"
    fresh_manager.cancel(rh.id)
    fresh_manager.wait(rh.id, timeout=3)
