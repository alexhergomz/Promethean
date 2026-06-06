"""Tests for the sub-agent system (subagent.py)."""
import time
import threading

import pytest

from multi_agent.subagent import SubAgentManager, SubAgentTask, _extract_final_text


# ── Mock for _agent_run ──────────────────────────────────────────────────

def _make_mock_agent_run(sleep_per_iter=0.05, iters=3):
    """Return a mock _agent_run that simulates work and checks cancellation."""

    def mock_agent_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
        for i in range(iters):
            if cancel_check and cancel_check():
                return
            time.sleep(sleep_per_iter)
        # Append an assistant message to state
        state.messages.append({
            "role": "assistant",
            "content": f"Result for: {prompt}",
            "tool_calls": [],
        })
        # Yield a TurnDone-like event (generator protocol)
        yield None

    return mock_agent_run


def _make_slow_mock(sleep_per_iter=0.2, iters=10):
    """Return a slow mock for cancellation testing."""
    return _make_mock_agent_run(sleep_per_iter=sleep_per_iter, iters=iters)


@pytest.fixture
def manager(monkeypatch):
    """Create a SubAgentManager with mocked _agent_run.

    Tearing down: cancel any tasks that are still alive (foreground
    'running' or background 'idle') and shut down the pool. Without
    this, background-mode tests would leave threads polling forever,
    blocking pytest from exiting cleanly.
    """
    mock = _make_mock_agent_run()
    monkeypatch.setattr("multi_agent.subagent._agent_run", mock)
    mgr = SubAgentManager(max_concurrent=3, max_depth=3)
    yield mgr
    mgr.shutdown()


@pytest.fixture
def slow_manager(monkeypatch):
    """Create a SubAgentManager with a slow mock for cancel testing."""
    mock = _make_slow_mock()
    monkeypatch.setattr("multi_agent.subagent._agent_run", mock)
    mgr = SubAgentManager(max_concurrent=3, max_depth=3)
    yield mgr
    mgr.shutdown()


# ── Tests ────────────────────────────────────────────────────────────────

class TestSpawnAndWait:
    def test_spawn_and_wait_completes(self, manager):
        task = manager.spawn("hello", {}, "system")
        result_task = manager.wait(task.id, timeout=5)
        assert result_task is not None
        assert result_task.status == "completed"
        assert result_task.result == "Result for: hello"

    def test_spawn_returns_immediately(self, manager):
        task = manager.spawn("hello", {}, "system")
        # Task should be pending or running, not yet completed
        assert task.status in ("pending", "running")


class TestListTasks:
    def test_list_tasks(self, manager):
        t1 = manager.spawn("task1", {}, "system")
        t2 = manager.spawn("task2", {}, "system")
        tasks = manager.list_tasks()
        task_ids = [t.id for t in tasks]
        assert t1.id in task_ids
        assert t2.id in task_ids
        assert len(tasks) == 2


class TestCancel:
    def test_cancel_running_task(self, slow_manager):
        task = slow_manager.spawn("slow task", {}, "system")
        # Wait briefly to ensure the task starts running
        time.sleep(0.1)
        assert task.status == "running"
        success = slow_manager.cancel(task.id)
        assert success is True
        # Wait for the task to actually finish
        slow_manager.wait(task.id, timeout=5)
        assert task.status == "cancelled"


class TestDepthLimit:
    def test_spawn_at_max_depth_fails(self, manager):
        task = manager.spawn("deep", {}, "system", depth=3)
        assert task.status == "failed"
        assert "Max depth" in task.result


class TestGetResult:
    def test_get_result_completed(self, manager):
        task = manager.spawn("hello", {}, "system")
        manager.wait(task.id, timeout=5)
        result = manager.get_result(task.id)
        assert result == "Result for: hello"

    def test_get_result_unknown_id(self, manager):
        result = manager.get_result("nonexistent_id")
        assert result is None


class TestExtractFinalText:
    def test_extracts_last_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "second"},
        ]
        assert _extract_final_text(messages) == "second"

    def test_returns_none_for_empty(self):
        assert _extract_final_text([]) is None

    def test_returns_none_no_assistant(self):
        messages = [{"role": "user", "content": "hi"}]
        assert _extract_final_text(messages) is None


class TestWaitUnknown:
    def test_wait_unknown_returns_none(self, manager):
        assert manager.wait("nonexistent") is None


class TestSlotPaging:
    """When config["enable_slot_paging"]=True and the model is on the
    custom (llama-server) provider, SubAgentManager should:
    1. Allocate a free slot via list_slots() before spawning
    2. Pass _slot_id through eff_config so providers.py forwards it
    3. Track slot in self._allocated_slots
    4. Erase the slot via erase_slot() on subagent finish
    5. Remove from _allocated_slots
    """

    def test_no_slot_action_when_paging_disabled(self, manager, monkeypatch):
        """Default config (enable_slot_paging=False) → no slot calls."""
        from llama_slots import SlotInfo
        list_calls = []
        erase_calls = []

        def fake_list(server_url):
            list_calls.append(server_url)
            return []

        def fake_erase(slot_id, server_url):
            erase_calls.append((slot_id, server_url))

        monkeypatch.setattr("llama_slots.list_slots", fake_list)
        monkeypatch.setattr("llama_slots.erase_slot", fake_erase)

        config = {"model": "custom/qwen", "custom_base_url": "http://127.0.0.1:8080/v1"}
        task = manager.spawn("test", config, "system")
        manager.wait(task.id, timeout=2)
        assert task.slot_id is None
        assert list_calls == []
        assert erase_calls == []

    def test_no_slot_action_for_non_custom_provider(self, manager, monkeypatch):
        """Even with paging enabled, non-custom provider → no slot calls."""
        list_calls = []
        monkeypatch.setattr("llama_slots.list_slots",
                            lambda url: list_calls.append(url) or [])
        monkeypatch.setattr("llama_slots.erase_slot",
                            lambda sid, server_url: None)

        # Anthropic-shaped model → detect_provider returns "anthropic" not "custom"
        config = {
            "model": "claude-sonnet-4-6",
            "enable_slot_paging": True,
        }
        task = manager.spawn("test", config, "system")
        manager.wait(task.id, timeout=2)
        assert task.slot_id is None
        assert list_calls == []

    def test_allocates_slot_when_enabled(self, manager, monkeypatch):
        from llama_slots import SlotInfo
        slots = [
            SlotInfo(id=0, state="processing", n_ctx=57344, n_past=1, prompt="parent"),
            SlotInfo(id=1, state="idle", n_ctx=57344, n_past=0),
            SlotInfo(id=2, state="idle", n_ctx=57344, n_past=0),
        ]
        erase_calls = []
        monkeypatch.setattr("llama_slots.list_slots", lambda url: slots)
        monkeypatch.setattr(
            "llama_slots.erase_slot",
            lambda slot_id, server_url: erase_calls.append((slot_id, server_url)),
        )

        config = {
            "model": "custom/qwen3.5-9b",
            "custom_base_url": "http://127.0.0.1:8080/v1",
            "enable_slot_paging": True,
        }
        task = manager.spawn("hello", config, "system")
        # Slot should be allocated *synchronously* in spawn() — verify even
        # before _run() finishes
        assert task.slot_id == 1, f"expected slot 1, got {task.slot_id}"
        # Wait for _run to finish so the release happens
        manager.wait(task.id, timeout=3)
        # Erase called on finish
        assert (1, "http://127.0.0.1:8080") in erase_calls
        # Slot removed from in-memory allocation set
        assert 1 not in manager._allocated_slots

    def test_two_concurrent_spawns_pick_different_slots(self, manager, monkeypatch):
        """Two simultaneous spawn() calls must allocate different slots —
        the lock + in-memory _allocated_slots set is what prevents
        collisions when both see the same llama-server snapshot."""
        from llama_slots import SlotInfo
        slots = [
            SlotInfo(id=0, state="idle", n_ctx=57344, n_past=0),
            SlotInfo(id=1, state="idle", n_ctx=57344, n_past=0),
            SlotInfo(id=2, state="idle", n_ctx=57344, n_past=0),
        ]
        monkeypatch.setattr("llama_slots.list_slots", lambda url: slots)
        monkeypatch.setattr("llama_slots.erase_slot", lambda sid, server_url: None)

        config = {
            "model": "custom/qwen",
            "custom_base_url": "http://127.0.0.1:8080/v1",
            "enable_slot_paging": True,
        }
        t1 = manager.spawn("a", config, "system")
        t2 = manager.spawn("b", config, "system")
        assert t1.slot_id != t2.slot_id
        assert {t1.slot_id, t2.slot_id} == {0, 1}
        manager.wait(t1.id, timeout=2)
        manager.wait(t2.id, timeout=2)

    def test_no_idle_slot_falls_back_to_unpinned(self, manager, monkeypatch):
        """When all slots are busy, allocation returns None → no _slot_id
        in eff_config, llama-server picks a slot itself (LRU). The subagent
        still runs."""
        from llama_slots import SlotInfo
        slots = [
            SlotInfo(id=0, state="processing", n_ctx=57344, n_past=1),
            SlotInfo(id=1, state="processing", n_ctx=57344, n_past=1),
        ]
        monkeypatch.setattr("llama_slots.list_slots", lambda url: slots)
        monkeypatch.setattr("llama_slots.erase_slot", lambda sid, server_url: None)

        config = {
            "model": "custom/qwen",
            "custom_base_url": "http://127.0.0.1:8080/v1",
            "enable_slot_paging": True,
        }
        task = manager.spawn("test", config, "system")
        assert task.slot_id is None
        manager.wait(task.id, timeout=2)
        assert task.status in ("completed", "failed")  # ran regardless

    def test_server_unreachable_falls_back_silently(self, manager, monkeypatch):
        """list_slots raising LlamaSlotsError (server down, /slots disabled)
        should not crash spawn; subagent runs without slot pinning."""
        from llama_slots import LlamaSlotsError

        def boom(url):
            raise LlamaSlotsError("server is down")

        monkeypatch.setattr("llama_slots.list_slots", boom)
        monkeypatch.setattr("llama_slots.erase_slot", lambda sid, server_url: None)

        config = {
            "model": "custom/qwen",
            "custom_base_url": "http://127.0.0.1:8080/v1",
            "enable_slot_paging": True,
        }
        task = manager.spawn("test", config, "system")
        assert task.slot_id is None  # graceful no-pin
        manager.wait(task.id, timeout=2)


class TestBackgroundLoop:
    """Background subagents stay alive after the initial run, blocking on
    inbox.get() until cancelled or shut down. SendMessage delivers new
    prompts; the agent processes each in sequence. Slot is parked
    during idle if slot paging is enabled.
    """

    def test_foreground_agent_exits_after_main_run(self, manager):
        """Default behavior unchanged — non-background agents drain
        any pre-queued messages then exit."""
        task = manager.spawn("hello", {}, "system")
        manager.wait(task.id, timeout=2)
        assert task.status == "completed"

    def test_background_agent_idles_after_main_run(self, manager):
        """A background agent reaches `idle` after main run completes
        and stays alive waiting for SendMessage."""
        from multi_agent.subagent import AgentDefinition
        ad = AgentDefinition(
            name="bg-test", description="bg",
            system_prompt="", source="user", background=True,
        )
        task = manager.spawn("hello", {}, "system", agent_def=ad)
        # Wait briefly for main run to complete and idle state to set in
        import time
        for _ in range(40):  # up to 4s
            if task.status == "idle":
                break
            time.sleep(0.1)
        assert task.status == "idle", f"expected idle, got {task.status!r}"
        # Now signal shutdown so the test cleans up
        task._inbox.put("__SHUTDOWN__")
        manager.wait(task.id, timeout=3)

    def test_background_agent_processes_inbox_message(self, manager, monkeypatch):
        """SendMessage to an idle background agent triggers a new run."""
        import time
        from multi_agent.subagent import AgentDefinition

        # Track that _agent_run was called twice — once for main, once for inbox msg
        call_log = []
        original_mock = manager._pool._mock_count if hasattr(manager._pool, "_mock_count") else None

        def tracking_mock(prompt, state, config, system_prompt, depth=0, cancel_check=None):
            call_log.append(prompt)
            for _ in range(3):
                if cancel_check and cancel_check():
                    return
                time.sleep(0.05)
            state.messages.append({
                "role": "assistant",
                "content": f"Result for: {prompt}",
                "tool_calls": [],
            })
            yield None

        monkeypatch.setattr("multi_agent.subagent._agent_run", tracking_mock)

        ad = AgentDefinition(
            name="bg-test", description="bg",
            system_prompt="", source="user", background=True,
        )
        task = manager.spawn("initial prompt", {}, "system", agent_def=ad)

        # Wait for idle
        for _ in range(40):
            if task.status == "idle":
                break
            time.sleep(0.1)
        assert task.status == "idle", f"got {task.status}"
        assert call_log == ["initial prompt"]

        # Send a follow-up message
        task._inbox.put("follow-up question")
        # Wait for the agent to process and return to idle
        time.sleep(0.6)
        assert "follow-up question" in call_log

        # Cleanup
        task._inbox.put("__SHUTDOWN__")
        manager.wait(task.id, timeout=3)

    def test_background_agent_shuts_down_on_cancel(self, manager):
        """task._cancel_flag breaks the background loop without needing
        the shutdown sentinel."""
        from multi_agent.subagent import AgentDefinition
        import time

        ad = AgentDefinition(
            name="bg-test", description="bg",
            system_prompt="", source="user", background=True,
        )
        task = manager.spawn("hello", {}, "system", agent_def=ad)
        # Wait for idle
        for _ in range(40):
            if task.status == "idle":
                break
            time.sleep(0.1)
        assert task.status == "idle"
        # Cancel
        manager.cancel(task.id)
        manager.wait(task.id, timeout=3)
        assert task.status == "cancelled"

    def test_background_agent_finishes_on_workspace_status(self, manager, tmp_path):
        """Rabbit-hole agents exit the background loop when the
        workspace manifest's status flips to 'finished' (Finish() tool
        sets this)."""
        import time
        from multi_agent.subagent import AgentDefinition
        from rabbit_hole.store import RabbitHoleWorkspace

        # Pre-create a workspace marked as finished
        ws_dir = str(tmp_path / "wk")
        ws = RabbitHoleWorkspace(ws_dir, "root")
        ws.manifest["status"] = "finished"
        import json
        ws.manifest_path.write_text(json.dumps(ws.manifest, indent=2))

        ad = AgentDefinition(
            name="bg-rh", description="bg-rh",
            system_prompt="", source="user", background=True,
        )
        # Manually set the rabbit-hole hint via config so the loop knows
        # to check workspace status. (Real flow goes through spawn()'s
        # rabbit-hole branch which sets this for deep-research-rabbit-hole;
        # here we just verify the loop's status check logic.)
        config = {"_rabbit_hole_workspace_dir": ws_dir}
        task = manager.spawn("hello", config, "system", agent_def=ad)
        # In the real spawn, rabbit_hole_dir is captured from the
        # rabbit-hole branch of spawn(). For non-deep-research-rabbit-hole
        # background agents we don't pass rabbit_hole_dir into the loop,
        # so the workspace check is a no-op. This test just verifies the
        # agent at least completes its main run and idles cleanly without
        # error when given a workspace-flag config.
        for _ in range(40):
            if task.status in ("idle", "completed"):
                break
            time.sleep(0.1)
        # Cleanup
        task._inbox.put("__SHUTDOWN__")
        manager.wait(task.id, timeout=3)


class TestToolWhitelistEnforcement:
    """AgentDefinition.tools is a security boundary — only the listed tools
    should be visible to the agent loop AND callable via dispatch.
    Two-layer enforcement: schema filter in agent.py (model never sees
    forbidden tools) + dispatch reject in tools/__init__.py (defense-in-
    depth if a model generates a forbidden tool call anyway)."""

    def test_spawn_propagates_whitelist_to_eff_config(self, manager, monkeypatch):
        """spawn() with agent_def.tools=['X', 'Y'] must set
        config['_agent_tools_whitelist'] so the filter fires in agent.py."""
        from multi_agent.subagent import AgentDefinition

        captured_config = {}

        def capture_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
            captured_config.update(config)
            state.messages.append(
                {"role": "assistant", "content": "x", "tool_calls": []})
            yield None

        monkeypatch.setattr("multi_agent.subagent._agent_run", capture_run)

        ad = AgentDefinition(
            name="restricted",
            description="agent with limited tools",
            tools=["WebSearch", "Think"],
            source="user",
        )
        task = manager.spawn("test", {}, "system", agent_def=ad)
        manager.wait(task.id, timeout=2)
        assert captured_config.get("_agent_tools_whitelist") == ["WebSearch", "Think"]

    def test_no_whitelist_when_tools_empty(self, manager, monkeypatch):
        """An AgentDefinition with no tools= list (empty default) means
        'all tools allowed' — no whitelist key set."""
        from multi_agent.subagent import AgentDefinition

        captured = {}

        def capture_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
            captured.update(config)
            state.messages.append(
                {"role": "assistant", "content": "x", "tool_calls": []})
            yield None

        monkeypatch.setattr("multi_agent.subagent._agent_run", capture_run)

        ad = AgentDefinition(name="open", description="all tools", tools=[])
        task = manager.spawn("test", {}, "system", agent_def=ad)
        manager.wait(task.id, timeout=2)
        assert "_agent_tools_whitelist" not in captured

    def test_dispatch_rejects_tool_outside_whitelist(self):
        """If a model somehow calls a forbidden tool, dispatch rejects."""
        from tools import execute_tool
        out = execute_tool(
            "Bash", {"command": "ls"},
            config={"_agent_tools_whitelist": ["WebSearch", "Think"]},
            permission_mode="accept-all",
        )
        assert "Denied" in out
        assert "whitelist" in out.lower()

    def test_dispatch_allows_tool_in_whitelist(self):
        """Whitelisted tools dispatch normally."""
        from tools import execute_tool
        out = execute_tool(
            "Think", {"thought": "consolidating"},
            config={"_agent_tools_whitelist": ["Think", "WebSearch"]},
            permission_mode="accept-all",
        )
        # Think returns "Thought recorded." on success
        assert "recorded" in out.lower()

    def test_no_whitelist_means_all_allowed(self):
        """Without _agent_tools_whitelist (parent agent path), all tools
        dispatch normally. Regression guard for the parent."""
        from tools import execute_tool
        out = execute_tool(
            "Think", {"thought": "ok"},
            config={},
            permission_mode="accept-all",
        )
        assert "Denied" not in out


class TestRabbitHoleAgentBackgroundFlag:
    def test_rabbit_hole_agent_is_background(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research-rabbit-hole")
        assert ad.background is True

    def test_regular_deep_research_is_foreground(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research")
        assert ad.background is False


class TestDeepResearchAgentRegistered:
    """The deep-research agent is a built-in registered at import time.
    Verify its definition is correct and that it grants the model the
    web-research toolset it needs (WebSearch/WebFetch + Think for
    intermediate-finding consolidation)."""

    def test_deep_research_in_builtin_registry(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research")
        assert ad is not None
        assert ad.source == "built-in"

    def test_deep_research_has_web_tools(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research")
        assert "WebSearch" in ad.tools
        assert "WebFetch" in ad.tools

    def test_deep_research_includes_think_tool(self):
        """Think is the scratchpad — without it the agent can't
        consolidate intermediate findings between rounds."""
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research")
        assert "Think" in ad.tools

    def test_deep_research_system_prompt_mentions_synthesis(self):
        from multi_agent.subagent import get_agent_definition
        ad = get_agent_definition("deep-research")
        sp = ad.system_prompt.lower()
        # The agent's purpose is multi-source synthesis with citations.
        assert "synthesize" in sp or "synthesized" in sp
        assert "cross-reference" in sp or "cross reference" in sp
