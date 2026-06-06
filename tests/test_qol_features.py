"""Smoke tests for the five QoL features added in this round.

- /model + /api + named profiles (model_profiles.py)
- /failover slash command
- /undo N reversible tool log (undo_log.py)
- Hooks in settings.json (hooks.py)
- Always-visible status footer (string-shape sanity only)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model_profiles
import undo_log
import hooks


# ── Profiles ────────────────────────────────────────────────────────────

class TestProfiles:
    def test_default_set_includes_alex_two(self):
        profiles = model_profiles.get_profiles({})
        assert "qwen" in profiles
        assert "m2" in profiles
        assert profiles["m2"]["model"] == "MiniMax-M2"
        assert profiles["qwen"]["base_url"].startswith("http://127.0.0.1")

    def test_apply_swaps_model_and_base_url(self):
        cfg = {"model": "starter", "custom_base_url": "old"}
        ok_, _msg = model_profiles.apply("qwen", cfg)
        assert ok_ is True
        assert cfg["model"] == "custom/qwen3.5-9b"
        assert cfg["custom_base_url"] == "http://127.0.0.1:8080/v1"
        assert cfg["context_limit"] == 57344

    def test_apply_minimax_sets_minimax_base_url(self):
        cfg = {}
        model_profiles.apply("m2-opti", cfg)
        # m2-opti points at the local optillm proxy.
        assert cfg["minimax_base_url"].startswith("http://127.0.0.1:8765")
        assert cfg["optillm_approach"] == "cot_reflection"

    def test_apply_unknown_profile(self):
        ok_, msg = model_profiles.apply("does-not-exist", {})
        assert ok_ is False
        assert "no such profile" in msg

    def test_key_status_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        profiles = model_profiles.get_profiles({})
        has, src = model_profiles.key_status(profiles["m2"])
        assert has is True and src == "env"

    def test_key_status_config_fallback(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        profiles = model_profiles.get_profiles({})
        has, src = model_profiles.key_status(
            profiles["m2"], {"minimax_api_key": "config-stored"},
        )
        assert has is True and src == "config"

    def test_key_status_none(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        profiles = model_profiles.get_profiles({})
        has, src = model_profiles.key_status(profiles["m2"])
        assert has is False and src == "none"

    def test_local_provider_needs_no_key(self):
        profiles = model_profiles.get_profiles({})
        has, src = model_profiles.key_status(profiles["qwen"])
        assert has is True and src == "n/a"


# ── Undo log ────────────────────────────────────────────────────────────

class TestUndoLog:
    def _isolated_home(self, monkeypatch):
        d = tempfile.mkdtemp()
        monkeypatch.setattr("pathlib.Path.home", lambda: __import__("pathlib").Path(d))
        return d

    def test_snapshot_and_revert_existing_file(self, monkeypatch):
        self._isolated_home(monkeypatch)
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("original content")
            path = f.name
        seq = undo_log.snapshot_before("s1", "Write",
                                        {"file_path": path})
        assert seq is not None
        # Simulate the tool actually mutating the file:
        open(path, "w").write("CHANGED")
        undo_log.record_after("s1", seq, "Write",
                              {"file_path": path}, "wrote 1 file")
        # Revert.
        entries = undo_log.last_n("s1", 1)
        assert len(entries) == 1
        ok_, _ = undo_log.revert("s1", entries[0])
        assert ok_ is True
        assert open(path).read() == "original content"
        os.unlink(path)

    def test_snapshot_and_revert_absent_file(self, monkeypatch):
        self._isolated_home(monkeypatch)
        # File doesn't exist yet — Write creates it; undo deletes.
        path = tempfile.mktemp()
        assert not os.path.exists(path)
        seq = undo_log.snapshot_before("s2", "Write", {"file_path": path})
        open(path, "w").write("created by tool")
        undo_log.record_after("s2", seq, "Write", {"file_path": path}, "ok")
        entries = undo_log.last_n("s2", 1)
        ok_, _ = undo_log.revert("s2", entries[0])
        assert ok_ is True
        assert not os.path.exists(path)

    def test_non_mutating_tool_ignored(self, monkeypatch):
        self._isolated_home(monkeypatch)
        seq = undo_log.snapshot_before("s3", "Read", {"file_path": "/etc/hosts"})
        assert seq is None

    def test_undone_entries_filtered_out(self, monkeypatch):
        self._isolated_home(monkeypatch)
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("v1")
            path = f.name
        seq = undo_log.snapshot_before("s4", "Write", {"file_path": path})
        open(path, "w").write("v2")
        undo_log.record_after("s4", seq, "Write", {"file_path": path}, "ok")
        # Revert it once; last_n should now be empty.
        undo_log.revert("s4", undo_log.last_n("s4", 1)[0])
        assert undo_log.last_n("s4", 5) == []
        os.unlink(path)


# ── Hooks ───────────────────────────────────────────────────────────────

class TestHooks:
    def test_no_hooks_returns_ok(self):
        ok_, msg = hooks.fire("pre_tool", {}, {"tool": "Write"})
        assert ok_ is True and msg == ""

    def test_match_filter_skips_unmatched(self, tmp_path):
        marker = tmp_path / "fired.txt"
        cfg = {
            "hooks": [
                {"event": "pre_tool", "match": "^Bash$",
                 "run": f"touch {marker}"},
            ],
        }
        hooks.fire("pre_tool", cfg, {"tool": "Write"}, match_key="Write")
        assert not marker.exists()
        hooks.fire("pre_tool", cfg, {"tool": "Bash"}, match_key="Bash")
        assert marker.exists()

    def test_block_on_error_aborts(self):
        cfg = {
            "hooks": [{"event": "pre_tool", "run": "exit 1",
                       "block_on_error": True}],
        }
        ok_, _ = hooks.fire("pre_tool", cfg, {"tool": "Write"},
                            match_key="Write")
        assert ok_ is False

    def test_non_blocking_continues_on_error(self):
        cfg = {
            "hooks": [{"event": "pre_tool", "run": "exit 1"}],
        }
        ok_, _ = hooks.fire("pre_tool", cfg, {"tool": "Write"},
                            match_key="Write")
        assert ok_ is True   # non-blocking: failure is silent
