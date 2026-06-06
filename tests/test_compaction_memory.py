"""Tests for memories-near-compaction (consolidate_for_compaction)."""
import pytest

import memory.store as _store
import memory.consolidator as consolidator
from memory.consolidator import consolidate_for_compaction
from memory.store import load_entries


@pytest.fixture(autouse=True)
def redirect_memory_dirs(tmp_path, monkeypatch):
    user_mem = tmp_path / "user_memory"
    user_mem.mkdir()
    proj_mem = tmp_path / "project_memory"
    proj_mem.mkdir()
    monkeypatch.setattr(_store, "USER_MEMORY_DIR", user_mem)
    monkeypatch.setattr(_store, "get_project_memory_dir", lambda: proj_mem)


def _stub_extract(monkeypatch, mems):
    monkeypatch.setattr(consolidator, "_extract_memories", lambda *a, **k: mems)


def _msgs(n):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} content"}
        for i in range(n)
    ]


def test_saves_to_project_scope(monkeypatch):
    _stub_extract(monkeypatch, [{
        "name": "chose_hqq", "type": "project",
        "description": "use HQQ for 4-bit",
        "content": "Picked HQQ over bnb on RDNA2.",
    }])
    saved = consolidate_for_compaction(_msgs(10), {"model": "x"})
    assert saved == ["chose_hqq"]
    proj = load_entries(scope="project")
    assert any(e.name == "chose_hqq" for e in proj)
    # source tag identifies its origin
    assert all(e.source == "compaction" for e in proj if e.name == "chose_hqq")


def test_skips_when_too_few_messages(monkeypatch):
    # Should not even call extraction with < 4 textual turns.
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return []
    monkeypatch.setattr(consolidator, "_extract_memories", spy)
    assert consolidate_for_compaction(_msgs(2), {"model": "x"}) == []
    assert called["n"] == 0


def test_extraction_failure_is_swallowed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr(consolidator, "_extract_memories", boom)
    assert consolidate_for_compaction(_msgs(10), {"model": "x"}) == []


def test_empty_extraction_saves_nothing(monkeypatch):
    _stub_extract(monkeypatch, [])
    assert consolidate_for_compaction(_msgs(10), {"model": "x"}) == []
    assert load_entries(scope="project") == []


def test_compaction_memory_default_enabled():
    from cc_config import DEFAULTS
    assert DEFAULTS["compaction_memory"] is True
