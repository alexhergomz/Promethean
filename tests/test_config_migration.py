"""Tests for the rebrand config-dir migration (~/.cheetahclaws → ~/.promethean)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cc_config


def _point_dirs(monkeypatch, tmp_path):
    """Redirect both the new and legacy config dirs into tmp_path."""
    new = tmp_path / ".promethean"
    legacy = tmp_path / ".cheetahclaws"
    monkeypatch.setattr(cc_config, "CONFIG_DIR", new)
    monkeypatch.setattr(cc_config, "_LEGACY_CONFIG_DIR", legacy)
    return new, legacy


def test_migrates_when_only_legacy_exists(monkeypatch, tmp_path):
    new, legacy = _point_dirs(monkeypatch, tmp_path)
    legacy.mkdir()
    (legacy / "config.json").write_text('{"model": "x", "key": "secret"}')
    (legacy / "memory").mkdir()
    (legacy / "memory" / "note.md").write_text("remember this")

    assert cc_config.migrate_legacy_config_dir() is True
    assert new.is_dir()
    assert not legacy.exists()
    # Content (incl. would-be API keys) preserved verbatim.
    assert (new / "config.json").read_text() == '{"model": "x", "key": "secret"}'
    assert (new / "memory" / "note.md").read_text() == "remember this"


def test_noop_when_new_already_exists(monkeypatch, tmp_path):
    new, legacy = _point_dirs(monkeypatch, tmp_path)
    new.mkdir()
    (new / "config.json").write_text("new")
    legacy.mkdir()
    (legacy / "config.json").write_text("legacy")

    assert cc_config.migrate_legacy_config_dir() is False
    # New dir untouched; legacy left alone (no clobber).
    assert (new / "config.json").read_text() == "new"
    assert legacy.exists()


def test_noop_when_no_legacy(monkeypatch, tmp_path):
    new, legacy = _point_dirs(monkeypatch, tmp_path)
    assert cc_config.migrate_legacy_config_dir() is False
    assert not new.exists()


def test_idempotent(monkeypatch, tmp_path):
    new, legacy = _point_dirs(monkeypatch, tmp_path)
    legacy.mkdir()
    (legacy / "config.json").write_text("once")
    assert cc_config.migrate_legacy_config_dir() is True
    # Second call is a no-op (new exists now).
    assert cc_config.migrate_legacy_config_dir() is False
    assert (new / "config.json").read_text() == "once"
