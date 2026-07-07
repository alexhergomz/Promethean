"""Tests for llama-server context-window auto-detection (macOS review §12)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import server_autostart as S  # noqa: E402


class _FakeResp:
    def __init__(self, code, data):
        self.status_code = code
        self._data = data

    def json(self):
        return self._data


def _fake_httpx(monkeypatch, resp):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda url, timeout=3.0: resp)


def test_probe_nested_n_ctx(monkeypatch):
    _fake_httpx(monkeypatch, _FakeResp(200, {"default_generation_settings": {"n_ctx": 32768}}))
    assert S._probe_n_ctx("http://x") == 32768


def test_probe_top_level_n_ctx(monkeypatch):
    _fake_httpx(monkeypatch, _FakeResp(200, {"n_ctx": 8192}))
    assert S._probe_n_ctx("http://x") == 8192


def test_probe_missing_returns_none(monkeypatch):
    _fake_httpx(monkeypatch, _FakeResp(200, {"something_else": 1}))
    assert S._probe_n_ctx("http://x") is None


def test_probe_non_200_returns_none(monkeypatch):
    _fake_httpx(monkeypatch, _FakeResp(503, {"n_ctx": 1}))
    assert S._probe_n_ctx("http://x") is None


def test_sync_sets_when_unset(monkeypatch):
    monkeypatch.setattr(S, "_health_ok", lambda root, timeout=2.0: True)
    _fake_httpx(monkeypatch, _FakeResp(200, {"default_generation_settings": {"n_ctx": 16384}}))
    cfg = {"model": "custom/foo", "custom_base_url": "http://127.0.0.1:8080/v1"}
    S.sync_context_limit(cfg)
    assert cfg.get("context_limit") == 16384


def test_sync_preserves_explicit(monkeypatch):
    monkeypatch.setattr(S, "_health_ok", lambda root, timeout=2.0: True)
    _fake_httpx(monkeypatch, _FakeResp(200, {"default_generation_settings": {"n_ctx": 16384}}))
    cfg = {"model": "custom/foo", "custom_base_url": "http://127.0.0.1:8080/v1",
           "context_limit": 200000}
    S.sync_context_limit(cfg)
    assert cfg["context_limit"] == 200000


def test_sync_skips_remote(monkeypatch):
    monkeypatch.setattr(S, "_health_ok", lambda root, timeout=2.0: True)
    _fake_httpx(monkeypatch, _FakeResp(200, {"n_ctx": 16384}))
    cfg = {"model": "custom/foo", "custom_base_url": "http://10.0.0.5:8080/v1"}
    S.sync_context_limit(cfg)
    assert "context_limit" not in cfg


def test_sync_skips_non_custom(monkeypatch):
    cfg = {"model": "ollama/qwen2.5-coder", "custom_base_url": "http://127.0.0.1:8080/v1"}
    S.sync_context_limit(cfg)
    assert "context_limit" not in cfg


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
