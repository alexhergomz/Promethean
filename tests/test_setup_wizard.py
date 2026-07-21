"""Regression tests for ``commands.core.run_setup_wizard``.

The harness is llama.cpp-only, so the wizard just points Promethean at a
local (or any OpenAI-compatible) server: base URL, model, optional key. These
pin that flow and that it writes a config without crashing offline.

The wizard is interactive, so we mock ``input``, ``urllib.request.urlopen``,
and ``cc_config.save_config`` to keep the test fully offline.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import commands.core as _core


@pytest.fixture(autouse=True)
def _ensure_real_providers_module():
    """Defend against test pollution: other suites have historically replaced
    ``sys.modules["providers"]`` with a stub and not restored it."""
    import importlib
    saved = sys.modules.pop("providers", None)
    try:
        importlib.invalidate_caches()
        importlib.import_module("providers")
        yield
    finally:
        if saved is not None and getattr(saved, "PROVIDERS", None) is None:
            sys.modules.pop("providers", None)


def _run_wizard(monkeypatch, inputs: list[str], config: dict) -> dict:
    """Drive the wizard end-to-end with canned input + offline mocks."""
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    queue = iter(inputs)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(queue)
        except StopIteration as exc:
            raise EOFError(
                "wizard asked for more input than the test provided "
                f"(canned answers: {inputs!r})"
            ) from exc

    monkeypatch.setattr("builtins.input", fake_input)

    # Offline: the server probe returns an empty body (no model detected).
    class _FakeResponse:
        def read(self): return b""
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse())

    import cc_config
    monkeypatch.setattr(cc_config, "save_config", lambda *_a, **_kw: None)

    _core.run_setup_wizard(config)
    return config


def test_wizard_sets_backend_and_model(monkeypatch):
    config: dict = {}
    result = _run_wizard(
        monkeypatch,
        inputs=["http://127.0.0.1:9000/v1", "qwen3.5-9b", ""],
        config=config,
    )
    assert result["custom_base_url"] == "http://127.0.0.1:9000/v1"
    assert result["model"] == "custom/qwen3.5-9b"      # bare name gets the prefix
    assert "custom_api_key" not in result


def test_wizard_uses_defaults_on_blank(monkeypatch):
    config: dict = {}
    result = _run_wizard(monkeypatch, inputs=["", "", ""], config=config)
    assert result["custom_base_url"] == "http://127.0.0.1:8080/v1"
    assert result["model"] == "custom/qwen3.5-9b"


def test_wizard_keeps_explicit_custom_prefix(monkeypatch):
    config: dict = {}
    result = _run_wizard(monkeypatch, inputs=["", "custom/qwen3.5-27b", ""], config=config)
    assert result["model"] == "custom/qwen3.5-27b"


def test_wizard_stores_optional_api_key(monkeypatch):
    config: dict = {}
    result = _run_wizard(monkeypatch, inputs=["", "", "proxy-secret"], config=config)
    assert result["custom_api_key"] == "proxy-secret"
