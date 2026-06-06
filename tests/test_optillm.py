"""OptiLLM integration tests.

The wiring is intentionally tiny: a single config knob (optillm_approach)
that gets forwarded as extra_body to the underlying OpenAI-compat
endpoint. The proxy itself (separate `pip install optillm` process) is
not part of these tests — they just verify the harness contract.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import providers
from commands.core import cmd_optillm, _OPTILLM_VALID


def _capture(monkeypatch, config):
    captured: dict = {}

    class _FakeStream:
        def __iter__(self): return iter([])
    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs); return _FakeStream()
    class _FakeChat:
        completions = _FakeCompletions()
    class _FakeClient:
        def __init__(self, *a, **kw): pass
        chat = _FakeChat()

    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(OpenAI=_FakeClient))
    list(providers.stream_openai_compat(
        api_key="dummy", base_url="http://localhost:8000/v1",
        model="MiniMax-M2", system="x",
        messages=[{"role": "user", "content": "hi"}],
        tool_schemas=[], config=config,
    ))
    return captured


class TestOptillmForwarding:
    def test_no_approach_no_field(self, monkeypatch):
        kw = _capture(monkeypatch, {})
        assert "optillm_approach" not in kw.get("extra_body", {})

    def test_approach_forwarded(self, monkeypatch):
        kw = _capture(monkeypatch, {"optillm_approach": "moa"})
        assert kw["extra_body"]["optillm_approach"] == "moa"

    def test_approach_coexists_with_reasoning_split(self, monkeypatch):
        # Minimax provider gets reasoning_split=True by default; the
        # optillm approach must be added alongside, not replace.
        kw = _capture(monkeypatch, {"optillm_approach": "mars"})
        eb = kw["extra_body"]
        assert eb["optillm_approach"] == "mars"
        assert eb["reasoning_split"] is True


class TestSlashCommand:
    def test_set_valid_approach(self):
        cfg = {}
        cmd_optillm("moa", None, cfg)
        assert cfg["optillm_approach"] == "moa"

    def test_clear_via_off(self):
        cfg = {"optillm_approach": "moa"}
        cmd_optillm("off", None, cfg)
        assert cfg["optillm_approach"] is None

    def test_rejects_unknown_slug(self):
        cfg = {}
        cmd_optillm("garbage-slug-xyz", None, cfg)
        # Unknown slug must NOT silently take effect.
        assert "optillm_approach" not in cfg

    def test_valid_slugs_cover_high_value_set(self):
        # Spot-check that the slugs we documented in the roadmap are accepted.
        for slug in ("moa", "mcts", "bon", "plansearch", "mars", "cepo"):
            assert slug in _OPTILLM_VALID
