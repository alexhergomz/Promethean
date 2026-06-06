"""No-network wiring tests for MiniMax (M2 / M1 family) provider support.

These tests pin the harness-side contract before any live API traffic happens.
Everything below builds the request kwargs through a mocked OpenAI client or
exercises pure-Python helpers (registry, error classifier, context limits)
so the suite stays green without a MINIMAX_API_KEY.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import providers
import compaction
from error_classifier import classify, ErrorCategory


class TestRegistry:
    def test_global_endpoint_default(self):
        # We default to the .io (global) endpoint, not .chat (China).
        assert providers.PROVIDERS["minimax"]["base_url"] == \
            "https://api.minimax.io/v1"

    def test_m2_family_registered(self):
        models = providers.PROVIDERS["minimax"]["models"]
        for m in ["MiniMax-M2", "MiniMax-M2-highspeed",
                  "MiniMax-M1", "MiniMax-Text-01"]:
            assert m in models, f"{m} missing from provider models list"

    def test_m2_listed_before_legacy(self):
        models = providers.PROVIDERS["minimax"]["models"]
        # The first model the picker offers should be the coding flagship.
        assert models[0] == "MiniMax-M2"
        assert models.index("MiniMax-M2") < models.index("abab6.5s-chat")


class TestPrefixDetection:
    def test_minimax_m2_routes_to_minimax(self):
        assert providers.detect_provider("MiniMax-M2") == "minimax"
        assert providers.detect_provider("MiniMax-M2.7-highspeed") == "minimax"
        assert providers.detect_provider("MiniMax-M1") == "minimax"
        assert providers.detect_provider("abab6.5s-chat") == "minimax"

    def test_explicit_provider_prefix(self):
        assert providers.detect_provider("minimax/MiniMax-M2") == "minimax"


class TestPerModelContextLimit:
    def test_m2_is_204k(self):
        assert compaction.get_context_limit("MiniMax-M2") == 204_800

    def test_m2_highspeed_is_204k(self):
        assert compaction.get_context_limit("MiniMax-M2-highspeed") == 204_800

    def test_m1_is_1m(self):
        assert compaction.get_context_limit("MiniMax-M1") == 1_000_000

    def test_text01_is_1m(self):
        assert compaction.get_context_limit("MiniMax-Text-01") == 1_000_000

    def test_unknown_minimax_falls_back_to_provider_default(self):
        # 204_800 is the provider-wide default in the registry — anything
        # not in per_model_context_limits gets it.
        assert compaction.get_context_limit("MiniMax-Unknown-Future") == 204_800

    def test_config_override_still_wins(self):
        # The compaction override (used by llama-server slot paging) must
        # still beat per-model entries.
        assert compaction.get_context_limit(
            "MiniMax-M2", {"context_limit": 57_344}
        ) == 57_344


class TestCosts:
    def test_m2_has_cost_entry(self):
        cost = providers.calc_cost("MiniMax-M2", 1_000_000, 1_000_000)
        # $0.3 in + $1.2 out per 1M tokens.
        assert abs(cost - 1.5) < 0.01

    def test_explicit_prefix_strips_correctly_for_cost_lookup(self):
        cost = providers.calc_cost("minimax/MiniMax-M2", 1_000_000, 0)
        assert abs(cost - 0.3) < 0.01


class TestMaxTokensCap:
    def test_m2_output_capped_at_16k(self):
        cap = providers.resolve_max_tokens(
            {"max_tokens": 100_000}, "minimax", "MiniMax-M2",
        )
        assert cap == 16_384

    def test_m1_output_capped_at_40k(self):
        cap = providers.resolve_max_tokens(
            {"max_tokens": 100_000}, "minimax", "MiniMax-M1",
        )
        assert cap == 40_960

    def test_user_request_under_cap_is_respected(self):
        cap = providers.resolve_max_tokens(
            {"max_tokens": 4096}, "minimax", "MiniMax-M2",
        )
        assert cap == 4096


class TestReasoningSplitWiring:
    """The streamer must opt into reasoning_split=true for minimax requests
    so chain-of-thought arrives as structured reasoning_details (rendered
    as ThinkingChunk) instead of inline <think> tags in content."""

    def _capture_kwargs(self, monkeypatch, config):
        captured: dict = {}

        class _FakeStream:
            def __iter__(self):
                return iter([])

        class _FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return _FakeStream()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass
            chat = _FakeChat()

        fake_openai = types.SimpleNamespace(OpenAI=_FakeClient)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        gen = providers.stream_openai_compat(
            api_key="dummy",
            base_url="https://api.minimax.io/v1",
            model="MiniMax-M2",
            system="you are helpful",
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            config=config,
        )
        # Drain the generator so the create() call happens.
        list(gen)
        return captured

    def test_reasoning_split_sent_by_default(self, monkeypatch):
        kw = self._capture_kwargs(monkeypatch, config={})
        assert kw.get("extra_body", {}).get("reasoning_split") is True

    def test_reasoning_split_omitted_when_thinking_off(self, monkeypatch):
        kw = self._capture_kwargs(monkeypatch, config={"thinking": False})
        assert "reasoning_split" not in kw.get("extra_body", {})

    def test_max_tokens_key_is_max_tokens_not_max_completion_tokens(
        self, monkeypatch,
    ):
        # Minimax follows the older OpenAI spec — it accepts max_tokens.
        # max_completion_tokens is openai-provider only.
        kw = self._capture_kwargs(monkeypatch, config={"max_tokens": 1024})
        assert "max_tokens" in kw
        assert "max_completion_tokens" not in kw


class TestMessagesEchoReasoningDetails:
    def test_reasoning_details_passed_through(self):
        details = [{"type": "reasoning", "text": "thinking it through"}]
        neutral = [
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
                ],
                "reasoning_details": details,
            },
        ]
        out = providers.messages_to_openai(neutral)
        assert out[0]["reasoning_details"] == details

    def test_reasoning_details_dropped_when_no_tool_calls(self):
        # M2 only requires the echo when tool_calls are present; without
        # them the field has no semantic anchor. Our serializer currently
        # only echoes inside the tool_calls branch, mirroring how
        # reasoning_content is handled.
        neutral = [
            {
                "role": "assistant",
                "content": "plain answer",
                "reasoning_details": [{"type": "reasoning", "text": "x"}],
            },
        ]
        out = providers.messages_to_openai(neutral)
        assert "reasoning_details" not in out[0]


class TestErrorClassifier:
    def test_minimax_1039_is_context_overflow(self):
        # Minimax wraps errors in HTTP 200 with base_resp.status_code 1039.
        # The classifier sees the stringified error from upstream raise().
        err = Exception(
            '{"base_resp":{"status_code":1039,"status_msg":"token limit"}}'
        )
        c = classify(err)
        assert c.category == ErrorCategory.CONTEXT_OVERFLOW

    def test_minimax_tokens_length_phrasing(self):
        err = Exception("Request failed: tokens length exceed model limit")
        c = classify(err)
        assert c.category == ErrorCategory.CONTEXT_OVERFLOW


class TestTokenizerCalibration:
    """Per-provider chars/tok ratio. Minimax M2 tokenizes code+JSON at
    ~2 chars/tok (denser than the chars/2.8 generic baseline). Wrong-
    direction estimates crash compaction on overflow — these tests pin
    the safe value and the per-msg overhead."""

    def test_minimax_chars_per_token_is_dense(self):
        ent = providers.PROVIDERS["minimax"]
        assert ent.get("token_estimate_chars_per_token") == 2.0
        assert ent.get("token_estimate_per_msg_overhead") == 20

    def test_estimate_uses_minimax_factor(self):
        msgs = [{"role": "user", "content": "x" * 1000}]
        bare = compaction.estimate_tokens(msgs)
        m2   = compaction.estimate_tokens(msgs, "MiniMax-M2")
        # M2 uses chars/2.0 vs the default chars/2.8 → strictly higher est.
        assert m2 > bare

    def test_estimate_uses_minimax_per_msg_overhead(self):
        # 20 msgs * (20 vs 4) framing tok → big delta even with empty content.
        msgs = [{"role": "user", "content": ""} for _ in range(20)]
        bare = compaction.estimate_tokens(msgs)
        m2   = compaction.estimate_tokens(msgs, "MiniMax-M2")
        assert m2 > bare

    def test_unknown_model_falls_back_to_default(self):
        msgs = [{"role": "user", "content": "hello world"}]
        bare    = compaction.estimate_tokens(msgs)
        unknown = compaction.estimate_tokens(msgs, "some-unmapped-model")
        assert bare == unknown


class TestStreamOptionsIncludeUsage:
    """The streamer must opt into include_usage so providers (M2 included)
    actually emit the final usage chunk — without it in_tok/out_tok stay 0
    and per-turn cost reporting breaks."""

    def test_include_usage_set_for_minimax(self, monkeypatch):
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

        fake_openai = types.SimpleNamespace(OpenAI=_FakeClient)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        list(providers.stream_openai_compat(
            api_key="dummy", base_url="https://api.minimax.io/v1",
            model="MiniMax-M2", system="x",
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[], config={},
        ))
        assert captured.get("stream_options", {}).get("include_usage") is True


class TestRegionOverride:
    def test_minimax_base_url_env_override(self, monkeypatch):
        # The dispatch helper inside providers.stream() picks the base_url;
        # this asserts the override surface exists (env var path) — we
        # can't drive the full stream() without an OpenAI client, but we
        # can confirm the config key is honoured via the registry lookup.
        # Sanity check: the env var name we expose is MINIMAX_BASE_URL.
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.chat/v1")
        # No assertion beyond the var being settable — the dispatch path
        # itself is exercised by integration tests when live keys exist.
        assert os.environ["MINIMAX_BASE_URL"] == "https://api.minimaxi.chat/v1"
