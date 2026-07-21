"""
Backend support for Promethean.

The harness targets a single backend: llama.cpp (llama-server) over the
OpenAI-compatible protocol, exposed as the ``custom`` provider. Point it at
llama-server (the default loopback) or any other OpenAI-compatible server via
``config["custom_base_url"]``. Keeping one transport is a deliberate
maintainability choice — see README ("llama.cpp-only").

Model string formats:
  "custom/qwen3.5-9b"        explicit provider prefix (recommended)
  "qwen3.5-9b"               bare name → custom
"""
from __future__ import annotations
import json
import os
import urllib.request
from typing import Generator

# ── Provider registry ──────────────────────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    # The one backend: llama.cpp / llama-server (or any OpenAI-compatible
    # server) reached over the OpenAI protocol. base_url is read from
    # config["custom_base_url"]; it defaults to the local llama-server loopback.
    "custom": {
        "type":       "openai",
        "api_key_env": "CUSTOM_API_KEY",   # llama-server needs none; kept for auth'd proxies
        "base_url":   None,                # read from config["custom_base_url"]
        "context_limit": 128000,
        "models": [],
    },
}

# Local inference has no per-token cost. Kept as an (empty) mapping so callers
# that look up COSTS keep working; calc_cost returns 0 for everything.
COSTS: dict[str, tuple[float, float]] = {}

# Prefix → provider auto-detection. With a single backend there is nothing to
# disambiguate; detect_provider falls through to "custom".
_PREFIXES: list[tuple[str, str]] = []


def detect_provider(model: str) -> str:
    """Return the provider name for a model string.

    Supports the explicit 'provider/model' form; any bare name resolves to the
    sole backend, ``custom`` (llama.cpp)."""
    if "/" in model:
        return model.split("/", 1)[0]
    for prefix, pname in _PREFIXES:
        if model.lower().startswith(prefix):
            return pname
    return "custom"


def bare_model(model: str) -> str:
    """Strip 'provider/' prefix if present."""
    return model.split("/", 1)[1] if "/" in model else model


# ── Auto max_tokens cap ────────────────────────────────────────────────────

# Per-model hard output caps. Empty for the llama.cpp backend: a local server
# reports its own limit via /v1/models (see _fetch_custom_model_limit), and
# the user's configured max_tokens is otherwise respected as-is.
_MODEL_OUTPUT_LIMITS: dict[str, int] = {}

# Cache: base_url → {model_id → max_model_len}
_custom_ctx_cache: dict[str, dict[str, int]] = {}


def _fetch_custom_model_limit(base_url: str, model: str, api_key: str) -> int | None:
    """Query /v1/models on a custom (vLLM/etc.) endpoint for max_model_len.
    Returns None on any failure. Results are cached per base_url."""
    cache = _custom_ctx_cache.setdefault(base_url, {})
    if model in cache:
        return cache[model]
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key or 'dummy'}"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        for entry in data.get("data", []):
            mid = entry.get("id", "")
            limit = entry.get("max_model_len") or entry.get("context_window")
            if limit:
                cache[mid] = int(limit)
        return cache.get(model)
    except Exception:
        return None


def resolve_max_tokens(config: dict, provider: str, model: str,
                       base_url: str = "", api_key: str = "") -> int | None:
    """Return the effective max_tokens to use, auto-capping to the model's limit.

    Priority:
      1. Per-model hard limit from _MODEL_OUTPUT_LIMITS (known models)
      2. For 'custom' provider: query /v1/models for max_model_len
      3. Provider-level context_limit from PROVIDERS registry
      4. User's configured value unchanged (no cap available)

    Always respects the user's configured value as an upper bound — never
    increases it beyond what was requested.
    """
    requested = config.get("max_tokens")
    if not requested:
        return None  # let the caller use its own default

    # 1. Known per-model limit
    bare = bare_model(model)
    known = _MODEL_OUTPUT_LIMITS.get(bare)
    if known:
        return min(requested, known)

    # 2. Custom endpoint: query /v1/models
    if provider == "custom" and base_url:
        ctx_limit = _fetch_custom_model_limit(base_url, model, api_key)
        if ctx_limit:
            # Reserve 256 tokens so max_tokens never equals max_model_len exactly
            # (vLLM rejects max_tokens == max_model_len in some versions)
            safe = max(256, ctx_limit - 256)
            return min(requested, safe)

    # 3. Provider-level context limit (conservative: cap output to 1/2 context)
    prov_ctx = PROVIDERS.get(provider, {}).get("context_limit")
    if prov_ctx:
        cap = prov_ctx // 2
        return min(requested, cap)

    return requested


def get_api_key(provider_name: str, config: dict) -> str:
    prov = PROVIDERS.get(provider_name, {})
    # 1. Check config dict (e.g. config["kimi_api_key"])
    cfg_key = config.get(f"{provider_name}_api_key", "")
    if cfg_key:
        return cfg_key
    # 2. Check env var
    env_var = prov.get("api_key_env")
    if env_var:
        import os
        return os.environ.get(env_var, "")
    # 3. Hardcoded (for local providers)
    return prov.get("api_key", "")


def calc_cost(model: str, in_tok: int, out_tok: int) -> float:
    ic, oc = COSTS.get(bare_model(model), (0.0, 0.0))
    return (in_tok * ic + out_tok * oc) / 1_000_000


# ── Tool schema conversion ─────────────────────────────────────────────────

def tools_to_openai(tool_schemas: list) -> list:
    """Convert Anthropic-style tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            },
        }
        for t in tool_schemas
    ]


# ── Message format conversion ──────────────────────────────────────────────
#
# Internal "neutral" message format:
#   {"role": "user",      "content": "text"}
#   {"role": "assistant", "content": "text", "tool_calls": [
#       {"id": "...", "name": "...", "input": {...}}
#   ]}
#   {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

def messages_to_openai(messages: list, ollama_native_images: bool = False) -> list:
    """Convert neutral messages → OpenAI API format.

    Args:
        ollama_native_images: if True, forward the 'images' list in user messages
                              using Ollama's /api/chat native format (a bare base64
                              list on the message object).  Set this only when
                              targeting the Ollama backend.
                              If False (default), images are converted to the
                              OpenAI/Gemini multipart ``image_url`` format so they
                              reach vision-capable cloud models correctly.
    """
    result = []
    for m in messages:
        role = m["role"]

        if role == "user":
            content = m["content"]
            if ollama_native_images and m.get("images"):
                # Ollama /api/chat native: bare base64 list on the message
                msg_out = {"role": "user", "content": content, "images": m["images"]}
            elif not ollama_native_images and m.get("images"):
                # OpenAI / Gemini multipart vision format
                parts = [{"type": "text", "text": content}]
                for img_b64 in m["images"]:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                msg_out = {"role": "user", "content": parts}
            else:
                msg_out = {"role": "user", "content": content}
            result.append(msg_out)

        elif role == "assistant":
            msg: dict = {"role": "assistant", "content": m.get("content") or None}
            tcs = m.get("tool_calls", [])
            if tcs:
                msg["tool_calls"] = []
                for tc in tcs:
                    tc_msg = {
                        "id":   tc["id"],
                        "type": "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": json.dumps(tc["input"], ensure_ascii=False),
                        },
                    }
                    # Pass through provider-specific fields (e.g. Gemini thought_signature)
                    if tc.get("extra_content"):
                        tc_msg["extra_content"] = tc["extra_content"]
                    msg["tool_calls"].append(tc_msg)
                # DeepSeek v4 spec: when an assistant turn carries tool_calls,
                # its `reasoning_content` must be echoed back on subsequent
                # requests.  Benign for other OpenAI-compat providers — they
                # ignore unknown fields.
                rc = m.get("reasoning_content")
                if rc:
                    msg["reasoning_content"] = rc
                # MiniMax M2 spec: when reasoning_split=true was used and the
                # assistant turn carries tool_calls, `reasoning_details` must
                # be echoed verbatim so the model can pick up the same chain
                # of thought. Other OpenAI-compat providers ignore unknown
                # fields, so this is safe to send unconditionally.
                rd = m.get("reasoning_details")
                if rd:
                    msg["reasoning_details"] = rd
            result.append(msg)

        elif role == "tool":
            result.append({
                "role":         "tool",
                "tool_call_id": m["tool_call_id"],
                "content":      m["content"],
            })

    return result


# ── Streaming adapters ─────────────────────────────────────────────────────

class TextChunk:
    def __init__(self, text): self.text = text

class ThinkingChunk:
    def __init__(self, text): self.text = text

class AssistantTurn:
    """Completed assistant turn with text + tool_calls.

    ``reasoning_content`` carries model-emitted chain-of-thought surfaced via an
    OpenAI-compat ``delta.reasoning_content`` field (DeepSeek v4, Kimi K2
    Thinking, GLM-4.6, etc.).  DeepSeek v4 requires it to be echoed back when
    the assistant turn contains tool_calls; see ``messages_to_openai``.
    """
    def __init__(self, text, tool_calls, in_tokens, out_tokens,
                 cache_read_tokens=0, cache_write_tokens=0,
                 reasoning_content="", finish_reason="",
                 reasoning_details=None, usage_estimated=False):
        self.text                 = text
        self.tool_calls           = tool_calls   # list of {id, name, input}
        self.in_tokens            = in_tokens
        self.out_tokens           = out_tokens
        self.cache_read_tokens    = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.reasoning_content    = reasoning_content
        # True when in/out tokens were estimated from emitted bytes because
        # the upstream (typically a proxy) stripped usage. The UI surfaces a
        # "~" marker on /cost so the user knows the figure is approximate.
        self.usage_estimated      = usage_estimated
        # Structured reasoning blocks as provided by MiniMax when
        # extra_body.reasoning_split=true. Shape: [{"type": "reasoning",
        # "text": "..."}]. Kept verbatim so messages_to_openai can replay
        # it on subsequent turns — M2 requires this when the assistant
        # message also carries tool_calls.
        self.reasoning_details    = reasoning_details or []
        # finish_reason is the provider-reported termination cause, normalized
        # to OpenAI's vocabulary: "stop" | "tool_calls" | "length" | "" (unknown).
        # The agent loop uses "length" to detect output-token truncation and
        # trigger the auto-continue path.
        self.finish_reason        = finish_reason


def _openai_cached_read_tokens(usage) -> int:
    """Extract the OpenAI-compatible cached read-token count.

    OpenAI-compatible providers surface cache hits as
    `usage.prompt_tokens_details.cached_tokens`; there is no separate
    "cache creation" counter in the OpenAI schema (caching is implicit on
    their side), so the write-side is always 0 for this family of providers.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return int(getattr(details, "cached_tokens", 0) or 0)


def stream_openai_compat(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    """Stream from any OpenAI-compatible API. Yields TextChunk, then AssistantTurn."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

    oai_messages = [{"role": "system", "content": system}] + messages_to_openai(messages)

    kwargs: dict = {
        "model":    model,
        "messages": oai_messages,
        "stream":   True,
    }

    # If pointed at an Ollama or LM Studio server (rather than llama-server),
    # forward the context window as num_ctx so the model isn't capped at the
    # backend default. Other OpenAI-compatible servers ignore unknown options.
    _is_ollama_port   = "11434" in base_url
    _is_lmstudio_port = "1234" in base_url and ("localhost" in base_url or "127.0.0.1" in base_url)
    if _is_ollama_port or _is_lmstudio_port:
        ctx_limit = config.get("context_limit") if isinstance(config.get("context_limit"), int) else None
        kwargs["extra_body"] = {"options": {"num_ctx": ctx_limit or 128000}}

    if tool_schemas and not config.get("no_tools"):
        kwargs["tools"] = tools_to_openai(tool_schemas)
        # "auto" requires vLLM --enable-auto-tool-choice; omit if server doesn't support it
        if not config.get("disable_tool_choice"):
            kwargs["tool_choice"] = "auto"
    _prov = detect_provider(model)

    # llama-server slot pinning (custom provider only).
    # SubAgentManager assigns _slot_id when slot paging is enabled, so the
    # subagent's chat completions go to a specific KV-cache slot rather than
    # whichever slot llama-server's prefix matcher picks. This makes the
    # parking lifecycle on subagent finish deterministic. Other providers
    # ignore the field if it appears in extra_body, so this is safe to forward
    # only when the provider is `custom` (i.e. our local llama-server).
    if _prov == "custom" and config.get("_slot_id") is not None:
        kwargs.setdefault("extra_body", {})["id_slot"] = config["_slot_id"]

    # Request the final usage chunk. Many OpenAI-compatible servers only emit
    # token usage when stream_options.include_usage=True. Without it,
    # in_tok/out_tok stay at 0 and per-turn accounting breaks. Safe to
    # set on all openai-compat providers; unknown providers ignore it.
    kwargs.setdefault("stream_options", {})["include_usage"] = True
    _effective_mt = resolve_max_tokens(config, _prov, model, base_url, api_key)
    if _effective_mt:
        # Further cap by provider-level max_completion_tokens if present
        prov_cap = PROVIDERS.get(_prov, {}).get("max_completion_tokens")
        val = min(_effective_mt, prov_cap) if prov_cap else _effective_mt
        # Newer OpenAI models (o1/o3/o4/gpt-5 family) dropped max_tokens in favour of
        # max_completion_tokens.  Use max_completion_tokens for the openai provider so
        # all current and future OpenAI models work without per-model special-casing.
        # All other OpenAI-compatible providers (Ollama, vLLM, Gemini, etc.) still
        # accept max_tokens, so we keep the old key for them.
        if _prov == "openai":
            kwargs["max_completion_tokens"] = val
        else:
            kwargs["max_tokens"] = val

    text            = ""
    reasoning_text  = ""
    # MiniMax structured reasoning, accumulated across stream chunks.
    # Each entry: {"type": "reasoning", "text": "...concatenated..."}.
    reasoning_details_buf: list[dict] = []
    tool_buf: dict = {}   # index → {id, name, args_str}
    in_tok = out_tok = 0
    cache_read_tok = cache_write_tok = 0
    finish_reason   = ""

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            # usage-only chunk (some providers send this last)
            if hasattr(chunk, "usage") and chunk.usage:
                in_tok  = chunk.usage.prompt_tokens
                out_tok = chunk.usage.completion_tokens
                cache_read_tok = _openai_cached_read_tokens(chunk.usage) or cache_read_tok
            continue

        choice = chunk.choices[0]
        delta  = choice.delta
        # finish_reason is set on the final chunk for the choice; capture it
        # so the agent loop can detect "length" (max_tokens truncation) and
        # trigger the auto-continue recovery path.
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason

        # MiniMax structured reasoning. Shape per chunk:
        #   delta.reasoning_details = [{"type": "reasoning.text", "index": 0, "text": "...partial..."}]
        # MiniMax mirrors the same text on delta.reasoning_content within
        # the same chunk — emitting both would double the thinking output.
        # When details is present we take it as the source of truth and
        # suppress the parallel reasoning_content emit.
        rd_delta = getattr(delta, "reasoning_details", None)
        if rd_delta is None and hasattr(delta, "model_extra"):
            rd_delta = (delta.model_extra or {}).get("reasoning_details")

        # Some providers (DeepSeek v4, Kimi K2 Thinking, GLM-4.6) stream
        # chain-of-thought on a sibling `reasoning_content` field before any
        # visible content.  Surface it as ThinkingChunk so the UI renders it
        # consistently with Anthropic extended-thinking / Ollama thinking.
        reasoning_delta = getattr(delta, "reasoning_content", None)
        if reasoning_delta and not rd_delta:
            reasoning_text += reasoning_delta
            yield ThinkingChunk(reasoning_delta)

        if rd_delta:
            for pos, item in enumerate(rd_delta):
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                elif not isinstance(item, dict):
                    item = {"type": "reasoning", "text": str(item)}
                txt = item.get("text") or ""
                # Prefer the embedded `index` field when present; some
                # providers stream multiple parallel reasoning streams and
                # rely on it to demultiplex. Fall back to list position.
                buf_idx = item.get("index", pos)
                while len(reasoning_details_buf) <= buf_idx:
                    reasoning_details_buf.append(
                        {"type": item.get("type", "reasoning"), "text": ""}
                    )
                reasoning_details_buf[buf_idx]["text"] += txt
                if txt:
                    reasoning_text += txt
                    yield ThinkingChunk(txt)

        if delta.content:
            text += delta.content
            yield TextChunk(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_buf:
                    tool_buf[idx] = {"id": "", "name": "", "args": "", "extra_content": None}
                if tc.id:
                    tool_buf[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_buf[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_buf[idx]["args"] += tc.function.arguments
                # Capture extra_content (e.g. Gemini thought_signature)
                extra = getattr(tc, "extra_content", None)
                if extra:
                    tool_buf[idx]["extra_content"] = extra

        # Some providers include usage in the last chunk
        if hasattr(chunk, "usage") and chunk.usage:
            in_tok  = chunk.usage.prompt_tokens  or in_tok
            out_tok = chunk.usage.completion_tokens or out_tok
            cache_read_tok = _openai_cached_read_tokens(chunk.usage) or cache_read_tok

    tool_calls = []
    for idx in sorted(tool_buf):
        v = tool_buf[idx]
        try:
            inp = json.loads(v["args"]) if v["args"] else {}
        except json.JSONDecodeError:
            inp = {"_raw": v["args"]}
        tc_entry = {"id": v["id"] or f"call_{idx}", "name": v["name"], "input": inp}
        if v.get("extra_content"):
            tc_entry["extra_content"] = v["extra_content"]
        tool_calls.append(tc_entry)

    if os.environ.get("CC_DEBUG_TRUNC"):
        import sys as _sys
        _sys.stderr.write(
            f"[trace] finish_reason={finish_reason!r} "
            f"text_len={len(text)} tool_calls={len(tool_calls)} "
            f"malformed={[t['name'] for t in tool_calls if '_raw' in t.get('input', {})]}\n"
        )
    # Usage fallback. Some proxies (notably OptiLLM) don't forward
    # `stream_options.include_usage` to the upstream nor re-emit the
    # final usage block, so in_tok/out_tok stay 0. Cost tracking silently
    # breaks. Detect and estimate from emitted bytes using the per-provider
    # chars/tok ratio (compaction's calibrated factor — see
    # _chars_per_token_for in compaction.py). Mark `usage_estimated=True`
    # on the turn so the UI can flag the figure as approximate.
    usage_estimated = False
    if in_tok == 0 and out_tok == 0:
        try:
            import compaction as _cmp
            cpt = _cmp._chars_per_token_for(model)
            pmo = _cmp._per_message_overhead_for(model)
            in_chars = len(system) + sum(
                len(str(m.get("content", "")))
                + sum(len(str(v)) for tc in (m.get("tool_calls") or [])
                      for v in (tc.get("input") or {}).values())
                for m in messages
            )
            out_chars = len(text) + len(reasoning_text) + sum(
                len(str(t.get("input", ""))) for t in tool_calls
            )
            in_tok  = max(1, int(in_chars  / cpt) + pmo * (len(messages) + 1))
            out_tok = max(1, int(out_chars / cpt) + pmo)
            usage_estimated = True
        except Exception:
            pass
    yield AssistantTurn(
        text, tool_calls, in_tok, out_tok, cache_read_tok, cache_write_tok,
        reasoning_content=reasoning_text,
        finish_reason=finish_reason,
        reasoning_details=reasoning_details_buf,
        usage_estimated=usage_estimated,
    )


def _recover_text_tool_calls(turn: "AssistantTurn", tool_schemas: list, config: dict) -> None:
    """Recover tool calls a model wrote as text and attach them to ``turn``.

    Mutates ``turn`` in place: on success, ``turn.tool_calls`` is populated and
    ``turn.text`` is stripped of the raw JSON/XML block. When a tool-shaped
    block names an *unregistered* tool (likely hallucinated), print a visible
    warning so the failure is never silent. Controlled by config
    ``recover_text_tool_calls`` (default on); any parser error is swallowed so a
    malformed response can never break the turn.
    """
    if not config.get("recover_text_tool_calls", True):
        return
    if not turn.text or not tool_schemas:
        return
    try:
        import tool_call_recovery as _tcr
        valid = {t.get("name") for t in tool_schemas if t.get("name")}
        recovered, cleaned, unknown = _tcr.recover_tool_calls(turn.text, valid)
    except Exception:
        return

    if recovered:
        turn.tool_calls = recovered
        turn.text = cleaned
        if os.environ.get("CC_DEBUG_TRUNC"):
            import sys as _sys
            _sys.stderr.write(
                f"[trace] recovered {len(recovered)} tool call(s) from text: "
                f"{[c['name'] for c in recovered]}\n"
            )
    elif unknown:
        # Looks like a tool call but names a tool we don't have — surface it
        # instead of the historical silent no-op.
        import sys as _sys
        _sys.stderr.write(
            f"\n\033[33m[warn] The model emitted what looks like a call to "
            f"unknown tool(s): {', '.join(unknown)}. Not dispatched — the model "
            f"may be hallucinating a tool, or emitting an unsupported call "
            f"format.\033[0m\n"
        )


def stream(
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    """
    Unified streaming entry point.
    Auto-detects provider from model string.
    Yields: TextChunk | ThinkingChunk | AssistantTurn

    Wraps every provider with:
      - Circuit breaker: fails fast when a provider has repeated errors.
      - Structured logging: logs api_call_start / api_call_done / api_call_error.
    """
    import logging_utils as _log
    import circuit_breaker as _cb

    provider_name = detect_provider(model)
    model_name    = bare_model(model)
    prov          = PROVIDERS.get(provider_name, PROVIDERS["custom"])
    api_key       = get_api_key(provider_name, config)
    session_id    = config.get("_session_id", "default")

    # ── Circuit breaker gate ───────────────────────────────────────────────
    breaker = _cb.get_breaker(provider_name, config)
    if not breaker.allow_request():
        raise _cb.CircuitOpenError(
            f"Circuit breaker OPEN for provider '{provider_name}'. "
            f"Cooldown: {breaker.cooldown:.0f}s. Use /circuit reset {provider_name} to force-close."
        )

    _log.debug("api_call_start", session_id=session_id,
               provider=provider_name, model=model_name)

    # ── Build inner generator ──────────────────────────────────────────────
    # One backend: llama.cpp / any OpenAI-compatible server. base_url comes
    # from config["custom_base_url"] (or the CUSTOM_BASE_URL env var); a named
    # profile may also seed prov["base_url"].
    import os as _os
    base_url = (config.get("custom_base_url")
                or _os.environ.get("CUSTOM_BASE_URL", "")
                or prov.get("base_url") or "")
    if not base_url:
        raise ValueError(
            "No backend configured. Point Promethean at llama-server (or any "
            "OpenAI-compatible server): set CUSTOM_BASE_URL, or run "
            "/config custom_base_url=http://127.0.0.1:8080/v1"
        )
    inner = stream_openai_compat(
        api_key, base_url, model_name, system, messages, tool_schemas, config
    )

    # ── Yield with failure tracking ────────────────────────────────────────
    try:
        for event in inner:
            if isinstance(event, AssistantTurn):
                breaker.record_success()
                # Parse-from-text tool-call fallback. Many local models
                # (Qwen-Coder via llama.cpp/Ollama, etc.) emit tool calls as
                # JSON/XML in the message *content* instead of native
                # tool_calls. Without this the turn silently no-ops. Recover
                # any tool-shaped blocks that name a registered tool.
                if not event.tool_calls:
                    _recover_text_tool_calls(event, tool_schemas, config)
                _log.info("api_call_done", session_id=session_id,
                          provider=provider_name, model=model_name,
                          in_tokens=event.in_tokens, out_tokens=event.out_tokens,
                          cache_read_tokens=getattr(event, 'cache_read_tokens', 0),
                          cache_write_tokens=getattr(event, 'cache_write_tokens', 0))
            yield event
    except Exception as exc:
        breaker.record_failure()
        _log.error("api_call_error", session_id=session_id,
                   provider=provider_name, model=model_name,
                   error_type=type(exc).__name__, error=str(exc)[:200])
        raise

