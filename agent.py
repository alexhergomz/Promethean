"""Core agent loop: neutral message format, multi-provider streaming."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Generator

from tool_registry import get_tool_schemas
from tools import execute_tool
import tools as _tools_init  # ensure built-in tools are registered on import
from providers import stream, AssistantTurn, TextChunk, ThinkingChunk, detect_provider
from compaction import maybe_compact, estimate_tokens, get_context_limit, compact_messages, sanitize_history
import logging_utils as _log
import quota as _quota
from circuit_breaker import CircuitOpenError as _CircuitOpenError
import runtime

# ── Re-export event types (used by promethean.py) ────────────────────────
__all__ = [
    "AgentState", "run",
    "TextChunk", "ThinkingChunk",
    "ToolStart", "ToolEnd", "TurnDone", "PermissionRequest",
]


@dataclass
class AgentState:
    """Mutable session state. messages use the neutral provider-independent format."""
    messages: list = field(default_factory=list)
    total_input_tokens:  int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens:  int = 0
    total_cache_write_tokens: int = 0
    turn_count: int = 0
    # Decode throughput of the most recent model turn (output tokens / sec),
    # measured from first output token to end of stream. 0 until first turn.
    last_tps: float = 0.0


@dataclass
class ToolStart:
    name:   str
    inputs: dict

@dataclass
class ToolEnd:
    name:      str
    result:    str
    permitted: bool = True

@dataclass
class TurnDone:
    input_tokens:  int
    output_tokens: int

@dataclass
class PermissionRequest:
    description: str
    granted: bool = False


# ── Agent loop ─────────────────────────────────────────────────────────────

def run(
    user_message: str,
    state: AgentState,
    config: dict,
    system_prompt: str,
    depth: int = 0,
    cancel_check=None,
) -> Generator:
    """
    Multi-turn agent loop (generator).
    Yields: TextChunk | ThinkingChunk | ToolStart | ToolEnd |
            PermissionRequest | TurnDone

    Args:
        depth: sub-agent nesting depth, 0 for top-level
        cancel_check: callable returning True to abort the loop early
    """
    # Append user turn in neutral format
    user_msg = {"role": "user", "content": user_message}
    # Attach pending image from /image command if present
    sctx = runtime.get_ctx(config)
    pending_img = sctx.pending_image
    sctx.pending_image = None
    if pending_img:
        user_msg["images"] = [pending_img]
    state.messages.append(user_msg)
    try:
        import hooks as _hooks
        _hooks.fire("user_message", {**config, "_session_id": config.get("_session_id", "default")},
                    {"message": user_message[:4000]})
    except Exception:
        pass

    # Inject runtime metadata into config so tools (e.g. Agent) can access it
    config = {**config, "_depth": depth, "_system_prompt": system_prompt}
    session_id = config.get("_session_id", "default")

    # Wire up structured logging from config (idempotent, cheap)
    _log.configure_from_config(config)

    while True:
        if cancel_check and cancel_check():
            return
        state.turn_count += 1
        assistant_turn: AssistantTurn | None = None

        # Compact context if approaching window limit
        try:
            maybe_compact(state, config)
        except Exception as _compact_err:
            _log.warn("compact_failed", error=str(_compact_err))

        # Enforce tool_calls ↔ tool-response pairing before every API call.
        # Defends against compaction artifacts, crashed tool execs, or any
        # other source of orphan 'tool' messages that OpenAI-compatible
        # providers (DeepSeek et al.) reject with a 400.
        _before_len = len(state.messages)
        state.messages = sanitize_history(state.messages)
        if len(state.messages) != _before_len:
            _log.warn("history_sanitized",
                      session_id=session_id,
                      removed=_before_len - len(state.messages))

        # ── Quota check — before spending tokens ──────────────────────────
        try:
            _quota.check_quota(session_id, config)
        except _quota.QuotaExceeded as qe:
            _log.warn("quota_exceeded", session_id=session_id, reason=qe.reason)
            yield TextChunk(f"\n[Quota exceeded — {qe.reason}]\n")
            break

        # Stream from provider — retry on ANY error (never crash the session).
        # Failover ladder: when retries are exhausted (or the error is non-
        # retryable, e.g. AUTH on the primary), advance to the next model in
        # config["failover_models"] and reset the retry counter. The switch
        # sticks for the rest of the session (updates config["model"]); user
        # can swap back via /model.
        _primary = config["model"]
        _ladder: list[str] = [_primary]
        for _fm in (config.get("failover_models") or []):
            if _fm and _fm not in _ladder:
                _ladder.append(_fm)
        _ladder_idx = 0
        max_retries = 3
        attempt = 0
        while True:
            try:
                # Honour subagent tool whitelist: AgentDefinition.tools
                # filtered through SubAgentManager.spawn → eff_config.
                # Without this, sub-agents see every registered tool which
                # overwhelms small models on tool selection. The whitelist
                # IS the sandboxing mechanism — without enforcement here,
                # the rabbit-hole agent could call Bash/Edit/Write etc.
                _all_schemas = get_tool_schemas()
                _whitelist = config.get("_agent_tools_whitelist")
                if _whitelist:
                    _allowed = set(_whitelist)
                    _filtered = [s for s in _all_schemas if s.get("name") in _allowed]
                    # Defensive: if filtering produces no schemas, fall
                    # back to all (a misconfigured whitelist shouldn't
                    # silently disable all tools).
                    _tool_schemas = _filtered if _filtered else _all_schemas
                else:
                    _tool_schemas = _all_schemas

                _t_first: float = 0.0  # set on first output token (decode start)
                for event in stream(
                    model=config["model"],
                    system=system_prompt,
                    messages=state.messages,
                    tool_schemas=_tool_schemas,
                    config=config,
                ):
                    if isinstance(event, (TextChunk, ThinkingChunk)):
                        if _t_first == 0.0:
                            _t_first = time.perf_counter()
                        yield event
                    elif isinstance(event, AssistantTurn):
                        assistant_turn = event
                        # Decode throughput for the status footer: output
                        # tokens since the first token / elapsed decode time.
                        _elapsed = time.perf_counter() - _t_first
                        if _t_first and _elapsed > 0 and event.out_tokens:
                            state.last_tps = event.out_tokens / _elapsed
                        # Record usage for quota tracking
                        _quota.record_usage(
                            session_id, config["model"],
                            event.in_tokens, event.out_tokens,
                        )
                break  # success — exit retry loop

            except _CircuitOpenError as e:
                _log.warn("circuit_open_skip", session_id=session_id,
                          error=str(e)[:200])
                yield TextChunk(f"\n[{e}]\n")
                return  # circuit manages its own cooldown — don't retry

            except Exception as e:
                from error_classifier import classify as _classify_err
                cerr = _classify_err(e)

                _terminal = attempt >= max_retries or not cerr.retryable
                if _terminal and _ladder_idx + 1 < len(_ladder):
                    _ladder_idx += 1
                    _old = config["model"]
                    config["model"] = _ladder[_ladder_idx]
                    attempt = 0
                    _log.warn("failover", session_id=session_id,
                              from_model=_old, to=config["model"],
                              category=cerr.category.value,
                              error=_truncate_err(str(e)))
                    yield TextChunk(
                        f"\n[Failover {_old} → {config['model']} after "
                        f"{cerr.category.value}: {_truncate_err(str(e))}]\n"
                    )
                    continue

                if _terminal:
                    _log.error("api_failed", session_id=session_id,
                               error_type=type(e).__name__,
                               category=cerr.category.value,
                               error=_truncate_err(str(e)))
                    hint = f" Hint: {cerr.hint}" if cerr.hint else ""
                    yield TextChunk(f"\n[Failed — {type(e).__name__}: {_truncate_err(str(e))}.{hint}]\n")
                    break

                if cerr.should_compress:
                    _force_compact(state, config)
                    yield TextChunk(f"\n[Context too long — compacted and retrying (attempt {attempt+1}/{max_retries})]\n")
                    attempt += 1
                    continue

                backoff = int(2 ** (attempt + 1) * cerr.backoff_multiplier)
                backoff = min(backoff, 30)
                _log.warn("api_retry", session_id=session_id,
                          attempt=attempt + 1, max_retries=max_retries,
                          category=cerr.category.value,
                          error_type=type(e).__name__,
                          error=_truncate_err(str(e)),
                          backoff_s=backoff)
                yield TextChunk(f"\n[Retry {attempt+1}/{max_retries} after {backoff}s — {cerr.category.value}: {_truncate_err(str(e))}]\n")
                time.sleep(backoff)
                attempt += 1

        if assistant_turn is None:
            break

        # Record assistant turn in neutral format
        _assistant_msg = {
            "role":       "assistant",
            "content":    assistant_turn.text,
            "tool_calls": assistant_turn.tool_calls,
        }
        # DeepSeek v4 requires reasoning_content to be echoed back on
        # subsequent requests when the turn contains tool_calls.  Storing it
        # on the neutral history lets messages_to_openai pass it through.
        _rc = getattr(assistant_turn, "reasoning_content", "")
        if _rc and assistant_turn.tool_calls:
            _assistant_msg["reasoning_content"] = _rc
        # MiniMax M2 wants reasoning_details echoed verbatim on tool_calls
        # turns; persist them on the neutral message so messages_to_openai
        # passes them through.
        _rd = getattr(assistant_turn, "reasoning_details", None)
        if _rd and assistant_turn.tool_calls:
            _assistant_msg["reasoning_details"] = _rd
        state.messages.append(_assistant_msg)

        state.total_input_tokens  += assistant_turn.in_tokens
        state.total_output_tokens += assistant_turn.out_tokens
        state.total_cache_read_tokens  += getattr(assistant_turn, 'cache_read_tokens', 0)
        state.total_cache_write_tokens += getattr(assistant_turn, 'cache_write_tokens', 0)
        # Sticky bit: once any turn in the session arrived with estimated
        # usage (proxy stripped it), the session totals are no longer ground
        # truth. UI marks them with a leading "~". Cleared by /clear.
        if getattr(assistant_turn, "usage_estimated", False):
            state._usage_estimated = True
        try:
            import hooks as _hooks
            _hooks.fire("turn_done", config, {
                "in_tokens":  assistant_turn.in_tokens,
                "out_tokens": assistant_turn.out_tokens,
                "model":      config.get("model", ""),
            })
        except Exception:
            pass
        yield TurnDone(assistant_turn.in_tokens, assistant_turn.out_tokens)

        # Detect output-token truncation and prep recovery. May mutate
        # assistant_turn.tool_calls / _assistant_msg in place by stripping
        # malformed (truncated-args) tool calls.
        _trunc_continue, _trunc_hint = _handle_length_truncation(
            state, assistant_turn, _assistant_msg, config,
        )

        if not assistant_turn.tool_calls:
            if _trunc_continue and _trunc_hint:
                # Output cut mid-text with no usable tool calls — inject the
                # continuation hint and loop instead of returning a half-baked
                # response to the user.
                cap = int(config.get("max_continuations", 3) or 0)
                yield TextChunk(
                    f"\n[Output truncated at max_tokens — auto-continuing "
                    f"({getattr(state, '_continuations', 1)}/{cap})]\n"
                )
                state.messages.append({"role": "user", "content": _trunc_hint})
                continue
            break   # No tools → conversation turn complete

        # ── Execute tools (parallel when safe) ────────────────────────────
        tool_calls = assistant_turn.tool_calls

        # Check permissions first (must be sequential — may prompt user)
        permissions: dict[str, bool] = {}
        for tc in tool_calls:
            permitted = _check_permission(tc, config)
            if not permitted:
                if config.get("permission_mode") == "plan":
                    permitted = False
                else:
                    req = PermissionRequest(description=_permission_desc(tc))
                    yield req
                    permitted = req.granted
            permissions[tc["id"]] = permitted

        # Determine which tools can run in parallel
        from tool_registry import get_tool as _get_tool
        parallel_batch = []
        sequential_batch = []
        for tc in tool_calls:
            if not permissions[tc["id"]]:
                sequential_batch.append(tc)
                continue
            tdef = _get_tool(tc["name"])
            if tdef and tdef.concurrent_safe and len(tool_calls) > 1:
                parallel_batch.append(tc)
            else:
                sequential_batch.append(tc)

        def _exec_one(tc):
            """Execute a single tool call, return (tc, result, permitted)."""
            tid = tc["id"]
            permitted = permissions[tid]
            if not permitted:
                if config.get("permission_mode") == "plan":
                    plan_file = runtime.get_ctx(config).plan_file or ""
                    result = (
                        f"[Plan mode] Write operations are blocked except to the plan file: {plan_file}\n"
                        "Finish your analysis and write the plan to the plan file. "
                        "The user will run /plan done to exit plan mode and begin implementation."
                    )
                else:
                    # Better diagnostic than "user rejected" — confused
                    # models otherwise assume the user is actively
                    # blocking them and descend into trial-and-error.
                    result = (
                        f"Denied: tool {tc['name']!r} requires permission and "
                        f"none was granted (permission_mode="
                        f"{config.get('permission_mode', 'auto')}, "
                        f"running headless). "
                        f"This is a configuration issue, not a user rejection. "
                        f"Try a different tool or stop this turn."
                    )
            else:
                result = execute_tool(
                    tc["name"], tc["input"],
                    permission_mode="accept-all",
                    config=config,
                )
            return tc, result, permitted

        results_ordered = []

        # Run parallel batch concurrently
        if parallel_batch:
            from concurrent.futures import ThreadPoolExecutor
            for tc in parallel_batch:
                yield ToolStart(tc["name"], tc["input"])
            with ThreadPoolExecutor(max_workers=min(len(parallel_batch), 8)) as pool:
                futures = {pool.submit(_exec_one, tc): tc for tc in parallel_batch}
                for future in futures:
                    tc, result, permitted = future.result()
                    _log.debug("tool_end", session_id=session_id,
                               tool=tc["name"], permitted=permitted,
                               result_len=len(result))
                    results_ordered.append((tc, result, permitted))

        # Run sequential batch one by one
        for tc in sequential_batch:
            yield ToolStart(tc["name"], tc["input"])
            _log.debug("tool_start", session_id=session_id,
                       tool=tc["name"], input_keys=list(tc["input"].keys()))
            tc, result, permitted = _exec_one(tc)
            _log.debug("tool_end", session_id=session_id,
                       tool=tc["name"], permitted=permitted,
                       result_len=len(result))
            results_ordered.append((tc, result, permitted))

        # Yield results and append to state in original order
        for tc, result, permitted in results_ordered:
            yield ToolEnd(tc["name"], result, permitted)
            state.messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         tc["name"],
                "content":      result,
            })

        # Anti-stuck: detect identical tool+args being called repeatedly.
        # When tripped, inject a one-shot steering message as the next user
        # turn. The model can ignore it; it's a nudge, not a correction.
        # Opt out with config["anti_stuck"] = False.
        if config.get("anti_stuck", True) and assistant_turn.tool_calls:
            import anti_stuck as _as
            stuck, why = _as.detect_stuck(state, assistant_turn.tool_calls)
            if stuck:
                _as.mark_fired(state)
                _log.warn("anti_stuck_fired", session_id=session_id, tool=why,
                          turn=state.turn_count)
                nudge = _as.steering_message(why)
                yield TextChunk(f"\n[{nudge}]\n")
                state.messages.append({"role": "user", "content": nudge})

        # Output-truncation recovery: if the assistant turn was cut at the
        # token limit (and either dropped a malformed tool call or just
        # truncated text after a tool call), append the synthetic hint so
        # the next iteration can recover. The hint follows all tool results
        # in history, which is the correct ordering for OpenAI-compatible
        # APIs (assistant → tool* → user → assistant ...).
        if _trunc_continue and _trunc_hint:
            cap = int(config.get("max_continuations", 3) or 0)
            yield TextChunk(
                f"\n[Output truncated at max_tokens — auto-continuing "
                f"({getattr(state, '_continuations', 1)}/{cap})]\n"
            )
            state.messages.append({"role": "user", "content": _trunc_hint})


# ── Helpers ───────────────────────────────────────────────────────────────

def _check_permission(tc: dict, config: dict) -> bool:
    """Return True if operation is auto-approved (no need to ask user)."""
    perm_mode = config.get("permission_mode", "auto")
    name = tc["name"]

    # Plan mode tools are always auto-approved
    if name in ("EnterPlanMode", "ExitPlanMode"):
        return True

    if perm_mode == "accept-all":
        return True
    if perm_mode == "manual":
        return False   # always ask

    # Subagent tool whitelist: when an AgentDefinition.tools list is in
    # play (set by SubAgentManager.spawn → eff_config), the whitelist IS
    # the security boundary. The schema filter in the agent loop prevents
    # the model from being told about non-whitelisted tools; the dispatch
    # whitelist enforcement rejects out-of-list calls. So any call that
    # reaches this function and is in the whitelist is implicitly trusted —
    # auto-approve. Without this, custom subagent tools (Think, Note,
    # SaveFinding, RabbitFetch, etc.) silently fail with "Denied: user
    # rejected this operation" because they're not in the auto-mode
    # allow-list below.
    whitelist = config.get("_agent_tools_whitelist")
    if whitelist and name in whitelist:
        return True

    if perm_mode == "plan":
        # Allow writes ONLY to the plan file
        if name in ("Write", "Edit"):
            plan_file = runtime.get_ctx(config).plan_file or ""
            target = tc["input"].get("file_path", "")
            if plan_file and target and \
               os.path.normpath(target) == os.path.normpath(plan_file):
                return True
            return False
        if name == "NotebookEdit":
            return False
        if name == "Bash":
            from tools import _is_safe_bash
            return _is_safe_bash(tc["input"].get("command", ""))
        return True  # reads are fine

    # "auto" mode: only ask for writes and non-safe bash. Auto-approve:
    #   - Original read-only tools
    #   - Symbol-graph navigation (read-only repo intelligence)
    #   - Think (no-op scratchpad — never touches state)
    #   - AskUserQuestion (its job IS to interact with the user)
    if name in (
        "Read", "Glob", "Grep", "WebFetch", "WebSearch",
        # Symbol-graph nav (vendored Aider repomap; all read-only)
        "RepoMap", "FindSymbol", "GetCallers", "Outline",
        "Neighborhood", "PathBetween", "Imports", "SearchFiles",
        # Misc no-side-effect tools
        "Think", "AskUserQuestion", "GetDiagnostics", "SleepTimer",
    ):
        return True
    if name == "Bash":
        from tools import _is_safe_bash
        return _is_safe_bash(tc["input"].get("command", ""))
    return False   # Write, Edit → ask


def _permission_desc(tc: dict) -> str:
    name = tc["name"]
    inp  = tc["input"]
    if name == "Bash":   return f"Run: {inp.get('command', '')}"
    if name == "Write":  return f"Write to: {inp.get('file_path', '')}"
    if name == "Edit":   return f"Edit: {inp.get('file_path', '')}"
    return f"{name}({list(inp.values())[:1]})"


def _force_compact(state: AgentState, config: dict) -> bool:
    """Force compaction regardless of threshold. Used when API rejects for context too long."""
    limit = get_context_limit(config.get("model", ""), config)
    before = estimate_tokens(state.messages)
    if before <= 0:
        return False
    from compaction import snip_old_tool_results
    snip_old_tool_results(state.messages, max_chars=1000, preserve_last_n_turns=3)
    if estimate_tokens(state.messages) < limit * 0.9:
        return True
    state.messages = compact_messages(state.messages, config)
    from compaction import _restore_plan_context
    state.messages.extend(_restore_plan_context(config))
    after = estimate_tokens(state.messages)
    return after < before


def _truncate_err(s: str, max_len: int = 120) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


# ── Output-truncation auto-continue (Aider/OpenCode pattern) ───────────────
#
# When a provider returns finish_reason="length" the model hit the max_tokens
# output cap. Two failure modes follow:
#
#   (a) text was truncated mid-sentence and there are no tool_calls —
#       agent loop would `break` and the user sees a half-finished response.
#   (b) a tool_call's JSON args were truncated mid-stream — providers.py
#       captures the raw bytes as input={"_raw": "<broken json>"}, which
#       then errors at execute time with no useful signal to the model.
#
# Recovery: strip malformed tool calls from history, append a synthetic
# user-role hint that explains the truncation and how to recover (continue
# from the cut-off point, or split into smaller Write/Edit chunks), and
# re-enter the loop. A counter caps consecutive continuations so a model
# that can't recover doesn't loop forever.

import re as _re


def _extract_partial_path(raw: str) -> str:
    """Best-effort: pull file_path out of a partial-JSON tool-call args string."""
    m = _re.search(r'"file_path"\s*:\s*"([^"]*)"', raw or "")
    return m.group(1) if m else ""


def _truncation_hint(dropped_tool_calls: list, had_text: bool,
                     max_tokens: int | None = None) -> str:
    """Build the user-role message that nudges the model to retry smarter.

    Tailored to the failure shape: malformed Write → split-into-chunks
    advice; bare text truncation → continue-from-cutoff advice. When the
    caller knows the model's output budget, the chunk size is recommended
    in concrete line/token numbers rather than vague language — small
    local models (e.g. Qwen3.5-9B Q4) need an explicit ceiling to stop
    re-emitting the whole file every retry.
    """
    # Hard ceiling derived from the budget. ~4 chars/token, ~80 chars/line
    # for typical markdown → tokens/20 ≈ lines that comfortably fit.
    if max_tokens and max_tokens > 0:
        line_budget = max(15, max_tokens // 20)
        budget_note = (
            f"Your output budget is {max_tokens} tokens — that is roughly "
            f"{line_budget} lines of markdown PER tool call before truncation."
        )
    else:
        line_budget = 30
        budget_note = (
            "Your output budget is small — assume roughly 30 lines of markdown "
            "PER tool call before truncation."
        )

    # Malformed tool call(s) — most common cause is an over-long Write content.
    write_drop = next(
        (tc for tc in dropped_tool_calls if tc.get("name") in ("Write", "Edit")),
        None,
    )
    if write_drop:
        raw  = write_drop.get("input", {}).get("_raw", "")
        path = _extract_partial_path(raw) or "<unknown>"
        name = write_drop["name"]
        return (
            f"STOP. Your last {name} call to `{path}` was cut off mid-content "
            f"because the model output token limit was reached. The partial tool "
            f"call has been DISCARDED — nothing was written. You MUST change "
            f"strategy on this retry; emitting the same long content again will "
            f"fail the same way.\n\n"
            f"{budget_note}\n\n"
            f"Required strategy:\n"
            f"  1. Call `Write` with ONLY THE FIRST {line_budget} LINES of the "
            f"file. No more. End cleanly mid-document.\n"
            f"  2. After that succeeds, call `Edit` repeatedly to append the next "
            f"{line_budget} lines each time: set `old_string` to the LAST 2-3 "
            f"LINES of what you wrote so far, and `new_string` to those same lines "
            f"followed by the next chunk.\n"
            f"  3. Repeat until the whole file is in place.\n\n"
            f"Do not output any explanatory prose — go straight to the first "
            f"`Write` call with at most {line_budget} lines of content."
        )
    if dropped_tool_calls:
        names = ", ".join(tc.get("name", "?") for tc in dropped_tool_calls)
        return (
            f"Your last tool call(s) ({names}) were truncated mid-arguments by the "
            f"output token limit and have been discarded. {budget_note} Retry "
            f"with shorter inputs."
        )
    if had_text:
        return (
            "Your previous response was cut off mid-output by the token limit. "
            f"{budget_note} Continue from exactly where you stopped — do NOT "
            "repeat anything you have already said. If you were writing a long "
            "file, switch to splitting it into a `Write` (first chunk) followed "
            "by `Edit` calls (each appending the next chunk by matching its last "
            "few lines)."
        )
    return (
        "Your previous response hit the output token limit before producing any "
        f"usable content. {budget_note} Try a more concise approach."
    )


def _handle_length_truncation(state: "AgentState",
                              assistant_turn,
                              assistant_msg: dict,
                              config: dict):
    """Mutate state to recover from finish_reason='length'.

    Returns (should_continue: bool, hint: str | None).
      should_continue=True means caller should NOT break the agent loop.
      hint is the synthetic user message to append (or None if cap hit).
    """
    if assistant_turn.finish_reason != "length":
        # Clean turn — reset the consecutive-continuation counter.
        state.__dict__.pop("_continuations", None)
        return (False, None)

    # Strip malformed tool calls from BOTH the live turn and the just-appended
    # history record. Leaving them in history would cause a 400 on the next
    # request because there'd be a tool_call without a matching tool response.
    valid, dropped = [], []
    for tc in assistant_turn.tool_calls or []:
        if "_raw" in tc.get("input", {}):
            dropped.append(tc)
        else:
            valid.append(tc)
    if dropped:
        assistant_turn.tool_calls    = valid
        assistant_msg["tool_calls"]  = valid

    # If stripping produced an entirely empty assistant message (no text AND
    # no surviving tool_calls), DO NOT pop it. Popping creates two consecutive
    # user messages once the caller appends the truncation hint, and that
    # alternation violation makes some models loop or spam unrelated tools
    # (observed empirically: the local Qwen 9B spams GetDiagnostics, others
    # silently re-emit their earlier preamble — "harness repeated text").
    # Instead, replace the empty content with a short stub so history stays
    # well-alternated: user → assistant(stub) → user(hint) → assistant(retry).
    has_text = bool((assistant_turn.text or "").strip())
    if not has_text and not valid and state.messages and state.messages[-1] is assistant_msg:
        assistant_msg["content"] = "[output cut off at max_tokens]"
        # Mirror onto the live turn too, so anything that re-reads it sees
        # the same content.
        assistant_turn.text = assistant_msg["content"]

    cap   = int(config.get("max_continuations", 3) or 0)
    count = getattr(state, "_continuations", 0)
    if cap <= 0 or count >= cap:
        return (False, None)
    state._continuations = count + 1

    # Thinking-burnout: model burned the whole token budget inside
    # `reasoning_details` without ever emitting visible content. Affects M2
    # (and any thinking model with no native thinking_budget knob — DeepSeek
    # R1, Qwen-think). No major harness recovers from this automatically;
    # we inject a strong "answer NOW" steering message and cap retries
    # separately so we don't burn the budget loop-thinking.
    rd = getattr(assistant_turn, "reasoning_details", None) or []
    is_thinking_burnout = (
        not has_text
        and not valid
        and not dropped
        and bool(rd)
    )
    if is_thinking_burnout:
        # Independent cap so the generic truncation cap doesn't share a
        # budget with thinking-burnout retries (which are a different
        # failure mode and want a tighter ceiling).
        tb_cap   = int(config.get("max_thinking_burnout_retries", 2) or 0)
        tb_count = getattr(state, "_thinking_burnout_retries", 0)
        if tb_cap <= 0 or tb_count >= tb_cap:
            return (False, None)
        state._thinking_burnout_retries = tb_count + 1
        # Grab a snippet of the burned thinking so the model sees it
        # didn't lose work — just needs to finalize. M2's reasoning_details
        # entries each carry a `text` field; concat then truncate.
        snippet = ""
        for item in rd:
            t = item.get("text") if isinstance(item, dict) else ""
            if t:
                snippet += t
            if len(snippet) > 1200:
                break
        snippet = snippet[:1200].strip()
        if snippet:
            snippet_block = (
                "Your reasoning so far (do not repeat or extend it):\n"
                f">>> {snippet} <<<\n\n"
            )
        else:
            snippet_block = ""
        hint = (
            "[thinking-burnout recovery] Your previous turn used the entire "
            "token budget on internal reasoning without producing any visible "
            "answer or tool call. " + snippet_block +
            "Give your final answer or next tool call NOW with minimal or "
            "zero further reasoning. Be direct and concise. Do not re-think; "
            "act on what you already concluded."
        )
        return (True, hint)

    hint = _truncation_hint(dropped, has_text, config.get("max_tokens"))
    return (True, hint)
