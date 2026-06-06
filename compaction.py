"""Context window management: two-layer compression for long conversations."""
from __future__ import annotations

import providers


# ── Token estimation ──────────────────────────────────────────────────────

def _count_str_chars(obj) -> int:
    """Recursively count total characters across all string values in a nested structure."""
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, dict):
        return sum(_count_str_chars(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_str_chars(item) for item in obj)
    return 0


def _chars_per_token_for(model: str | None) -> float:
    """Resolve the provider-calibrated chars/token ratio.

    Calibrated empirically (see TODO_NEXT.md → per-provider tokenizer calibration).
    MiniMax M2 measured: 1.8 (json) – 5.4 (long English) chars/tok depending on
    content type. Coding-agent traffic skews heavily toward code+JSON, so we
    pick the conservative low end. Underestimating tokens trips compaction
    earlier (cheap); overestimating skips compaction and crashes on overflow
    (catastrophic). Always err low.
    """
    if not model:
        return 2.8
    try:
        prov = providers.detect_provider(model)
    except Exception:
        return 2.8
    entry = providers.PROVIDERS.get(prov, {})
    return float(entry.get("token_estimate_chars_per_token", 2.8))


def _per_message_overhead_for(model: str | None) -> int:
    """Per-message framing token overhead. Calibrated: M2 = 20 tokens/msg
    (vs the legacy 4) — measured on the en_short probe where 44 chars
    yielded 31 input tokens, ~22 of which were framing."""
    if not model:
        return 4
    try:
        prov = providers.detect_provider(model)
    except Exception:
        return 4
    entry = providers.PROVIDERS.get(prov, {})
    return int(entry.get("token_estimate_per_msg_overhead", 4))


def estimate_tokens(messages: list, model: str | None = None) -> int:
    """Estimate token count.

    When `model` is provided, uses the per-provider calibrated chars/token
    ratio from `PROVIDERS[prov]["token_estimate_chars_per_token"]` (default
    2.8) and per-message overhead from `token_estimate_per_msg_overhead`
    (default 4). Falls back to the legacy chars/2.8 + 4 tok/msg estimate
    when called without a model — older callers stay correct without
    threading the model param everywhere.

    Args:
        messages: list of message dicts with "content" field
        model:    optional model id used to pick provider-calibrated ratios
    Returns:
        approximate token count, int
    """
    total_chars = 0
    msg_count = 0
    for m in messages:
        msg_count += 1
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for v in block.values():
                        if isinstance(v, str):
                            total_chars += len(v)
        for tc in m.get("tool_calls", []):
            total_chars += _count_str_chars(tc)
    cpt = _chars_per_token_for(model)
    pmo = _per_message_overhead_for(model)
    content_tokens = int(total_chars / cpt)
    framing_tokens = msg_count * pmo
    return int((content_tokens + framing_tokens) * 1.1)


def get_context_limit(model: str, config: dict = None) -> int:
    """Look up context window size for a model.

    Resolution order (most specific first):
      1. config["context_limit"] — explicit override. CRITICAL for
         self-hosted llama-server with -np > 1 where each slot only
         gets n_ctx / n_parallel tokens (e.g. -c 229376 -np 4 → each
         slot is 57344, NOT 229376). Without this override the harness
         thinks the limit is 128000 (provider default) and lets the
         conversation grow past the actual slot capacity.
      2. Per-model entry in PROVIDERS[prov]["per_model_context_limits"].
         Needed for providers whose lineup spans multiple windows — e.g.
         MiniMax-M2 is 204_800 but MiniMax-M1 is 1M under the same key.
      3. Provider default from PROVIDERS table
      4. Hardcoded 128000 fallback

    Args:
        model: model string (e.g. "claude-opus-4-6", "ollama/llama3.3")
        config: optional config dict; checked for "context_limit" override
    Returns:
        context limit in tokens
    """
    if config is not None:
        override = config.get("context_limit")
        if override and isinstance(override, int) and override > 0:
            return override
    provider_name = providers.detect_provider(model)
    prov = providers.PROVIDERS.get(provider_name, {})
    per_model = prov.get("per_model_context_limits") or {}
    bare = providers.bare_model(model)
    if bare in per_model:
        return per_model[bare]
    return prov.get("context_limit", 128000)


# ── Layer 1: Snip old tool results ────────────────────────────────────────

def snip_old_tool_results(
    messages: list,
    max_chars: int = 2000,
    preserve_last_n_turns: int = 6,
) -> list:
    """Truncate tool-role messages older than preserve_last_n_turns from end.

    For old tool messages whose content exceeds max_chars, keep the first half
    and last quarter, inserting '[... N chars snipped ...]' in between.
    Mutates in place and returns the same list.

    Args:
        messages: list of message dicts (mutated in place)
        max_chars: maximum character length before truncation
        preserve_last_n_turns: number of messages from end to preserve
    Returns:
        the same messages list (mutated)
    """
    cutoff = max(0, len(messages) - preserve_last_n_turns)
    for i in range(cutoff):
        m = messages[i]
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= max_chars:
            continue
        first_half = content[: max_chars // 2]
        last_quarter = content[-(max_chars // 4):]
        snipped = len(content) - len(first_half) - len(last_quarter)
        m["content"] = f"{first_half}\n[... {snipped} chars snipped ...]\n{last_quarter}"
    return messages


def elide_old_tool_observations(
    messages: list,
    preserve_last_n_tool_messages: int = 5,
    min_chars_to_elide: int = 200,
) -> list:
    """SWE-agent ACI-style aggressive elision of old tool observations.

    For tool-role messages older than the most recent
    `preserve_last_n_tool_messages` tool messages, replace the content
    with a single-line summary giving the size and a 60-char preview.
    Skips messages already shorter than `min_chars_to_elide` (no point
    eliding a 50-char output) and skips messages already containing an
    elision marker (idempotent across calls).

    This is more aggressive than snip_old_tool_results — that keeps
    ~2000 chars per old observation; this one keeps ~80. Use when
    sessions get long and the model has accumulated many file Reads
    and Grep results that are no longer load-bearing.

    The model can still reason about WHAT it saw (preview gives a hint)
    without re-loading the bytes. If the model needs the actual content
    again, it can re-call the tool (FileContextTracker will return the
    real file unchanged or with a banner if mtime moved).

    Mutates in place; returns the same list.
    """
    # Find indices of all tool messages; keep the last N untouched.
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= preserve_last_n_tool_messages:
        return messages
    cutoff_idx = tool_indices[-preserve_last_n_tool_messages]

    for i, m in enumerate(messages):
        if i >= cutoff_idx:
            break
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        if len(content) < min_chars_to_elide:
            continue
        # Idempotent — don't re-elide an already-elided message.
        if content.startswith("[elided:"):
            continue
        n_lines = content.count("\n") + (1 if content else 0)
        n_chars = len(content)
        # 60-char preview from the very start of the output (often the most
        # informative — file headers, command echo, error class name).
        preview = content[:60].replace("\n", " ").strip()
        if len(content) > 60:
            preview += "…"
        m["content"] = (
            f"[elided: {n_lines} lines / {n_chars} chars — preview: {preview!r}]"
        )
    return messages


# ── Layer 2: Auto-compact ─────────────────────────────────────────────────

def _respect_tool_pairs(messages: list, split: int) -> int:
    """Advance split so it never falls inside a tool_calls → tool-response block.

    OpenAI-compatible APIs (DeepSeek, etc.) reject any 'tool' message that is
    not preceded by an 'assistant' with matching tool_calls. If the split lands
    between an assistant(tool_calls) and its tool responses, the recent half
    would contain orphan tool messages after compaction.
    """
    n = len(messages)
    if split <= 0 or split >= n:
        return split
    prev = messages[split - 1]
    if prev.get("role") == "assistant" and (prev.get("tool_calls") or []):
        j = split
        while j < n and messages[j].get("role") == "tool":
            j += 1
        split = j
    while split < n and messages[split].get("role") == "tool":
        split += 1
    return split


def find_split_point(messages: list, keep_ratio: float = 0.3) -> int:
    """Find index that splits messages so ~keep_ratio of tokens are in the recent portion.

    Walks backwards from end, accumulating token estimates, and returns the
    index where the recent portion reaches ~keep_ratio of total tokens. The
    index is then adjusted so it never cuts a tool-call response block.

    Args:
        messages: list of message dicts
        keep_ratio: fraction of tokens to keep in the recent portion
    Returns:
        split index (messages[:idx] = old, messages[idx:] = recent).
        Returns 0 if no safe split exists (caller should skip compaction).
    """
    if not messages:
        return 0
    keep_ratio = max(0.0, min(1.0, keep_ratio))
    total = estimate_tokens(messages)
    target = int(total * keep_ratio)
    running = 0
    raw = 0
    for i in range(len(messages) - 1, -1, -1):
        running += estimate_tokens([messages[i]])
        if running >= target:
            raw = i
            break
    adjusted = _respect_tool_pairs(messages, raw)
    if adjusted >= len(messages):
        return 0
    return adjusted


def sanitize_history(messages: list) -> list:
    """Enforce the tool-calls ↔ tool-response invariant required by OpenAI-compatible APIs.

    Walks the list in order maintaining a set of pending tool_call_ids from the
    most recent assistant(tool_calls). Drops any 'tool' message whose
    tool_call_id is not in that set (orphan). When a non-tool message arrives
    with pending ids still open, strips those unanswered tool_calls from the
    preceding assistant message (so DeepSeek won't reject it).

    Returns a new list; the input is not mutated.
    """
    cleaned: list = []
    pending: set[str] = set()

    def _strip_unanswered():
        if not pending:
            return
        # Walk back past any trailing tool messages to reach the assistant that owns them.
        target = None
        for k in range(len(cleaned) - 1, -1, -1):
            role_k = cleaned[k].get("role")
            if role_k == "tool":
                continue
            if role_k == "assistant":
                target = k
            break
        if target is None:
            return
        prev = cleaned[target]
        tcs = prev.get("tool_calls") or []
        kept = [tc for tc in tcs if tc.get("id") not in pending]
        if len(kept) == len(tcs):
            return
        new_prev = dict(prev)
        if kept:
            new_prev["tool_calls"] = kept
        else:
            new_prev.pop("tool_calls", None)
            # All tool_calls were unanswered and got stripped. If the
            # assistant has no text either, leave a stub instead of an
            # empty string — empty assistants confuse some models into
            # repeating prior text or spamming unrelated tools.
            if not (new_prev.get("content") or "").strip():
                new_prev["content"] = "[output cut off at max_tokens]"
        cleaned[target] = new_prev

    for m in messages:
        role = m.get("role")
        if role == "tool":
            tid = m.get("tool_call_id")
            if tid in pending:
                cleaned.append(m)
                pending.discard(tid)
            continue
        _strip_unanswered()
        pending = set()
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                pending = {tc["id"] for tc in tcs if tc.get("id")}
        cleaned.append(m)

    _strip_unanswered()
    return cleaned


def _build_workspace_state_block(workspace_dir: str) -> str:
    """Render a fresh, lossless ground-truth view of the rabbit-hole
    workspace state. This block sits right after the system prompt and
    is rebuilt on every compaction event — it's how the agent stays
    anchored to its mission across very long runs.

    Kept compact (~500 tokens at typical workspace size) because it ships
    in every compacted-context request.
    """
    from rabbit_hole.store import RabbitHoleWorkspace
    ws = RabbitHoleWorkspace(workspace_dir)
    out = ["[Workspace state — your persistent memory of this research run]", ""]
    rq = ws.manifest.get("root_question", "")
    if rq:
        out.append(f"Original question: {rq}")
        out.append("")

    open_qs = ws.list_open_questions()
    closed_qs = [q for q in ws.list_all_questions() if q.status == "closed"]

    if open_qs:
        out.append(f"Open sub-questions ({len(open_qs)}):")
        for q in open_qs[:15]:
            parent = f"  ↳ child of {q.parent_id}" if q.parent_id else ""
            out.append(f"  {q.id}: {q.text[:140]}{parent}  [{len(q.finding_ids)} findings]")
        if len(open_qs) > 15:
            out.append(f"  ... and {len(open_qs) - 15} more (use ListOpenQuestions)")
        out.append("")

    if closed_qs:
        out.append(f"Closed sub-questions ({len(closed_qs)}):")
        for q in closed_qs[-5:]:  # most recent 5 closures
            summary = q.summary[:120] if q.summary else "(no summary)"
            out.append(f"  ✓ {q.id}: {q.text[:80]} — {summary}")
        out.append("")

    findings = ws.list_findings()
    recent_findings = sorted(findings, key=lambda f: -f.created_turn)[:5]
    if recent_findings:
        out.append("Most recent findings:")
        for f in recent_findings:
            n_urls = len(f.evidence_urls)
            out.append(f"  {f.id} ({f.sub_question_id}): {f.claim[:140]}  [{n_urls} sources]")
        out.append("")

    n_sources = len(ws.list_sources())
    out.append(f"Stats: turn {ws.state.turn}, "
               f"{len(findings)} findings total, {n_sources} unique sources, "
               f"stuck_for={ws.stuck_for()}")
    out.append("")
    out.append("Use SearchFindings / ListSources to explore the full workspace. "
               "Save NEW findings as you go — the workspace is your durable memory.")

    return "\n".join(out)


def compact_for_research_continuity(messages: list, config: dict) -> list:
    """Rabbit-hole-aware compaction. Differs from compact_messages by:

      • Keeping the original system prompt and the initial user prompt
        intact at the head — these define the mission and must never
        be summarized away.
      • Inserting a fresh workspace-state block immediately after
        the system prompt — this is the lossless ground truth (open
        sub-questions, recent findings, etc.) that the LLM summary
        cannot lose.
      • Using a research-continuity-tuned summary prompt that prioritizes
        the agent's CURRENT line of investigation, unfilled hypotheses,
        and tool-pattern lessons — NOT already-saved findings (those
        are in the workspace state).

    Result: [system, workspace_state, summary, ack, *recent_messages].
    The original first user message is included in the recent slice via
    find_split_point's natural cutoff so the agent always has its
    starting prompt visible.
    """
    workspace_dir = config.get("_rabbit_hole_workspace_dir")
    if not workspace_dir or len(messages) < 4:
        return messages

    # Keep system prompt + first user message untouched at the head.
    # Find them: the first system message and the first user message.
    head_end = 0
    saw_user = False
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            head_end = i + 1
        elif m.get("role") == "user" and not saw_user:
            saw_user = True
            head_end = i + 1
            break
    head = messages[:head_end]

    # The rest splits as usual: older half summarized, recent half kept verbatim.
    rest = messages[head_end:]
    if len(rest) < 2:
        return messages

    split_in_rest = find_split_point(rest, keep_ratio=0.3)
    if split_in_rest <= 0:
        return messages
    old = rest[:split_in_rest]
    recent = rest[split_in_rest:]

    # Build summary of older turns
    old_text = ""
    for m in old:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            old_text += f"[{role}]: {content[:500]}\n"
        elif isinstance(content, list):
            old_text += f"[{role}]: (structured content)\n"

    summary_prompt = (
        "You are summarizing the conversation history of a long-running "
        "deep-research agent. The agent's persistent findings are saved "
        "in a workspace (sub-questions + findings + sources) and shown "
        "separately — DON'T re-summarize those.\n"
        "\n"
        "What to PRESERVE in your summary, in priority order:\n"
        "1. The CURRENT line of investigation — which sub-question the "
        "agent was pursuing right before the cutoff, and any partial "
        "reasoning about it.\n"
        "2. Findings or sources the agent encountered but hadn't yet "
        "saved formally (these are at risk of being lost on cutoff).\n"
        "3. The agent's strategic decisions — why it chose this branch "
        "over another, what hypotheses it has formed but not yet tested.\n"
        "4. Tool-call patterns that worked or failed (so it doesn't "
        "repeat mistakes).\n"
        "\n"
        "What to DROP:\n"
        "- Already-saved findings (in workspace state).\n"
        "- Routine tool-call narration ('I'll now fetch X').\n"
        "- The original user prompt and root sub-question decomposition.\n"
        "\n"
        "Output: 150-250 words, plain prose, no headers. Begin with "
        "'Mid-investigation summary:'.\n"
        "\n"
        "History to summarize:\n"
        + old_text
    )

    from auxiliary import stream_auxiliary
    summary_text = stream_auxiliary(
        system="You are a concise research-continuity summarizer.",
        messages=[{"role": "user", "content": summary_prompt}],
        config=config,
    )

    # Workspace state block — lossless, fresh
    state_block = _build_workspace_state_block(workspace_dir)

    state_msg = {
        "role": "user",
        "content": state_block,
    }
    state_ack = {
        "role": "assistant",
        "content": (
            "Workspace state noted. I have the open sub-questions, recent "
            "findings, and run statistics. The workspace is my durable "
            "memory; I'll continue from here."
        ),
    }
    summary_msg = {
        "role": "user",
        "content": (
            f"[Mid-investigation summary of older conversation turns "
            f"(workspace state above is the source of truth for findings)]\n"
            f"{summary_text}"
        ),
    }
    summary_ack = {
        "role": "assistant",
        "content": (
            "Got it. Continuing the investigation from where I left off."
        ),
    }
    return [*head, state_msg, state_ack, summary_msg, summary_ack, *recent]


def compact_messages(messages: list, config: dict, focus: str = "") -> list:
    """Compress old messages into a summary via LLM call.

    Splits at find_split_point, summarizes old portion, returns
    [summary_msg, ack_msg, *recent_messages].

    Args:
        messages: full message list
        config: agent config dict (must contain "model")
        focus: optional focus instructions for the summarizer
    Returns:
        new compacted message list
    """
    split = find_split_point(messages)
    if split <= 0:
        return messages

    old = messages[:split]
    recent = messages[split:]

    # Build summary request
    old_text = ""
    for m in old:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            old_text += f"[{role}]: {content[:500]}\n"
        elif isinstance(content, list):
            old_text += f"[{role}]: (structured content)\n"

    summary_prompt = (
        "Summarize the following conversation history concisely. "
        "Preserve key decisions, file paths, tool results, and context "
        "needed to continue the conversation."
    )
    if focus:
        summary_prompt += f"\n\nFocus especially on: {focus}"
    summary_prompt += "\n\n" + old_text

    # Call auxiliary (fast/cheap) model for summary instead of the primary model
    from auxiliary import stream_auxiliary
    summary_text = stream_auxiliary(
        system="You are a concise summarizer.",
        messages=[{"role": "user", "content": summary_prompt}],
        config=config,
    )

    # Context survival: rescue durable facts from the messages we're about to
    # drop into a lossy summary. Persists to memory so they outlive this and
    # future sessions (mirrors Claude Code's pre-compaction memory write).
    # Gated by config; failures must never block compaction.
    if config.get("compaction_memory", True):
        try:
            from memory.consolidator import consolidate_for_compaction
            saved = consolidate_for_compaction(old, config)
            if saved:
                import logging_utils as _log
                _log.info("compaction_memory_saved", count=len(saved), names=saved)
        except Exception:
            pass

    summary_msg = {
        "role": "user",
        "content": f"[Previous conversation summary]\n{summary_text}",
    }
    ack_msg = {
        "role": "assistant",
        "content": "Understood. I have the context from the previous conversation. Let's continue.",
    }
    return [summary_msg, ack_msg, *recent]


# ── Main entry ────────────────────────────────────────────────────────────

def maybe_compact(state, config: dict) -> bool:
    """Check if context window is getting full and compress if needed.

    Runs snip_old_tool_results first, then auto-compact if still over threshold.

    Args:
        state: AgentState with .messages list
        config: agent config dict (must contain "model")
    Returns:
        True if compaction was performed
    """
    model = config.get("model", "")
    limit = get_context_limit(model, config)
    # Rabbit-hole agents compact earlier (50% vs 70%) because the cost of
    # lossy summarization is small (workspace state stays intact) and
    # research runs benefit from the agent's context staying fresh.
    is_rabbit_hole = bool(config.get("_rabbit_hole_workspace_dir"))
    threshold = limit * (0.5 if is_rabbit_hole else 0.7)

    if estimate_tokens(state.messages, model) <= threshold:
        return False

    # Layer 1a (optional, off by default): SWE-agent-style aggressive elision
    # of old tool observations. Opt in with config["aggressive_elision"]=True
    # for very-long-session use cases where the bulk of the budget is old
    # Read/Grep outputs the model has already moved past.
    if config.get("aggressive_elision"):
        elide_old_tool_observations(
            state.messages,
            preserve_last_n_tool_messages=config.get(
                "elision_preserve_last_n", 5),
        )
        if estimate_tokens(state.messages, model) <= threshold:
            return True

    # Layer 1b: snip old tool results (per-message head/tail truncation,
    # less aggressive than elision; runs always).
    snip_old_tool_results(state.messages)

    if estimate_tokens(state.messages, model) <= threshold:
        return True

    # Layer 2: auto-compact. Rabbit-hole runs use a research-continuity-
    # tuned variant that keeps the system prompt + initial user message
    # intact and prepends a fresh workspace-state block (lossless ground
    # truth) before the LLM-summarized older turns.
    if is_rabbit_hole:
        state.messages = compact_for_research_continuity(state.messages, config)
    else:
        state.messages = compact_messages(state.messages, config)
    state.messages.extend(_restore_plan_context(config))
    return True


# ── Plan context restoration ─────────────────────────────────────────────

def _restore_plan_context(config: dict) -> list:
    """If in plan mode, return messages that restore plan file context."""
    from pathlib import Path
    import runtime
    plan_file = runtime.get_ctx(config).plan_file or ""
    if not plan_file or config.get("permission_mode") != "plan":
        return []
    p = Path(plan_file)
    if not p.exists():
        return []
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return [
        {"role": "user", "content": f"[Plan file restored after compaction: {plan_file}]\n\n{content}"},
        {"role": "assistant", "content": "I have the plan context. Let's continue."},
    ]


# ── Manual compact ───────────────────────────────────────────────────────

def manual_compact(state, config: dict, focus: str = "") -> tuple[bool, str]:
    """User-triggered compaction via /compact. Not gated by threshold.

    Returns (success, info_message).
    """
    if len(state.messages) < 4:
        return False, "Not enough messages to compact."

    model = config.get("model", "")
    before = estimate_tokens(state.messages, model)
    snip_old_tool_results(state.messages)
    state.messages = compact_messages(state.messages, config, focus=focus)
    state.messages.extend(_restore_plan_context(config))
    after = estimate_tokens(state.messages, model)
    saved = before - after
    return True, f"Compacted: ~{before} → ~{after} tokens (~{saved} saved)"
