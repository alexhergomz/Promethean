"""Memory consolidator: extract long-term insights from completed sessions.

Called manually via `/memory consolidate` or programmatically after a session.
Uses a lightweight AI call to identify user preferences, feedback corrections,
and project decisions worth promoting to persistent semantic memory.

Design principles:
- Hard cap of 3 memories per session to avoid noise accumulation
- Auto-extracted memories start at 0.8 confidence (below explicit user saves)
- Won't overwrite a higher-confidence existing memory
- Skips short sessions (< MIN_MESSAGES_TO_CONSOLIDATE turns)
"""
from __future__ import annotations

MIN_MESSAGES_TO_CONSOLIDATE = 8  # don't consolidate trivial sessions

_SYSTEM = """\
You are a memory consolidation assistant. Analyze the conversation below and extract
insights that are worth storing as persistent memories for future sessions.

Focus ONLY on:
1. New user preferences or working-style corrections revealed in this session
2. Project decisions or facts made explicit (NOT derivable from code/git)
3. Behavioral feedback given to the AI (what to do or avoid, and why)

Return a JSON object with key "memories" containing a list of objects, each with:
  "name":        short slug, e.g. "user_prefers_concise_responses"
  "type":        "user" | "feedback" | "project"
  "description": one-line description (used for search relevance)
  "content":     memory body; for feedback/project lead with the rule/fact then
                 **Why:** and **How to apply:** lines
  "confidence":  float 0.0–1.0 (use ~0.8 for inferred, ~0.9 for clearly stated)

Return {"memories": []} if nothing new or worth saving.

Do NOT extract:
- Code patterns, architecture, file paths — derivable from the codebase
- Git history or debugging fixes — already in commits
- Anything already obvious from CLAUDE.md
- Ephemeral task state or tool results

Keep to AT MOST 3 memories. Quality over quantity."""

# Variant used when the context window is being compacted: the analyzed turns
# are about to be DELETED, so the bar for "worth keeping" is lower — we want to
# rescue anything needed to keep working after the older turns disappear.
_SYSTEM_COMPACTION = """\
You are a memory-rescue assistant. The conversation turns below are about to be
compacted away (deleted) to free the context window. Before they vanish, extract
facts that the assistant will need to KEEP WORKING but that won't be obvious from
the running summary or the codebase.

Focus on:
1. Concrete decisions made and their rationale (chosen approach, rejected ones)
2. Project/task state established this session (what was done, what's pending)
3. User preferences or constraints revealed mid-session
4. Specific values that are easy to lose: paths, config keys, parameter choices,
   names, IDs — when they were the result of deliberation (not just any path seen)

Return a JSON object with key "memories" containing a list of objects, each with:
  "name":        short slug
  "type":        "user" | "feedback" | "project"
  "description": one-line description (used for search relevance)
  "content":     memory body; for feedback/project lead with the rule/fact then
                 **Why:** and **How to apply:** lines
  "confidence":  float 0.0-1.0 (use ~0.75 for inferred mid-session state)

Return {"memories": []} if nothing is worth rescuing. Keep to AT MOST 3."""


def _extract_memories(transcript: str, system: str, config: dict) -> list[dict]:
    """Run one extraction LLM pass; return the parsed list of memory dicts."""
    from providers import stream, AssistantTurn
    import json

    result_text = ""
    for event in stream(
        model=config.get("model", ""),
        system=system,
        messages=[{"role": "user", "content": f"Conversation:\n\n{transcript}"}],
        tool_schemas=[],
        config={**config, "max_tokens": 1024, "no_tools": True},
    ):
        if isinstance(event, AssistantTurn):
            result_text = event.text
            break

    if not result_text:
        return []
    try:
        parsed = json.loads(result_text)
    except Exception:
        return []
    data = parsed.get("memories", [])
    return data if isinstance(data, list) else []


def _save_extracted(
    memories_data: list[dict],
    *,
    source: str,
    scope: str,
    default_confidence: float,
) -> list[str]:
    """Persist extracted memory dicts, skipping more-confident conflicts."""
    from datetime import datetime
    from .store import MemoryEntry, save_memory, check_conflict

    saved: list[str] = []
    for m in memories_data[:3]:  # hard cap
        required = ("name", "type", "description", "content")
        if not all(k in m for k in required):
            continue
        entry = MemoryEntry(
            name=str(m["name"]),
            description=str(m["description"]),
            type=str(m.get("type", "user")),
            content=str(m["content"]),
            created=datetime.now().strftime("%Y-%m-%d"),
            confidence=float(m.get("confidence", default_confidence)),
            source=source,
        )
        conflict = check_conflict(entry, scope=scope)
        if conflict and conflict["existing_confidence"] >= entry.confidence:
            continue
        save_memory(entry, scope=scope)
        saved.append(entry.name)
    return saved


def consolidate_for_compaction(messages: list, config: dict) -> list[str]:
    """Rescue durable memories from messages about to be compacted away.

    Unlike consolidate_session (which looks at the recent tail of a finished
    session), this is handed the EXACT slice that compaction is about to drop,
    so it analyzes all of it and writes to project scope at lower confidence.

    Returns the list of saved memory names. Empty on skip or error.
    """
    try:
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                prefix = "User" if role == "user" else "Assistant"
                parts.append(f"{prefix}: {content[:600].replace(chr(10), ' ')}")
        if len(parts) < 4:
            return []
        transcript = "\n".join(parts[-60:])  # cap the prompt size
        data = _extract_memories(transcript, _SYSTEM_COMPACTION, config)
        return _save_extracted(
            data, source="compaction", scope="project", default_confidence=0.75,
        )
    except Exception:
        return []


def consolidate_session(messages: list, config: dict) -> list[str]:
    """Analyze a session's messages and extract memories worth keeping long-term.

    Args:
        messages: the conversation message list (neutral format)
        config:   the active config dict (must contain a "model" key)

    Returns:
        List of memory names that were saved. Empty list on skip or error.
    """
    if len(messages) < MIN_MESSAGES_TO_CONSOLIDATE:
        return []

    try:
        # Build condensed transcript from the last 40 messages (≈ 20 turns)
        parts: list[str] = []
        for m in messages[-40:]:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                prefix = "User" if role == "user" else "Assistant"
                parts.append(f"{prefix}: {content[:600].replace(chr(10), ' ')}")

        if not parts:
            return []

        data = _extract_memories("\n".join(parts), _SYSTEM, config)
        return _save_extracted(
            data, source="consolidator", scope="user", default_confidence=0.8,
        )

    except Exception:
        return []
