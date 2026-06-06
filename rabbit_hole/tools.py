"""Tools given to the rabbit-hole subagent.

These tools mutate the workspace and form the agent's vocabulary for
managing its own research lifecycle. Each tool is a thin wrapper over
RabbitHoleWorkspace; they all read `_rabbit_hole_workspace_dir` from
the agent's config to find the workspace.

Tools registered:
  RabbitFetch       — dedup-aware WebFetch wrapper (cache before fetching)
  AddSubQuestion    — queue a new line of inquiry
  ListOpenQuestions — see what's still pending
  MarkQuestionDone  — close a sub-question with a 1-2 sentence summary
  SaveFinding       — record a structured finding (claim + evidence URLs)
  SearchFindings    — BM25 over saved findings
  ListSources       — see what's been fetched (for dedup awareness)
  Finish            — explicitly trigger synthesis and stop

The model's allowed tool list is restricted to these + WebSearch +
Think + Read (path-jailed). See AgentDefinition for `deep-research-rabbit-hole`.
"""
from __future__ import annotations

from typing import List, Optional

from rabbit_hole.store import RabbitHoleWorkspace


def _ws(config: dict) -> Optional[RabbitHoleWorkspace]:
    """Get the workspace from config, or None if not in rabbit-hole mode.
    Tools should bail with a clear error when called outside the agent.
    """
    root = config.get("_rabbit_hole_workspace_dir")
    if not root:
        return None
    return RabbitHoleWorkspace(root)


# ── Schemas (registered into TOOL_SCHEMAS via tools/__init__.py) ──────────

RABBIT_HOLE_SCHEMAS = [
    {
        "name": "RabbitFetch",
        "description": (
            "Fetch a URL with automatic dedup against the rabbit-hole "
            "workspace cache. If the URL has already been fetched in this "
            "session, returns the cached content with a [CACHED] marker so "
            "you don't waste a request. Strongly prefer this over plain "
            "WebFetch when in rabbit-hole mode — it's the dedup mechanism."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch."},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate returned content to this many chars (default 8000)."
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "AddSubQuestion",
        "description": (
            "Queue a new sub-question for investigation. Returns the "
            "question id. Provide parent_id when refining a sub-question "
            "into more specific children."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The question."},
                "parent_id": {"type": "string", "description": "Parent question id (optional)."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "ListOpenQuestions",
        "description": (
            "List all sub-questions still in 'open' state. Pick one to "
            "investigate next. If empty, you've answered everything you "
            "queued — either decompose further or call Finish."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "MarkQuestionDone",
        "description": (
            "Close a sub-question with a 1-2 sentence summary of what you "
            "learned. The summary appears in the final report's TL;DR for "
            "root questions and as section bodies for leaves."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string"},
                "summary": {
                    "type": "string",
                    "description": "1-2 sentence summary. Be specific."
                },
            },
            "required": ["question_id", "summary"],
        },
    },
    {
        "name": "SaveFinding",
        "description": (
            "Record a structured finding: a single specific claim plus the "
            "URLs that support it. One finding = one claim. If you have two "
            "related claims, save them separately. Findings are the unit of "
            "synthesis — granular findings produce a better final report "
            "than vague paragraph-long ones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The claim, in one sentence."},
                "sub_question_id": {"type": "string"},
                "evidence_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs supporting this claim (the same URLs you fetched)."
                },
            },
            "required": ["claim", "sub_question_id", "evidence_urls"],
        },
    },
    {
        "name": "SearchFindings",
        "description": (
            "BM25 search over your saved findings. Useful when you want to "
            "check what you already concluded about a topic before investigating "
            "further (avoids re-treading the same ground)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "description": "Default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ListSources",
        "description": (
            "List URLs already fetched into the workspace cache. Use to "
            "see what you have without re-fetching."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "Note",
        "description": (
            "Write a short progress note that the user can see via "
            "/rabbit-hole status. Use for things like 'switching focus "
            "from inference to training', 'found surprising claim in "
            "[source X], will verify next', 'dead end on q-abc, marking "
            "done with insufficient evidence'. The note shows up in "
            "chronological order alongside automated events (sources "
            "fetched, findings saved, questions closed) so the user "
            "can follow your reasoning live without disrupting the run."
            "\n\n"
            "This is COMPLEMENTARY to Think — Think is your private "
            "scratchpad for between-step deliberation; Note is your "
            "outward-facing progress log for the user. Use Note "
            "sparingly — one short note every few turns, not every step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "1-2 sentence progress note for the user."
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "Finish",
        "description": (
            "Explicitly conclude the rabbit-hole run and trigger final "
            "synthesis. Call when you've answered enough sub-questions, "
            "you've hit diminishing returns, or you've reasonably exhausted "
            "the investigation. The final report is built from your saved "
            "findings, not from your context — you don't need to summarize "
            "in the call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "1-2 sentence reason for finishing."
                },
            },
            "required": ["reason"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────

def _err_no_workspace() -> str:
    return ("Error: this tool only works inside a rabbit-hole research agent. "
            "Spawn the deep-research-rabbit-hole agent to use it.")


# Anti-stuck: number of turns without a "real progress" event before
# the harness starts adding a nudge to tool outputs.
_STUCK_NUDGE_THRESHOLD = 6
# Hard cap: after this many turns of being stuck, ListOpenQuestions tells
# the agent to call Finish. This is advisory — the agent decides.
_STUCK_TERMINAL_THRESHOLD = 12


def _stuck_banner(ws) -> str:
    """If the agent is making no progress, return a banner string to
    prepend to tool output. Empty string when not stuck."""
    n = ws.stuck_for()
    if n < _STUCK_NUDGE_THRESHOLD:
        return ""
    if n >= _STUCK_TERMINAL_THRESHOLD:
        return (
            f"\n\n[⚠ STUCK for {n} turns — no new sources, findings, or "
            f"closed questions. You should call Finish now and let the "
            f"synthesizer build a report from what you have.]\n"
        )
    return (
        f"\n\n[⚠ STUCK for {n} turns — pivot to a different open "
        f"sub-question, broaden your search, or close the current "
        f"question with summary 'insufficient evidence' and move on.]\n"
    )


def rabbit_fetch(url: str, config: dict, max_chars: int = 8000) -> str:
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    cached = ws.get_cached_source(url)
    if cached:
        body = cached.get("content", "")[:max_chars]
        return (f"[CACHED — fetched on turn {cached.get('fetched_at_turn', '?')}]\n"
                f"URL: {cached['url']}\nTitle: {cached.get('title', '?')}\n\n{body}")

    # Cache miss — actually fetch via the existing WebFetch tool
    from tools.web import _webfetch
    try:
        content = _webfetch(url)
    except Exception as e:
        return f"Error fetching {url}: {e}"
    if content.startswith("Error:"):
        return content

    # Try to extract a title from the content (first <title> tag if HTML,
    # else first non-empty line). Keep it short.
    title = ""
    if "<title>" in content.lower():
        i = content.lower().find("<title>")
        j = content.lower().find("</title>", i)
        if i >= 0 and j > i:
            title = content[i + 7:j].strip()[:200]
    if not title:
        for line in content.splitlines():
            line = line.strip()
            if line:
                title = line[:200]
                break

    ws.add_source(url, content, title=title)
    truncated = content[:max_chars]
    return f"URL: {url}\nTitle: {title}\n\n{truncated}"


def add_sub_question(text: str, config: dict,
                     parent_id: Optional[str] = None) -> str:
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    qid = ws.add_question(text, parent_id=parent_id)
    return f"Sub-question added: {qid}\n  text: {text}"


def list_open_questions(config: dict) -> str:
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    qs = ws.list_open_questions()
    if not qs:
        return "(no open sub-questions — decompose further or call Finish)"
    out = [f"Open sub-questions ({len(qs)}):"]
    for q in qs:
        parent = f" (child of {q.parent_id})" if q.parent_id else ""
        n_findings = len(q.finding_ids)
        out.append(f"  {q.id}: {q.text}{parent}  [{n_findings} findings]")
    return "\n".join(out) + _stuck_banner(ws)


def mark_question_done(question_id: str, summary: str, config: dict) -> str:
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    if ws.mark_question_done(question_id, summary=summary):
        return f"Closed {question_id} with summary."
    return f"Could not close {question_id} (unknown id or already closed)."


def save_finding(claim: str, sub_question_id: str,
                 evidence_urls: List[str], config: dict) -> str:
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    if sub_question_id not in ws.questions:
        return f"Error: unknown sub_question_id {sub_question_id!r}."
    fid = ws.add_finding(claim, sub_question_id, evidence_urls)
    return f"Finding {fid} saved (linked to {sub_question_id}, {len(evidence_urls)} sources)."


def search_findings(query: str, config: dict, top_k: int = 5) -> str:
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    from rabbit_hole.synthesis import _bm25_rank_findings
    findings = ws.list_findings()
    ranked = _bm25_rank_findings(query, findings, top_k=top_k)
    if not ranked:
        return f"(no findings match {query!r})"
    out = [f"Top findings for {query!r}:"]
    for f, score in ranked:
        out.append(f"  [{score:.2f}] {f.id} ({f.sub_question_id}): {f.claim}")
    return "\n".join(out) + _stuck_banner(ws)


def list_sources(config: dict) -> str:
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    sources = ws.list_sources()
    if not sources:
        return "(no sources fetched yet)"
    out = [f"Cached sources ({len(sources)}):"]
    for s in sources:
        out.append(f"  - {s.get('title', '?')[:80]} — {s['url']}")
    return "\n".join(out) + _stuck_banner(ws)


def note(text: str, config: dict) -> str:
    """Record a free-form progress note for the user."""
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.advance_turn()
    text = text.strip()
    if not text:
        return "(empty note — provide a non-empty 'text' argument)"
    # Cap to keep status output readable
    if len(text) > 500:
        text = text[:500] + "…"
    ws.append_event("note", text=text)
    return f"Note recorded (visible in /rabbit-hole status)."


def finish(reason: str, config: dict) -> str:
    """Mark workspace finished. The harness wrapper detects this and
    triggers synthesis on next iteration."""
    ws = _ws(config)
    if ws is None:
        return _err_no_workspace()
    ws.manifest["status"] = "finished"
    ws.manifest["finish_reason"] = reason
    ws.manifest_path.write_text(__import__("json").dumps(ws.manifest, indent=2))
    return (f"Finish requested. Reason: {reason}\n"
            f"Synthesis will run when the loop yields. You should stop "
            f"calling tools after this.")
