"""Final-report synthesis for a rabbit-hole workspace.

Reuses the BM25 + identifier-aware tokenizer from agent_tools.helpers —
the same machinery that powers SearchFiles. We don't have a symbol
graph here so PageRank doesn't apply, but the lexical retrieval is the
load-bearing piece for finding which findings answer which questions.

Strategy
  Walk the sub-question tree depth-first. For each LEAF question:
    1. Use BM25 to rank findings by (claim relevance to question text).
    2. Cluster top findings by URL — same source supporting multiple
       claims becomes one entry with multiple claims listed.
    3. Surface contradictions — the same URL cited in findings whose
       claim text disagrees (lexical contradiction proxy: very low
       Jaccard between claim tokens).
  Walk back up the tree, rolling each leaf's section into its parent's
  section as a sub-bullet block.

Output is a single markdown report with:
  • TL;DR built from the closed root question's summary
  • A section per question (recursive)
  • Sources index sorted by # of findings citing them
  • Open questions still unanswered
  • Footer with run statistics
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from rabbit_hole.store import Finding, Question, RabbitHoleWorkspace


def _bm25_rank_findings(
    query: str, findings: List[Finding], top_k: int = 8,
) -> List[Tuple[Finding, float]]:
    """Rank findings by BM25 relevance of claim text to query.

    Reuses _tokenize_for_search from the standalone search_tokenize module —
    same identifier-aware splitter used in SearchFiles, so a question
    mentioning 'TurboQuant' will match a finding whose claim says
    'turbo_quant'. Imported from search_tokenize (not agent_tools.helpers) so
    BM25 search doesn't drag in the optional tree-sitter graph deps."""
    from search_tokenize import _tokenize_for_search

    q_tokens = set(_tokenize_for_search(query))
    if not q_tokens or not findings:
        return []

    docs = [(f, _tokenize_for_search(f.claim)) for f in findings]
    df: Dict[str, int] = defaultdict(int)
    for _, toks in docs:
        for t in set(toks) & q_tokens:
            df[t] += 1

    n = len(docs)
    avg_dl = sum(len(t) for _, t in docs) / max(1, n)

    def _idf(t: str) -> float:
        return max(0.0, math.log((n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1))

    K1, B = 1.5, 0.75
    scored: List[Tuple[Finding, float]] = []
    for f, toks in docs:
        tf: Dict[str, int] = defaultdict(int)
        for t in toks:
            if t in q_tokens:
                tf[t] += 1
        if not tf:
            continue
        dl = len(toks)
        denom = K1 * (1 - B + B * dl / max(1, avg_dl))
        score = sum(
            _idf(t) * (tf[t] * (K1 + 1)) / (tf[t] + denom)
            for t in tf
        )
        scored.append((f, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _claim_token_set(claim: str) -> set:
    from search_tokenize import _tokenize_for_search
    return set(_tokenize_for_search(claim))


def _is_contradiction(a: Finding, b: Finding) -> bool:
    """Heuristic: two findings citing overlapping URLs but with low-overlap
    claim tokens indicates a contradiction (or at least a tension worth
    surfacing). Pure lexical proxy — false positives possible."""
    if not (set(a.evidence_urls) & set(b.evidence_urls)):
        return False
    ta, tb = _claim_token_set(a.claim), _claim_token_set(b.claim)
    if not ta or not tb:
        return False
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    return jaccard < 0.2


def _section_for_question(
    ws: RabbitHoleWorkspace,
    q: Question,
    all_findings: List[Finding],
    all_questions: List[Question],
    depth: int = 0,
) -> str:
    """Render one section for a question, recursively folding in children."""
    indent = "  " * depth
    out = []
    h = "#" * (depth + 2)
    status_marker = "✓" if q.status == "closed" else "○"
    out.append(f"\n{h} {status_marker} {q.text}\n")

    if q.summary:
        out.append(f"{q.summary}\n")

    # BM25-rank findings tied to this question OR whose claim matches its text
    own_findings = [f for f in all_findings if f.sub_question_id == q.id]
    related = _bm25_rank_findings(q.text, all_findings, top_k=12)
    related_findings = [f for f, _ in related if f.sub_question_id != q.id]

    if own_findings:
        out.append(f"{indent}**Findings:**\n")
        for f in own_findings:
            out.append(f"{indent}- {f.claim}")
            if f.evidence_urls:
                cites = " ".join(f"[{i+1}]({u})"
                                 for i, u in enumerate(f.evidence_urls[:3]))
                out.append(f"{indent}  Sources: {cites}")
        out.append("")

    if related_findings:
        out.append(f"{indent}*See also (cross-referenced from other questions):*")
        for f in related_findings[:3]:
            # Show the FULL claim — truncating at 140 chars was clipping
            # mid-sentence and people noticed (the user complained about
            # "synthesis getting cut off in certain places"). The number
            # of related findings is already capped at 3 above; that's
            # what controls report length, not per-claim truncation.
            out.append(f"{indent}- {f.claim} _(from question {f.sub_question_id})_")
        out.append("")

    # Surface contradictions among findings tied to this question
    contradictions = []
    for i, a in enumerate(own_findings):
        for b in own_findings[i + 1:]:
            if _is_contradiction(a, b):
                contradictions.append((a, b))
    if contradictions:
        out.append(f"{indent}**⚠ Possible contradictions:**")
        for a, b in contradictions[:3]:
            # Use a multi-line layout instead of inline ↔ so neither
            # claim is forced into a single backtick'd line. This was
            # the other place where 80-char truncation cut content.
            out.append(f"{indent}- *Claim A:* {a.claim}")
            out.append(f"{indent}  *Claim B:* {b.claim}")
            out.append(f"{indent}  _(both cite overlapping URLs but the "
                       f"claim tokens disagree — verify manually)_")
        out.append("")

    # Recurse into children
    children = [c for c in all_questions if c.parent_id == q.id]
    for child in children:
        out.append(_section_for_question(
            ws, child, all_findings, all_questions, depth=depth + 1))
    return "\n".join(out)


def synthesize_workspace(
    ws: RabbitHoleWorkspace,
    final: bool = True,
    cancelled: bool = False,
) -> str:
    """Build a markdown report from the workspace state.

    If `final` is True, write to reports/final.md and return the report
    text. If False, write a numbered intermediate report.
    """
    findings = ws.list_findings()
    questions = ws.list_all_questions()
    sources   = ws.list_sources()

    # The "root" question is the one with no parent_id. There may be
    # multiple if the agent generated several top-level decompositions.
    roots = [q for q in questions if q.parent_id is None]

    out = []
    out.append("# Rabbit-Hole Research Report")
    if cancelled:
        out.append("> **Note:** the run was cancelled by the user — "
                   "the report is built from partial findings.")
    out.append("")

    # ── TL;DR ────────────────────────────────────────────────────────────
    out.append("## TL;DR")
    if not findings:
        out.append("(no findings recorded)")
    else:
        # Use root-question summaries if available; else first-3 longest claims
        summaries = [q.summary for q in roots if q.summary]
        if summaries:
            for s in summaries:
                out.append(f"- {s}")
        else:
            for f in sorted(findings, key=lambda f: -len(f.claim))[:3]:
                out.append(f"- {f.claim}")
    out.append("")

    # ── Per-question sections ────────────────────────────────────────────
    out.append("## Findings by question")
    if not roots:
        out.append("(no questions decomposed — agent may have ended early)")
    else:
        for r in roots:
            out.append(_section_for_question(ws, r, findings, questions, depth=0))

    # ── Sources index ────────────────────────────────────────────────────
    out.append("\n## Sources")
    if not sources:
        out.append("(no sources fetched)")
    else:
        # Count findings citing each source
        cite_count: Dict[str, int] = defaultdict(int)
        url_to_title: Dict[str, str] = {}
        for s in sources:
            url_to_title[s["url"]] = s.get("title", s["url"])
        for f in findings:
            for u in f.evidence_urls:
                cite_count[u] += 1
        ranked_urls = sorted(
            url_to_title.items(),
            key=lambda kv: (-cite_count.get(kv[0], 0), kv[0]),
        )
        for url, title in ranked_urls:
            n = cite_count.get(url, 0)
            tag = f" _(cited {n}×)_" if n else ""
            # Cap title display at 200 chars (was 100) — fetched-page
            # titles can run long with site nav/breadcrumbs concatenated.
            # Show enough to identify the source; URL link gives the
            # full address regardless. Use ellipsis only when actually
            # over the cap so short titles aren't cluttered.
            display_title = title if len(title) <= 200 else title[:197] + "…"
            out.append(f"- [{display_title}]({url}){tag}")

    # ── Open questions ──────────────────────────────────────────────────
    open_qs = [q for q in questions if q.status == "open"]
    if open_qs:
        out.append("\n## Open questions (unanswered)")
        for q in open_qs:
            out.append(f"- {q.text}")

    # ── Footer ──────────────────────────────────────────────────────────
    out.append("\n---")
    out.append(f"_run statistics: {len(findings)} findings, "
               f"{len(sources)} unique sources, "
               f"{len(questions)} sub-questions ({len(open_qs)} still open), "
               f"{ws.state.turn} agent turns_")

    report = "\n".join(out)

    if final:
        (ws.reports_dir / "final.md").write_text(report, encoding="utf-8")
    else:
        n = len(list(ws.reports_dir.glob("intermediate-*.md")))
        (ws.reports_dir / f"intermediate-{n + 1:03d}.md").write_text(
            report, encoding="utf-8")
    return report
