"""Rabbit-hole — long-running autonomous deep research.

A subagent runs until killed, fetching web sources and saving structured
findings to a workspace on disk. The agent's context is bounded; the
knowledge accumulates in the workspace directory. When the user kills
the run (or a hard cap fires), `synthesize_workspace()` builds a final
markdown report by BM25-retrieving findings per sub-question and
cross-referencing source URLs.

Disk layout (per task):

    .promethean/rabbit-hole/<task_id>/
        manifest.json                created_at, root_question, status
        questions.json               id → {text, status, parent_id, finding_ids}
        sources/<sha256>.json        {url, title, fetched_at}
        sources/<sha256>.txt         raw fetched content
        findings/<id>.json           {claim, sub_question_id, evidence_urls, source_shas}
        notes/turn-N.md              raw Think outputs
        state.json                   turn counters (for anti-stuck detection)
        reports/intermediate-N.md
        reports/final.md
"""
from rabbit_hole.store import RabbitHoleWorkspace
from rabbit_hole.synthesis import synthesize_workspace

__all__ = ["RabbitHoleWorkspace", "synthesize_workspace"]
