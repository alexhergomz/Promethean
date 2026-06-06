"""Self-registering module for the rabbit-hole tools.

Loaded as an extension module by tools/__init__.py:_EXTENSION_MODULES.
Importing this file registers all eight rabbit-hole tools into the
central tool registry. The agent definition (in
multi_agent/subagent.py) restricts which agents see these tools via
its `tools` whitelist — they're available to any agent in principle
but only the deep-research-rabbit-hole agent will be told to use them.
"""
from __future__ import annotations

from tool_registry import ToolDef, register_tool
from rabbit_hole.tools import (
    RABBIT_HOLE_SCHEMAS,
    add_sub_question,
    finish,
    list_open_questions,
    list_sources,
    mark_question_done,
    note,
    rabbit_fetch,
    save_finding,
    search_findings,
)


_schemas = {s["name"]: s for s in RABBIT_HOLE_SCHEMAS}


_tool_defs = [
    ToolDef(
        name="RabbitFetch",
        schema=_schemas["RabbitFetch"],
        func=lambda p, c: rabbit_fetch(
            p["url"], c, max_chars=p.get("max_chars", 8000)),
        read_only=False,  # mutates the workspace cache
        concurrent_safe=False,
    ),
    ToolDef(
        name="AddSubQuestion",
        schema=_schemas["AddSubQuestion"],
        func=lambda p, c: add_sub_question(
            p["text"], c, parent_id=p.get("parent_id")),
        read_only=False, concurrent_safe=False,
    ),
    ToolDef(
        name="ListOpenQuestions",
        schema=_schemas["ListOpenQuestions"],
        func=lambda p, c: list_open_questions(c),
        read_only=True, concurrent_safe=True,
        cacheable=False,  # state mutates between calls
    ),
    ToolDef(
        name="MarkQuestionDone",
        schema=_schemas["MarkQuestionDone"],
        func=lambda p, c: mark_question_done(
            p["question_id"], p["summary"], c),
        read_only=False, concurrent_safe=False,
    ),
    ToolDef(
        name="SaveFinding",
        schema=_schemas["SaveFinding"],
        func=lambda p, c: save_finding(
            p["claim"], p["sub_question_id"],
            p.get("evidence_urls", []), c),
        read_only=False, concurrent_safe=False,
    ),
    ToolDef(
        name="SearchFindings",
        schema=_schemas["SearchFindings"],
        func=lambda p, c: search_findings(
            p["query"], c, top_k=p.get("top_k", 5)),
        read_only=True, concurrent_safe=True,
        cacheable=False,
    ),
    ToolDef(
        name="ListSources",
        schema=_schemas["ListSources"],
        func=lambda p, c: list_sources(c),
        read_only=True, concurrent_safe=True,
        cacheable=False,
    ),
    ToolDef(
        name="Note",
        schema=_schemas["Note"],
        func=lambda p, c: note(p["text"], c),
        read_only=False, concurrent_safe=False,
    ),
    ToolDef(
        name="Finish",
        schema=_schemas["Finish"],
        func=lambda p, c: finish(p["reason"], c),
        read_only=False, concurrent_safe=False,
    ),
]


for _td in _tool_defs:
    register_tool(_td)
