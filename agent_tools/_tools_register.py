"""Register the agent_tools helpers as first-class promethean tools.

Importing this module side-effects: the tools become visible to the model
in the JSON tool schema, sitting next to Read/Grep/Glob/Edit. This is the
fix for "model has overlay but still uses Grep" — the helpers need to be
in the tool schema, not just mentioned in prose.

Imported automatically by `promethean.bootstrap.bootstrap()` when the
agent_tools package is available.
"""
from __future__ import annotations

import json

from tool_registry import ToolDef, register_tool

from .helpers import (
    find_symbol,
    get_callers,
    imports,
    neighborhood,
    outline,
    path_between,
    repo_map,
    search_files,
)
from .visualize import (
    viz_imports,
    viz_neighborhood,
    viz_path_between,
    viz_search_files,
)


def _hits_to_text(hits) -> str:
    if not hits:
        return "(no matches)"
    out = []
    for h in hits:
        line = f"{h.file}:{h.line}  [{h.kind}]  {h.name}"
        if h.preview:
            line += f"\n  {h.preview}"
        out.append(line)
    return "\n".join(out)


def _tool_repo_map(params: dict, config: dict) -> str:
    return repo_map(
        root=params.get("root", "."),
        focus_files=params.get("focus_files"),
        max_tokens=params.get("max_tokens", 2048),
    )


def _tool_find_symbol(params: dict, config: dict) -> str:
    return _hits_to_text(find_symbol(
        params["name"],
        root=params.get("root", "."),
        kind=params.get("kind", "def"),
    ))


def _tool_get_callers(params: dict, config: dict) -> str:
    return _hits_to_text(get_callers(
        params["name"],
        root=params.get("root", "."),
    ))


def _tool_outline(params: dict, config: dict) -> str:
    return outline(params["file"], root=params.get("root", "."))


def _tool_neighborhood(params: dict, config: dict) -> str:
    nb = neighborhood(params["name"], root=params.get("root", "."))
    viz_neighborhood(params["name"], nb)
    parts = []
    for label, key in (("DEFINITION", "def"), ("CALLERS", "callers"),
                       ("CALLEES", "callees")):
        hits = nb[key]
        parts.append(f"── {label} ({len(hits)}) ──")
        parts.append(_hits_to_text(hits))
    return "\n".join(parts)


def _tool_path_between(params: dict, config: dict) -> str:
    root = params.get("root", ".")
    chain = path_between(
        params["a"], params["b"], root=root,
        max_hops=int(params.get("max_hops", 4)),
    )
    # Build sibling context for the graph viz (cheap on second pass —
    # SymbolIndex's underlying tag lookups are sqlite-cached). For each
    # node in the chain, collect names of other callees that BFS could
    # have taken but didn't (i.e. excluding the next hop). Fed to the
    # viz as dimmed branch labels next to the lit-up path.
    siblings = None
    if chain and len(chain) > 1:
        from .helpers import get_symbol_index
        idx = get_symbol_index(root)
        siblings = []
        for i, h in enumerate(chain):
            if i == len(chain) - 1:
                siblings.append([])
                continue
            seen, sibs = set(), []
            for d in idx.defs.get(h.name, []):
                for r in idx.callees_of(d):
                    if r.name in seen or r.name == h.name:
                        continue
                    seen.add(r.name)
                    sibs.append(r.name)
            siblings.append(sibs)
    viz_path_between(params["a"], params["b"], chain, siblings)
    if not chain:
        return (f"(no path from {params['a']!r} to {params['b']!r} within "
                f"{params.get('max_hops', 4)} hops)")
    lines = [f"{i}. {h.name}  ({h.file}:{h.line})"
             + (f"\n   {h.preview}" if h.preview else "")
             for i, h in enumerate(chain)]
    return f"Call chain {params['a']} → {params['b']} ({len(chain)-1} hops):\n" + "\n".join(lines)


def _tool_search_files(params: dict, config: dict) -> str:
    results = search_files(
        params["query"],
        root=params.get("root", "."),
        top_k=int(params.get("top_k", 10)),
    )
    viz_search_files(params["query"], results)
    if not results:
        return f"(no files matched query {params['query']!r})"
    lines = [f"{score:>7.2f}  {rel}" + (f"\n         {preview}" if preview else "")
             for rel, score, preview in results]
    return f"Top {len(results)} files for query {params['query']!r}:\n" + "\n".join(lines)


def _tool_imports(params: dict, config: dict) -> str:
    depth = int(params.get("depth", 1))
    deps = imports(
        params["file"], root=params.get("root", "."), depth=depth
    )
    viz_imports(params["file"], deps, depth=depth)
    parts = [
        f"── USES (this file depends on) ({len(deps['uses'])}) ──",
        _hits_to_text(deps["uses"]),
        f"── USED_BY (depends on this file) ({len(deps['used_by'])}) ──",
        _hits_to_text(deps["used_by"]),
    ]
    return "\n".join(parts)


# ── Schemas ───────────────────────────────────────────────────────────────────

register_tool(ToolDef(
    name="RepoMap",
    schema={
        "name": "RepoMap",
        "description": (
            "Aider-style ranked repo map: tree-sitter + PageRank over symbol "
            "references. Returns a markdown-ish summary of the most important "
            "files and their key class/function signatures, sized to ~max_tokens. "
            "Use this BEFORE diving into an unfamiliar codebase. Pass focus_files "
            "to bias the ranking toward files you're working on. Cheaper and "
            "more reliable than reading dozens of files individually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root":        {"type": "string", "description": "Repo root path. Default: cwd."},
                "focus_files": {"type": "array", "items": {"type": "string"},
                                "description": "Files of interest — biases ranking. Optional."},
                "max_tokens":  {"type": "integer", "description": "Output size cap (default 2048)."},
            },
            "required": [],
        },
    },
    func=_tool_repo_map,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="FindSymbol",
    schema={
        "name": "FindSymbol",
        "description": (
            "Find every DEFINITION of a symbol across the repo (exact name match). "
            "Returns file:line locations with a one-line preview of each definition. "
            "Use for 'where is X defined?' / 'show me MyClass'. "
            "Faster and more precise than grep for symbol lookup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact symbol name."},
                "root": {"type": "string", "description": "Repo root path. Default: cwd."},
                "kind": {"type": "string", "enum": ["def", "ref"],
                         "description": "'def' (default) = definitions; 'ref' = references."},
            },
            "required": ["name"],
        },
    },
    func=_tool_find_symbol,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="GetCallers",
    schema={
        "name": "GetCallers",
        "description": (
            "Find every REFERENCE (usage/call site) of a symbol across the repo. "
            "Returns file:line locations. Use for 'who calls foo?' / "
            "'where is this used?'. Cross-file impact analysis. "
            "More accurate than grep because it uses tree-sitter AST."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact symbol name."},
                "root": {"type": "string", "description": "Repo root path. Default: cwd."},
            },
            "required": ["name"],
        },
    },
    func=_tool_get_callers,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="Neighborhood",
    schema={
        "name": "Neighborhood",
        "description": (
            "GitNexus-style one-shot view of a symbol: its definition(s), "
            "callers (who references it), and callees (what it references "
            "inside its body). Single tool call replaces 2-3 grep+read "
            "rounds when the model is exploring a symbol's role. Callees "
            "are approximate — based on def-line spans, not full AST scope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact symbol name."},
                "root": {"type": "string", "description": "Repo root path. Default: cwd."},
            },
            "required": ["name"],
        },
    },
    func=_tool_neighborhood,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="PathBetween",
    schema={
        "name": "PathBetween",
        "description": (
            "Find a call chain from symbol `a` to symbol `b` via the "
            "tree-sitter symbol graph (BFS, ≤ max_hops). Use for impact "
            "analysis: 'how does the API handler reach the database?'. "
            "Returns the chain as ordered (symbol, file:line) hops. Empty "
            "if no path within max_hops."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "a":        {"type": "string", "description": "Source symbol name."},
                "b":        {"type": "string", "description": "Target symbol name."},
                "root":     {"type": "string", "description": "Repo root path. Default: cwd."},
                "max_hops": {"type": "integer", "description": "Max BFS depth (default 4)."},
            },
            "required": ["a", "b"],
        },
    },
    func=_tool_path_between,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="Imports",
    schema={
        "name": "Imports",
        "description": (
            "Forward + reverse dependencies of a single file by symbol "
            "graph (NOT by Python `import` statements). 'uses' = external "
            "defs this file references; 'used_by' = external refs to "
            "symbols this file defines. Use to understand a file's "
            "coupling before refactoring. Pass depth>1 to follow the "
            "dep chain transitively (e.g. depth=2 returns 2-hop deps too)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file":  {"type": "string", "description": "File path (absolute or relative to root)."},
                "root":  {"type": "string", "description": "Repo root path. Default: cwd."},
                "depth": {"type": "integer", "description": "Transitive depth (default 1). Cap at 3 for big repos."},
            },
            "required": ["file"],
        },
    },
    func=_tool_imports,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="SearchFiles",
    schema={
        "name": "SearchFiles",
        "description": (
            "TF-IDF keyword search over file contents. Use as a fallback "
            "when FindSymbol misses — for config keys, error messages, "
            "doc topics, or any free-text phrase that isn't a code "
            "symbol. Tokenizes identifiers + words, ranks files by "
            "TF-IDF score, returns top_k with a first-match preview line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":  {"type": "string", "description": "Free-text query (multiple words OK)."},
                "root":   {"type": "string", "description": "Repo root path. Default: cwd."},
                "top_k":  {"type": "integer", "description": "Max files to return (default 10)."},
            },
            "required": ["query"],
        },
    },
    func=_tool_search_files,
    read_only=True,
    concurrent_safe=True,
))

register_tool(ToolDef(
    name="Outline",
    schema={
        "name": "Outline",
        "description": (
            "List every definition (class/function/method/...) in a single file, "
            "in order, with line numbers. Quick way to understand a file's "
            "structure without reading the whole thing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File path (absolute or relative to root)."},
                "root": {"type": "string", "description": "Repo root path. Default: cwd."},
            },
            "required": ["file"],
        },
    },
    func=_tool_outline,
    read_only=True,
    concurrent_safe=True,
))
