"""agent_tools — Aider-style repo map for the local Qwen agent.

Designed for programmatic use from Python scripts that the model writes
and executes via the Bash tool. Example session:

    $ python3 -c "
    from agent_tools import repo_map, find_symbol
    print(repo_map(root='.',
                   focus_files=['compaction.py'],
                   max_tokens=1500))
    print('---')
    for hit in find_symbol('compact_messages'):
        print(hit.file, hit.line, hit.kind)
    "

Public API:
    repo_map(root, focus_files=None, max_tokens=2048) -> str
    find_symbol(name, root='.', kind='def') -> list[Tag]
    get_callers(name, root='.') -> list[Tag]
    outline(file, root='.') -> str
    neighborhood(name, root='.') -> {"def", "callers", "callees"}
    path_between(a, b, root='.', max_hops=4) -> list[Hit]
    imports(file, root='.') -> {"uses", "used_by"}

All functions are deterministic and require zero VRAM — pure tree-sitter
parsing + a small PageRank computation in NetworkX.
"""
from __future__ import annotations

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

__all__ = [
    "repo_map",
    "find_symbol",
    "get_callers",
    "outline",
    "neighborhood",
    "path_between",
    "imports",
    "search_files",
]
