"""Demo: every visualization case for the graph tools.

Run from a real terminal to see the actual colors / animation:

    cd /home/alex/Escritorio/LLM/promethean
    $HOME/miniconda3/bin/python tests/demo_graph_viz.py

What you'll see (in order):

  Neighborhood
    1. rich symbol with multiple callers + callees (validate_token)
    2. orphan — defined but never called (connect)
    3. missing — symbol not in repo (does_not_exist)

  PathBetween (boxed graph view, path lit up green, siblings dimmed)
    4. 2-hop chain with sibling branches at each step
    5. cycle terminates cleanly (validate_token ↔ refresh_token)
    6. no path within max_hops (handle_request → connect)

  Imports
    7. file with many forward + reverse deps (api/handler.py)
    8. orphan file: only used_by, no uses (db/connection.py)
    9. leaf file: only uses, never used by anyone (api/handler.py)

  SearchFiles
   10. rare token (floccinauci...) — README dominates
   11. common-ish multi-token query
   12. no matches at all
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import bootstrap as _bs  # noqa: F401
# Force the viz on even if we're being captured (pipes etc) — the whole
# point of this script is to show the rendering.
os.environ["CC_GRAPH_VIEW"] = "1"
sys.stdout.isatty = lambda: True   # type: ignore[assignment]

# Re-instantiate the viz console with force_terminal so colors render
# regardless of how stdout is wired.
import agent_tools.visualize as v
from rich.console import Console
v._console = Console(file=sys.stdout, force_terminal=True, color_system="256")

from agent_tools._tools_register import (
    _tool_imports,
    _tool_neighborhood,
    _tool_path_between,
    _tool_search_files,
)

FIXTURE = str(Path(_HERE) / "fixtures" / "symbol_graph_repo")


def _section(n: int, title: str) -> None:
    bar = "═" * 70
    print()
    print(f"\033[1;36m{bar}\033[0m")
    print(f"\033[1;36m  {n:>2}.  {title}\033[0m")
    print(f"\033[1;36m{bar}\033[0m")
    print()
    time.sleep(0.25)


def _pause(s: float = 0.5) -> None:
    time.sleep(s)


def main() -> int:
    print("\033[1mGraph tool visualization demo\033[0m")
    print(f"Fixture: {FIXTURE}")
    print(f"CC_GRAPH_VIEW = {os.environ.get('CC_GRAPH_VIEW', '(unset)')}")

    # ── Neighborhood ────────────────────────────────────────────────────────

    _section(1, "Neighborhood — rich symbol with callers + callees")
    _tool_neighborhood({"name": "validate_token", "root": FIXTURE}, {})
    _pause()

    _section(2, "Neighborhood — orphan (defined, never called)")
    _tool_neighborhood({"name": "connect", "root": FIXTURE}, {})
    _pause()

    _section(3, "Neighborhood — missing symbol")
    _tool_neighborhood({"name": "does_not_exist", "root": FIXTURE}, {})
    _pause()

    # ── PathBetween ─────────────────────────────────────────────────────────

    _section(4, "PathBetween — short chain (2 hops, animated walk)")
    _tool_path_between(
        {"a": "handle_request", "b": "query_db", "root": FIXTURE}, {}
    )
    _pause()

    _section(5, "PathBetween — cycle terminates cleanly")
    _tool_path_between(
        {"a": "validate_token", "b": "refresh_token", "root": FIXTURE}, {}
    )
    _pause()

    _section(6, "PathBetween — unreachable target")
    _tool_path_between(
        {"a": "handle_request", "b": "connect",
         "root": FIXTURE, "max_hops": 6}, {}
    )
    _pause()

    # ── Imports ─────────────────────────────────────────────────────────────

    _section(7, "Imports — file with many forward deps")
    _tool_imports({"file": "api/handler.py", "root": FIXTURE}, {})
    _pause()

    _section(8, "Imports — file used by others, no own deps")
    _tool_imports({"file": "db/connection.py", "root": FIXTURE}, {})
    _pause()

    _section(9, "Imports — leaf module (utils/format.py)")
    _tool_imports({"file": "utils/format.py", "root": FIXTURE}, {})
    _pause()

    # ── SearchFiles ─────────────────────────────────────────────────────────

    _section(10, "SearchFiles — rare token (README sentinel)")
    _tool_search_files(
        {"query": "floccinaucinihilipilification",
         "root": FIXTURE, "top_k": 5}, {}
    )
    _pause()

    _section(11, "SearchFiles — multi-token symbol query")
    _tool_search_files(
        {"query": "validate_token refresh_token",
         "root": FIXTURE, "top_k": 5}, {}
    )
    _pause()

    _section(12, "SearchFiles — no matches")
    _tool_search_files(
        {"query": "zzzzzzzzz_definitely_not_in_corpus",
         "root": FIXTURE, "top_k": 5}, {}
    )

    print()
    print("\033[1;32mDemo complete.\033[0m  Run with "
          "\033[1mCC_GRAPH_VIEW=0\033[0m to see the silenced behavior.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
