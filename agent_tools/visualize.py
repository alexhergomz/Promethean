"""Live CLI visualization for the symbol-graph tools.

When a JSON-tool wrapper in `_tools_register.py` finishes its computation,
it calls one of the `viz_*` functions here BEFORE returning the text
payload to the model. The viz prints directly to stdout, sandwiched
between `print_tool_start` and `print_tool_end` in the agent loop, so
the user sees a pretty rendering while the model still gets the plain
text result.

Design choices:
  * Rich is a soft import — if missing, viz functions are no-ops.
  * Pure Python imports of `agent_tools.helpers` (e.g. user scripts)
    do NOT trigger viz; only the JSON-tool wrappers do.
  * Suppressed automatically when stdout isn't a TTY (test runners,
    pipes), so pytest output stays clean.
  * Toggleable via `CC_GRAPH_VIEW` env var (`0` / `off` / `false` / `no`
    disables; default is on).

PathBetween animates the walk with ~80 ms per hop — gives the "walk"
feel for a 3–4 hop chain at ~250–320 ms total cost, which is well
inside the model's per-turn latency budget.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, List

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    _RICH_OK = True
    _console = Console(file=sys.stdout, force_terminal=False)
except ImportError:
    _RICH_OK = False
    _console = None


# Explicit user override set via /graph-view slash command. None = auto.
_explicit: Optional[bool] = None


def set_enabled(state: Optional[bool]) -> None:
    """Override the auto-detected enabled state.

    `True` / `False` force the viz on/off; `None` reverts to the default
    detection logic (env var → TTY check). Used by the /graph-view
    slash command to flip the state in-session.
    """
    global _explicit
    _explicit = state


def is_enabled() -> bool:
    """Public read of the current enabled state — for /graph-view to display."""
    return _enabled()


def _enabled() -> bool:
    if not _RICH_OK:
        return False
    # Explicit slash-command override wins
    if _explicit is not None:
        return _explicit
    # Env var (initial / scripted control)
    flag = os.environ.get("CC_GRAPH_VIEW", "1").strip().lower()
    if flag in ("0", "off", "false", "no", ""):
        return False
    # Auto: only render on a real TTY
    return sys.stdout.isatty()


# ── Neighborhood: tree with def at root ──────────────────────────────────────

def viz_neighborhood(name: str, nb: dict) -> None:
    if not _enabled():
        return
    defs    = nb.get("def", [])
    callers = nb.get("callers", [])
    callees = nb.get("callees", [])

    label = f"[bold cyan]{name}[/]"
    if defs:
        d = defs[0]
        label += f"  [dim]→ {d.file}:{d.line}[/]"
    elif not callers and not callees:
        _console.print(f"[dim red]✗ no symbol [bold]{name}[/] in repo[/]")
        return

    tree = Tree(label, guide_style="dim")

    cb = tree.add(f"[yellow]← called by ({len(callers)})[/]")
    if not callers:
        cb.add("[dim](none)[/]")
    for h in callers[:10]:
        prev = (h.preview or "").strip()[:60]
        cb.add(f"[white]{h.file}:{h.line}[/]  [dim]{prev}[/]")
    if len(callers) > 10:
        cb.add(f"[dim]… and {len(callers) - 10} more[/]")

    ce = tree.add(f"[green]→ calls ({len(callees)})[/]")
    if not callees:
        ce.add("[dim](none)[/]")
    seen = set()
    n_shown = 0
    for h in callees:
        if h.name in seen:
            continue
        seen.add(h.name)
        ce.add(f"[white]{h.name}[/]  [dim]{h.file}:{h.line}[/]")
        n_shown += 1
        if n_shown >= 10:
            break
    leftover = len({h.name for h in callees}) - n_shown
    if leftover > 0:
        ce.add(f"[dim]… and {leftover} more unique[/]")

    _console.print(tree)


# ── PathBetween: boxed graph with lit-up path + sibling context ─────────────

def viz_path_between(
    a: str, b: str, chain: List[Any], siblings: List[List[str]] = None
) -> None:
    """Vertical boxed chain showing the path from `a` to `b` as a real graph.

    Each path node is rendered as a 4-line box (top border, name, file:line,
    bottom border) in bright green — the "lit up" traversed nodes. Between
    boxes, an arrow shows the next hop and (if `siblings` is provided)
    a dim list of OTHER callees of the current node — sibling branches
    that BFS could have taken but didn't. That's what turns the linear
    walk into a graph view: you see the local fan-out at each step, with
    your traversed branch highlighted.

    siblings: optional list[list[str]] aligned to chain. siblings[i] is
        the names of other callees of chain[i] (excluding chain[i+1]).
        None means no sibling info — falls back to a plain path render.
    """
    if not _enabled():
        return
    if not chain:
        _console.print(
            f"[dim red]✗ no path from [bold]{a}[/] to [bold]{b}[/][/]"
        )
        return

    sibs = siblings or [[] for _ in chain]
    indent = "  "

    _console.print(
        f"[bold]path:[/]  [cyan]{a}[/] [yellow]→[/] [cyan]{b}[/]  "
        f"[dim]({len(chain) - 1} hop{'s' if len(chain) != 2 else ''})[/]"
    )

    for i, h in enumerate(chain):
        is_first = i == 0
        is_last = i == len(chain) - 1

        name_str = h.name
        info_str = f"{h.file}:{h.line}"
        w = max(len(name_str), len(info_str)) + 4
        bar = "─" * (w - 2)
        pn = f" {name_str} ".ljust(w - 2)
        pi = f" {info_str} ".ljust(w - 2)

        marker = ""
        if is_first:
            marker = "   [bold yellow]◄ start[/]"
        elif is_last:
            marker = "   [bold yellow]◄ end[/]"

        _console.print(f"{indent}[bold green]╭{bar}╮[/]")
        _console.print(
            f"{indent}[bold green]│[/]"
            f"[bold bright_green]{pn}[/]"
            f"[bold green]│[/]{marker}"
        )
        _console.print(
            f"{indent}[bold green]│[/][dim]{pi}[/][bold green]│[/]"
        )
        _console.print(f"{indent}[bold green]╰{bar}╯[/]")

        if not is_last:
            others = [
                s for s in (sibs[i] if i < len(sibs) else [])
                if s != chain[i + 1].name and s != h.name
            ]
            others = others[:4]
            _console.print(f"{indent}[bold green]│[/]")
            if others:
                _console.print(
                    f"{indent}[bold green]├──[/] "
                    f"[dim]siblings: {', '.join(others)}[/]"
                )
            _console.print(f"{indent}[bold green]│[/]")
            _console.print(f"{indent}[bold green]▼[/]")
            time.sleep(0.10)   # tiny delay per hop = "walk" feel


# ── Imports: two-panel uses / used_by ────────────────────────────────────────

def viz_imports(file: str, deps: dict, depth: int = 1) -> None:
    if not _enabled():
        return
    uses    = deps.get("uses", [])
    used_by = deps.get("used_by", [])

    def _format(items, fmt):
        if not items:
            return "[dim](none)[/]"
        rows = [fmt(h) for h in items[:12]]
        if len(items) > 12:
            rows.append(f"[dim]… and {len(items) - 12} more[/]")
        return "\n".join(rows)

    uses_text = _format(
        uses,
        lambda h: f"[white]{h.name}[/]  [dim]{h.file}:{h.line}[/]",
    )
    by_text = _format(
        used_by,
        lambda h: f"[dim]{h.name:>20}[/]  [white]{h.file}:{h.line}[/]",
    )

    depth_marker = f"  [dim](depth={depth})[/]" if depth > 1 else ""
    p1 = Panel(uses_text, title=f"[yellow]uses ({len(uses)})[/]",
               border_style="dim", expand=True)
    p2 = Panel(by_text, title=f"[green]used_by ({len(used_by)})[/]",
               border_style="dim", expand=True)
    _console.print(f"[bold]{file}[/]{depth_marker}")
    _console.print(Columns([p1, p2], expand=True))


# ── SearchFiles: leaderboard with score bars ─────────────────────────────────

def viz_search_files(query: str, results: List[Any]) -> None:
    if not _enabled():
        return
    if not results:
        _console.print(f"[dim red]no matches for {query!r}[/]")
        return

    max_score = max((s for _, s, _ in results), default=1.0) or 1.0
    bar_width = 16

    table = Table(
        show_header=True, header_style="dim", box=None, padding=(0, 1),
        title=f"[bold]search:[/] [cyan]{query}[/]  "
              f"[dim]({len(results)} result{'s' if len(results) != 1 else ''})[/]",
        title_justify="left",
    )
    table.add_column("score", justify="right", style="dim", no_wrap=True)
    table.add_column("", no_wrap=True)
    table.add_column("file", overflow="fold")

    for rel, score, _preview in results:
        filled = int(round((score / max_score) * bar_width))
        filled = max(1, min(bar_width, filled))
        bar = (
            "[green]" + "▇" * filled + "[/]"
            + "[dim]" + "·" * (bar_width - filled) + "[/]"
        )
        table.add_row(f"{score:.2f}", bar, rel)

    _console.print(table)
