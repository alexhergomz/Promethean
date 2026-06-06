"""Tool plugin registry for promethean.

Provides a central registry for tool definitions, lookup, schema export,
dispatch with output truncation, and result caching for read-only tools.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDef:
    """Definition of a single tool plugin.

    Attributes:
        name: unique tool identifier
        schema: JSON-schema dict sent to the API (name, description, input_schema)
        func: callable(params: dict, config: dict) -> str
        read_only: True if the tool never mutates state
        concurrent_safe: True if safe to run in parallel with other tools
        cacheable: whether the registry should memoize results by (name, params)
            for read-only tools. Most read-only tools should leave this True.
            Set False when the tool's func has its own state-aware caching
            (e.g. Read uses the per-session FileContextTracker that knows
            about mtime and is smarter than naive (name, params) memoization).
    """
    name: str
    schema: Dict[str, Any]
    func: Callable[[Dict[str, Any], Dict[str, Any]], str]
    read_only: bool = False
    concurrent_safe: bool = False
    cacheable: bool = True


# --------------- internal state ---------------

_registry: Dict[str, ToolDef] = {}

# --------------- result cache (read-only tools only) ---------------

_CACHE_MAX = 64  # max cached entries
_cache: Dict[str, str] = {}   # hash → result
_cache_order: list[str] = []  # LRU eviction order


def _cache_key(name: str, params: Dict[str, Any]) -> str:
    """Create a stable hash from tool name + params."""
    raw = json.dumps({"n": name, "p": params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def clear_tool_cache() -> None:
    """Clear the tool result cache. Called on file writes to invalidate."""
    _cache.clear()
    _cache_order.clear()


# --------------- public API ---------------

def register_tool(tool_def: ToolDef) -> None:
    """Register a tool, overwriting any existing tool with the same name."""
    _registry[tool_def.name] = tool_def


def get_tool(name: str) -> Optional[ToolDef]:
    """Look up a tool by name. Returns None if not found."""
    return _registry.get(name)


def get_all_tools() -> List[ToolDef]:
    """Return all registered tools (insertion order)."""
    return list(_registry.values())


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return the schemas of all registered tools (for API tool parameter)."""
    return [t.schema for t in _registry.values()]


def execute_tool(
    name: str,
    params: Dict[str, Any],
    config: Dict[str, Any],
    max_output: int = 32000,
) -> str:
    """Dispatch a tool call by name.

    Args:
        name: tool name
        params: tool input parameters dict
        config: runtime configuration dict
        max_output: maximum allowed output length in characters

    Returns:
        Tool result string, possibly truncated.
    """
    # The `max_tool_output` config governs the output ceiling for EVERY tool
    # (text Read, PDF, CSV, Grep, WebFetch, …), enforced here in one place
    # rather than each tool hardcoding its own limit. config wins over the
    # parameter default — this is what makes max_tool_output a live setting.
    try:
        max_output = int(config.get("max_tool_output") or max_output)
    except (TypeError, ValueError):
        pass

    tool = get_tool(name)
    if tool is None:
        return f"Error: tool '{name}' not found."

    # Cache hit for read-only tools (same name + same params = same result),
    # but only if the tool opted in. Tools with their own state-aware
    # caching (e.g. Read with the FileContextTracker) opt out via
    # cacheable=False so the registry doesn't short-circuit them.
    use_cache = tool.read_only and getattr(tool, "cacheable", True)
    if use_cache:
        key = _cache_key(name, params)
        if key in _cache:
            return _cache[key]
    else:
        # Write tools invalidate cache (file content may have changed)
        if name in ("Write", "Edit", "Bash", "NotebookEdit"):
            clear_tool_cache()

    try:
        result = tool.func(params, config)
    except Exception as e:
        return f"Error executing {name}: {e}"

    # Store in cache for read-only tools
    if use_cache:
        _cache[key] = result
        _cache_order.append(key)
        # Evict oldest if over limit
        while len(_cache_order) > _CACHE_MAX:
            old = _cache_order.pop(0)
            _cache.pop(old, None)

    if len(result) > max_output:
        total = len(result)
        first_half = max_output // 2
        last_quarter = max_output // 4
        truncated = total - first_half - last_quarter
        # Tell the model what's missing AND how to get it coherently, rather
        # than silently cutting — readers (Read/ReadPDF/ReadSpreadsheet) page
        # via offset/limit, searches narrow via query/glob. Kept terse so it
        # never dominates a small max_tool_output budget.
        marker = (
            f"\n[... {truncated:,} of {total:,} chars truncated — "
            f"page with offset/limit ...]\n"
        )
        result = result[:first_half] + marker + result[-last_quarter:]

    return result


def clear_registry() -> None:
    """Remove all registered tools. Intended for testing."""
    _registry.clear()
