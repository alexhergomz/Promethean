"""Auxiliary model routing for cheap/fast side tasks.

Routes tasks like context compression, session title generation, and
vision analysis to a fast, inexpensive model instead of the user's
primary model. Falls back to the primary model if no auxiliary is available.

Config key: "auxiliary_model" (default: auto-detect)
"""
from __future__ import annotations

from typing import Optional

import providers

_resolved: Optional[str] = None


def get_auxiliary_model(config: dict) -> str:
    """Return the model to use for cheap side tasks.

    A dedicated ``auxiliary_model`` in config wins; otherwise the primary
    model handles these too. (Earlier builds routed side tasks to a cheap
    cloud model when a key was present — with the harness now llama.cpp-only,
    there's a single local model to use.)
    """
    return config.get("auxiliary_model") or config.get("model", "")


def reset_cache():
    """Clear the cached auxiliary model (kept for API/test compatibility)."""
    global _resolved
    _resolved = None


def stream_auxiliary(
    system: str,
    messages: list,
    config: dict,
) -> str:
    """Run a simple text completion with the auxiliary model.

    Returns the full response text (no streaming to user, no tools).
    """
    model = get_auxiliary_model(config)
    text = ""
    try:
        for event in providers.stream(
            model=model,
            system=system,
            messages=messages,
            tool_schemas=[],
            config=config,
        ):
            if isinstance(event, providers.TextChunk):
                text += event.text
    except Exception:
        # Auxiliary model failure should not crash the caller.
        # Return whatever text was collected so far.
        pass
    return text
