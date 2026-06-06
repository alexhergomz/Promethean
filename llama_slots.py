"""HTTP admin client for llama-server slot save/restore/erase.

The inference server (llama-server) maintains N parallel "slots", each
with its own KV cache. By default the parent agent and any subagents
share the same model weights but get isolated slots, so a subagent run
doesn't clobber the parent's prefix cache.

Slots can be `paged` to disk via the /slots/{id}?action=save endpoint
(requires the server to be started with --slot-save-path PATH). Once
saved, the slot can be `erased` to free its allocation for reuse. Later,
the saved KV state can be `restored` into any free slot.

Use case in this harness:
  1. Parent uses slot 0 with a long, well-warmed prefix cache.
  2. Parent spawns a deep-research subagent.
  3. SubAgent uses slot 1 (or any free slot) — parent's slot 0 stays warm.
  4. When the subagent finishes, save+erase its slot, freeing it for the
     next subagent OR for parent to expand into if needed.

This module is a thin httpx wrapper over the slot endpoints. It does
NOT decide WHEN to page — that policy lives in the SubAgentManager (or
the user's harness configuration). Failures (server down, slot endpoint
disabled, missing --slot-save-path) are surfaced as exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_SERVER_URL = "http://127.0.0.1:8080"


@dataclass
class SlotInfo:
    """One slot's runtime state, as reported by GET /slots."""
    id: int
    state: str           # "idle" | "processing" | …
    n_ctx: int           # this slot's max context
    n_past: int          # tokens currently held in KV
    prompt: str = ""     # cached prompt prefix the slot represents


class LlamaSlotsError(RuntimeError):
    """Raised when a /slots admin call fails."""


def _client(server_url: str = DEFAULT_SERVER_URL, timeout: float = 30.0):
    import httpx
    return httpx.Client(base_url=server_url.rstrip("/"), timeout=timeout)


def list_slots(server_url: str = DEFAULT_SERVER_URL) -> list[SlotInfo]:
    """Return live slot states. Raises LlamaSlotsError on server failure.

    Note: GET /slots requires --slots to be enabled at server start
    (it is enabled by default but may be disabled via --no-slots).
    """
    with _client(server_url) as c:
        r = c.get("/slots")
        if r.status_code != 200:
            raise LlamaSlotsError(
                f"GET /slots failed: HTTP {r.status_code} — {r.text[:200]}"
            )
        try:
            data = r.json()
        except Exception as e:
            raise LlamaSlotsError(f"GET /slots returned non-JSON: {e}")
    out: list[SlotInfo] = []
    for entry in data:
        out.append(SlotInfo(
            id=entry.get("id", -1),
            state=entry.get("state", "unknown"),
            n_ctx=entry.get("n_ctx", 0),
            n_past=entry.get("n_past", 0),
            prompt=(entry.get("prompt") or "")[:200],
        ))
    return out


def find_idle_slot(server_url: str = DEFAULT_SERVER_URL) -> Optional[int]:
    """Return the id of an idle slot, or None if every slot is busy.

    "idle" here matches llama-server's reported state for unused slots.
    """
    for s in list_slots(server_url):
        if s.state == "idle":
            return s.id
    return None


def save_slot(
    slot_id: int,
    filename: str,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Save slot {slot_id}'s KV cache to {filename} (relative to the
    server's --slot-save-path). Returns the server's response dict.

    Use this before erasing a slot whose state you may want back later
    (e.g. parking a long-running subagent's context for resumption).
    """
    if not filename or "/" in filename or ".." in filename:
        raise LlamaSlotsError(
            f"Invalid filename {filename!r} — must be a basename with no "
            f"directory separators or .. traversal."
        )
    with _client(server_url) as c:
        r = c.post(
            f"/slots/{slot_id}?action=save",
            json={"filename": filename},
        )
        if r.status_code != 200:
            raise LlamaSlotsError(
                f"save_slot({slot_id}, {filename}) failed: "
                f"HTTP {r.status_code} — {r.text[:200]}"
            )
        return r.json()


def restore_slot(
    slot_id: int,
    filename: str,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Restore slot {slot_id}'s KV from {filename}. The slot must be
    free (idle or recently erased) — restoring on top of an in-use slot
    fails server-side.
    """
    if not filename or "/" in filename or ".." in filename:
        raise LlamaSlotsError(
            f"Invalid filename {filename!r} — must be a basename with no "
            f"directory separators or .. traversal."
        )
    with _client(server_url) as c:
        r = c.post(
            f"/slots/{slot_id}?action=restore",
            json={"filename": filename},
        )
        if r.status_code != 200:
            raise LlamaSlotsError(
                f"restore_slot({slot_id}, {filename}) failed: "
                f"HTTP {r.status_code} — {r.text[:200]}"
            )
        return r.json()


def erase_slot(
    slot_id: int,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Erase slot {slot_id}'s KV cache, freeing the slot for reuse.

    This does NOT free VRAM — the slot allocation is fixed at server
    start (n_ctx / n_parallel per slot). It only marks the KV as empty
    so the next request can use the slot without prefix conflict.
    """
    with _client(server_url) as c:
        r = c.post(f"/slots/{slot_id}?action=erase")
        if r.status_code != 200:
            raise LlamaSlotsError(
                f"erase_slot({slot_id}) failed: "
                f"HTTP {r.status_code} — {r.text[:200]}"
            )
        return r.json()


def park_slot(
    slot_id: int,
    filename: str,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Save the slot's state to disk THEN erase it — the standard "I'm
    done with this context for now, free the slot but let me come back
    to it later" sequence. Returns the save response.
    """
    save_resp = save_slot(slot_id, filename, server_url=server_url)
    erase_slot(slot_id, server_url=server_url)
    return save_resp
