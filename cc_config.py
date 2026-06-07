"""Configuration management for Promethean (multi-provider)."""
import os
import json
from pathlib import Path

CONFIG_DIR        = Path.home() / ".promethean"
_LEGACY_CONFIG_DIR = Path.home() / ".cheetahclaws"  # pre-rebrand location
CONFIG_FILE       = CONFIG_DIR  / "config.json"
HISTORY_FILE      = CONFIG_DIR  / "input_history.txt"
SESSIONS_DIR      = CONFIG_DIR  / "sessions"
DAILY_DIR         = SESSIONS_DIR / "daily"       # daily/YYYY-MM-DD/session_*.json
SESSION_HIST_FILE = SESSIONS_DIR / "history.json" # master: all sessions ever

# kept for backward-compat (/resume still reads from here)
MR_SESSION_DIR = SESSIONS_DIR / "mr_sessions"


def migrate_legacy_config_dir() -> bool:
    """One-time rebrand migration: move ~/.cheetahclaws → ~/.promethean.

    Best-effort and idempotent — only acts when the new dir is absent and the
    legacy one exists, so existing users keep their config (incl. API keys),
    sessions, memory and undo history after the rename. Returns True if it
    migrated, False otherwise. Never raises.
    """
    try:
        if CONFIG_DIR.exists() or not _LEGACY_CONFIG_DIR.is_dir():
            return False
        _LEGACY_CONFIG_DIR.rename(CONFIG_DIR)
        return True
    except OSError:
        # Cross-device move or permissions — leave the legacy dir untouched;
        # the app simply starts with fresh config rather than crashing.
        return False

DEFAULTS = {
    "model":            "ollama/gemma4:e4b",
    "max_tokens":       40000,
    "permission_mode":  "auto",   # auto | accept-all | manual
    "verbose":          False,
    # Tri-state: None = unset (use provider default), True = ON, False = explicit OFF.
    # The explicit-OFF state matters for DeepSeek v4 where the server default
    # is ON; providers.py only injects the disable toggle when value is False.
    "thinking":         None,
    "thinking_budget":  10000,
    "custom_base_url":  "",       # for "custom" provider
    # ── Local llama-server autostart ──────────────────────────────────────
    # When the active model is a local llama-server (provider "custom" on a
    # loopback custom_base_url) and it isn't running, bootstrap starts it so
    # `promethean` is one command end-to-end. No-op for remote/cloud
    # providers. Needs llama_model_path set (and a binary: llama_server_bin
    # or a "llama-server" on PATH). See server_autostart.py.
    "llama_autostart":         True,
    "llama_server_bin":        "",   # default: shutil.which("llama-server")
    "llama_model_path":        "",   # path to the .gguf to serve (required)
    "llama_server_args":       [],   # extra args; default ["-ngl","99","-fa","on"]
    "llama_autostart_timeout": 180,  # seconds to wait for /health
    "max_tool_output":  32000,
    "max_agent_depth":  3,
    "max_concurrent_agents": 3,
    "session_daily_limit":   10000,    # max sessions kept per day in daily/
    "session_history_limit": 100000,  # max sessions kept in history.json
    # ── Context limit override ─────────────────────────────────────────────
    # When non-null, takes priority over the provider's default context
    # limit. CRITICAL for self-hosted llama-server with -np > 1: each
    # slot's capacity is n_ctx / n_parallel (e.g. -c 229376 -np 4 → 57344
    # per slot). Without this, the harness uses the provider default
    # (128000 for "custom") and conversations grow past the slot capacity,
    # producing "request exceeds context size" 400 errors that the
    # auto-compact path can't always recover from.
    #
    # For the default qcoder run.sh setup (-np 4), set 50000 (~87% of
    # 57344) — leaves headroom for the response.
    "context_limit": None,
    # ── Slot save-restore around subagent runs (llama-server only) ────────
    # When True, spawning a rabbit-hole or other heavy subagent saves the
    # parent's KV cache to disk before the subagent claims the slot, then
    # restores it on subagent exit. Net effect: parent and subagent each
    # get the FULL slot capacity (when np=1, the full -c size; default
    # ~229K with the qcoder run.sh setup) — just not concurrently. After
    # the subagent ends, the parent's next REPL turn finds its prefix
    # cached and skips re-prefill. Requires --slot-save-path on the
    # server (run.sh sets it by default).
    "auto_slot_swap": True,
    # ── Slot paging (llama-server only) ────────────────────────────────────
    # When True and provider="custom", subagents pin to specific KV-cache
    # slots via id_slot. On subagent finish the slot is erased, freeing it
    # for the next subagent. Without this, llama-server's LRU may evict
    # the parent's warm slot. Requires --slot-save-path to be set on the
    # server (run.sh does this by default).
    "enable_slot_paging": False,
    # ── Security settings ──────────────────────────────────────────────────
    # allowed_root: restrict file operations (Read/Write/Edit/Glob/Grep) to this
    # directory tree.  null = unrestricted (CLI default).  Set to the project
    # root in production deployments to prevent path traversal.
    "allowed_root": None,
    # shell_policy: controls Bash tool execution.
    #   "allow"   — execute freely (CLI default)
    #   "log"     — execute but write every command to stderr with session_id
    #   "deny"    — block all Bash execution
    "shell_policy": "allow",
    # ── Structured logging ─────────────────────────────────────────────────
    # log_level: "off" | "error" | "warn" | "info" | "debug"
    #   Default "warn" keeps the interactive CLI quiet; set to "info" on
    #   production servers to capture every API call, retry, and quota event.
    "log_level": "warn",
    # log_file: absolute path or null.  null → stderr (only warn/error visible
    #   at default level).  Point to a file in production for persistent logs.
    "log_file": None,
    # ── OptiLLM proxy ──────────────────────────────────────────────────────
    # When set, requests forward through an upstream OptiLLM proxy that
    # applies an inference-time technique. Requires the proxy to be running
    # (pip install optillm; `optillm --base-url <upstream>`) AND
    # MINIMAX_BASE_URL (or equivalent) pointed at the proxy.
    # Slugs: moa | mcts | bon | plansearch | cot_reflection | re2 |
    #        self_consistency | mars | cepo (see TODO_NEXT.md).
    # Cost: each technique multiplies token use; use selectively for
    # hard sub-problems. Set null to disable.
    "optillm_approach": None,
    # ── Symbol-graph footer ───────────────────────────────────────────────
    # When the agent edits a function/class def, append a short footer to
    # the tool result listing other files that reference the same name.
    # Helps weaker models notice they need to update callers too. Skip if
    # working on a small project or you don't want the noise.
    "symbol_context": True,
    # ── Memories near compaction (context survival) ────────────────────────
    # When True, just before the context window is compacted into a lossy
    # summary, an extra LLM pass rescues durable facts (decisions, project
    # state, user constraints, deliberated values) from the messages about to
    # be dropped and writes them to project-scope memory — so they survive
    # this and future sessions. Costs one small extra call per auto-compact.
    # Set False to disable.
    "compaction_memory": True,
    # ── Hooks ──────────────────────────────────────────────────────────────
    # User-defined shell commands that fire on agent lifecycle events. See
    # hooks.py for the full schema. Empty list = no hooks. Example: auto-
    # format files after Edit/Write with
    #   {"event": "post_tool", "match": "Write|Edit",
    #    "run": "ruff format \"$FILE\""}
    "hooks": [],
    # ── Anti-stuck heuristic ──────────────────────────────────────────────
    # When the model calls the same tool with identical arguments 3 times in
    # the last 6 invocations, the harness injects a one-shot user message
    # asking the model to step back. Helps weaker API models (M2, 9B Qwen)
    # break out of "I'll read the same file again" loops. Set False to
    # disable.
    "anti_stuck": True,
    # ── Failover ladder ───────────────────────────────────────────────────
    # When the active model hits terminal failure (retries exhausted, or a
    # non-retryable error like AUTH on the primary), the agent advances to
    # the next entry in this list and continues the turn there. The switch
    # is sticky for the rest of run() but resets on the next user message —
    # so transient primary outages don't lock the session onto a fallback.
    # Empty list disables the feature (legacy behavior: give up on exhaustion).
    "failover_models": [],
    # ── Circuit breaker ────────────────────────────────────────────────────
    # circuit_failure_threshold: consecutive failures (in window) to trip open.
    "circuit_failure_threshold": 5,
    # circuit_window_seconds: rolling window for failure counting.
    "circuit_window_seconds": 60,
    # circuit_cooldown_seconds: how long to stay OPEN before probing again.
    "circuit_cooldown_seconds": 120,
    # ── Output-truncation auto-continue (Aider/OpenCode pattern) ───────────
    # When the provider returns finish_reason="length" the model hit max_tokens
    # mid-output. We detect this, drop any malformed tool call, and inject a
    # synthetic "continue / split into smaller chunks" hint as a user message
    # so the next turn can recover. max_continuations caps consecutive
    # auto-continues to prevent a runaway loop on a model that can't recover.
    # Set to 0 to disable the feature.
    "max_continuations": 3,
    # ── Quota / budget control ─────────────────────────────────────────────
    # All limits are null (unlimited) by default.  Set to enforce hard caps.
    "session_token_budget": None,  # max tokens (in+out) per session
    "session_cost_budget":  None,  # max USD per session
    "daily_token_budget":   None,  # max tokens today (all sessions)
    "daily_cost_budget":    None,  # max USD today (all sessions)
    # Per-provider API keys (optional; env vars take priority)
    # "anthropic_api_key": "sk-ant-..."
    # "openai_api_key":    "sk-..."
    # "gemini_api_key":    "..."
    # "kimi_api_key":      "..."
    # "qwen_api_key":      "..."
    # "zhipu_api_key":     "..."
    # "deepseek_api_key":  "..."
}


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass
    # Backward-compat: legacy single api_key → anthropic_api_key
    if cfg.get("api_key") and not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = cfg.pop("api_key")
    # Also accept ANTHROPIC_API_KEY env for backward-compat
    if not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    return cfg


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(exist_ok=True)
    # Strip internal runtime keys (e.g. _run_query_callback) before saving
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def current_provider(cfg: dict) -> str:
    from providers import detect_provider
    return detect_provider(cfg.get("model", "claude-opus-4-6"))


def has_api_key(cfg: dict) -> bool:
    """Check whether the active provider has an API key configured."""
    from providers import get_api_key
    pname = current_provider(cfg)
    key = get_api_key(pname, cfg)
    return bool(key)


def calc_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    from providers import calc_cost as _cc
    return _cc(model, in_tokens, out_tokens)
