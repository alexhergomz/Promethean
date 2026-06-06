"""
quota.py — Per-session and per-day token / cost quota enforcement.

check_quota() is called before each API request.  When a limit would be
exceeded it raises QuotaExceeded so the agent can surface the error cleanly
instead of making a billable call.

Config keys (all optional; None / 0 = no limit):
  session_token_budget  int    max tokens (in+out) per session
  session_cost_budget   float  max USD per session
  daily_token_budget    int    max tokens today (all sessions in this process)
  daily_cost_budget     float  max USD today (all sessions in this process)

Daily counters are stored in ~/.promethean/quota/YYYY-MM-DD.json.
Thread-safe within a single process; no cross-process locking.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class QuotaExceeded(Exception):
    """Raised before an API call when a configured budget would be exceeded."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ── In-memory counters (per session, reset on session end) ─────────────────

_lock          = threading.Lock()
_sess_tokens:  dict[str, int]   = {}   # session_id → total tokens
_sess_cost:    dict[str, float] = {}   # session_id → total cost (USD)


# ── Daily file helpers ─────────────────────────────────────────────────────

def _quota_dir() -> Path:
    from cc_config import CONFIG_DIR
    d = CONFIG_DIR / "quota"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _load_daily() -> tuple[int, float]:
    """Return (tokens, cost) from today's on-disk record. Lock must be held."""
    p = _quota_dir() / f"{_today_key()}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return int(data.get("tokens", 0)), float(data.get("cost", 0.0))
    except Exception:
        return 0, 0.0


def _save_daily(tokens: int, cost: float) -> None:
    """Persist today's cumulative usage. Lock must be held."""
    p = _quota_dir() / f"{_today_key()}.json"
    try:
        p.write_text(
            json.dumps({"tokens": tokens, "cost": cost}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


# ── Public API ─────────────────────────────────────────────────────────────

def check_quota(session_id: str, config: dict) -> None:
    """
    Raise QuotaExceeded if any configured limit has already been reached.
    Call this BEFORE making an API request.
    """
    lim_st = config.get("session_token_budget") or 0
    lim_sc = config.get("session_cost_budget")  or 0.0
    lim_dt = config.get("daily_token_budget")   or 0
    lim_dc = config.get("daily_cost_budget")    or 0.0

    # Fast path: no limits configured
    if not any((lim_st, lim_sc, lim_dt, lim_dc)):
        return

    with _lock:
        st = _sess_tokens.get(session_id, 0)
        sc = _sess_cost.get(session_id, 0.0)
        dt, dc = _load_daily()

    if lim_st and st >= lim_st:
        raise QuotaExceeded(
            f"Session token budget reached ({st:,}/{lim_st:,} tokens)"
        )
    if lim_sc and sc >= lim_sc:
        raise QuotaExceeded(
            f"Session cost budget reached (${sc:.4f}/${lim_sc:.4f})"
        )
    if lim_dt and dt >= lim_dt:
        raise QuotaExceeded(
            f"Daily token budget reached ({dt:,}/{lim_dt:,} tokens)"
        )
    if lim_dc and dc >= lim_dc:
        raise QuotaExceeded(
            f"Daily cost budget reached (${dc:.4f}/${lim_dc:.4f})"
        )


def record_usage(session_id: str, model: str, in_tokens: int, out_tokens: int) -> None:
    """
    Record token usage after a successful API call.
    Updates in-memory session counters and the on-disk daily record.
    """
    from providers import calc_cost
    tokens = in_tokens + out_tokens
    cost   = calc_cost(model, in_tokens, out_tokens)

    with _lock:
        _sess_tokens[session_id] = _sess_tokens.get(session_id, 0) + tokens
        _sess_cost[session_id]   = _sess_cost.get(session_id, 0.0) + cost
        dt, dc = _load_daily()
        _save_daily(dt + tokens, dc + cost)

    import logging_utils as _log
    _log.info("usage_recorded",
              session_id=session_id,
              model=model,
              in_tokens=in_tokens,
              out_tokens=out_tokens,
              session_tokens=_sess_tokens[session_id],
              session_cost_usd=round(_sess_cost[session_id], 6))


def get_usage(session_id: str) -> dict:
    """Return current usage stats for a session (for /quota status command)."""
    with _lock:
        dt, dc = _load_daily()
        return {
            "session_tokens": _sess_tokens.get(session_id, 0),
            "session_cost":   _sess_cost.get(session_id, 0.0),
            "daily_tokens":   dt,
            "daily_cost":     dc,
        }


def reset_session(session_id: str) -> None:
    """Clear in-memory counters for a session that has ended."""
    with _lock:
        _sess_tokens.pop(session_id, None)
        _sess_cost.pop(session_id, None)
