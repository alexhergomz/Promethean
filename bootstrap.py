"""
bootstrap.py — Explicit startup sequence for Promethean.

Call ``bootstrap(config)`` once after loading config and before starting
the REPL.  It makes the startup order visible, documented, and testable —
instead of relying on implicit module-level side effects scattered across
tools.py and other modules.

All steps are idempotent: calling bootstrap() more than once is harmless.

Startup sequence:
  1. Configure structured logging (earliest possible so all steps emit events)
  2. Ensure the tool registry is populated (imports tools.py)
  3. Start the optional health-check HTTP server (if health_check_port is set)
"""
from __future__ import annotations

import logging_utils as _log

_bootstrapped: bool = False


def bootstrap(config: dict) -> None:
    """Run the Promethean startup sequence.  Idempotent."""
    global _bootstrapped

    # ── 1. Structured logging ──────────────────────────────────────────────
    # Configure from config first so all subsequent startup events are logged.
    _log.configure_from_config(config)
    _log.info("bootstrap_start",
              model=config.get("model", ""),
              version=_get_version(),
              shell_policy=config.get("shell_policy", "allow"),
              allowed_root=config.get("allowed_root") or "(unrestricted)")

    # ── 2. Tool registry ───────────────────────────────────────────────────
    # tools.py self-registers built-ins + extension tools on first import.
    # This import is the single explicit trigger; subsequent imports are no-ops.
    import tools as _tools  # noqa: F401

    # Register agent_tools (Aider-style repo map) tools if the package is
    # importable. Best-effort — missing tree-sitter / grep_ast just skips.
    try:
        from agent_tools import _tools_register as _agent_tools_reg  # noqa: F401
        _log.info("agent_tools_registered")
    except Exception as exc:
        _log.debug("agent_tools_skipped", error=str(exc)[:150])

    _log.debug("bootstrap_tools_ready")

    # ── 3. Health-check HTTP server ────────────────────────────────────────
    port = config.get("health_check_port")
    if port:
        try:
            from health import start_health_server
            start_health_server(int(port), config)
            _log.info("health_server_started", port=int(port))
        except Exception as exc:
            _log.warn("health_server_failed", error=str(exc)[:200])

    _bootstrapped = True
    _log.info("bootstrap_done")


def _get_version() -> str:
    try:
        import promethean
        return getattr(promethean, "VERSION", "unknown")
    except Exception:
        return "unknown"
