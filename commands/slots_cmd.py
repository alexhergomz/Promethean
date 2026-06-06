"""/slots — runtime configuration for llama-server slots and serial mode.

Two layers of state:

  • SERVER-LEVEL (requires server restart):
      N_PARALLEL — number of slots / parallel contexts
      n_ctx      — total KV-cache budget (we leave at run.sh default)
    Persisted to ~/Escritorio/LLM/.qcoder-server.conf so qcoder picks
    up the new value on next start.

  • HARNESS-LEVEL (no restart needed):
      auto_slot_swap — when True, subagent spawn saves+erases parent's
                       slot, restores on subagent exit (serial mode).
                       When False, subagents claim free slots in
                       parallel (parallel mode).
    Lives in promethean config dict, persisted to
    ~/.promethean/config.json on /slots ... commands.

Subcommands:
  /slots                          show current state
  /slots np <N>                   set N_PARALLEL (server restart needed)
  /slots serial on|off            toggle auto_slot_swap (effective immediately)
  /slots restart                  restart llama-server with current settings

Three combinations the user might want:
  np=1 serial=on   (default) — single context, full 224K, save/restore around subagents
  np=4 serial=off            — 4 parallel contexts at 57K each, true parallelism
  np=4 serial=on             — 4 parallel slots PLUS serial save/restore for >4 agents
                               queueing onto them (slower, but unbounded concurrency)
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ui.render import clr, err, info, ok, warn


# Where qcoder writes the server-level config (sourced by qcoder before
# starting llama-server). Stays a shell-fragment file because qcoder is
# bash and run.sh reads env vars. Path mirrors qcoder's $QCODER_ROOT.
QCODER_ROOT = Path(os.environ.get("QCODER_ROOT") or (Path.home() / "Escritorio" / "LLM"))
QCODER_CONF = QCODER_ROOT / ".qcoder-server.conf"
USER_CONFIG = Path.home() / ".promethean" / "config.json"
SERVER_URL = "http://127.0.0.1:8080"


def _help() -> None:
    info("Usage:")
    info("  /slots                       show current slot configuration")
    info("  /slots np <N>                set parallel slot count (server restart needed)")
    info("  /slots serial on|off         toggle serial save/restore mode")
    info("  /slots restart               restart llama-server with current settings")
    info("")
    info("Combinations:")
    info("  /slots np 1   /slots serial on    single 224K slot, parent+subagent share serially (default)")
    info("  /slots np 4   /slots serial off   4 parallel 57K slots, true concurrency")
    info("  /slots np 4   /slots serial on    4 slots, queue beyond — slower, unbounded agents")


def _read_qcoder_conf() -> dict:
    """Parse the qcoder shell-config fragment into a dict. We only care
    about KEY=value pairs; values are unquoted as best-effort."""
    out: dict[str, str] = {}
    if not QCODER_CONF.exists():
        return out
    for line in QCODER_CONF.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        # Strip any leading 'export '
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        # Strip surrounding quotes if any
        val = val.strip().strip('"').strip("'")
        out[key.strip()] = val
    return out


def _write_qcoder_conf(updates: dict) -> None:
    """Merge updates into the qcoder shell-config fragment, preserving
    any keys we don't know about. Writes a clean canonical form."""
    current = _read_qcoder_conf()
    current.update(updates)
    QCODER_CONF.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-managed by /slots in promethean.",
        "# Sourced by qcoder before starting llama-server.",
        "# Edit by hand at your own risk — /slots will overwrite changes.",
        "",
    ]
    for key in sorted(current.keys()):
        val = current[key]
        # Always quote — values may contain spaces / special chars
        lines.append(f'export {key}="{val}"')
    lines.append("")
    QCODER_CONF.write_text("\n".join(lines))


def _save_user_config(config: dict) -> None:
    """Persist the user's promethean config to ~/.promethean/config.json.

    MERGES with the existing on-disk content rather than overwriting —
    the in-memory `config` dict at slash-command call time may not
    contain every key the user has set (e.g. when called from a test
    or a non-fully-initialised REPL state). Without merging we'd
    silently lose model/api_key/etc. on every /slots toggle.

    Strips the leading-underscore runtime keys that don't belong on
    disk (private state like _session_id).
    """
    USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if USER_CONFIG.exists():
        try:
            existing = json.loads(USER_CONFIG.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    # Merge: in-memory wins over disk (lets the user actually toggle
    # values), but disk-only keys (model, api_key, etc.) are preserved.
    merged = dict(existing)
    for k, v in config.items():
        if not k.startswith("_"):
            merged[k] = v
    USER_CONFIG.write_text(json.dumps(merged, indent=2))


def _server_running() -> bool:
    import httpx
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _live_slots_info() -> tuple[int, str]:
    """Return (num_slots, summary) from the running server, or (0, msg)
    if the server isn't reachable."""
    if not _server_running():
        return 0, "(server not running)"
    try:
        from llama_slots import list_slots
        slots = list_slots(SERVER_URL)
    except Exception as e:
        return 0, f"(error querying /slots: {e})"
    n = len(slots)
    busy = sum(1 for s in slots if s.state != "idle")
    return n, f"{n} slot(s), {busy} active"


def _show_status(config: dict) -> bool:
    """Print current slot configuration."""
    qc = _read_qcoder_conf()
    n_parallel_conf = qc.get("N_PARALLEL", "(default 1)")
    serial_mode = config.get("auto_slot_swap", True)

    info(f"Slot configuration:")
    print(f"  {'N_PARALLEL':<24} {clr(str(n_parallel_conf), 'cyan')}    "
          f"{clr('(server-side; restart to apply)', 'dim')}")
    print(f"  {'auto_slot_swap':<24} {clr(str(serial_mode), 'cyan')}    "
          f"{clr('(harness; effective immediately)', 'dim')}")

    n_live, summary = _live_slots_info()
    print()
    info(f"Server status: {summary}")
    if n_live > 0:
        try:
            from llama_slots import list_slots
            for s in list_slots(SERVER_URL):
                color = "yellow" if s.state == "processing" else "dim"
                print(f"  slot {s.id}: state={clr(s.state, color)}  "
                      f"n_past={s.n_past}/{s.n_ctx}")
        except Exception:
            pass

    print()
    if n_parallel_conf == "1" or n_parallel_conf == "(default 1)":
        if serial_mode:
            print(clr("  Mode: single context, serial — full slot capacity per agent, "
                      "subagent save/restore around runs", "dim"))
        else:
            print(clr("  ⚠ np=1 with serial=off — parent and subagent will thrash "
                      "the single slot", "yellow"))
    else:
        if serial_mode:
            print(clr(f"  Mode: {n_parallel_conf} contexts + serial — agents queue "
                      f"on slots, save/restore for >{n_parallel_conf} concurrent", "dim"))
        else:
            print(clr(f"  Mode: {n_parallel_conf} parallel contexts — true "
                      f"concurrency up to {n_parallel_conf} agents", "dim"))
    return True


def _set_np(n: int, config: dict) -> bool:
    """Set N_PARALLEL in qcoder.conf. Server must restart to pick up."""
    if n < 1 or n > 32:
        err(f"N must be 1..32 (got {n})")
        return True
    _write_qcoder_conf({"N_PARALLEL": str(n)})
    ok(f"N_PARALLEL set to {n} in {QCODER_CONF}.")
    info(f"Per-slot context: {229376 // n} tokens (with -c 229376 unchanged).")
    info(f"Run /slots restart  OR  qcoder server restart  to apply.")
    return True


def _set_serial(value: str, config: dict) -> bool:
    """Toggle auto_slot_swap. Effective immediately for new subagent spawns."""
    v = value.strip().lower()
    if v in ("on", "true", "1", "yes", "y"):
        config["auto_slot_swap"] = True
        _save_user_config(config)
        ok("Serial mode ON — subagent spawn saves+restores parent's slot.")
        info("Effective immediately for new subagents (running ones unaffected).")
    elif v in ("off", "false", "0", "no", "n"):
        config["auto_slot_swap"] = False
        # Auto-enable slot pinning so each subagent gets a dedicated slot
        # in parallel mode. Without this, llama-server picks slots by
        # prefix-matching which is fine but less deterministic.
        config["enable_slot_paging"] = True
        _save_user_config(config)
        ok("Serial mode OFF — subagents claim free slots in parallel.")
        info("Auto-set enable_slot_paging=True so each subagent gets a pinned slot.")
        warn("If N_PARALLEL=1, parent and subagent will thrash the single slot.")
    else:
        err(f"Expected 'on' or 'off', got {value!r}.")
    return True


def _restart_server(config: dict) -> bool:
    """Bounce llama-server. qcoder server restart sources qcoder.conf,
    so the new N_PARALLEL takes effect."""
    info("Running: qcoder server restart")
    try:
        proc = subprocess.run(
            ["qcoder", "server", "restart"],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        err("qcoder not found in PATH. Restart the server manually.")
        return True
    except subprocess.TimeoutExpired:
        err("qcoder server restart timed out. Check status manually.")
        return True
    if proc.stdout.strip():
        for line in proc.stdout.splitlines():
            print(f"  {line}")
    if proc.stderr.strip():
        for line in proc.stderr.splitlines():
            print(clr(f"  {line}", "dim"))
    if proc.returncode == 0:
        ok("Server restart complete.")
    else:
        err(f"qcoder server restart exited with code {proc.returncode}")
    return True


def cmd_slots(args: str, state, config) -> bool:
    a = args.strip()
    if not a:
        return _show_status(config)

    parts = a.split(None, 1)
    head = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if head in ("help", "-h", "--help"):
        _help()
        return True
    if head == "np":
        if not rest:
            err("Usage: /slots np <N>")
            return True
        try:
            n = int(rest.split()[0])
        except ValueError:
            err(f"N must be an integer, got {rest!r}")
            return True
        return _set_np(n, config)
    if head == "serial":
        if not rest:
            err("Usage: /slots serial on|off")
            return True
        return _set_serial(rest, config)
    if head == "restart":
        return _restart_server(config)
    if head == "status":
        return _show_status(config)

    err(f"Unknown subcommand: {head}")
    _help()
    return True
