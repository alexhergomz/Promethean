"""server_autostart.py — Bring up the local llama-server on demand.

When the active model is served by a local llama-server (provider ``custom``
pointing at a loopback address) and that server is not answering, this module
starts it for the user so ``promethean`` is a single command end-to-end. It is
best-effort: any failure degrades to a printed hint and never blocks the REPL.

Configuration (all under the normal config dict / ~/.promethean/config.json):
  llama_autostart        bool  enable autostart (default True)
  llama_server_bin       str   path to the llama-server binary
                               (default: first ``llama-server`` on PATH)
  llama_model_path       str   path to the .gguf to serve (required to start)
  llama_server_args      list  extra args (default ["-ngl","99","-fa","on"])
  llama_autostart_timeout int  seconds to wait for /health (default 180)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import logging_utils as _log

_LOOPBACK = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
_DEFAULT_ARGS = ["-ngl", "99", "-fa", "on"]


def _health_ok(base_root: str, timeout: float = 2.0) -> bool:
    """True if llama-server answers /health with status ok."""
    try:
        import httpx
        r = httpx.get(f"{base_root}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _server_root(base_url: str) -> str:
    """Strip a trailing /v1 (OpenAI path) to get the server root for /health."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def ensure_local_server(config: dict) -> None:
    """Start the local llama-server if the active model needs it and it's down.

    Idempotent and best-effort: returns silently for non-local providers and
    never raises.
    """
    try:
        _ensure(config)
    except Exception as exc:  # never let startup convenience break the REPL
        _log.debug("autostart_skipped", error=str(exc)[:200])


def _ensure(config: dict) -> None:
    if not config.get("llama_autostart", True):
        return

    # Only manage a *local* custom (llama-server) endpoint.
    from providers import detect_provider
    if detect_provider(config.get("model", "")) != "custom":
        return

    base_url = (config.get("custom_base_url")
                or os.environ.get("CUSTOM_BASE_URL") or "").strip()
    if not base_url:
        return
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host not in _LOOPBACK:
        return  # remote server — not ours to start
    port = parsed.port or 8080
    root = _server_root(base_url)

    if _health_ok(root):
        return  # already running

    # Resolve the binary and model.
    bin_path = config.get("llama_server_bin") or shutil.which("llama-server")
    model_path = config.get("llama_model_path") or ""
    missing = []
    if not bin_path or not Path(bin_path).exists():
        missing.append("llama_server_bin (or a 'llama-server' on PATH)")
    if not model_path or not Path(model_path).exists():
        missing.append("llama_model_path (.gguf)")
    if missing:
        print(f"[promethean] llama-server is down and autostart can't run: "
              f"missing {', '.join(missing)}.\n"
              f"            Set them with /config, or start the server "
              f"manually, then retry.", file=sys.stderr)
        _log.warn("autostart_unconfigured", missing=missing)
        return

    extra = config.get("llama_server_args") or _DEFAULT_ARGS
    cmd = [bin_path, "-m", model_path,
           "--host", host, "--port", str(port), *extra]

    # Log to the config dir so the user can inspect a failed launch.
    try:
        import cc_config
        log_dir = cc_config.CONFIG_DIR
    except Exception:
        log_dir = Path.home()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "llama-server.log"

    print(f"[promethean] starting llama-server ({Path(model_path).name}) ...",
          file=sys.stderr)
    _log.info("autostart_spawn", bin=bin_path, model=model_path, port=port)
    with open(log_file, "ab") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)

    timeout = int(config.get("llama_autostart_timeout", 180))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health_ok(root):
            print("[promethean] llama-server ready.", file=sys.stderr)
            _log.info("autostart_ready")
            return
        time.sleep(2)

    print(f"[promethean] llama-server not ready after {timeout}s; it may still "
          f"be loading. See {log_file}", file=sys.stderr)
    _log.warn("autostart_timeout", timeout=timeout, log=str(log_file))
