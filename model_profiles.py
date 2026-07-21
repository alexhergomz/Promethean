"""Named model profiles — the daily-driver UX shortcut.

A profile is a short alias (e.g. `qwen`, `m2`) that resolves to a full
model spec plus the bits needed to make it reachable: provider, base_url
override, API key env var, default config knobs. The user switches via
`/model qwen` instead of typing the full provider/model string and
remembering whether to set context_limit or thinking budget.

Defaults are seeded in cc_config.DEFAULT_PROFILES and can be overridden
in ~/.promethean/config.json under the `model_profiles` key.
"""
from __future__ import annotations

import os
import socket
from typing import Optional
from urllib.parse import urlparse


# Default profiles shipped with the harness. The harness is llama.cpp-only,
# so the seed is a single local profile: Qwen3.5-9B via llama-server on :8080.
# Add your own in ~/.promethean/config.json under `model_profiles`.
DEFAULT_PROFILES: dict[str, dict] = {
    "qwen": {
        "model":           "custom/qwen3.5-9b",
        "provider":        "custom",
        "base_url":        "http://127.0.0.1:8080/v1",
        "api_key_env":     None,            # llama-server doesn't need one
        "description":     "Local Qwen3.5-9B (TurboQuant) via llama-server :8080",
        "config_overrides": {
            "context_limit":   57344,        # 229376 / 4 slots
            "max_tokens":      8192,
            "thinking":        None,
        },
    },
}


def get_profiles(config: dict) -> dict[str, dict]:
    """Merged profile map: defaults overlaid with user customizations."""
    merged = {k: dict(v) for k, v in DEFAULT_PROFILES.items()}
    for k, v in (config.get("model_profiles") or {}).items():
        merged.setdefault(k, {}).update(v or {})
    return merged


def apply(profile_name: str, config: dict) -> tuple[bool, str]:
    """Apply a profile to `config` in place. Returns (ok, message)."""
    profiles = get_profiles(config)
    if profile_name not in profiles:
        return False, f"no such profile: {profile_name!r}. Try /model to list."
    p = profiles[profile_name]
    config["model"] = p["model"]
    base_url = p.get("base_url")
    if base_url:
        # Provider-specific base_url overrides (mirrors config_cmd.py conventions).
        prov = p.get("provider")
        if prov == "custom":
            config["custom_base_url"] = base_url
        elif prov == "minimax":
            config["minimax_base_url"] = base_url
        else:
            config[f"{prov}_base_url"] = base_url
    for k, v in (p.get("config_overrides") or {}).items():
        config[k] = v
    return True, f"profile {profile_name} applied → {p['model']}"


def key_status(profile: dict, config: dict | None = None) -> tuple[bool, str]:
    """Returns (has_key, source). Source is one of 'env', 'config', 'n/a', 'none'."""
    env_var = profile.get("api_key_env")
    if not env_var:
        return True, "n/a"   # no key needed (local provider)
    if os.environ.get(env_var):
        return True, "env"
    if config is not None:
        prov = profile.get("provider")
        if prov and config.get(f"{prov}_api_key"):
            return True, "config"
    return False, "none"


def reachable(profile: dict, timeout: float = 0.4) -> Optional[bool]:
    """Quick TCP probe to base_url host:port. None if no base_url to probe.

    Intentionally NOT an HTTP request — we don't want to burn API quota or
    require keys just to render the /model listing.
    """
    base_url = profile.get("base_url")
    if not base_url:
        return None
    try:
        u = urlparse(base_url)
        host = u.hostname or "127.0.0.1"
        port = u.port or (443 if u.scheme == "https" else 80)
    except Exception:
        return None
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
