"""
commands/config_cmd.py — Configuration and model commands for Promethean.

Commands: /model, /config, /verbose, /thinking, /permissions, /cwd
"""
from __future__ import annotations

import json
import os

from ui.render import clr, info, ok, warn, err


def cmd_model(args: str, _state, config) -> bool:
    from providers import PROVIDERS, detect_provider
    import model_profiles as _mp

    arg = (args or "").strip()

    # ── Profile quick-switch ──────────────────────────────────────────────
    # `/model <profile>` is the fast path: one word swaps model+base_url+
    # config knobs in one step. Falls through to full lookup if `arg`
    # doesn't match a profile name.
    if arg and arg in _mp.get_profiles(config):
        ok_, msg = _mp.apply(arg, config)
        (ok if ok_ else err)(msg)
        if ok_:
            from cc_config import save_config
            save_config(config)
        return True

    if not arg:
        model = config["model"]
        pname = detect_provider(model)
        info(f"Current model:    {model}  (provider: {pname})")

        # Profiles section — show first, this is the daily-driver shortcut.
        profiles = _mp.get_profiles(config)
        if profiles:
            info("\nProfiles (quick switch):")
            for name, p in profiles.items():
                has_key, _src = _mp.key_status(p, config)
                reach = _mp.reachable(p)
                marks = []
                if not has_key:
                    marks.append(clr("no key", "red"))
                elif p.get("api_key_env"):
                    marks.append(clr("key ✓", "green"))
                if reach is True:
                    marks.append(clr("reachable", "green"))
                elif reach is False:
                    marks.append(clr("unreachable", "red"))
                marker = "  " + " ".join(marks) if marks else ""
                cur = "  " + clr("← current", "yellow") if p["model"] == model else ""
                info(f"  {clr('/model ' + name, 'cyan'):24s} {p['model']:30s}{marker}{cur}")
                info(f"    {clr(p.get('description',''), 'dim')}")

        info("\nAvailable models by provider:")
        for pn, pdata in PROVIDERS.items():
            if pn == "ollama":
                # Show live local models instead of hardcoded list
                from providers import list_ollama_models
                base_url = (
                    os.environ.get("OLLAMA_BASE_URL")
                    or config.get("ollama_base_url")
                    or pdata.get("base_url", "http://localhost:11434")
                )
                local = list_ollama_models(base_url)
                if local:
                    info(f"  {'ollama':12s}  " + ", ".join(local[:6]) + ("..." if len(local) > 6 else ""))
                    info(f"  {'':12s}  " + clr(f"({len(local)} local models — /model ollama to pick)", "dim"))
                else:
                    info(f"  {'ollama':12s}  " + clr("(not running or no models pulled)", "dim"))
                continue
            ms = pdata.get("models", [])
            if ms:
                info(f"  {pn:12s}  " + ", ".join(ms[:4]) + ("..." if len(ms) > 4 else ""))
        info("\nFormat: 'provider/model' or just model name (auto-detected)")
        info("  e.g. /model gpt-4o")
        info("  e.g. /model ollama/qwen2.5-coder")
        info("  e.g. /model kimi:moonshot-v1-32k")
    else:
        m = args.strip()
        # "/model ollama" with no model name → interactive picker
        if m == "ollama":
            if _interactive_ollama_picker(config):
                return True
            return True
        if "/" not in m and ":" in m:
            left, right = m.split(":", 1)
            if left in PROVIDERS:
                m = f"{left}/{right}"
        config["model"] = m
        pname = detect_provider(m)
        ok(f"Model set to {m}  (provider: {pname})")
        from cc_config import save_config
        save_config(config)
    return True


def cmd_failover(args: str, _state, config) -> bool:
    """Manage the cross-provider failover ladder.

    Usage:
        /failover                    show current ladder
        /failover <m1> <m2> ...      set ladder (space-separated model ids
                                     or profile names — profiles resolve)
        /failover add <model>        append to current ladder
        /failover off                clear (disable failover)
    """
    from cc_config import save_config
    import model_profiles as _mp

    parts = args.strip().split()
    ladder = list(config.get("failover_models") or [])
    profiles = _mp.get_profiles(config)

    def _resolve(tok: str) -> str:
        # If `tok` matches a profile name, use the profile's full model id.
        if tok in profiles:
            return profiles[tok]["model"]
        return tok

    if not parts:
        if not ladder:
            info("failover: off  (set via /failover <model1> <model2> ...)")
        else:
            info("failover ladder (engages on terminal failure of current model):")
            info(f"  primary    {config['model']}  ← active")
            for i, m in enumerate(ladder, 1):
                info(f"  fallback {i:<2} {m}")
        info("\nUse profile names too:  /failover m2 qwen")
        return True

    sub = parts[0].lower()
    if sub in ("off", "clear", "disable", "none"):
        config["failover_models"] = []
        save_config(config)
        ok("failover: off")
        return True
    if sub == "add" and len(parts) >= 2:
        new = _resolve(parts[1])
        if new in ladder or new == config.get("model"):
            warn(f"{new} already in ladder or is primary; not added.")
            return True
        ladder.append(new)
        config["failover_models"] = ladder
        save_config(config)
        ok(f"failover: appended {new}  →  {ladder}")
        return True

    # Set the full ladder.
    new_ladder = [_resolve(t) for t in parts]
    config["failover_models"] = new_ladder
    save_config(config)
    ok(f"failover ladder set: {new_ladder}")
    return True


def cmd_api(args: str, _state, config) -> bool:
    """One-step API key setup.

    Usage:
        /api                         show key status for all providers
        /api <provider>              prompt for a key, save to config
        /api <provider> <key>        set a key non-interactively (avoid in
                                     shared terminals; keys are stored in
                                     ~/.promethean/config.json plaintext)
    """
    from providers import PROVIDERS
    from cc_config import save_config

    parts = args.strip().split(None, 1)
    if not parts:
        # Status table for every provider that needs a key.
        info("API key status:")
        for pn, pdata in PROVIDERS.items():
            env_var = pdata.get("api_key_env")
            if not env_var:
                continue
            cfg_key = f"{pn}_api_key"
            env_v   = os.environ.get(env_var, "")
            cfg_v   = config.get(cfg_key, "")
            if env_v:
                mark = clr(f"✓ from env (${env_var})", "green")
                tail = f"{env_v[:4]}…{env_v[-4:]}"
            elif cfg_v:
                mark = clr("✓ from config", "green")
                tail = f"{cfg_v[:4]}…{cfg_v[-4:]}"
            else:
                mark = clr("✗ not set", "red")
                tail = ""
            info(f"  {pn:12s}  {mark}  {tail}")
        info("\nTo set: /api <provider> [key]   (e.g. /api minimax)")
        return True

    provider = parts[0].lower()
    if provider not in PROVIDERS:
        err(f"unknown provider: {provider!r}. Try: {', '.join(PROVIDERS.keys())}")
        return True

    pdata = PROVIDERS[provider]
    env_var = pdata.get("api_key_env")
    if not env_var:
        warn(f"{provider} doesn't need an API key.")
        return True

    if len(parts) == 2:
        key = parts[1].strip()
    else:
        try:
            key = input(clr(f"  Paste {provider} API key (or Enter to skip): ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); return True
        if not key:
            return True

    cfg_key = f"{provider}_api_key"
    config[cfg_key] = key
    # Also export to the live process so the next request picks it up
    # without restarting. Persisted in config across restarts via save_config.
    os.environ[env_var] = key
    save_config(config)
    ok(f"{provider} key saved (env {env_var}, also in ~/.promethean/config.json).")
    return True


def _interactive_ollama_picker(config: dict) -> bool:
    """Prompt the user to select from locally available Ollama models."""
    from providers import PROVIDERS, list_ollama_models
    from tools import ask_input_interactive
    prov = PROVIDERS.get("ollama", {})
    base_url = (
        os.environ.get("OLLAMA_BASE_URL")
        or config.get("ollama_base_url")
        or prov.get("base_url", "http://localhost:11434")
    )

    models = list_ollama_models(base_url)
    if not models:
        err(f"No local Ollama models found at {base_url}.")
        return False

    menu_buf = clr("\n  ── Local Ollama Models ──", "dim")
    for i, m in enumerate(models):
        menu_buf += "\n" + clr(f"  [{i+1:2d}] ", "yellow") + m
    print(menu_buf)
    print()

    try:
        ans = ask_input_interactive(clr("  Select a model number or Enter to cancel > ", "cyan"), config, menu_buf).strip()
        if not ans: return False
        idx = int(ans) - 1
        if 0 <= idx < len(models):
            new_model = f"ollama/{models[idx]}"
            config["model"] = new_model
            from cc_config import save_config
            save_config(config)
            ok(f"Model updated to {new_model}")
            return True
        else:
            err("Invalid selection.")
    except (ValueError, KeyboardInterrupt, EOFError):
        pass
    return False


def cmd_config(args: str, _state, config) -> bool:
    from cc_config import save_config
    if not args:
        _SECRETS = {"api_key", "anthropic_api_key", "telegram_token", "wechat_token"}
        display = {k: v for k, v in config.items()
                   if k not in _SECRETS and not k.startswith("_")
                   and not k.endswith(("_key", "_token", "_secret"))}
        print(json.dumps(display, indent=2))
    elif "=" in args:
        key, _, val = args.partition("=")
        key, val = key.strip(), val.strip()
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.isdigit():
            val = int(val)
        config[key] = val
        save_config(config)
        ok(f"Set {key} = {val}")
    else:
        k = args.strip()
        v = config.get(k, "(not set)")
        info(f"{k} = {v}")
    return True


def cmd_verbose(_args: str, _state, config) -> bool:
    from cc_config import save_config
    config["verbose"] = not config.get("verbose", False)
    state_str = "ON" if config["verbose"] else "OFF"
    ok(f"Verbose mode: {state_str}")
    save_config(config)
    return True


def cmd_thinking(_args: str, _state, config) -> bool:
    from cc_config import save_config
    config["thinking"] = not config.get("thinking", False)
    state_str = "ON" if config["thinking"] else "OFF"
    ok(f"Extended thinking: {state_str}")
    save_config(config)
    return True


def cmd_permissions(args: str, _state, config) -> bool:
    from cc_config import save_config
    from tools import ask_input_interactive
    modes = ["auto", "accept-all", "manual"]
    mode_desc = {
        "auto":       "Prompt for each tool call (default)",
        "accept-all": "Allow all tool calls silently",
        "manual":     "Prompt for each tool call (strict)",
    }
    if not args.strip():
        current = config.get("permission_mode", "auto")
        menu_buf = clr("\n  ── Permission Mode ──", "dim")
        for i, m in enumerate(modes):
            marker = clr("●", "green") if m == current else clr("○", "dim")
            menu_buf += f"\n  {marker} {clr(f'[{i+1}]', 'yellow')} {clr(m, 'cyan')}  {clr(mode_desc[m], 'dim')}"
        print(menu_buf)
        print()
        try:
            ans = ask_input_interactive(clr("  Select a mode number or Enter to cancel > ", "cyan"), config, menu_buf).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return True
        if not ans:
            return True
        if ans.isdigit() and 1 <= int(ans) <= len(modes):
            m = modes[int(ans) - 1]
            config["permission_mode"] = m
            save_config(config)
            ok(f"Permission mode set to: {m}")
        else:
            err("Invalid selection.")
    else:
        m = args.strip()
        if m not in modes:
            err(f"Unknown mode: {m}. Choose: {', '.join(modes)}")
        else:
            config["permission_mode"] = m
            save_config(config)
            ok(f"Permission mode set to: {m}")
    return True


def cmd_cwd(args: str, _state, config) -> bool:
    if not args.strip():
        info(f"Working directory: {os.getcwd()}")
    else:
        p = args.strip()
        try:
            os.chdir(p)
            ok(f"Changed directory to: {os.getcwd()}")
        except Exception as e:
            err(str(e))
    return True
