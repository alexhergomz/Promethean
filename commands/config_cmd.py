"""
commands/config_cmd.py — Configuration and model commands for Promethean.

Commands: /model, /config, /verbose, /thinking, /permissions, /cwd
"""
from __future__ import annotations

import json
import os

from ui.render import clr, info, ok, warn, err


def _is_local_profile(p: dict) -> bool:
    """A profile is local when it runs against a loopback backend — the
    llama.cpp/OpenAI-compatible `custom` provider on this machine."""
    if p.get("provider") == "custom":
        return True
    base = (p.get("base_url") or "")
    return "127.0.0.1" in base or "localhost" in base


def cmd_model(args: str, _state, config) -> bool:
    from providers import PROVIDERS, detect_provider
    import model_profiles as _mp

    arg = (args or "").strip()

    # ── Hardware-matched recommendations ──────────────────────────────────
    # `/model recommend [budget_gb] [family]` — narrow the huge quant lists to
    # what actually fits this machine.
    parts = arg.split()
    if parts and parts[0].lower() in ("recommend", "suggest", "recommendations"):
        return _cmd_model_recommend(parts[1:], config)

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

        # Hardware-matched local recommendation leads — this is a local-first
        # harness, so the first thing offered is the model sized to this machine.
        info(clr("\n  /model recommend", "cyan")
             + clr("  — best local GGUF for your hardware (Qwen3.5, Gemma 4, Nemotron)", "dim"))

        # Profiles — local only. A profile counts as local when it runs against
        # a loopback backend (custom llama-server / Ollama / LM Studio).
        profiles = _mp.get_profiles(config)
        local_profiles = {n: p for n, p in profiles.items()
                          if _is_local_profile(p)}
        if local_profiles:
            info("\nProfiles (quick switch):")
            for name, p in local_profiles.items():
                reach = _mp.reachable(p)
                marks = []
                if reach is True:
                    marks.append(clr("reachable", "green"))
                elif reach is False:
                    marks.append(clr("unreachable", "red"))
                marker = "  " + " ".join(marks) if marks else ""
                cur = "  " + clr("← current", "yellow") if p["model"] == model else ""
                info(f"  {clr('/model ' + name, 'cyan'):24s} {p['model']:30s}{marker}{cur}")
                info(f"    {clr(p.get('description',''), 'dim')}")

        # One backend: llama.cpp (llama-server) over the OpenAI-compatible
        # protocol. The `custom` provider points at it (or any OpenAI-compatible
        # server) via custom_base_url.
        info("\nBackend:  " + clr("llama.cpp", "cyan")
             + clr("  — llama-server, or any OpenAI-compatible endpoint via custom_base_url", "dim"))
        info("\nSet a model with:  /model custom/<name>")
        info("  e.g. /model custom/qwen3.5-9b")
    else:
        m = args.strip()
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


def _cmd_model_recommend(argv: list, config) -> bool:
    """Render hardware-matched local-model suggestions (`/model recommend`)."""
    import model_recommend as _mr

    # Parse optional args in any order: <budget_gb>, ctx=<K>, fp16, <family>.
    # Bare small number = context target (K); larger/decimal number = budget GB.
    budget_override = None
    target_ctx_k = None
    kv_div = _mr.DEFAULT_KV_DIV
    families = None
    for a in argv:
        al = a.lower()
        if al in ("fp16", "nokv", "nocompress"):
            kv_div = 1.0
            continue
        if al.startswith("ctx="):
            try:
                target_ctx_k = float(al[4:].rstrip("k"))
            except ValueError:
                pass
            continue
        try:
            n = float(a)
            # Heuristic: a plain integer ≥ 40 is a context length in K; else GB.
            if a.isdigit() and n >= 40:
                target_ctx_k = n
            else:
                budget_override = n
            continue
        except ValueError:
            pass
        families = (families or []) + [a]

    hw = _mr.detect_hardware()
    budget = budget_override if budget_override else hw.budget_gb
    if not budget:
        err("Couldn't detect memory and no budget given.")
        info("  Try:  /model recommend 8   (your GPU VRAM or usable RAM, in GB)")
        return True

    # If the user runs a local server with a known context window, default the
    # sizing target to it (their actual context) unless they passed ctx=.
    if target_ctx_k is None and config.get("context_limit"):
        target_ctx_k = config["context_limit"] / 1000.0

    # Header: what we detected and what budget we're using.
    ram = f"{hw.ram_gb:.0f} GB RAM" if hw.ram_gb else "RAM unknown"
    vram = f"{hw.vram_gb:.1f} GB VRAM" if hw.vram_gb else "no discrete GPU detected"
    info(f"Hardware:  {ram} · {vram}")
    src = "override" if budget_override else ("GPU-resident" if hw.vram_gb else "system RAM")
    info(f"Budget:    {clr(f'{budget:.1f} GB', 'cyan')}  ({src})   "
         + clr("override: /model recommend <GB>", "dim"))
    kv_desc = "fp16 KV" if kv_div == 1.0 else f"Q4 KV cache (÷{kv_div:.0f}: TurboQuant / llama.cpp -ctk q4_0)"
    ctx_desc = f"{target_ctx_k:.0f}K" if target_ctx_k else "each model's max"
    info(f"Sizing:    quant chosen so {clr(ctx_desc, 'cyan')} context fits, assuming {kv_desc}   "
         + clr("(ctx=<K> or fp16 to change)", "dim"))
    info(clr("Fetching quantizations from Hugging Face …", "dim"))

    recs = _mr.build_recommendations(budget, families,
                                     target_ctx_k=target_ctx_k, kv_quant_div=kv_div)
    real = [r for r in recs if "recommended" in r]
    if not real:
        warn("No catalog model fits that budget (or HF was unreachable).")
        if any(r.get("fetch_failed") for r in recs):
            info("  Some repos couldn't be reached — check your connection and retry.")
        else:
            info("  Try a larger budget, e.g. /model recommend 16")
        return True

    info("")
    info(clr(f"Recommended for ~{budget:.0f} GB", "bold")
         + clr("  (KV-efficient families first; quant sized to fit the full context)", "dim"))
    info("")

    def _fmt(pick):
        cov = "" if pick.fits_target else clr("  (max it reaches)", "dim")
        return (f"{pick.quant.label} ({pick.quant.size_gb:.2f} GB) · "
                f"fits ~{pick.ctx_k:.0f}K ctx{cov}")

    top = None
    for i, r in enumerate(recs):
        m = r["model"]
        if r.get("fetch_failed"):
            info(f"  {clr(m.key, 'dim')}  {clr('(HF unreachable)', 'red')}")
            continue
        star = clr(" ★", "yellow") if top is None else ""
        if top is None:
            top = r
        badges = " ".join(clr(f"[{t}]", "dim") for t in m.tags)
        tgt = r["target_ctx_k"]
        info(f"  {clr(m.key, 'cyan')}{star}  {clr(m.family, 'dim')}  "
             + clr(f"max {tgt}K", "dim") + f"  {badges}")
        info(f"      → {clr(_fmt(r['recommended']), 'green')}")
        rq = r["recommended"].quant
        if rq.filename:
            info(f"        {clr(_mr.download_url(m.repo, rq.filename), 'dim')}")
        if r.get("alt_quality"):
            info(f"        higher fidelity: {_fmt(r['alt_quality'])}")
        if r.get("alt_context"):
            info(f"        more context:    {_fmt(r['alt_context'])}")
        if m.note:
            info(f"      {clr(m.note, 'dim')}")

    # Concrete next steps for the top pick.
    if top:
        m = top["model"]
        q = top["recommended"].quant
        fname = q.filename or (m.key + ".gguf")
        url = _mr.download_url(m.repo, q.filename) if q.filename else \
            f"https://huggingface.co/{m.repo}"
        local = f"~/.promethean/models/{fname}"
        info("")
        info(clr(f"To use {m.key} ({q.label}):", "bold"))
        info(f"  1. {clr('mkdir -p ~/.promethean/models', 'cyan')}")
        info(f"  2. {clr(f'curl -L -o {local} \\\\', 'cyan')}")
        info(f"       {clr(url, 'cyan')}")
        info(f"  3. {clr(f'/config llama_model_path={local}', 'cyan')}")
        info(f"     {clr('/config model=custom/' + m.key, 'cyan')}   "
             + clr("(loopback custom_base_url autostarts llama-server)", "dim"))
        info(clr("  Full guide: docs/MODELS.md", "dim"))
    return True
