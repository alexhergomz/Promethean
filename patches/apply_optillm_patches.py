#!/usr/bin/env python3
"""Idempotent patcher for the local OptiLLM install.

Applies the strict-provider compatibility fixes (currently: BoN avoids
mid-conversation role:system to keep MiniMax M2 happy). Safe to run
after every `pip install -U optillm` — it detects when a patch is
already applied and skips.

Usage:
    python3 patches/apply_optillm_patches.py            # apply
    python3 patches/apply_optillm_patches.py --check    # report only

Exits non-zero on failure or when --check finds an unpatched site.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _find_optillm_root() -> Path:
    """Locate the installed optillm package by importing it."""
    import importlib.util
    spec = importlib.util.find_spec("optillm")
    if not spec or not spec.origin:
        raise SystemExit("optillm not installed in this Python env")
    return Path(spec.origin).parent


# (description, file_relative_to_optillm_root, OLD literal, NEW literal,
#  sentinel that marks "already patched")
PATCHES: list[tuple[str, str, str, str, str]] = [
    (
        "bon: drop mid-conversation system role, fold instruction into user turn",
        "bon.py",
        # OLD: appends a stray system message before the rating loop, plus
        # a terse "Rate the above response:" user message per item.
        '''    rating_messages.append({"role": "system", "content": "Rate the following responses on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness. Respond with only a number."})

    ratings = []
    for completion in completions:
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "user", "content": "Rate the above response:"})''',
        # NEW: no extra system message; rating instruction lives in the
        # per-completion user message.
        '''    ratings = []
    for completion in completions:
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "user", "content": (
            "Rate the above response on a scale from 0 to 10, where 0 is "
            "poor and 10 is excellent. Consider factors such as relevance, "
            "coherence, and helpfulness. Respond with only a number."
        )})''',
        # Sentinel — present after the patch, absent before.
        "Rate the above response on a scale from 0 to 10",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Report status without modifying files")
    args = ap.parse_args()

    root = _find_optillm_root()
    print(f"optillm at: {root}")

    all_ok = True
    for desc, rel, old, new, sentinel in PATCHES:
        path = root / rel
        if not path.exists():
            print(f"  [{rel}]  SKIP — file not found (different optillm version?)")
            continue
        src = path.read_text()
        if sentinel in src:
            print(f"  [{rel}]  already patched: {desc}")
            continue
        if old not in src:
            # Source has drifted; we can't safely apply this verbatim.
            print(f"  [{rel}]  CANNOT APPLY (source drifted): {desc}")
            all_ok = False
            continue
        if args.check:
            print(f"  [{rel}]  needs patch: {desc}")
            all_ok = False
            continue
        # Apply.
        path.write_text(src.replace(old, new))
        print(f"  [{rel}]  patched: {desc}")

    if args.check:
        return 0 if all_ok else 1
    if not all_ok:
        return 1
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
