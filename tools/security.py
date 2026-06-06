"""tools_security.py — Path-traversal guard and bash safety check.

Three guards live here:

  • _is_safe_bash(cmd)         allow-list — read-only commands that skip
                               the permission prompt
  • _is_dangerous_bash(cmd)    deny-list — catastrophic patterns blocked
                               REGARDLESS of permission_mode (incl.
                               accept-all). Hard floor against confused
                               models, NOT a hardened sandbox
  • _check_path_allowed(...)   path-jail — restricts FS tools to the
                               configured allowed_root / _worktree_cwd
  • _is_sensitive_path(p)      hard deny-list of universally sensitive
                               paths (.ssh, .aws, /etc/shadow, …) that
                               are blocked regardless of jail config
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

# ── Bash allow-list (skip permission for these read-only commands) ────────

_SAFE_PREFIXES = (
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "date",
    "which", "type", "env", "printenv", "uname", "whoami", "id",
    "git log", "git status", "git diff", "git show", "git branch",
    "git remote", "git stash list", "git tag",
    "find ", "grep ", "rg ", "ag ", "fd ",
    "python ", "python3 ", "node ", "ruby ", "perl ",
    "pip show", "pip list", "npm list", "cargo metadata",
    "df ", "du ", "free ", "top -bn", "ps ",
    "curl -I", "curl --head",
)


_CHAIN_OPERATORS = (";", "&&", "||", "|", "`", "$(", "\n")


def _is_safe_bash(cmd: str) -> bool:
    """Return True if cmd is read-only and never needs a permission prompt.

    Rejects commands that contain shell chaining operators (;, &&, ||, |,
    backticks, $(…)) — these could execute arbitrary code after a safe prefix.
    """
    c = cmd.strip()
    if any(op in c for op in _CHAIN_OPERATORS):
        return False
    return any(c.startswith(p) for p in _SAFE_PREFIXES)


# ── Bash deny-list (catastrophic patterns — blocked even in accept-all) ──

# Each entry: (compiled regex, human-readable reason). The regex is matched
# with .search() so the pattern can appear anywhere in the command. We aim
# for tight, obvious patterns — false positives are worse than missed
# obfuscations, since this is a guardrail against confused models, not a
# hardened sandbox. A determined adversary can always obfuscate.

_DANGEROUS_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # rm -rf targeting filesystem root or $HOME (with any flag ordering)
    (re.compile(
        r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*[fF]|-[a-zA-Z]*[fF][a-zA-Z]*[rR]|"
        r"--recursive\s+--force|--force\s+--recursive)"
        r"[^\n]*\s(/(\s|$)|/\*|~(\s|/|$)|\$HOME(\s|/|$))"
    ), "rm -rf targeting / or $HOME"),

    # dd writing to a raw block device
    (re.compile(
        r"\bdd\b[^\n]*\bof=/dev/(sd[a-z]|nvme\d|hd[a-z]|disk\d|mmcblk\d)"
    ), "dd writing to a block device"),

    # mkfs on a block device (filesystem create destroys all data)
    (re.compile(
        r"\bmkfs(\.\w+)?\s+/dev/(sd[a-z]|nvme\d|hd[a-z]|disk\d|mmcblk\d)"
    ), "mkfs on a block device"),

    # Canonical fork bomb
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:&?\s*\}\s*;\s*:"),
     "fork bomb"),

    # Pipe network content directly into a shell interpreter
    (re.compile(
        r"\b(curl|wget|fetch)\b[^|;&\n]*\|\s*"
        r"(sh|bash|zsh|fish|python\d?|perl|ruby|node)\b"
    ), "piping network content into a shell interpreter"),

    # chmod 777 on / or $HOME
    (re.compile(
        r"\bchmod\s+(-R\s+)?(777|a\+rwx)\s+(/(\s|$)|~(\s|/|$)|\$HOME)"
    ), "chmod 777 on / or $HOME"),

    # Redirect output into a raw block device
    (re.compile(
        r">\s*/dev/(sd[a-z]|nvme\d|hd[a-z]|disk\d|mmcblk\d)"
    ), "redirecting output to a block device"),

    # shred targeting a block device
    (re.compile(
        r"\bshred\b[^\n]*\s/dev/(sd[a-z]|nvme\d|hd[a-z]|disk\d|mmcblk\d)"
    ), "shred targeting a block device"),
)


def _is_dangerous_bash(cmd: str) -> Optional[str]:
    """Return a reason string if cmd matches a catastrophic-pattern deny list,
    else None.

    Fires regardless of permission_mode and protects against accidental
    catastrophes from confused models. Not a hardened sandbox — a determined
    adversarial model can obfuscate around regex.
    """
    for pat, reason in _DANGEROUS_PATTERNS:
        if pat.search(cmd):
            return reason
    return None


# ── Path jail ────────────────────────────────────────────────────────────

# Sensitive paths that are ALWAYS blocked, regardless of allowed_root.
# These hold credentials and keys whose disclosure would compromise the
# user's identity / accounts. Resolved against the user's home AND the
# absolute system paths.
_SENSITIVE_RELATIVES = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".gnupg-trezor",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".docker/config.json",
    ".kube",
)

_SENSITIVE_ABSOLUTES = (
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/root",
)


def _is_sensitive_path(file_path: str) -> Optional[str]:
    """Return a reason string if file_path resolves under a sensitive
    location (SSH keys, AWS creds, /etc/shadow, …), else None.

    Fires regardless of allowed_root so users always get a hard floor.
    """
    try:
        resolved = Path(file_path).resolve()
    except Exception:
        return None
    home = Path.home().resolve()
    for rel in _SENSITIVE_RELATIVES:
        target = (home / rel).resolve()
        try:
            resolved.relative_to(target)
            return f"sensitive credential path (~/{rel})"
        except ValueError:
            pass
        if resolved == target:
            return f"sensitive credential path (~/{rel})"
    for absolute in _SENSITIVE_ABSOLUTES:
        target = Path(absolute)
        try:
            resolved.relative_to(target)
            return f"sensitive system path ({absolute})"
        except ValueError:
            pass
        if str(resolved) == absolute:
            return f"sensitive system path ({absolute})"
    return None


def _check_path_allowed(file_path: str, config: dict) -> str | None:
    """Return an error string if file_path is blocked, else None.

    Two layers:
      1. Sensitive deny-list (always enforced) — SSH keys, AWS creds, etc.
      2. allowed_root jail (only when config sets allowed_root or
         _worktree_cwd) — production deployments should set one of these.
    """
    sensitive = _is_sensitive_path(file_path)
    if sensitive:
        return (
            f"Error: refusing access to {file_path} — {sensitive}. "
            "These paths are blocked unconditionally."
        )

    allowed_root = config.get("allowed_root") or config.get("_worktree_cwd")
    if not allowed_root:
        return None
    try:
        resolved = Path(file_path).resolve()
        root     = Path(allowed_root).resolve()
        resolved.relative_to(root)
        return None
    except ValueError:
        return (
            f"Error: path '{file_path}' is outside the allowed root '{root}'. "
            "Set config['allowed_root'] to a broader directory if this is intentional."
        )
    except Exception as e:
        return f"Error: path validation failed: {e}"
