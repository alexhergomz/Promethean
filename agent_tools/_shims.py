"""Minimal stand-ins for the three aider-internal modules that repomap.py imports.

Vendored locally so we don't pull in the full `aider-chat` package (which has
build issues on Python 3.13 and brings in heavy LLM-client deps we don't need).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


# ── Shim for aider.dump (debug helper) ─────────────────────────────────────────

def dump(*args, **kwargs):
    """No-op replacement for aider.dump; original prints repr of locals."""
    if os.environ.get("AGENT_TOOLS_DEBUG"):
        print(*args, file=sys.stderr, **kwargs)


# ── Shim for aider.waiting.Spinner (UI element, not needed for library use) ───

class Spinner:
    """No-op spinner. The original animates a CLI spinner; library users don't care."""
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass


# ── Shim for aider.special.filter_important_files ─────────────────────────────

ROOT_IMPORTANT_FILES = {
    # Subset of aider/special.py — the ones that matter for repo summaries.
    "README", "README.md", "README.rst", "README.txt",
    "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE", "LICENSE.md",
    "CODEOWNERS", "SECURITY.md",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "package-lock.json", "yarn.lock",
    "Cargo.toml", "go.mod", "Gemfile", "pom.xml", "build.gradle",
    "composer.json", "mix.exs",
    ".gitignore", ".editorconfig",
    "tsconfig.json", "Dockerfile", "docker-compose.yml",
    "Makefile", "CMakeLists.txt",
    ".github/dependabot.yml",
}
_NORMALIZED = {os.path.normpath(p) for p in ROOT_IMPORTANT_FILES}


def _is_important(file_path: str) -> bool:
    file_name = os.path.basename(file_path)
    dir_name = os.path.normpath(os.path.dirname(file_path))
    if dir_name == os.path.normpath(".github/workflows") and file_name.endswith(".yml"):
        return True
    return os.path.normpath(file_path) in _NORMALIZED


def filter_important_files(file_paths):
    return list(filter(_is_important, file_paths))


# ── Minimal IO shim — stand-in for aider.io.InputOutput ───────────────────────

class SimpleIO:
    """Replacement for aider's IO object. Just reads files and (optionally)
    prints messages to stderr. The original handles colored UI; we don't."""

    def __init__(self, encoding: str = "utf-8", verbose: bool = False):
        self.encoding = encoding
        self.verbose = verbose

    def read_text(self, fname):
        try:
            return Path(fname).read_text(encoding=self.encoding, errors="replace")
        except (OSError, UnicodeDecodeError):
            return None

    def tool_output(self, msg, *args, **kwargs):
        if self.verbose:
            print(f"[repomap] {msg}", file=sys.stderr)

    def tool_warning(self, msg, *args, **kwargs):
        if self.verbose:
            print(f"[repomap WARN] {msg}", file=sys.stderr)

    def tool_error(self, msg, *args, **kwargs):
        print(f"[repomap ERROR] {msg}", file=sys.stderr)


# ── Token counter — replaces main_model.token_count ───────────────────────────

class CharRatioCounter:
    """4 chars ≈ 1 token. Good enough for repo-map binary search.

    The original RepoMap calls main_model.token_count() to size the output.
    Real LLMs vary slightly (Qwen, Llama tokenize a bit differently from GPT)
    but the binary-search target is approximate anyway, so a constant ratio
    works fine. If you need precision, swap this for a real tokenizer.
    """
    def token_count(self, text: str) -> int:
        return max(1, len(text) // 4)
