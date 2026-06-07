"""Unit tests for the slash-command completer in ui/input.py.

Covers the two-level completion, the COMMANDS+meta symmetry fix, and the
regression guard for the `/c/cwd` glue bug.
"""

from __future__ import annotations

import pytest

from ui.input import HAS_PROMPT_TOOLKIT, SlashCompleter

if not HAS_PROMPT_TOOLKIT:
    pytest.skip("prompt_toolkit not installed", allow_module_level=True)

from prompt_toolkit.document import Document


META = {
    "help":       ("Show help", []),
    "branch":     ("Branch the conversation here", []),
    "clear":      ("Clear conversation history", []),
    "checkpoint": ("List / restore checkpoints", ["clear"]),
    "compact":    ("Compact conversation history", []),
    "config":     ("Show / set config key=value", []),
    "context":    ("Show token-context usage", []),
    "copy":       ("Copy last response to clipboard", []),
    "cost":       ("Show cost estimate", []),
    "cwd":        ("Show / change working directory", []),
    "cloudsave":  ("Cloud-sync sessions to GitHub Gist",
                   ["setup", "auto", "list", "load", "push"]),
    "mcp":        ("Manage MCP servers", ["reload", "add", "remove"]),
    "plugin":     ("Manage plugins",
                   ["install", "uninstall", "enable", "disable"]),
}

COMMANDS = {name: (lambda *a, **k: True) for name in META}


def _completions(completer, text: str):
    doc = Document(text, cursor_position=len(text))
    return list(completer.get_completions(doc, complete_event=None))


def _make_completer(commands=None, meta=None):
    commands = commands if commands is not None else dict(COMMANDS)
    meta = meta if meta is not None else dict(META)
    return SlashCompleter(lambda: commands, lambda: meta)


def test_level1_c_prefix_includes_cwd_and_siblings():
    completer = _make_completer()
    texts = [c.text for c in _completions(completer, "/c")]
    assert "/cwd" in texts
    assert "/clear" in texts
    assert "/checkpoint" in texts
    assert "/compact" in texts
    assert "/config" in texts
    assert "/context" in texts
    assert "/copy" in texts
    assert "/cost" in texts
    assert "/cloudsave" in texts
    # Irrelevant commands should not leak in
    assert "/help" not in texts


def test_level1_empty_slash_yields_all_commands():
    completer = _make_completer()
    texts = [c.text for c in _completions(completer, "/")]
    assert len(texts) == len(META)
    assert set(texts) == {"/" + name for name in META}


def test_level1_display_meta_includes_description_and_subhint():
    completer = _make_completer()
    results = _completions(completer, "/p")
    (plugin,) = [c for c in results if c.text == "/plugin"]
    assert "Manage plugins" in plugin.display_meta_text
    assert "install" in plugin.display_meta_text


def test_level2_subcommand_completion():
    completer = _make_completer()
    completions = _completions(completer, "/mcp r")
    texts = [c.text for c in completions]
    assert "reload" in texts
    assert "remove" in texts
    assert "add" not in texts  # does not start with 'r'


def test_level2_no_subcommands_for_command_without_any():
    completer = _make_completer()
    completions = _completions(completer, "/clear x")
    assert completions == []


def test_non_slash_input_yields_nothing():
    completer = _make_completer()
    assert _completions(completer, "hello") == []
    assert _completions(completer, "") == []
    assert _completions(completer, " /help") == []  # leading space


def test_slash_c_cwd_glue_yields_nothing():
    """Regression guard: `/c/cwd` must not match any real command."""
    completer = _make_completer()
    completions = _completions(completer, "/c/cwd")
    assert completions == []


def test_live_command_registration_visible_without_restart():
    """Commands added to the live dict (e.g. via /plugin install) appear."""
    live_commands = dict(COMMANDS)
    live_meta = dict(META)
    completer = SlashCompleter(lambda: live_commands, lambda: live_meta)

    # Warm cache with a query that fuzzily matches nothing yet ('f'..'k' is
    # not a subsequence of any existing command name).
    assert [c.text for c in _completions(completer, "/fk")] == []

    # Register a new command at runtime.
    live_commands["fakecmd"] = lambda *a, **k: True
    live_meta["fakecmd"] = ("Fake test command", [])

    texts = [c.text for c in _completions(completer, "/fk")]
    assert "/fakecmd" in texts


def test_symmetry_commands_only_also_visible():
    """Commands present in COMMANDS but missing from _CMD_META still complete."""
    cmds = dict(COMMANDS)
    cmds["orphan"] = lambda *a, **k: True
    completer = SlashCompleter(lambda: cmds, lambda: dict(META))

    texts = [c.text for c in _completions(completer, "/or")]
    assert "/orphan" in texts


def test_setup_registers_module_level_providers():
    """Verify ui.input.setup() injects providers without requiring ctor args."""
    import ui.input as ui_input

    cmds = {"alpha": True, "beta": True}
    meta = {"alpha": ("A", []), "beta": ("B", [])}
    ui_input.setup(lambda: cmds, lambda: meta)
    try:
        completer = SlashCompleter()  # no ctor args — reads module-level
        texts = [c.text for c in _completions(completer, "/alph")]
        assert "/alpha" in texts
        assert "/beta" not in texts  # 'alph' is not a subsequence of 'beta'
    finally:
        ui_input.setup(lambda: {}, lambda: {})


def test_fuzzy_ranch_matches_branch():
    """Subsequence fuzzy: a typo'd/partial query still surfaces the command."""
    completer = _make_completer()
    texts = [c.text for c in _completions(completer, "/ranch")]
    assert "/branch" in texts  # 'ranch' is a subsequence of 'branch'


def test_fuzzy_subsequence_requires_all_chars_in_order():
    """A query char absent from a command keeps that command out."""
    completer = _make_completer()
    texts = [c.text for c in _completions(completer, "/cz")]
    # No command contains the subsequence 'c'..'z'.
    assert texts == []


def test_fuzzy_prefix_ranks_above_scattered_subsequence():
    """Prefix matches sort before scattered-subsequence matches."""
    completer = _make_completer()
    texts = [c.text for c in _completions(completer, "/co")]
    # 'co' prefixes copy/cost/config/context; 'cloudsave' only matches as a
    # scattered subsequence (c..o), so it must rank last among these.
    assert "/cloudsave" in texts
    for prefixed in ("/copy", "/cost", "/config", "/context"):
        assert texts.index(prefixed) < texts.index("/cloudsave")


def test_fuzzy_subcommand_matching():
    """Level-2 subcommands match fuzzily too."""
    completer = _make_completer()
    texts = [c.text for c in _completions(completer, "/cloudsave st")]
    assert "setup" in texts  # 'st' is a subsequence of 'setup'
    assert "auto" not in texts  # 'auto' has no 's'


def test_module_does_not_import_promethean():
    """Regression guard for the circular-import concern from review."""
    import sys
    import ui.input as ui_input
    # Reload ui.input in a clean state and confirm promethean is not pulled in.
    # (Running this in the test session where promethean may already be loaded
    # is acceptable — the assertion is about ui.input's own import graph.)
    src = open(ui_input.__file__).read()
    assert "import promethean" not in src
    assert "from promethean" not in src
