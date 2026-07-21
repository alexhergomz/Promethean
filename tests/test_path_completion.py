"""Tests for REPL file-path completion helpers (ui/input.py).

The helpers are pure (no prompt_toolkit needed), so they cover the behaviour
that drives Tab-completion of paths in the input line.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.input import _filesystem_completions, _looks_like_path, _path_token  # noqa: E402


def test_path_token_extracts_suffix():
    assert _path_token("fix the bug in src/uti") == "src/uti"
    assert _path_token("plain") == "plain"
    assert _path_token("trailing space ") == ""


def test_looks_like_path():
    assert _looks_like_path("src/x")
    assert _looks_like_path("~/notes")
    assert _looks_like_path("./run")
    assert _looks_like_path(".config")
    assert not _looks_like_path("utils")
    assert not _looks_like_path("the")


def test_completions_list_dirs_first_with_slash(tmp_path, monkeypatch):
    (tmp_path / "alpha.py").write_text("")
    (tmp_path / "alpha_dir").mkdir()
    (tmp_path / ".hidden").write_text("")
    monkeypatch.chdir(tmp_path)
    toks = [t for t, _ in _filesystem_completions("al")]
    assert "alpha_dir/" in toks           # directory gets a trailing slash
    assert "alpha.py" in toks
    assert toks.index("alpha_dir/") < toks.index("alpha.py")   # dirs first
    assert all(not t.startswith(".") for t in toks)            # hidden excluded


def test_completions_include_hidden_on_dot_prefix(tmp_path, monkeypatch):
    (tmp_path / ".hidden").write_text("")
    monkeypatch.chdir(tmp_path)
    assert any(t == ".hidden" for t, _ in _filesystem_completions("."))


def test_completions_keep_typed_directory_prefix(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "utils.py").write_text("")
    monkeypatch.chdir(tmp_path)
    assert ("src/utils.py", False) in _filesystem_completions("src/uti")


def test_completions_missing_dir_is_empty():
    assert _filesystem_completions("no_such_dir_xyz/foo") == []


def test_completer_dispatches_non_slash_to_paths(tmp_path, monkeypatch):
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from ui.input import SlashCompleter

    (tmp_path / "alpha.py").write_text("")
    monkeypatch.chdir(tmp_path)
    completer = SlashCompleter(commands_provider=lambda: {}, meta_provider=lambda: {})
    doc = Document("open al", len("open al"))
    comps = list(completer.get_completions(doc, CompleteEvent(completion_requested=True)))
    assert any(c.text == "alpha.py" for c in comps)
