"""Unit tests for symbol-graph auto-injection on Edit/Write."""
from __future__ import annotations

import os
import sys
import types
from collections import defaultdict, namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import symbol_context


_FakeTag = namedtuple("Tag", "rel_fname fname line name kind")


class _FakeIndex:
    def __init__(self, refs):
        self.refs = defaultdict(list)
        for name, tags in refs.items():
            self.refs[name] = tags


def _install_fake_index(monkeypatch, refs):
    """Stub agent_tools.helpers.get_symbol_index → returns fake."""
    fake = _FakeIndex(refs)
    fake_helpers = types.SimpleNamespace(get_symbol_index=lambda root=".": fake)
    fake_agent_tools = types.SimpleNamespace(helpers=fake_helpers)
    monkeypatch.setitem(sys.modules, "agent_tools", fake_agent_tools)
    monkeypatch.setitem(sys.modules, "agent_tools.helpers", fake_helpers)


class TestSymbolExtraction:
    def test_def_extracted(self):
        names = symbol_context._extract_symbols(
            "def my_function(x):\n    return x"
        )
        assert names == ["my_function"]

    def test_class_extracted(self):
        names = symbol_context._extract_symbols("class FooBar:\n    pass")
        assert names == ["FooBar"]

    def test_async_def(self):
        names = symbol_context._extract_symbols("async def handler():\n    pass")
        assert names == ["handler"]

    def test_short_names_ignored(self):
        # Single-letter and 2-char names get filtered (too noisy as refs).
        names = symbol_context._extract_symbols("def x():\n    pass\ndef ab():\n    pass")
        assert names == []

    def test_multiple_defs_capped(self):
        big = "\n".join(f"def fn_{i}():\n    pass" for i in range(20))
        names = symbol_context._extract_symbols(big)
        assert len(names) == symbol_context._MAX_SYMBOLS_PER_EDIT

    def test_no_defs_returns_empty(self):
        names = symbol_context._extract_symbols("just a comment, no defs here")
        assert names == []


class TestForEdit:
    def test_empty_when_no_symbols(self, monkeypatch):
        _install_fake_index(monkeypatch, {})
        out = symbol_context.for_edit(
            {"old_string": "foo = 1", "new_string": "foo = 2",
             "file_path": "a.py"},
        )
        assert out == ""

    def test_empty_when_no_callers(self, monkeypatch):
        _install_fake_index(monkeypatch, {"lonely_fn": []})
        out = symbol_context.for_edit(
            {"old_string": "def lonely_fn():\n    pass",
             "new_string": "def lonely_fn(x):\n    return x",
             "file_path": "a.py"},
        )
        assert out == ""

    def test_emits_footer_when_callers_in_other_files(self, monkeypatch):
        _install_fake_index(monkeypatch, {
            "shared_fn": [
                _FakeTag("util.py", "/a/util.py", 42, "shared_fn", "ref"),
                _FakeTag("main.py", "/a/main.py", 100, "shared_fn", "ref"),
            ],
        })
        out = symbol_context.for_edit(
            {"old_string": "def shared_fn(x):\n    return x",
             "new_string": "def shared_fn(x, y):\n    return x+y",
             "file_path": "core.py"},
        )
        assert "shared_fn" in out
        assert "util.py:42" in out and "main.py:100" in out
        assert "symbol-graph" in out

    def test_refs_in_same_file_excluded(self, monkeypatch):
        _install_fake_index(monkeypatch, {
            "local_fn": [
                _FakeTag("a.py", "/repo/a.py", 5, "local_fn", "ref"),
                _FakeTag("a.py", "/repo/a.py", 12, "local_fn", "ref"),
            ],
        })
        out = symbol_context.for_edit(
            {"old_string": "def local_fn():\n    pass",
             "new_string": "def local_fn(x):\n    return x",
             "file_path": "a.py"},
        )
        # All refs are in the edited file, so footer is empty.
        assert out == ""

    def test_many_callers_summarized(self, monkeypatch):
        tags = [_FakeTag(f"f{i}.py", f"/r/f{i}.py", 1, "popular", "ref")
                for i in range(20)]
        _install_fake_index(monkeypatch, {"popular": tags})
        out = symbol_context.for_edit(
            {"old_string": "def popular():\n    pass",
             "new_string": "def popular(x):\n    return x",
             "file_path": "lib.py"},
        )
        # Cap shown to MAX_CALLERS_PER_SYM, plus "+N more" tail.
        assert "+12 more" in out
        # First 8 should appear.
        assert "f0.py:1" in out

    def test_write_uses_content_field(self, monkeypatch):
        _install_fake_index(monkeypatch, {
            "new_class": [
                _FakeTag("other.py", "/a/other.py", 9, "new_class", "ref"),
            ],
        })
        out = symbol_context.for_edit(
            {"content": "class new_class:\n    pass",
             "file_path": "fresh.py"},
        )
        assert "new_class" in out and "other.py:9" in out
