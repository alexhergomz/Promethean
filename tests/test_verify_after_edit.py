"""Tests for auto verify-after-edit.

summarize_for_edit must stay silent on clean/unsupported files and surface
real problems; the execute_tool wiring must append the footer only for
supported source files and only when there is something to report.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: E402
from tools.diagnostics import summarize_for_edit  # noqa: E402


@pytest.fixture(autouse=True)
def _builtins_registered():
    # test_tool_registry.py clears the shared registry in teardown, so make
    # the execute_tool tests below independent of suite ordering.
    tools._register_builtins()
    yield


def test_clean_python_is_silent(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n")
    # A syntactically valid file yields no actionable footer, whatever
    # checker happens to be installed (worst case py_compile: syntax OK).
    assert summarize_for_edit(str(f)) is None


def test_broken_python_reports(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n    pass\n")   # syntax error
    out = summarize_for_edit(str(f))
    assert out is not None and out.strip()


def test_unsupported_language_is_silent(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hello\n")
    assert summarize_for_edit(str(f)) is None


def test_edit_footer_attached_on_breakage(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "summarize_for_edit",
                        lambda fp, timeout=15: "py_compile (syntax check):\nboom")
    f = tmp_path / "bad.py"
    f.write_text("x = 1\n")
    out = tools.execute_tool(
        "Edit",
        {"file_path": str(f), "old_string": "x = 1", "new_string": "x = 2"},
        permission_mode="accept-all",
        config={"symbol_context": False, "undo_log": False},
    )
    assert "Changes applied" in out
    assert "[verify]" in out and "boom" in out


def test_edit_footer_skipped_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "summarize_for_edit",
                        lambda fp, timeout=15: "should not appear")
    f = tmp_path / "bad.py"
    f.write_text("x = 1\n")
    out = tools.execute_tool(
        "Edit",
        {"file_path": str(f), "old_string": "x = 1", "new_string": "x = 2"},
        permission_mode="accept-all",
        config={"symbol_context": False, "undo_log": False,
                "verify_after_edit": False},
    )
    assert "[verify]" not in out


def test_edit_footer_skipped_for_non_source(tmp_path, monkeypatch):
    called = {"n": 0}

    def _spy(fp, timeout=15):
        called["n"] += 1
        return "unexpected"

    monkeypatch.setattr(tools, "summarize_for_edit", _spy)
    f = tmp_path / "readme.md"
    f.write_text("hello\n")
    out = tools.execute_tool(
        "Edit",
        {"file_path": str(f), "old_string": "hello", "new_string": "world"},
        permission_mode="accept-all",
        config={"symbol_context": False, "undo_log": False},
    )
    assert "[verify]" not in out
    assert called["n"] == 0     # extension gate short-circuits before the call
