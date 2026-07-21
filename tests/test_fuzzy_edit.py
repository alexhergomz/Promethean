"""Tests for Edit's inexact-match recovery (tools/fs.py).

Local models frequently miss a verbatim old_string by copying line-number
gutters, getting indentation wrong, or drifting a character or two. These
cover each recovery path plus the guards that stop a wrong or ambiguous
edit from landing silently.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fs import _AMBIGUOUS, _edit, _locate_inexact  # noqa: E402


def test_exact_match_still_applies(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    out = _edit(str(f), "y = 2", "y = 3")
    assert "Changes applied" in out
    assert "no verbatim match" not in out
    assert f.read_text() == "x = 1\ny = 3\n"


def test_recovers_line_number_prefix(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def foo():\n    return 1\n")
    # old_string copied straight out of a Read result, gutter and all.
    old = "     1\tdef foo():\n     2\t    return 1"
    out = _edit(str(f), old, "def foo():\n    return 2")
    assert "Changes applied" in out
    assert "line-number-stripped" in out
    assert f.read_text() == "def foo():\n    return 2\n"


def test_recovers_wrong_indentation(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("class A:\n        def m(self):\n                return 42\n")
    # Model supplies the block at 4-space indent; file uses 8.
    old = "def m(self):\n    return 42"
    out = _edit(str(f), old, "def m(self):\n    return 99")
    assert "indentation-insensitive" in out
    assert "return 99" in f.read_text()
    assert "return 42" not in f.read_text()


def test_recovers_small_drift(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(
        "def calculate(value):\n"
        "    result = value * 2\n"
        "    return result\n"
    )
    # Spacing drift the stripped-line pass cannot catch, but only one
    # plausible target exists.
    old = "    result = value*2\n    return result"
    out = _edit(str(f), old, "    result = value * 3\n    return result")
    assert "closest match" in out
    assert "value * 3" in f.read_text()


def test_ambiguous_block_is_rejected(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def a():\n    pass\n\ndef a():\n    pass\n")
    old = "def a():\npass"     # no indent → no verbatim hit, two stripped hits
    out = _edit(str(f), old, "def a():\n    return 1")
    assert out.startswith("Error")
    assert "more than one place" in out
    assert f.read_text() == "def a():\n    pass\n\ndef a():\n    pass\n"   # untouched


def test_no_plausible_match_errors(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("alpha = 1\nbeta = 2\n")
    out = _edit(str(f), "gamma = delta + epsilon", "gamma = 0")
    assert out.startswith("Error")
    assert "not found" in out
    assert f.read_text() == "alpha = 1\nbeta = 2\n"


def test_locate_returns_ambiguous_sentinel():
    content = "def a():\n    pass\n\ndef a():\n    pass\n"
    assert _locate_inexact(content, "def a():\npass") is _AMBIGUOUS


def test_replace_all_unaffected(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v = 1\nv = 1\n")
    out = _edit(str(f), "v = 1", "v = 2", replace_all=True)
    assert "Changes applied" in out
    assert f.read_text() == "v = 2\nv = 2\n"
