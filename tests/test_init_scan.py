"""Tests for the repo-aware /init scan and CLAUDE.md rendering."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commands.core import _infer_test_command, _render_claude_md, _scan_project  # noqa: E402


def test_scan_detects_python_and_manifest(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "tests").mkdir()
    scan = _scan_project(tmp_path)
    assert "Python" in scan["languages"]
    assert "pyproject.toml" in scan["manifests"]
    assert "app.py" in scan["entries"]
    assert scan["test_command"] == "pytest"


def test_scan_skips_vendor_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("x=1\n")
    (tmp_path / "real.py").write_text("x=1\n")
    scan = _scan_project(tmp_path)
    # node_modules content must not dominate/appear; only the real file counts.
    assert scan["languages"] == ["Python"]


def test_infer_test_command_node(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
    assert _infer_test_command(tmp_path, ["package.json"], False) == "npm test"


def test_infer_test_command_rust_and_go(tmp_path):
    assert _infer_test_command(tmp_path, ["Cargo.toml"], False) == "cargo test"
    assert _infer_test_command(tmp_path, ["go.mod"], False) == "go test ./..."


def test_render_includes_headings_and_detected(tmp_path):
    scan = {
        "languages": ["Python"], "manifests": ["pyproject.toml"],
        "entries": ["main.py"], "has_tests": True, "test_command": "pytest",
    }
    md = _render_claude_md("demoproj", scan)
    assert "# demoproj" in md
    assert "## Project Overview" in md
    assert "- Python" in md
    assert "`main.py`" in md
    assert "pytest" in md


def test_render_empty_scan_keeps_prompts(tmp_path):
    scan = {"languages": [], "manifests": [], "entries": [],
            "has_tests": False, "test_command": ""}
    md = _render_claude_md("bare", scan)
    assert "## Tech Stack" in md
    assert "<!--" in md          # placeholder prompts remain when nothing found
