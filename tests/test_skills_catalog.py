"""Tests for the skills catalog injected into the system prompt."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skill import loader
from skill.loader import SkillDef, skills_catalog


def _stub(monkeypatch, skills):
    monkeypatch.setattr(loader, "load_skills", lambda *a, **k: skills)


def test_empty_when_no_skills(monkeypatch):
    _stub(monkeypatch, [])
    assert skills_catalog() == ""


def test_lists_name_and_when_to_use(monkeypatch):
    sk = SkillDef(
        name="commit", description="d", triggers=["/commit"], tools=[],
        prompt="", file_path="<b>", when_to_use="when changes are ready to commit",
    )
    _stub(monkeypatch, [sk])
    out = skills_catalog()
    assert "**commit**" in out
    assert "when changes are ready to commit" in out


def test_falls_back_to_description(monkeypatch):
    sk = SkillDef(
        name="review", description="review a PR", triggers=["/review"], tools=[],
        prompt="", file_path="<b>",
    )
    _stub(monkeypatch, [sk])
    assert "review a PR" in skills_catalog()


def test_skill_without_hint_or_desc_excluded(monkeypatch):
    sk = SkillDef(name="bare", description="", triggers=["/bare"], tools=[],
                  prompt="", file_path="<b>")
    _stub(monkeypatch, [sk])
    assert skills_catalog() == ""


def test_argument_hint_included(monkeypatch):
    sk = SkillDef(
        name="deploy", description="deploy", triggers=["/deploy"], tools=[],
        prompt="", file_path="<b>", argument_hint="[env]",
    )
    _stub(monkeypatch, [sk])
    assert "[env]" in skills_catalog()


def test_hint_truncated(monkeypatch):
    sk = SkillDef(
        name="x", description="", triggers=["/x"], tools=[], prompt="",
        file_path="<b>", when_to_use="z" * 500,
    )
    _stub(monkeypatch, [sk])
    out = skills_catalog()
    assert "..." in out and len(out) < 250


def test_cap_respected(monkeypatch):
    many = [
        SkillDef(name=f"s{i}", description=f"desc {i}", triggers=[f"/s{i}"],
                 tools=[], prompt="", file_path="<b>")
        for i in range(60)
    ]
    _stub(monkeypatch, many)
    out = skills_catalog(max_skills=10)
    assert out.count("\n") == 9  # 10 lines


def test_robust_to_load_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(loader, "load_skills", boom)
    assert skills_catalog() == ""
