"""Unit tests for the GitNexus-style graph helpers in agent_tools.

Builds a tiny throwaway python package and asserts that
neighborhood / path_between / imports / search_files return the
expected shape on a corpus we control completely.

Run: pytest tests/test_symbol_graph.py -v
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import bootstrap as _bs  # noqa: F401  (sets up sys.path / shims)

# Optional repo-map deps (diskcache, grep_ast, pygments, tree_sitter); skip
# the module when they're absent, matching the runtime's graceful degradation.
pytest.importorskip("agent_tools.helpers", reason="graph extra not installed")

from agent_tools.helpers import (
    SymbolIndex,
    imports,
    neighborhood,
    path_between,
    search_files,
)


# ── Fixture: a 3-file mini repo with known call graph ────────────────────────
#
#   alpha()       (in a.py)         calls beta()
#   beta()        (in b.py)         calls gamma()
#   gamma()       (in c.py)         leaf
#   helper_only() (in c.py)         orphan, no callers
#
# Plus a README that mentions only "documentation" so search_files has
# something non-symbolic to find.

@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(textwrap.dedent("""
        from b import beta

        def alpha():
            return beta() + 1
    """).lstrip())

    (tmp_path / "b.py").write_text(textwrap.dedent("""
        from c import gamma

        def beta():
            return gamma() * 2
    """).lstrip())

    (tmp_path / "c.py").write_text(textwrap.dedent("""
        def gamma():
            return 42

        def helper_only():
            return "unused"
    """).lstrip())

    (tmp_path / "README.md").write_text(textwrap.dedent("""
        # Mini Repo

        This is documentation explaining the configuration system.
        Buzzword: serendipity. Rare term: floccinaucinihilipilification.
    """).lstrip())

    return tmp_path


# ── SymbolIndex ──────────────────────────────────────────────────────────────

def test_symbol_index_finds_all_defs(mini_repo):
    idx = SymbolIndex(root=str(mini_repo))
    assert "alpha" in idx.defs
    assert "beta" in idx.defs
    assert "gamma" in idx.defs
    assert "helper_only" in idx.defs


def test_symbol_index_def_span_uses_next_def_line(mini_repo):
    idx = SymbolIndex(root=str(mini_repo))
    # gamma and helper_only are both in c.py; gamma's span ends where
    # helper_only's def line begins.
    gamma_def = idx.defs["gamma"][0]
    helper_def = idx.defs["helper_only"][0]
    start, end = idx.def_span(gamma_def)
    assert start == gamma_def.line
    assert end == helper_def.line


def test_symbol_index_callees_finds_inner_refs(mini_repo):
    idx = SymbolIndex(root=str(mini_repo))
    alpha_def = idx.defs["alpha"][0]
    callee_names = {r.name for r in idx.callees_of(alpha_def)}
    assert "beta" in callee_names


# ── neighborhood ─────────────────────────────────────────────────────────────

def test_neighborhood_basic_shape(mini_repo):
    nb = neighborhood("beta", root=str(mini_repo))
    assert len(nb["def"]) == 1
    assert nb["def"][0].file == "b.py"
    caller_files = {h.file for h in nb["callers"]}
    assert "a.py" in caller_files
    callee_names = {h.name for h in nb["callees"]}
    assert "gamma" in callee_names


def test_neighborhood_orphan_has_no_callers(mini_repo):
    nb = neighborhood("helper_only", root=str(mini_repo))
    assert len(nb["def"]) == 1
    assert nb["callers"] == []


def test_neighborhood_unknown_symbol_is_empty(mini_repo):
    nb = neighborhood("does_not_exist", root=str(mini_repo))
    assert nb["def"] == []
    assert nb["callers"] == []
    assert nb["callees"] == []


# ── path_between ─────────────────────────────────────────────────────────────

def test_path_between_direct_chain(mini_repo):
    chain = path_between("alpha", "gamma", root=str(mini_repo), max_hops=4)
    names = [h.name for h in chain]
    assert names == ["alpha", "beta", "gamma"]


def test_path_between_self_returns_singleton(mini_repo):
    chain = path_between("alpha", "alpha", root=str(mini_repo), max_hops=4)
    assert len(chain) == 1
    assert chain[0].name == "alpha"


def test_path_between_no_path(mini_repo):
    chain = path_between("alpha", "helper_only", root=str(mini_repo), max_hops=4)
    assert chain == []


def test_path_between_respects_max_hops(mini_repo):
    # alpha -> beta -> gamma is 2 hops. max_hops=1 should fail to reach gamma.
    chain = path_between("alpha", "gamma", root=str(mini_repo), max_hops=1)
    assert chain == []


def test_path_between_unknown_endpoints(mini_repo):
    assert path_between("nope", "gamma", root=str(mini_repo)) == []
    assert path_between("alpha", "nope", root=str(mini_repo)) == []


# ── imports ──────────────────────────────────────────────────────────────────

def test_imports_uses_external_defs(mini_repo):
    deps = imports("a.py", root=str(mini_repo))
    used_names = {h.name for h in deps["uses"]}
    # a.py refs beta which is defined in b.py
    assert "beta" in used_names


def test_imports_used_by_lists_external_refs(mini_repo):
    deps = imports("c.py", root=str(mini_repo))
    used_by_files = {h.file for h in deps["used_by"]}
    # gamma (in c.py) is referenced from b.py
    assert "b.py" in used_by_files


def test_imports_excludes_self_references(mini_repo):
    deps = imports("c.py", root=str(mini_repo))
    # c.py has both gamma and helper_only; refs inside c.py shouldn't show
    # in either uses or used_by.
    assert all(h.file != "c.py" for h in deps["uses"])
    assert all(h.file != "c.py" for h in deps["used_by"])


# ── search_files ─────────────────────────────────────────────────────────────

def test_search_files_finds_non_symbol_text(mini_repo):
    res = search_files("documentation configuration", root=str(mini_repo), top_k=5)
    files = [rel for rel, _, _ in res]
    assert "README.md" in files


def test_search_files_rare_token_outranks_common(mini_repo):
    # The rare token should make README.md rank above any code file.
    res = search_files("floccinaucinihilipilification", root=str(mini_repo), top_k=3)
    assert res
    assert res[0][0] == "README.md"


def test_search_files_empty_query(mini_repo):
    assert search_files("", root=str(mini_repo)) == []


def test_search_files_no_matches_returns_empty(mini_repo):
    res = search_files("zzzzzzzzzzz_no_such_token", root=str(mini_repo), top_k=5)
    assert res == []
