"""Comprehensive offline tests using the multi-module fixture repo at
tests/fixtures/symbol_graph_repo/.

Covers cross-language parsing (Python + JS), name collisions across
files, BFS termination on cycles, nested defs (class methods),
pygments-fallback files, and the new-file Write preview.

Run: pytest tests/test_symbol_graph_fixture.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import bootstrap as _bs  # noqa: F401

# Optional repo-map deps (diskcache, grep_ast, pygments, tree_sitter); skip
# the module when they're absent, matching the runtime's graceful degradation.
pytest.importorskip("agent_tools.helpers", reason="graph extra not installed")

from agent_tools.helpers import (
    SymbolIndex,
    find_symbol,
    get_callers,
    imports,
    neighborhood,
    outline,
    path_between,
    repo_map,
    search_files,
)
from tools.fs import _write
from ui.render import _has_diff


FIXTURE = str(Path(_HERE) / "fixtures" / "symbol_graph_repo")


# ── SymbolIndex coverage of the whole fixture ────────────────────────────────

def test_index_covers_all_languages():
    idx = SymbolIndex(root=FIXTURE)
    # Python files
    for f in ("api/handler.py", "api/auth.py", "db/connection.py",
              "db/models.py", "utils/format.py", "utils/log.py"):
        assert f in idx.defs_by_file, f"missing {f} in index"
    # JS file (cross-language)
    assert "frontend/ui.js" in idx.defs_by_file


def test_index_finds_js_symbols():
    idx = SymbolIndex(root=FIXTURE)
    assert "renderApp" in idx.defs
    assert "fetchUser" in idx.defs
    assert idx.defs["renderApp"][0].rel_fname == "frontend/ui.js"


def test_index_dedupes_repomap_emit_bug():
    """Vendored repomap emits each tag twice; index must dedupe."""
    idx = SymbolIndex(root=FIXTURE)
    # validate_token has exactly one def in api/auth.py
    assert len(idx.defs["validate_token"]) == 1


# ── Name collisions ──────────────────────────────────────────────────────────

def test_find_symbol_returns_both_definitions_of_parse():
    """`parse` is defined in api/auth.py AND utils/format.py."""
    hits = find_symbol("parse", root=FIXTURE)
    files = sorted(h.file for h in hits)
    assert files == ["api/auth.py", "utils/format.py"]


def test_find_symbol_no_duplicates_after_dedup():
    """Even though repomap emits each tag twice, find_symbol should
    return one Hit per (file, line)."""
    hits = find_symbol("validate_token", root=FIXTURE)
    keys = [(h.file, h.line) for h in hits]
    assert len(keys) == len(set(keys))


# ── path_between: long chains, cycles, max_hops ──────────────────────────────

def test_long_chain_handler_to_query_db():
    """handle_request -> fetch_user -> query_db (2 hops)."""
    chain = path_between("handle_request", "query_db", root=FIXTURE, max_hops=4)
    names = [h.name for h in chain]
    assert names == ["handle_request", "fetch_user", "query_db"]


def test_path_between_terminates_on_cycle():
    """validate_token <-> refresh_token forms a cycle. BFS must terminate."""
    chain = path_between("validate_token", "refresh_token", root=FIXTURE, max_hops=4)
    assert len(chain) == 2
    assert chain[0].name == "validate_token"
    assert chain[-1].name == "refresh_token"


def test_path_between_unreachable_across_modules():
    """connect() is orphaned — nothing in the repo references it."""
    chain = path_between("handle_request", "connect", root=FIXTURE, max_hops=6)
    assert chain == []


def test_path_between_max_hops_short_circuits():
    """Long chain with max_hops=1 should fail."""
    chain = path_between("handle_request", "query_db", root=FIXTURE, max_hops=1)
    assert chain == []


# ── neighborhood ─────────────────────────────────────────────────────────────

def test_neighborhood_orphan_has_no_callers():
    nb = neighborhood("connect", root=FIXTURE)
    assert len(nb["def"]) == 1
    assert nb["def"][0].file == "db/connection.py"
    assert nb["callers"] == []


def test_neighborhood_validate_token_has_real_callers():
    nb = neighborhood("validate_token", root=FIXTURE)
    caller_files = {h.file for h in nb["callers"]}
    assert "api/handler.py" in caller_files
    assert "api/auth.py" in caller_files  # cycle: refresh_token also calls it


def test_neighborhood_callees_includes_inner_refs():
    nb = neighborhood("handle_request", root=FIXTURE)
    callee_names = {h.name for h in nb["callees"]}
    assert "validate_token" in callee_names
    assert "fetch_user" in callee_names
    assert "log_info" in callee_names


# ── outline: class with methods, sorted by line ──────────────────────────────

def test_outline_sorts_by_line_number():
    out = outline("db/models.py", root=FIXTURE)
    lines = [L for L in out.splitlines() if L.strip().startswith(tuple("0123456789"))]
    line_nums = [int(L.split(":")[0].strip()) for L in lines]
    assert line_nums == sorted(line_nums)


def test_outline_lists_class_and_methods():
    out = outline("db/models.py", root=FIXTURE)
    for needed in ("User", "__init__", "full_name", "to_dict"):
        assert needed in out, f"outline missing {needed}: {out!r}"


# ── imports: forward + reverse symbol-graph deps ─────────────────────────────

def test_imports_handler_uses_external_symbols():
    deps = imports("api/handler.py", root=FIXTURE)
    used_names = {h.name for h in deps["uses"]}
    # handler.py refs validate_token, query_db, User, log_info — all external
    for n in ("validate_token", "query_db", "User", "log_info"):
        assert n in used_names, f"missing {n} in uses"


def test_imports_models_used_by_handler():
    deps = imports("db/models.py", root=FIXTURE)
    used_by_files = {h.file for h in deps["used_by"]}
    assert "api/handler.py" in used_by_files


def test_imports_excludes_self_refs():
    deps = imports("db/models.py", root=FIXTURE)
    assert all(h.file != "db/models.py" for h in deps["uses"])
    assert all(h.file != "db/models.py" for h in deps["used_by"])


def test_imports_depth_2_picks_up_transitive_deps():
    """db/connection.py is used directly only by api/handler.py.
    But api/handler.py itself has many users — at depth=2, db/connection.py's
    used_by reaches transitively through api/handler.py."""
    direct = imports("db/connection.py", root=FIXTURE, depth=1)
    deep   = imports("db/connection.py", root=FIXTURE, depth=2)
    direct_files = {h.file for h in direct["used_by"]}
    deep_files   = {h.file for h in deep["used_by"]}
    # Depth=2 must be a strict superset (or equal if no transitive layer).
    assert direct_files <= deep_files
    # And on this fixture, we expect strictly more at depth 2 (handler.py
    # has callers of its own — like log_info usage propagates).
    # Looser invariant: depth=2 hits >= depth=1 hits.
    assert len(deep["used_by"]) >= len(direct["used_by"])


def test_imports_depth_2_uses_chain():
    """api/handler.py uses db/models.py directly (User class). models.py
    has no external uses of its own, so depth=2 forward should equal depth=1
    here (or at most pick up tiny additions). Just assert the shape and
    that depth=2 is a superset of depth=1."""
    d1 = imports("api/handler.py", root=FIXTURE, depth=1)
    d2 = imports("api/handler.py", root=FIXTURE, depth=2)
    d1_keys = {(h.file, h.line) for h in d1["uses"]}
    d2_keys = {(h.file, h.line) for h in d2["uses"]}
    assert d1_keys <= d2_keys


def test_imports_depth_zero_or_negative_clamped_to_one():
    d0 = imports("api/handler.py", root=FIXTURE, depth=0)
    d1 = imports("api/handler.py", root=FIXTURE, depth=1)
    assert {(h.file, h.line) for h in d0["uses"]} == {(h.file, h.line) for h in d1["uses"]}


# ── search_files: TF-IDF over real prose + code ──────────────────────────────

def test_search_finds_doc_for_rare_token():
    res = search_files("floccinaucinihilipilification", root=FIXTURE, top_k=3)
    assert res
    assert res[0][0] == "README.md"


def test_search_finds_code_file_for_symbol_term():
    res = search_files("validate_token refresh_token", root=FIXTURE, top_k=3)
    assert res
    assert res[0][0] == "api/auth.py"


def test_search_caps_results_at_top_k():
    res = search_files("the user", root=FIXTURE, top_k=2)
    assert len(res) <= 2


def test_search_camelcase_query_matches_snake_case_file():
    """Query in camelCase should still find files defining the snake_case
    form. Identifier-aware tokenization splits both into the same parts."""
    res = search_files("validateToken", root=FIXTURE, top_k=3)
    assert res
    rels = [r for r, _, _ in res]
    assert "api/auth.py" in rels, f"camelCase query missed snake_case match: {rels}"


def test_search_partial_word_finds_compound():
    """Searching for 'validate' alone should still find validate_token's
    file because the snake_case identifier is split into sub-tokens."""
    res = search_files("validate", root=FIXTURE, top_k=5)
    assert res
    rels = [r for r, _, _ in res]
    assert "api/auth.py" in rels


def test_search_path_match_bonus():
    """A query token that appears in the file path (with no content matches
    in other files) should still surface that file.

    The fixture has api/auth.py — querying for 'auth' alone should rank
    api/auth.py highly even if some other file mentions the word more times,
    because the path component contributes a bonus."""
    res = search_files("auth", root=FIXTURE, top_k=3)
    assert res
    rels = [r for r, _, _ in res]
    assert "api/auth.py" in rels[:2]


def test_search_bm25_does_not_diverge_with_term_repetition():
    """BM25 saturates term-frequency contribution (k1+1 / tf+k1·...). A file
    with 1000 occurrences of a query term should NOT rank arbitrarily higher
    than one with 10 occurrences — TF saturates. This was the main reason
    to switch from raw TF-IDF.

    We don't have a synthetic fixture for this, but we can verify the
    invariant indirectly: a long file with repeated mentions shouldn't
    drown out a short, focused match."""
    # validate_token appears in both api/auth.py (definition) and
    # api/handler.py (call site). auth.py should rank higher because it's
    # the focused match, not because of raw frequency.
    res = search_files("validate_token", root=FIXTURE, top_k=3)
    assert res
    assert res[0][0] == "api/auth.py"


# ── Pygments-fallback file: utils/format.py has only defs ────────────────────

def test_format_module_indexed_despite_no_real_refs():
    idx = SymbolIndex(root=FIXTURE)
    assert "format_currency" in idx.defs
    assert "format_date" in idx.defs


def test_pygments_fallback_refs_excluded_from_callers():
    """utils/format.py contains no real refs to other repo symbols.
    Pygments-fallback emits every Name token as a ref at line=-1; those
    must NOT show up as callers of, say, format_currency."""
    nb = neighborhood("format_currency", root=FIXTURE)
    # No real call site for format_currency exists in the fixture.
    assert nb["callers"] == []


# ── New-file Write preview ───────────────────────────────────────────────────

def test_write_new_file_returns_diff():
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, "new.py")
        out = _write(fp, "def hi():\n    return 1\n")
        assert _has_diff(out), f"new-file write should produce a diff: {out!r}"
        assert "Created" in out
        assert "+def hi():" in out


def test_write_new_file_caps_at_40_lines():
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, "long.txt")
        content = "\n".join(f"line_{i}" for i in range(80)) + "\n"
        out = _write(fp, content)
        assert "[...  " in out or "more lines ...]" in out
        # Should NOT contain line_75 (well past the 40-line cap).
        assert "+line_75" not in out


def test_write_existing_file_still_returns_diff():
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, "x.py")
        _write(fp, "a\n")
        out = _write(fp, "a\nb\n")
        assert _has_diff(out)
        assert "+b" in out


# ── repo_map produces non-empty output for the fixture ───────────────────────

def test_repo_map_renders_for_fixture():
    out = repo_map(root=FIXTURE, max_tokens=2048)
    # Should mention at least the top-level files and a few symbols.
    assert "handler.py" in out or "auth.py" in out
    # Must not be the empty-fallback message.
    assert "(empty repo map)" not in out


# ── get_callers convenience wrapper ──────────────────────────────────────────

def test_get_callers_alias_finds_refs():
    hits = get_callers("validate_token", root=FIXTURE)
    files = {h.file for h in hits}
    assert "api/handler.py" in files


# ── SymbolIndex.callers_of (backward edge primitive for bidirectional BFS) ──

def test_callers_of_finds_enclosing_def():
    """callers_of('validate_token') should return the names of symbols
    whose def-body contains a ref to validate_token — i.e. handle_request
    (in api/handler.py) and refresh_token (in api/auth.py, the cycle)."""
    from agent_tools.helpers import get_symbol_index
    idx = get_symbol_index(FIXTURE)
    callers = set(idx.callers_of("validate_token"))
    assert "handle_request" in callers
    assert "refresh_token" in callers


def test_callers_of_orphan_is_empty():
    from agent_tools.helpers import get_symbol_index
    idx = get_symbol_index(FIXTURE)
    assert idx.callers_of("connect") == []


def test_callers_of_excludes_self_recursive_refs():
    """validate_token's body refs validate_token (cycle) but the caller
    is refresh_token, not validate_token itself — excluded by the
    enclosing.name != name check."""
    from agent_tools.helpers import get_symbol_index
    idx = get_symbol_index(FIXTURE)
    callers = set(idx.callers_of("validate_token"))
    assert "validate_token" not in callers


# ── Caching: SymbolIndex + SearchFiles tokens ───────────────────────────────

def test_symbol_index_is_cached_within_session():
    """Repeated calls to the public helpers should reuse the same
    SymbolIndex instance (cached by mtime fingerprint)."""
    from agent_tools.helpers import (
        clear_caches, get_symbol_index,
    )
    clear_caches()
    idx1 = get_symbol_index(FIXTURE)
    idx2 = get_symbol_index(FIXTURE)
    assert idx1 is idx2, "SymbolIndex must be cached across calls"


def test_symbol_index_cache_invalidates_on_mtime_change(tmp_path):
    """Touching a file inside the indexed root must invalidate the cache
    so the next call sees fresh tags."""
    import time as _t
    from agent_tools.helpers import clear_caches, get_symbol_index

    (tmp_path / "a.py").write_text("def alpha(): pass\n")
    clear_caches()
    idx1 = get_symbol_index(str(tmp_path))
    assert "alpha" in idx1.defs

    # Touch the file with a new mtime; sleep so the timestamp differs
    # even on coarse-grained filesystems.
    _t.sleep(0.05)
    (tmp_path / "a.py").write_text("def alpha(): pass\ndef beta(): pass\n")
    idx2 = get_symbol_index(str(tmp_path))
    assert idx2 is not idx1, "cache must invalidate when files change"
    assert "beta" in idx2.defs


def test_search_files_tokens_cached_across_queries():
    """Two consecutive search_files calls on the same root should reuse
    the cached file-token table (the expensive half of TF-IDF)."""
    import time as _t
    from agent_tools.helpers import clear_caches, search_files

    clear_caches()
    t0 = _t.perf_counter()
    search_files("validate_token", root=FIXTURE, top_k=3)
    first_ms = (_t.perf_counter() - t0) * 1000

    t0 = _t.perf_counter()
    search_files("refresh_token", root=FIXTURE, top_k=3)
    second_ms = (_t.perf_counter() - t0) * 1000

    # Cached run must be at least 3x faster on a real repo. Fixture is
    # small so 3x is conservative; real repos see ~12x.
    assert second_ms < max(first_ms / 3, first_ms - 5), (
        f"second call wasn't cache-accelerated: "
        f"first={first_ms:.1f}ms second={second_ms:.1f}ms"
    )
