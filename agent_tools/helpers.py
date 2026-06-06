"""Public helper functions on top of the vendored RepoMap.

Designed for the programmatic-tool-calling pattern: the model writes Python
that imports these and runs via the Bash tool. Output is plain text/markdown
suitable for direct printing to stdout.
"""
from __future__ import annotations

import os
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ._shims import CharRatioCounter, SimpleIO
from .repomap import RepoMap, find_src_files

Hit = namedtuple("Hit", "file line name kind preview")


# ── Caches: SymbolIndex + SearchFiles token table ────────────────────────────
# Both cached by (abs_root) and validated by a cheap mtime fingerprint
# (sum of source-file mtimes). When repo content changes, fingerprint
# differs and we rebuild. Fingerprint is O(n stat calls) — much cheaper
# than the actual index/tokenization work it gates.

_INDEX_CACHE: Dict[str, Tuple[float, "SymbolIndex"]] = {}
_TOKENS_CACHE: Dict[str, Tuple[float, Dict[str, List[str]]]] = {}


def _mtime_fingerprint(abs_root: str) -> float:
    total = 0.0
    for fname in find_src_files(abs_root):
        try:
            total += os.path.getmtime(fname)
        except OSError:
            pass
    return total


def get_symbol_index(root: str = ".") -> "SymbolIndex":
    """Return a cached SymbolIndex for `root`, or build + cache one.

    The cache is invalidated when any indexed file's mtime changes (sum
    fingerprint). Building is otherwise expensive because it walks every
    source file and pulls tags through tree-sitter — even with the
    sqlite tag cache, the walk + dict population dominates per-call cost
    for repeat usage in a single agent turn.
    """
    abs_root = str(Path(root).resolve())
    fp = _mtime_fingerprint(abs_root)
    cached = _INDEX_CACHE.get(abs_root)
    if cached is not None and cached[0] == fp:
        return cached[1]
    idx = SymbolIndex(root=abs_root)
    _INDEX_CACHE[abs_root] = (fp, idx)
    return idx


def _get_or_build_tokens(abs_root: str) -> Dict[str, List[str]]:
    """Return cached `rel_path → token list` for SearchFiles, or build it.

    Building reads + tokenizes every source file (after a small extension
    blocklist for minified bundles and lock files) — this is the
    expensive half of TF-IDF search. The DF table is computed per-query
    from these cached tokens since DF only depends on query terms.
    """
    fp = _mtime_fingerprint(abs_root)
    cached = _TOKENS_CACHE.get(abs_root)
    if cached is not None and cached[0] == fp:
        return cached[1]

    rm = RepoMap(
        map_tokens=4096,
        root=abs_root,
        main_model=CharRatioCounter(),
        io=SimpleIO(),
        refresh="files",
    )
    file_tokens: Dict[str, List[str]] = {}
    for fname in find_src_files(abs_root):
        if not Path(fname).is_file():
            continue
        low = fname.lower()
        if low.endswith((".min.js", ".min.css", ".bundle.js",
                         "package-lock.json", "yarn.lock", ".lock")):
            continue
        try:
            text = Path(fname).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > 1_000_000:
            continue
        tokens = _tokenize_for_tfidf(text)
        if not tokens:
            continue
        rel = rm.get_rel_fname(fname)
        file_tokens[rel] = tokens

    _TOKENS_CACHE[abs_root] = (fp, file_tokens)
    return file_tokens


def clear_caches() -> None:
    """Test hook — drop all cached indices."""
    _INDEX_CACHE.clear()
    _TOKENS_CACHE.clear()


# ── Main entry point: full repo map ───────────────────────────────────────────

def repo_map(
    root: str = ".",
    focus_files: Optional[Iterable[str]] = None,
    max_tokens: int = 2048,
    verbose: bool = False,
) -> str:
    """Render an Aider-style ranked repo map for the given root.

    `focus_files` are the files you're currently working on — the PageRank
    walk starts from them, so symbols they reference get higher weight in
    the output. Omit to get a "global" view.

    Returns a markdown-ish text block listing the most important
    files and their key symbols (defs only, with a few lines of context),
    sized to roughly `max_tokens` (4-char-per-token estimate).
    """
    root = str(Path(root).resolve())
    chat_files = []
    other_files = []

    if focus_files:
        for f in focus_files:
            p = (Path(root) / f).resolve() if not Path(f).is_absolute() else Path(f).resolve()
            if p.is_file():
                chat_files.append(str(p))

    # Gather everything else as "other_files"
    chat_set = set(chat_files)
    for f in find_src_files(root):
        if f not in chat_set and Path(f).is_file():
            other_files.append(f)

    rm = RepoMap(
        map_tokens=max_tokens,
        root=root,
        main_model=CharRatioCounter(),
        io=SimpleIO(verbose=verbose),
        verbose=verbose,
        refresh="files",
    )
    out = rm.get_repo_map(chat_files, other_files)
    return out or "(empty repo map)"


# ── Symbol search: where is `name` defined? ───────────────────────────────────

def find_symbol(name: str, root: str = ".", kind: str = "def") -> List[Hit]:
    """Find every definition of `name` across the repo (kind='def') or
    every reference (kind='ref'). Returns a list of Hits with file/line/kind.

    The match is exact on the symbol name. For substring/regex search,
    fall back to `grep`.
    """
    root = str(Path(root).resolve())
    rm = RepoMap(
        map_tokens=4096,
        root=root,
        main_model=CharRatioCounter(),
        io=SimpleIO(),
        refresh="files",
    )
    hits: List[Hit] = []
    seen: Set[Tuple[str, int]] = set()
    for fname in find_src_files(root):
        if not Path(fname).is_file():
            continue
        rel = rm.get_rel_fname(fname)
        for tag in rm.get_tags(fname, rel):
            if tag.name != name or tag.kind != kind:
                continue
            if tag.line < 0 and kind == "ref":
                continue  # pygments fallback ref, not a real call site
            key = (rel, tag.line)
            if key in seen:
                continue
            seen.add(key)
            preview = _preview_line(fname, tag.line)
            hits.append(Hit(file=rel, line=tag.line + 1, name=tag.name,
                            kind=tag.kind, preview=preview))
    return hits


def get_callers(name: str, root: str = ".") -> List[Hit]:
    """Find every reference to `name`. Useful for 'who calls foo?' queries."""
    return find_symbol(name, root=root, kind="ref")


# ── Single-file outline: list every def in a file ────────────────────────────

def outline(file: str, root: str = ".") -> str:
    """Return an outline of one file as 'line: name (kind)' rows.

    Quick way to scan a file's structure without reading the whole thing.
    """
    root = str(Path(root).resolve())
    p = (Path(root) / file).resolve() if not Path(file).is_absolute() else Path(file).resolve()
    if not p.is_file():
        return f"(not a file: {p})"
    rm = RepoMap(
        map_tokens=4096,
        root=root,
        main_model=CharRatioCounter(),
        io=SimpleIO(),
        refresh="files",
    )
    rel = rm.get_rel_fname(str(p))
    rows: List[Tuple[int, str]] = []
    seen: Set[Tuple[int, str]] = set()
    for tag in rm.get_tags(str(p), rel):
        if tag.kind != "def":
            continue
        key = (tag.line, tag.name)
        if key in seen:
            continue
        seen.add(key)
        rows.append((tag.line, tag.name))
    if not rows:
        return f"(no definitions found in {rel})"
    rows.sort(key=lambda r: r[0])
    return f"{rel}:\n" + "\n".join(f"{ln + 1:>5}: {nm}" for ln, nm in rows)


# ── Symbol graph: cached index over the whole repo ───────────────────────────

class SymbolIndex:
    """One-shot scan of every tag in the repo, indexed for fast queries.

    Walks `find_src_files(root)` once, calling `RepoMap.get_tags` per file
    (which is sqlite-cached on file mtime — repeat calls are essentially
    free). The four lookup tables it builds are what the GitNexus-style
    graph queries (neighborhood, path_between, imports) need.

    Build once per top-level tool invocation; do not cache across calls
    because the user's repo state can change between turns.
    """
    def __init__(self, root: str = ".", verbose: bool = False):
        self.root = str(Path(root).resolve())
        self._rm = RepoMap(
            map_tokens=4096,
            root=self.root,
            main_model=CharRatioCounter(),
            io=SimpleIO(verbose=verbose),
            refresh="files",
        )
        # name -> list[Tag] (definitions of this symbol, anywhere in repo)
        self.defs: Dict[str, List] = defaultdict(list)
        # name -> list[Tag] (references to this symbol, anywhere in repo)
        self.refs: Dict[str, List] = defaultdict(list)
        # rel_fname -> list[Tag] (every def in this file, sorted by line)
        self.defs_by_file: Dict[str, List] = defaultdict(list)
        # rel_fname -> list[Tag] (every ref in this file, sorted by line)
        self.refs_by_file: Dict[str, List] = defaultdict(list)

        # Dedup key: (file, line, kind, name). The vendored aider repomap
        # has a known emit-duplicate bug; without dedup, def_span returns
        # zero-width spans (because the "next def" is the duplicate of the
        # current one) and BFS can't follow callees.
        seen: Set[Tuple[str, int, str, str]] = set()
        for fname in find_src_files(self.root):
            if not Path(fname).is_file():
                continue
            rel = self._rm.get_rel_fname(fname)
            for tag in self._rm.get_tags(fname, rel):
                key = (rel, tag.line, tag.kind, tag.name)
                if key in seen:
                    continue
                seen.add(key)
                if tag.kind == "def":
                    self.defs[tag.name].append(tag)
                    self.defs_by_file[rel].append(tag)
                elif tag.kind == "ref":
                    # line == -1 means this came from the pygments fallback
                    # path (files that have only defs and no refs). Those
                    # tokens aren't real call sites — skip them, otherwise
                    # every identifier in the file shows up as a caller.
                    if tag.line < 0:
                        continue
                    self.refs[tag.name].append(tag)
                    self.refs_by_file[rel].append(tag)

        for tags in self.defs_by_file.values():
            tags.sort(key=lambda t: t.line)
        for tags in self.refs_by_file.values():
            tags.sort(key=lambda t: t.line)

    def def_span(self, def_tag) -> Tuple[int, int]:
        """Approximate (start, end) line range of a def's body.

        We don't have AST extents — only the def-name line. Approximation:
        body runs from the def line to the line of the *next* def in the
        same file (exclusive). Last def in file runs to infinity. Good
        enough for callee extraction in flat-ish layouts; nested defs
        will get attributed to the outer enclosing def.
        """
        defs = self.defs_by_file.get(def_tag.rel_fname, [])
        for i, d in enumerate(defs):
            if d is def_tag or (d.line == def_tag.line and d.name == def_tag.name):
                start = d.line
                end = defs[i + 1].line if i + 1 < len(defs) else 10**9
                return (start, end)
        return (def_tag.line, def_tag.line + 1)

    def callees_of(self, def_tag) -> List:
        """Refs that fall inside this def's approximate body span."""
        start, end = self.def_span(def_tag)
        out = []
        for r in self.refs_by_file.get(def_tag.rel_fname, []):
            if start < r.line < end and r.name != def_tag.name:
                out.append(r)
        return out

    def callers_of(self, name: str) -> List[str]:
        """Names of symbols whose body contains a call to `name`.

        For each ref to `name`, finds the enclosing def in the same
        file by scanning the file's def list (sorted by line) and
        picking the latest def whose line <= ref.line. That's the
        function/method/class that contains the call. Used by
        path_between's backward BFS pass.
        """
        result = set()
        for ref in self.refs.get(name, []):
            enclosing = None
            for d in self.defs_by_file.get(ref.rel_fname, []):
                if d.line <= ref.line:
                    enclosing = d
                else:
                    break
            if enclosing is not None and enclosing.name != name:
                result.add(enclosing.name)
        return list(result)


# ── GitNexus-style graph queries ─────────────────────────────────────────────

def neighborhood(name: str, root: str = ".") -> Dict[str, List[Hit]]:
    """One-shot lookup: definition(s), callers, and callees of `name`.

    Returns a dict with three lists of Hit:
        "def"     — every def site of `name`
        "callers" — every ref to `name` (who uses it)
        "callees" — refs that appear inside `name`'s def body (what it uses)

    Callees are approximate: see SymbolIndex.def_span.
    """
    idx = get_symbol_index(root)
    def_hits = [_tag_to_hit(t) for t in idx.defs.get(name, [])]
    caller_hits = [_tag_to_hit(t) for t in idx.refs.get(name, [])]
    callee_hits: List[Hit] = []
    for d in idx.defs.get(name, []):
        for r in idx.callees_of(d):
            callee_hits.append(_tag_to_hit(r))
    return {"def": def_hits, "callers": caller_hits, "callees": callee_hits}


def path_between(
    a: str, b: str, root: str = ".", max_hops: int = 4
) -> List[Hit]:
    """Find the shortest call chain from `a` to `b` (≤ max_hops edges).

    Edge: symbol X → symbol Y if Y is referenced inside X's def body.
    Returns the chain of def-Hits from a to b (inclusive), or [] if no
    path within `max_hops`.

    Implementation: bidirectional layer-by-layer BFS. Forward expansion
    from `a` follows callees; backward expansion from `b` follows
    callers (`SymbolIndex.callers_of`). After each layer we check
    whether the two visited sets overlap and pick the intersection
    minimizing dist_f + dist_b. For a graph with branching factor k,
    this explores ~2·k^(d/2) nodes vs the unidirectional k^d — a real
    win on long chains in big repos.
    """
    idx = get_symbol_index(root)
    if not idx.defs.get(a) or not idx.defs.get(b):
        return []
    if a == b:
        return [_tag_to_hit(idx.defs[a][0])]

    # Distances from a (forward) and from b (backward) for all visited
    # symbols. Parent maps reconstruct the path on success.
    dist_f: Dict[str, int] = {a: 0}
    dist_b: Dict[str, int] = {b: 0}
    parent_f: Dict[str, Optional[str]] = {a: None}
    parent_b: Dict[str, Optional[str]] = {b: None}
    frontier_f: List[str] = [a]
    frontier_b: List[str] = [b]

    def _best_meet() -> Optional[str]:
        common = set(dist_f) & set(dist_b)
        if not common:
            return None
        return min(common, key=lambda n: dist_f[n] + dist_b[n])

    while frontier_f and frontier_b:
        # Always expand the smaller side (standard bidirectional optim).
        if len(frontier_f) <= len(frontier_b):
            new_frontier: List[str] = []
            for cur in frontier_f:
                for d in idx.defs.get(cur, []):
                    for r in idx.callees_of(d):
                        nxt = r.name
                        if nxt in dist_f:
                            continue
                        dist_f[nxt] = dist_f[cur] + 1
                        parent_f[nxt] = cur
                        new_frontier.append(nxt)
            frontier_f = new_frontier
        else:
            new_frontier = []
            for cur in frontier_b:
                for caller in idx.callers_of(cur):
                    if caller in dist_b:
                        continue
                    dist_b[caller] = dist_b[cur] + 1
                    parent_b[caller] = cur
                    new_frontier.append(caller)
            frontier_b = new_frontier

        meet = _best_meet()
        if meet is not None and dist_f[meet] + dist_b[meet] <= max_hops:
            # Walk a → meet via parent_f, then meet → b via parent_b.
            chain: List[str] = []
            n: Optional[str] = meet
            while n is not None:
                chain.append(n)
                n = parent_f[n]
            chain.reverse()
            n = parent_b[meet]
            while n is not None:
                chain.append(n)
                n = parent_b[n]
            return [_tag_to_hit(idx.defs[s][0]) for s in chain
                    if idx.defs.get(s)]

    return []


def imports(
    file: str, root: str = ".", depth: int = 1
) -> Dict[str, List[Hit]]:
    """Forward + reverse deps for a single file, optionally transitive.

    Returns:
        "uses"     — Hits at external def sites this file (transitively) uses
        "used_by"  — Hits at external ref sites for this file's defs (transitively)

    `depth=1` (default) is the direct-only behavior: the file's own refs
    that resolve to defs elsewhere, and other files' refs to this file's
    defs. `depth>1` does BFS over files: starting from the input file,
    each layer follows the same forward/backward edges as the original
    1-hop logic, accumulating Hits from every newly-reached file.

    Pure symbol-graph relation; doesn't follow Python `import` statements
    syntactically — it tracks actual usage.
    """
    if depth < 1:
        depth = 1
    idx = get_symbol_index(root)
    p = (Path(idx.root) / file).resolve() if not Path(file).is_absolute() else Path(file).resolve()
    rel = idx._rm.get_rel_fname(str(p))

    # ── Forward BFS (uses) ──────────────────────────────────────────────────
    visited_f: Set[str] = {rel}
    frontier_f: Set[str] = {rel}
    uses: List[Hit] = []
    seen_use: Set[Tuple[str, int]] = set()

    for _ in range(depth):
        next_frontier: Set[str] = set()
        for cur in frontier_f:
            cur_def_names: Set[str] = {
                t.name for t in idx.defs_by_file.get(cur, [])
            }
            cur_ref_names: Set[str] = {
                t.name for t in idx.refs_by_file.get(cur, [])
            }
            for ref_name in cur_ref_names:
                if ref_name in cur_def_names:
                    continue   # ref to symbol defined in same file
                for d in idx.defs.get(ref_name, []):
                    if d.rel_fname in visited_f:
                        continue
                    key = (d.rel_fname, d.line)
                    if key in seen_use:
                        continue
                    seen_use.add(key)
                    uses.append(_tag_to_hit(d))
                    next_frontier.add(d.rel_fname)
        if not next_frontier:
            break
        visited_f.update(next_frontier)
        frontier_f = next_frontier

    # ── Backward BFS (used_by) ──────────────────────────────────────────────
    visited_b: Set[str] = {rel}
    frontier_b: Set[str] = {rel}
    used_by: List[Hit] = []
    seen_by: Set[Tuple[str, int]] = set()

    for _ in range(depth):
        next_frontier = set()
        for cur in frontier_b:
            cur_def_names = {
                t.name for t in idx.defs_by_file.get(cur, [])
            }
            for def_name in cur_def_names:
                for r in idx.refs.get(def_name, []):
                    if r.rel_fname in visited_b:
                        continue
                    key = (r.rel_fname, r.line)
                    if key in seen_by:
                        continue
                    seen_by.add(key)
                    used_by.append(_tag_to_hit(r))
                    next_frontier.add(r.rel_fname)
        if not next_frontier:
            break
        visited_b.update(next_frontier)
        frontier_b = next_frontier

    uses.sort(key=lambda h: (h.file, h.line))
    used_by.sort(key=lambda h: (h.file, h.line))
    return {"uses": uses, "used_by": used_by}


# ── TF-IDF keyword search over file contents ────────────────────────────────
#
# The identifier-aware tokenizer lives in the standalone top-level
# ``search_tokenize`` module so it can be imported without dragging in this
# package's heavy tree-sitter/grep-ast/networkx chain (rabbit_hole reuses it).
# Re-exported here so existing ``from agent_tools.helpers import _tokenize_*``
# call sites keep working.
from search_tokenize import (  # noqa: E402
    _TFIDF_STOPWORDS,
    _TOKEN_RE,
    _SPLIT_RE,
    _split_identifier,
    _tokenize_for_search,
    _tokenize_for_tfidf,
)


def _path_components(rel_path: str) -> Set[str]:
    """Return the lowercase tokens appearing in a file's path components.

    Used to give a small score bonus when query terms also appear in the
    path. `api/auth.py` contributes `{api, auth, py}` (with sub-token
    splits inherited from `_split_identifier`).
    """
    out: Set[str] = set()
    for piece in rel_path.replace("\\", "/").split("/"):
        # Strip extension for the leaf
        bare = piece.rsplit(".", 1)[0]
        for tok in _split_identifier(bare):
            t = tok.lower()
            if len(t) >= 2:
                out.add(t)
    return out


def search_files(
    query: str, root: str = ".", top_k: int = 10
) -> List[Tuple[str, float, str]]:
    """Rank files by BM25 relevance to a free-text query.

    Tokenizes file contents with identifier-aware splitting (camelCase →
    [camel, case]; snake_case → [snake, case]; acronyms preserved), then
    scores each file with BM25 (k1=1.5, b=0.75 — standard params).
    A small bonus is added per query token that also appears in the
    file's path components.

    Encoder-free: zero embedding models, zero external services. Lexical
    only — but with identifier-awareness, `searchFile` matches `search_file`
    and a query for `validate token` ranks `api/auth.py:validate_token`
    highly.

    Returns list of (relative_path, score, first_match_preview), top_k by
    score, descending. Use this as a fallback when FindSymbol misses —
    e.g. searching for a config key, error message, or doc topic that
    isn't a symbol.
    """
    import math
    root_abs = str(Path(root).resolve())

    q_tokens_list = _tokenize_for_search(query)
    q_tokens = set(q_tokens_list)
    if not q_tokens:
        return []

    file_tokens = _get_or_build_tokens(root_abs)
    if not file_tokens:
        return []

    # Document frequency for query terms only — we never need the full DF table.
    df: Dict[str, int] = defaultdict(int)
    for tokens in file_tokens.values():
        seen = set(tokens) & q_tokens
        for t in seen:
            df[t] += 1

    n_docs = len(file_tokens)
    avg_dl = sum(len(t) for t in file_tokens.values()) / max(1, n_docs)

    # BM25 IDF — Robertson-Spärck-Jones variant, clamped at 0 to avoid the
    # negative-IDF pathology when df > n_docs / 2.
    def _idf(t: str) -> float:
        n = df.get(t, 0)
        return max(0.0, math.log((n_docs - n + 0.5) / (n + 0.5) + 1))

    idf = {t: _idf(t) for t in q_tokens}

    # BM25 hyperparams — standard defaults.
    K1 = 1.5
    B  = 0.75
    PATH_BONUS = 0.5  # multiplier per query-token-in-path match (additive after BM25)

    scored: List[Tuple[str, float, str]] = []
    for rel, tokens in file_tokens.items():
        tf: Dict[str, int] = defaultdict(int)
        for t in tokens:
            if t in q_tokens:
                tf[t] += 1
        if not tf:
            # Path-only match — still score it if any query token is in the path.
            path_toks = _path_components(rel) & q_tokens
            if not path_toks:
                continue
            score = sum(idf.get(t, 0.0) for t in path_toks) * PATH_BONUS
            preview = _first_match_line(Path(root_abs) / rel, q_tokens)
            scored.append((rel, score, preview))
            continue

        dl = len(tokens)
        # BM25 formula:
        #   score = Σ idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avg_dl))
        denom_norm = K1 * (1 - B + B * dl / max(1, avg_dl))
        score = sum(
            idf[t] * (tf[t] * (K1 + 1)) / (tf[t] + denom_norm)
            for t in tf
        )

        # Additive path bonus — small reward when query tokens also appear in
        # the file's path components, scaled by their IDF so it can't drown
        # out content matches but does break ties.
        path_hits = _path_components(rel) & q_tokens
        if path_hits:
            score += PATH_BONUS * sum(idf.get(t, 0.0) for t in path_hits)

        preview = _first_match_line(Path(root_abs) / rel, q_tokens)
        scored.append((rel, score, preview))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _first_match_line(path: Path, q_tokens: Set[str]) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                lower = line.lower()
                if any(t in lower for t in q_tokens):
                    return f"L{i+1}: {line.strip()[:120]}"
    except OSError:
        pass
    return ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tag_to_hit(tag) -> Hit:
    fname = str(Path(tag.rel_fname))
    abs_path = tag.fname if hasattr(tag, "fname") else fname
    preview = _preview_line(abs_path, tag.line) if tag.line >= 0 else ""
    return Hit(
        file=tag.rel_fname,
        line=tag.line + 1 if tag.line >= 0 else -1,
        name=tag.name,
        kind=tag.kind,
        preview=preview,
    )


def _preview_line(fname: str, line_idx: int) -> str:
    if line_idx < 0:
        return ""
    try:
        with open(fname, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    return line.rstrip("\n")[:120]
    except OSError:
        return ""
    return ""
