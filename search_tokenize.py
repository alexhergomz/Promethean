"""Identifier-aware tokenizer for BM25 / TF-IDF keyword search.

Standalone, pure-stdlib module so it can be imported WITHOUT pulling in the
heavy ``agent_tools`` package (which eagerly imports tree-sitter / grep-ast /
networkx at package-init). Both ``agent_tools.helpers`` (SearchFiles) and
``rabbit_hole.synthesis`` (BM25 finding search) import the tokenizer from
here, so keyword search works even when the optional ``graph`` extra — and
its tree-sitter deps — is not installed.
"""
from __future__ import annotations

import re
from typing import List

_TFIDF_STOPWORDS = frozenset({
    "the", "and", "for", "are", "was", "but", "not", "you", "all", "any",
    "can", "had", "her", "his", "its", "our", "out", "two", "use", "way",
    "with", "this", "that", "have", "from", "they", "will", "would", "their",
    "what", "when", "where", "which", "while",
    "def", "class", "return", "import", "from", "self", "cls", "true",
    "false", "none", "lambda", "yield", "pass", "break", "continue",
    "raise", "try", "except", "finally", "elif", "else", "with",
    "var", "let", "const", "function", "new", "this", "void", "int",
    "str", "list", "dict", "tuple", "set", "bool", "type",
})

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Identifier-splitting regex — break `validateToken` into [validate, token],
# `HTTPSRequest` into [https, request], `parse_url_v2` into [parse, url, v2].
# The pattern emits a sub-token at: an uppercase letter preceded by a lowercase
# letter (camelCase boundary), an uppercase letter followed by a lowercase
# letter when preceded by another uppercase (acronym→Word boundary), a digit
# preceded by a letter, or a letter preceded by a digit.
_SPLIT_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])|"           # validateToken → validate | Token
    r"(?<=[A-Z])(?=[A-Z][a-z])|"      # HTTPSRequest  → HTTPS    | Request
    r"(?<=[a-zA-Z])(?=[0-9])|"        # parse2json    → parse    | 2json
    r"(?<=[0-9])(?=[a-zA-Z])"         # 2json         → 2        | json
)


def _split_identifier(ident: str) -> List[str]:
    """Split an identifier into sub-tokens (camelCase / snake_case / acronyms).

    Always returns the full original token AND its parts. So `validate_token`
    yields `["validate_token", "validate", "token"]`. This means an exact
    match for `validate_token` still scores highly, while a partial query
    like `validate` or `token` also hits.
    """
    parts = [ident]
    # snake_case + kebab-case + dotted: split on _, -, .
    for piece in re.split(r"[_\-.]+", ident):
        if not piece:
            continue
        if piece != ident:
            parts.append(piece)
        # Now camelCase / acronym splits within the piece
        for sub in _SPLIT_RE.split(piece):
            if sub and sub != piece and len(sub) >= 2:
                parts.append(sub)
    return parts


def _tokenize_for_search(text: str) -> List[str]:
    """Identifier-aware tokenizer for BM25 search.

    Emits the original identifier AND its sub-parts (camelCase / snake_case
    splits). Filters stopwords on the lowercased form. Sub-tokens shorter
    than 2 chars are dropped to avoid noise from `i`, `j`, etc.
    """
    out: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        ident = m.group(0)
        for part in _split_identifier(ident):
            t = part.lower()
            if len(t) < 2:
                continue
            if t in _TFIDF_STOPWORDS:
                continue
            out.append(t)
    return out


# Backward-compat alias — older code paths may still import this name.
_tokenize_for_tfidf = _tokenize_for_search
