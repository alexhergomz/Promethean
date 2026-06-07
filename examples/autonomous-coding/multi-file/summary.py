"""Aggregations over Transaction records."""
from collections import defaultdict


def total(txns):
    return sum(t.amount for t in txns)


def by_category(txns):
    out = defaultdict(float)
    for t in txns:
        out[t.category] += t.amount
    return dict(out)
