"""Formatting helpers — only defs, no refs to other repo symbols.
This exercises the pygments-fallback path in repomap.get_tags_raw
(which emits ref tags at line=-1 for files with only definitions).
"""


def format_currency(amount):
    return f"${amount:,.2f}"


def format_date(d):
    return d.isoformat()


def parse(s):
    """Same name as api.auth.parse — both should be discoverable."""
    return s.strip()
