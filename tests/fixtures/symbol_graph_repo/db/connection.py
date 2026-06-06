"""Database connection helpers."""


def query_db(sql, params):
    """Used by api.handler.fetch_user."""
    return {"id": params[0], "name": "stub"}


def connect():
    """Orphan: defined but never referenced anywhere in this fixture.
    SearchFiles + FindSymbol should still find it; Neighborhood should
    return an empty callers list."""
    return {"conn": "fake"}
