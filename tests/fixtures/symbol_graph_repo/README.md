# Symbol Graph Test Repo

A tiny multi-module web service used as a fixture for tests of
`agent_tools.helpers` (Neighborhood, PathBetween, Imports, SearchFiles).

## Architecture

```
api/handler.py  →  api/auth.py
                →  db/connection.py  →  db/models.py
                →  utils/log.py
```

There is a deliberate cycle between `validate_token` and `refresh_token`
in `api/auth.py` to exercise the BFS visited-set in `path_between`.

## Rare-token sentinel for TF-IDF

This README intentionally contains the rare token
**floccinaucinihilipilification** so SearchFiles tests can verify that
documentation files dominate when the query mentions only such a token.
