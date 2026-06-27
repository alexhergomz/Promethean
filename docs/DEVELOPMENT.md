# Development

Local tooling for working on Promethean. For contribution guidelines and the
PR checklist, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Setup

```bash
pip install -e ".[all]"   # or pick the extras you need: web, autosuggest, graph
pip install pytest
```

## Tests

```bash
python -m pytest tests/ -x -q
```

## Linting and formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and
formatting, configured under `[tool.ruff]` in `pyproject.toml` (lint rules
`F` + `I`, line length 100).

```bash
ruff check .          # lint
ruff check --fix .    # auto-fix the fixable findings
ruff format .         # format
```

In CI, `ruff check` runs as an advisory step: it surfaces findings as
annotations but does not fail the build, so the pre-existing baseline does not
block work. New and changed code is kept clean by the pre-commit hook below.

## Pre-commit hooks

[pre-commit](https://pre-commit.com/) runs ruff (lint + format) and a few
file-hygiene checks against the files you touch.

```bash
pip install pre-commit
pre-commit install          # install the git hook
pre-commit run --all-files  # optional: run against the whole tree
```

Once installed, the hooks run automatically on every `git commit`.
