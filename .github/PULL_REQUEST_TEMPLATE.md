## Summary

<!-- What does this PR change, and why? -->

## Related issues

<!-- e.g. Closes #123 -->

## Checklist

- [ ] `python -m pytest tests/ -x -q` passes
- [ ] `ruff check` is clean on the files this PR touches
- [ ] No new dependencies added to core without discussion
- [ ] Runtime state uses `RuntimeContext`, not `config["_xxx"]`
- [ ] Plugin tools export `TOOL_DEFS`, not direct `register_tool()` calls
- [ ] New modules are added to `pyproject.toml`
- [ ] No secrets or API keys in committed code
- [ ] Bug fixes and new features are in separate PRs (one concern per PR)
- [ ] Documentation is updated where relevant
