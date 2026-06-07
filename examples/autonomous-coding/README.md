# Example — autonomous coding (local, $0)

A worked example of Promethean running the full agentic loop **entirely on a
single 8 GB consumer GPU** (Qwen3.5-9B), no cloud, no API keys.

## `multi-file/` — cross-file bugfix + a new feature

A small 4-file project (`model.py`, `parse.py`, `summary.py`,
`test_summary.py`) shipped in its **failing** state. From one prompt,
unsupervised, Promethean:

1. Ran the suite and reproduced the failures.
2. **Diagnosed across files** — the tests failed in `test_summary.py`, but the
   bug was in `parse.py` (`int("12.50")` → `ValueError`); fixed it to `float`.
3. **Added a feature** in `summary.py` — `top_category(txns)`, including the
   empty-input edge case, unprompted.
4. **Wrote a test** for it, then re-ran: **all tests pass**.

It edited three files coherently and used the symbol-graph to flag a cross-file
caller automatically (see `TRANSCRIPT.md`).

### Reproduce

```bash
# 1. start the local inference server (see the repo README)
qcoder server start serve

# 2. run the agent on this folder
cd examples/autonomous-coding/multi-file
promethean --accept-all -p "Run pytest and fix the bug (the cause may be in a \
different file than the failing test). Then add a top_category(txns) function to \
summary.py that returns the highest-spending category, with a test. Re-run the suite."
```

The full, cleaned run is in [`TRANSCRIPT.md`](TRANSCRIPT.md).

> Scope note: a 9B model on a deliberately small, self-contained project. The
> point is a coherent **local** agentic loop — debugging *and* feature work
> across files — running end to end at zero per-token cost, not frontier-scale
> reasoning.
