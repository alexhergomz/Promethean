"""Regression: a single hung research source must not abort the whole run.

Before the fix, ``as_completed(timeout=...)`` raised TimeoutError when any
source failed to finish in the budget, aborting the entire /research run
(macOS review §8.4). Now a straggler is recorded as a timeout and the sources
that finished still return their results.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

pytest.importorskip("research", reason="research package deps not installed")

from research import aggregator as agg  # noqa: E402
from research.sources import SOURCES  # noqa: E402


class _FakeSpec:
    def __init__(self, name, delay, results):
        self.name = name
        self.domains = ["web"]
        self._delay = delay
        self._results = results

    def search(self, q, limit, cfg, time_range=None):
        if self._delay:
            time.sleep(self._delay)
        return list(self._results)


def test_hung_source_degrades_to_partial(monkeypatch):
    fast = _FakeSpec("t_fast", 0.0, [])
    slow = _FakeSpec("t_slow", 5.0, [])  # far beyond the 0.5s budget below
    monkeypatch.setitem(SOURCES, "t_fast", fast)
    monkeypatch.setitem(SOURCES, "t_slow", slow)

    t0 = time.time()
    # source_timeout=0.25 → overall budget = 0.5s. slow sleeps 5s.
    brief = agg.research(
        "anything",
        sources=["t_fast", "t_slow"],
        use_cache=False,
        synthesize=False,
        source_timeout=0.25,
    )
    elapsed = time.time() - t0

    # Did NOT raise, and did NOT block on the 5s straggler.
    assert elapsed < 3.0, f"blocked on straggler: {elapsed:.1f}s"

    by_name = {s.name: s for s in brief.statuses}
    assert "t_fast" in by_name and by_name["t_fast"].ok
    assert "t_slow" in by_name and not by_name["t_slow"].ok
    assert "timeout" in (by_name["t_slow"].error or "").lower()


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
