"""pytest fixtures shared across the test suite."""
import os
import pytest


@pytest.fixture(autouse=True)
def _redirect_rabbit_hole_workspace(tmp_path, monkeypatch):
    """Redirect rabbit-hole workspaces to tmp_path so tests don't pollute
    ~/.promethean/rabbit-hole/. The SubAgentManager.spawn() rabbit-hole
    branch checks PROMETHEAN_RABBIT_HOLE_DIR before falling back to the
    home-dir default.

    Autouse so any test that goes through cmd_rabbit_hole or directly
    spawns a deep-research-rabbit-hole agent gets the redirection
    automatically — no per-test wiring required. Without this, the
    workspace base path in spawn() defaults to ~/.promethean/rabbit-hole/
    and every spawn-touching test leaks a directory there.
    """
    rh_root = tmp_path / "rh-test-root"
    rh_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PROMETHEAN_RABBIT_HOLE_DIR", str(rh_root))
    yield
