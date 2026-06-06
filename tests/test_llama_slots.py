"""Slot admin client (llama_slots.py) — HTTP wrappers tested with mocks.

We don't talk to a real server here; we patch httpx.Client to return
canned responses and verify the URL / payload shape and the helpers'
parsing behavior.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from llama_slots import (
    LlamaSlotsError,
    SlotInfo,
    erase_slot,
    find_idle_slot,
    list_slots,
    park_slot,
    restore_slot,
    save_slot,
)


def _mock_client(get_status=200, get_json=None, post_status=200, post_json=None):
    """Build a fake httpx.Client whose context-manager protocol returns
    a magic mock with `get` and `post` methods stubbed."""
    fake_client = MagicMock()
    if get_json is None:
        get_json = []
    if post_json is None:
        post_json = {"ok": True}

    get_resp = MagicMock(status_code=get_status, text=json.dumps(get_json))
    get_resp.json.return_value = get_json
    post_resp = MagicMock(status_code=post_status, text=json.dumps(post_json))
    post_resp.json.return_value = post_json

    fake_client.get.return_value = get_resp
    fake_client.post.return_value = post_resp
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    return fake_client


# ── list_slots ─────────────────────────────────────────────────────────────

def test_list_slots_parses_response():
    canned = [
        {"id": 0, "state": "idle", "n_ctx": 57344, "n_past": 0, "prompt": ""},
        {"id": 1, "state": "processing", "n_ctx": 57344, "n_past": 1234,
         "prompt": "system prompt..."},
    ]
    fake = _mock_client(get_json=canned)
    with patch("llama_slots._client", return_value=fake):
        result = list_slots()
    assert len(result) == 2
    assert isinstance(result[0], SlotInfo)
    assert result[0].id == 0 and result[0].state == "idle"
    assert result[1].n_past == 1234


def test_list_slots_truncates_long_prompt_preview():
    canned = [{"id": 0, "state": "idle", "n_ctx": 57344, "n_past": 0,
               "prompt": "x" * 5000}]
    fake = _mock_client(get_json=canned)
    with patch("llama_slots._client", return_value=fake):
        result = list_slots()
    assert len(result[0].prompt) == 200


def test_list_slots_raises_on_error_status():
    fake = _mock_client(get_status=404, get_json={"error": "no slots"})
    with patch("llama_slots._client", return_value=fake):
        with pytest.raises(LlamaSlotsError):
            list_slots()


# ── find_idle_slot ─────────────────────────────────────────────────────────

def test_find_idle_slot_returns_first_idle():
    canned = [
        {"id": 0, "state": "processing", "n_ctx": 57344, "n_past": 1, "prompt": "..."},
        {"id": 1, "state": "idle", "n_ctx": 57344, "n_past": 0, "prompt": ""},
        {"id": 2, "state": "idle", "n_ctx": 57344, "n_past": 0, "prompt": ""},
    ]
    fake = _mock_client(get_json=canned)
    with patch("llama_slots._client", return_value=fake):
        assert find_idle_slot() == 1


def test_find_idle_slot_returns_none_if_all_busy():
    canned = [
        {"id": 0, "state": "processing", "n_ctx": 57344, "n_past": 1, "prompt": "x"},
        {"id": 1, "state": "processing", "n_ctx": 57344, "n_past": 1, "prompt": "y"},
    ]
    fake = _mock_client(get_json=canned)
    with patch("llama_slots._client", return_value=fake):
        assert find_idle_slot() is None


# ── save_slot ──────────────────────────────────────────────────────────────

def test_save_slot_posts_correct_payload():
    fake = _mock_client(post_json={"filename": "a.bin", "n_saved": 1234})
    with patch("llama_slots._client", return_value=fake):
        out = save_slot(0, "a.bin")
    assert out["n_saved"] == 1234
    fake.post.assert_called_once()
    args, kwargs = fake.post.call_args
    assert "/slots/0?action=save" in args[0]
    assert kwargs["json"] == {"filename": "a.bin"}


def test_save_slot_rejects_filename_with_slash():
    with pytest.raises(LlamaSlotsError):
        save_slot(0, "subdir/x.bin")


def test_save_slot_rejects_filename_with_traversal():
    with pytest.raises(LlamaSlotsError):
        save_slot(0, "..hack")


def test_save_slot_raises_on_server_error():
    fake = _mock_client(post_status=500, post_json={"error": "no slot save path"})
    with patch("llama_slots._client", return_value=fake):
        with pytest.raises(LlamaSlotsError):
            save_slot(0, "valid.bin")


# ── restore_slot ───────────────────────────────────────────────────────────

def test_restore_slot_posts_correct_payload():
    fake = _mock_client(post_json={"filename": "x.bin", "n_restored": 5678})
    with patch("llama_slots._client", return_value=fake):
        out = restore_slot(2, "x.bin")
    assert out["n_restored"] == 5678
    args, kwargs = fake.post.call_args
    assert "/slots/2?action=restore" in args[0]
    assert kwargs["json"] == {"filename": "x.bin"}


# ── erase_slot ─────────────────────────────────────────────────────────────

def test_erase_slot_posts_action_erase():
    fake = _mock_client(post_json={"id": 1, "erased": True})
    with patch("llama_slots._client", return_value=fake):
        out = erase_slot(1)
    assert out["erased"] is True
    args, _ = fake.post.call_args
    assert "/slots/1?action=erase" in args[0]


# ── park_slot (save then erase) ────────────────────────────────────────────

def test_park_slot_calls_save_then_erase():
    fake = _mock_client(post_json={"ok": True})
    with patch("llama_slots._client", return_value=fake):
        park_slot(0, "parked.bin")
    # Two POST calls: first save, then erase.
    assert fake.post.call_count == 2
    save_call = fake.post.call_args_list[0]
    erase_call = fake.post.call_args_list[1]
    assert "action=save" in save_call.args[0]
    assert "action=erase" in erase_call.args[0]


def test_park_slot_does_not_erase_if_save_fails():
    """If save_slot raises, park_slot should propagate WITHOUT erasing.
    Erasing on save-failure would lose context."""
    # First POST (save) returns 500; if park_slot were buggy and called
    # erase anyway, we'd see a second POST with action=erase.
    fake = _mock_client(post_status=500, post_json={"error": "no path"})
    with patch("llama_slots._client", return_value=fake):
        with pytest.raises(LlamaSlotsError):
            park_slot(0, "x.bin")
    # Only one POST should have happened (the save attempt).
    assert fake.post.call_count == 1
