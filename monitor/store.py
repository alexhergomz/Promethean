"""
monitor/store.py — Persistent subscription storage.

Subscriptions are stored in ~/.promethean/monitor_subscriptions.json.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

STORE_PATH = Path.home() / ".promethean" / "monitor_subscriptions.json"


def _load() -> dict:
    if not STORE_PATH.exists():
        return {"subscriptions": []}
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"subscriptions": []}


def _save(data: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_subscriptions() -> list[dict]:
    return _load().get("subscriptions", [])


def get_subscription(topic: str) -> dict | None:
    for s in list_subscriptions():
        if s["topic"] == topic:
            return s
    return None


def add_subscription(topic: str, schedule: str = "daily",
                     channels: list[str] | None = None) -> dict:
    """Add or update a subscription. Returns the subscription dict."""
    data = _load()
    subs = data.setdefault("subscriptions", [])

    # Update if exists
    for s in subs:
        if s["topic"] == topic:
            s["schedule"] = schedule
            if channels is not None:
                s["channels"] = channels
            _save(data)
            return s

    sub = {
        "id": uuid.uuid4().hex[:8],
        "topic": topic,
        "schedule": schedule,
        "channels": channels or [],
        "created_at": datetime.now().isoformat(),
        "last_run": None,
        "next_run": None,
        "last_report": None,
    }
    subs.append(sub)
    _save(data)
    return sub


def remove_subscription(topic: str) -> bool:
    data = _load()
    before = len(data.get("subscriptions", []))
    data["subscriptions"] = [s for s in data.get("subscriptions", [])
                              if s["topic"] != topic]
    if len(data["subscriptions"]) < before:
        _save(data)
        return True
    return False


def update_last_run(topic: str, report: str) -> None:
    data = _load()
    now = datetime.now().isoformat()
    for s in data.get("subscriptions", []):
        if s["topic"] == topic:
            s["last_run"] = now
            s["last_report"] = report[:500]  # preview
            break
    _save(data)
