"""
Citizen SMS-notification subscribers, persisted to a JSON file so they survive
a server restart (no database dependency needed for this scale).
Real phone numbers a citizen submits through the citizen UI are stored here,
keyed by zone id, and used by the auto-notify loop in main.py to send real
Twilio SMS when a zone's alert level rises to YELLOW/ORANGE/RED.
"""
import json
import os
import threading

SUBSCRIBERS_PATH = os.path.join(os.path.dirname(__file__), "data", "subscribers.json")
_lock = threading.Lock()


def _load() -> dict:
    if not os.path.exists(SUBSCRIBERS_PATH):
        return {}
    try:
        with open(SUBSCRIBERS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict):
    os.makedirs(os.path.dirname(SUBSCRIBERS_PATH), exist_ok=True)
    with open(SUBSCRIBERS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def subscribe(zone_id: str, phone_number: str) -> dict:
    with _lock:
        data = _load()
        numbers = data.setdefault(zone_id, [])
        if phone_number not in numbers:
            numbers.append(phone_number)
        _save(data)
        return {"ok": True, "zone_id": zone_id, "subscriber_count": len(numbers)}


def unsubscribe(zone_id: str, phone_number: str) -> dict:
    with _lock:
        data = _load()
        numbers = data.get(zone_id, [])
        if phone_number in numbers:
            numbers.remove(phone_number)
        _save(data)
        return {"ok": True, "zone_id": zone_id, "subscriber_count": len(numbers)}


def get_subscribers(zone_id: str) -> list:
    return _load().get(zone_id, [])


def get_all_counts() -> dict:
    data = _load()
    return {zone_id: len(numbers) for zone_id, numbers in data.items()}
