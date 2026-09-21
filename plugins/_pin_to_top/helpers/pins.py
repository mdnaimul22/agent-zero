from __future__ import annotations

import os
import threading
import time
from typing import Literal

from helpers import kvp, projects

PinKind = Literal["chat", "task", "project"]

STORE_KEY = "plugin_pin_to_top"
KINDS: tuple[PinKind, ...] = ("chat", "task", "project")
_lock = threading.RLock()


def get_pins() -> dict[PinKind, dict[str, float]]:
    """Return normalized persisted pins grouped by sidebar list."""
    with _lock:
        return _normalize(kvp.get_persistent(STORE_KEY, {}))


def toggle_pin(kind: str, item_id: str) -> tuple[bool, float]:
    """Toggle a pin and return its new state and timestamp."""
    normalized_kind = _require_kind(kind)
    normalized_id = _require_item_id(item_id)
    if normalized_kind == "project":
        normalized_id = projects.validate_project_name(normalized_id)

    with _lock:
        pins = _normalize(kvp.get_persistent(STORE_KEY, {}))
        kind_pins = pins[normalized_kind]
        if normalized_id in kind_pins:
            del kind_pins[normalized_id]
            timestamp = 0.0
            pinned = False
        else:
            if normalized_kind == "project" and not os.path.isfile(
                projects.get_project_meta(normalized_id, projects.PROJECT_HEADER_FILE)
            ):
                raise ValueError("Project not found")
            timestamp = time.time()
            kind_pins[normalized_id] = timestamp
            pinned = True

        kvp.set_persistent(STORE_KEY, pins)
        return pinned, timestamp


def _normalize(value: object) -> dict[PinKind, dict[str, float]]:
    normalized: dict[PinKind, dict[str, float]] = {kind: {} for kind in KINDS}
    if not isinstance(value, dict):
        return normalized

    for kind in KINDS:
        entries = value.get(kind)
        if not isinstance(entries, dict):
            continue
        for item_id, timestamp in entries.items():
            try:
                clean_timestamp = float(timestamp)
            except (TypeError, ValueError):
                continue
            clean_id = str(item_id).strip()
            if clean_id and clean_timestamp > 0:
                normalized[kind][clean_id] = clean_timestamp

    return normalized


def _require_kind(kind: str) -> PinKind:
    if kind not in KINDS:
        raise ValueError("kind must be 'chat', 'task', or 'project'")
    return kind


def _require_item_id(item_id: str) -> str:
    normalized = item_id.strip()
    if not normalized:
        raise ValueError("item_id is required")
    if len(normalized) > 512:
        raise ValueError("item_id is too long")
    return normalized
