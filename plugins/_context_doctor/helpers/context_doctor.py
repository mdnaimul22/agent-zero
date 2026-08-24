"""Repair complete Agent Zero tool-call JSON into compact JSON."""

from __future__ import annotations

import json
from typing import Any


def _is_tool_call(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("tool_name"), str)
        and isinstance(value.get("tool_args"), dict)
        and (
            "thoughts" not in value
            or (
                isinstance(value["thoughts"], list)
                and all(isinstance(item, str) for item in value["thoughts"])
            )
        )
        and ("headline" not in value or isinstance(value["headline"], str))
    )


def repair_and_minify(response: str, *, suppress_xml: bool) -> str | None:
    """Return compact tool-call JSON, XML fallback, or ``None`` for other output."""
    if not response:
        return None

    try:
        from plugins._context_doctor.helpers.json_repair_patch import apply_patch
        from json_repair import repair_json

        apply_patch()
        repaired = repair_json(response, return_objects=True)
    except Exception:
        return "{}" if suppress_xml and "<" in response and ">" in response else None

    if isinstance(repaired, list):
        repaired = next((item for item in repaired if _is_tool_call(item)), None)
    if _is_tool_call(repaired):
        return json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))
    return "{}" if suppress_xml and "<" in response and ">" in response else None


def update_log_item(agent: Any, log_item: Any, response: str) -> None:
    """Replace final log details with repaired JSON and derived display fields."""
    try:
        parsed = json.loads(response)
        if not _is_tool_call(parsed):
            return
        heading = parsed.get("headline") or f"Using {parsed['tool_name']}"
        log_item.update(
            content=response,
            kvps=parsed,
            heading=f"{getattr(agent, 'agent_name', 'A0')}: {heading}",
        )
    except (AttributeError, TypeError, ValueError):
        pass
