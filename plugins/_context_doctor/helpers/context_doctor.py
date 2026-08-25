"""Repair completed model output into compact JSON."""

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


_A0_SALVAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "thoughts": {"type": "array", "items": {"type": "string"}},
        "headline": {"type": "string"},
        "tool_name": {"type": "string"},
        "tool_args": {"type": "object"},
    },
    "required": [],
}


def _a0_completeness_score(value: dict[str, Any]) -> int:
    return sum(field in value for field in ("thoughts", "headline", "tool_name", "tool_args")) + bool(
        value.get("tool_args")
    )


def transform_response(response: str, *, suppress_xml: bool) -> str:
    """Return compact repaired tool JSON or a compact raw-text fallback."""
    if response:
        try:
            from plugins._context_doctor.helpers.json_repair_patch import apply_patch
            from json_repair import repair_json

            apply_patch()
            try:
                salvage_repaired = repair_json(
                    response,
                    return_objects=True,
                    schema=_A0_SALVAGE_SCHEMA,
                    schema_repair_mode="salvage",
                )
            except Exception:
                salvage_repaired = repair_json(response, return_objects=True)

            candidates = (
                salvage_repaired
                if isinstance(salvage_repaired, list)
                else [salvage_repaired]
            )
            try:
                no_schema = repair_json(response, return_objects=True)
            except Exception:
                no_schema = None
            if isinstance(no_schema, list):
                valid = [item for item in no_schema if _is_tool_call(item)]
                if len(valid) > 1 or (
                    len(valid) == 1 and not _is_tool_call(salvage_repaired)
                ):
                    candidates = no_schema
            repaired = max(
                (item for item in candidates if _is_tool_call(item)),
                key=_a0_completeness_score,
                default=None,
            )
        except Exception:
            repaired = None
            salvage_repaired = None

        if _is_tool_call(repaired):
            return json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))

        for candidate in (repaired, salvage_repaired):
            if isinstance(candidate, dict) and (
                "thoughts" in candidate or "headline" in candidate
            ):
                return json.dumps(
                    candidate, ensure_ascii=False, separators=(",", ":")
                )

    if suppress_xml and "<" in response and ">" in response:
        return "{}"
    return json.dumps({"thoughts": [response]}, ensure_ascii=False, separators=(",", ":"))


def update_log_item(
    agent: Any,
    log_item: Any,
    response: str,
    *,
    update_log: bool,
    raw_response: str,
) -> None:
    """Refresh log fields from transformed JSON; optionally replace raw details."""
    try:
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            return
        heading = parsed.get("headline")
        if not isinstance(heading, str) or not heading:
            tool_name = parsed.get("tool_name")
            heading = f"Using {tool_name}" if isinstance(tool_name, str) else ""
        kwargs: dict[str, Any] = {"kvps": parsed}
        if heading:
            kwargs["heading"] = f"{getattr(agent, 'agent_name', 'A0')}: {heading}"
        kwargs["content"] = response if update_log else raw_response
        log_item.update(**kwargs)
    except (AttributeError, TypeError, ValueError):
        pass
