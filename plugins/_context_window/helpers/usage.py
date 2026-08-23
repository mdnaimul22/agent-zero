import hashlib
from typing import Any

from helpers import files, history, skills, tokens


PARTS_KEY = "context_window_usage"
CACHE_KEY = "_context_window_usage_cache"
USAGE_KEYS = (
    "messages",
    "system_tools",
    "skills",
    "mcp_tools",
    "system_prompt",
    "extras",
)
MEASURED_KEYS = tuple(key for key in USAGE_KEYS if key != "system_prompt")


def reset(agent: Any) -> None:
    params = _temporary_params(agent)
    if params is not None:
        params[PARTS_KEY] = {}


def discard(agent: Any) -> None:
    params = _temporary_params(agent)
    if params is not None:
        params.pop(PARTS_KEY, None)


def record_prompt(agent: Any, key: str, prompt: Any) -> None:
    parts = _parts(agent)
    if parts is None or key not in MEASURED_KEYS:
        return
    text = files.remove_code_fences(str(prompt or ""), language="json")
    parts[key] = _cached_tokens(agent, f"prompt:{key}", text)


def capture_context(agent: Any, loop_data: Any) -> None:
    parts = _parts(agent)
    if parts is None or loop_data is None:
        return

    output = list(getattr(loop_data, "history_output", None) or [])
    skill_output = [message for message in output if skills.skill_instruction_name(message)]
    skill_tokens = _output_tokens(agent, "history_skills", skill_output)
    parts["messages"] = max(_history_tokens(agent, output) - skill_tokens, 0)
    parts["skills"] = parts.get("skills", 0) + skill_tokens

    protocol_values = {
        **getattr(loop_data, "protocol_persistent", {}),
        **getattr(loop_data, "protocol_temporary", {}),
    }
    extras_values = {
        **getattr(loop_data, "extras_persistent", {}),
        **getattr(loop_data, "extras_temporary", {}),
    }
    protocol = agent._build_context_message(
        "agent.context.protocol.md",
        "protocol",
        protocol_values,
        include_empty=False,
    )
    extras = agent._build_context_message(
        "agent.context.extras.md",
        "extras",
        extras_values,
        include_empty=True,
    )
    parts["extras"] = _output_tokens(agent, "extras", protocol + extras)


def finalize(agent: Any) -> None:
    params = _temporary_params(agent)
    parts = params.pop(PARTS_KEY, None) if params is not None else None
    window = agent.get_data(agent.DATA_NAME_CTX_WINDOW) if agent else None
    if not isinstance(parts, dict) or not isinstance(window, dict):
        return

    total = _non_negative_int(window.get("tokens"))
    usage = {key: _non_negative_int(parts.get(key)) for key in MEASURED_KEYS}
    measured_total = sum(usage.values())
    if measured_total > total and measured_total:
        usage = _scale_to_total(usage, total, measured_total)
        measured_total = total
    usage["system_prompt"] = total - measured_total
    usage = {key: usage.get(key, 0) for key in USAGE_KEYS}

    updated = dict(window)
    updated["usage"] = usage
    agent.set_data(agent.DATA_NAME_CTX_WINDOW, updated)


def usage_snapshot(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {key: _non_negative_int(value.get(key)) for key in USAGE_KEYS}


def _parts(agent: Any) -> dict[str, int] | None:
    params = _temporary_params(agent)
    value = params.get(PARTS_KEY) if params is not None else None
    return value if isinstance(value, dict) else None


def _temporary_params(agent: Any) -> dict[str, Any] | None:
    loop_data = getattr(agent, "loop_data", None)
    params = getattr(loop_data, "params_temporary", None)
    return params if isinstance(params, dict) else None


def _history_tokens(agent: Any, output: list[history.OutputMessage]) -> int:
    get_tokens = getattr(getattr(agent, "history", None), "get_tokens", None)
    if callable(get_tokens):
        return _non_negative_int(get_tokens())
    return _count_output_tokens(output)


def _output_tokens(
    agent: Any, cache_key: str, output: list[history.OutputMessage]
) -> int:
    text = history.output_text(output, ai_label="assistant", human_label="user")
    return _cached_tokens(agent, cache_key, text)


def _count_output_tokens(output: list[history.OutputMessage]) -> int:
    text = history.output_text(output, ai_label="assistant", human_label="user")
    return tokens.approximate_prompt_tokens(text)


def _cached_tokens(agent: Any, key: str, text: str) -> int:
    cache = _cache(agent)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = cache.get(key) if cache is not None else None
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == digest:
        return _non_negative_int(cached[1])

    count = tokens.approximate_prompt_tokens(text)
    if cache is not None:
        cache[key] = (digest, count)
    return count


def _cache(agent: Any) -> dict[str, tuple[str, int]] | None:
    data = getattr(agent, "data", None)
    if not isinstance(data, dict):
        return None
    cache = data.get(CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        data[CACHE_KEY] = cache
    return cache


def _non_negative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _scale_to_total(values: dict[str, int], total: int, current: int) -> dict[str, int]:
    scaled = {key: value * total // current for key, value in values.items()}
    remainder = total - sum(scaled.values())
    order = sorted(values, key=lambda key: values[key] * total % current, reverse=True)
    for key in order[:remainder]:
        scaled[key] += 1
    return scaled
