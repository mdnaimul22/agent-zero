from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import Agent, LoopData
from helpers import extension, history
from plugins._context_window.api.context_window import ContextWindow
from plugins._context_window.helpers import usage


ROOT = Path(__file__).resolve().parents[3]


class _Log:
    def set_progress(self, _message: str) -> None:
        pass


@pytest.mark.asyncio
async def test_usage_follows_prompt_sources_and_reconciles_to_total(monkeypatch):
    agent = object.__new__(Agent)
    loop_data = LoopData()
    agent.loop_data = loop_data
    agent.context = SimpleNamespace(log=_Log())
    agent.history = history.History(agent)
    agent.data = {}
    agent.history.add_message(False, "User asks a question.")
    agent.history.add_message(True, "Assistant answers.")
    agent.history.add_message(
        False,
        {
            "tool_name": "skills_tool",
            "tool_result": "Skill instructions without a special heading.",
            "skill_instructions": {
                "name": "test-skill",
                "content_included": True,
            },
        },
    )

    system_parts = {
        "system_prompt": "Main instructions without a special heading.",
        "system_tools": "Tool definitions without a special heading.",
        "mcp_tools": "Remote definitions without a special heading.",
        "skills": "Available skill names without a special heading.",
    }

    async def get_system_prompt(_loop_data):
        for key in ("system_tools", "mcp_tools", "skills"):
            usage.record_prompt(agent, key, system_parts[key])
        return list(system_parts.values())

    def read_prompt(prompt_file: str, **kwargs) -> str:
        if prompt_file == "agent.context.protocol.md":
            return "[PROTOCOL]\n" + kwargs["protocol"]
        if prompt_file == "agent.context.extras.md":
            return "[EXTRAS]\n" + kwargs["extras"]
        raise AssertionError(f"Unexpected prompt: {prompt_file}")

    async def call_extensions(extension_point: str, agent=None, **kwargs):
        if extension_point == "message_loop_prompts_after":
            current = kwargs["loop_data"]
            current.protocol_persistent["project"] = "Project instructions."
            current.extras_temporary["time"] = "Current time."
            usage.capture_context(agent, current)

    agent.get_system_prompt = get_system_prompt
    agent.read_prompt = read_prompt
    monkeypatch.setattr(extension, "call_extensions_async", call_extensions)
    monkeypatch.setattr(history.History, "_get_max_embeds", lambda self: 0)

    usage.reset(agent)
    await Agent.prepare_prompt.__wrapped__(agent, loop_data)
    usage.finalize(agent)

    window = agent.get_data(Agent.DATA_NAME_CTX_WINDOW)
    breakdown = window["usage"]
    assert tuple(breakdown) == usage.USAGE_KEYS
    assert sum(breakdown.values()) == window["tokens"]
    assert all(breakdown[key] > 0 for key in usage.USAGE_KEYS)
    assert usage.PARTS_KEY not in loop_data.params_temporary


@pytest.mark.asyncio
async def test_api_returns_only_counts_and_effective_limit(monkeypatch):
    agent = SimpleNamespace(
        DATA_NAME_CTX_WINDOW="ctx_window",
        get_data=lambda _key: {
            "text": "private prompt",
            "tokens": 120,
            "usage": {"messages": 42},
        },
    )
    handler = object.__new__(ContextWindow)
    handler.use_context = lambda _context_id: SimpleNamespace(
        streaming_agent=None,
        agent0=agent,
    )
    monkeypatch.setattr(
        "plugins._context_window.api.context_window.get_chat_model_config",
        lambda _agent: {"ctx_length": 128_000},
    )

    result = await handler.process({"context": "ctx-1"}, SimpleNamespace())

    assert result == {
        "tokens": 120,
        "context_window": 128_000,
        "usage": {
            "messages": 42,
            "system_tools": 0,
            "skills": 0,
            "mcp_tools": 0,
            "system_prompt": 0,
            "extras": 0,
        },
    }
    assert "text" not in result


def test_webui_and_accounting_are_plugin_owned():
    model_switcher = (
        ROOT
        / "plugins/_model_config/extensions/webui/chat-input-progress-start/model-switcher.html"
    ).read_text(encoding="utf-8")
    model_store = (ROOT / "plugins/_model_config/webui/switcher-mixin.js").read_text(
        encoding="utf-8"
    )
    component = (
        ROOT
        / "plugins/_context_window/extensions/webui/model-context-strip-end/context-window.html"
    ).read_text(encoding="utf-8")
    context_store = (
        ROOT / "plugins/_context_window/webui/context-window-store.js"
    ).read_text(encoding="utf-8")
    helper = (ROOT / "plugins/_context_window/helpers/usage.py").read_text(
        encoding="utf-8"
    )

    assert 'id="model-context-strip-end"' in model_switcher
    assert "contextWindowUsage" not in model_switcher
    assert "contextUsage" not in model_store
    assert "Context window" in component
    assert "position: static" in component
    assert "width: min(19rem, calc(100vw - 2rem))" in component
    assert "right: 1.25rem" in component
    assert "width: min(17rem, calc(100vw - 3rem))" in component
    assert 'label: "Free space"' in context_store
    assert "Breakdown available after the next message." in component
    assert "startswith(" not in helper
    assert "rpartition(" not in helper


def test_source_prompt_extensions_are_registered():
    expected = {
        "_functions/agent/Agent/prepare_prompt/start": "ResetContextUsage",
        "_functions/agent/Agent/prepare_prompt/end": "StoreContextUsage",
        "message_loop_prompts_after": "CaptureContextUsage",
    }
    for point, class_name in expected.items():
        classes = extension._get_extension_classes(point)  # type: ignore[attr-defined]
        assert any(cls.__name__ == class_name for cls in classes)

    system_prompt_classes = {
        cls.__name__: cls
        for cls in extension._get_extension_classes("system_prompt")  # type: ignore[attr-defined]
    }
    for owner, recorder in {
        "ToolsPrompt": "RecordSystemToolsUsage",
        "MCPToolsPrompt": "RecordMcpToolsUsage",
        "SkillsPrompt": "RecordSkillsUsage",
    }.items():
        builder = system_prompt_classes[owner].execute.__globals__["build_prompt"]
        module = builder.__wrapped__.__module__.replace(".", "/")
        point = f"_functions/{module}/build_prompt/end"
        classes = extension._get_extension_classes(point)  # type: ignore[attr-defined]
        assert any(cls.__name__ == recorder for cls in classes)


def test_prompt_fragment_cache_is_bounded_and_content_addressed(monkeypatch):
    calls = []
    agent = SimpleNamespace(data={}, loop_data=LoopData())
    monkeypatch.setattr(
        usage.tokens,
        "approximate_prompt_tokens",
        lambda text: calls.append(text) or len(text),
    )

    usage.reset(agent)
    usage.record_prompt(agent, "system_tools", "same prompt")
    usage.record_prompt(agent, "system_tools", "same prompt")
    usage.record_prompt(agent, "system_tools", "changed prompt")

    assert calls == ["same prompt", "changed prompt"]
    cache = agent.data[usage.CACHE_KEY]
    assert len(cache) == 1
    assert cache["prompt:system_tools"][1] == len("changed prompt")
    assert all(len(value[0]) == 64 for value in cache.values())


def test_history_ledger_changes_without_invalidating_fragment_cache(monkeypatch):
    calls = []
    history_tokens = 1_000
    agent = SimpleNamespace(
        data={},
        loop_data=LoopData(),
        history=SimpleNamespace(get_tokens=lambda: history_tokens),
        _build_context_message=lambda *args, **kwargs: [],
    )
    skill_message = {
        "ai": False,
        "content": {
            "tool_name": "skills_tool",
            "tool_result": "Loaded skill body.",
            "skill_instructions": {
                "name": "test-skill",
                "content_included": True,
            },
        },
    }
    loop_data = SimpleNamespace(
        history_output=[skill_message],
        protocol_persistent={},
        protocol_temporary={},
        extras_persistent={},
        extras_temporary={},
    )
    monkeypatch.setattr(
        usage.tokens,
        "approximate_prompt_tokens",
        lambda text: calls.append(text) or len(text),
    )

    usage.reset(agent)
    usage.record_prompt(agent, "system_tools", "stable tools")
    usage.capture_context(agent, loop_data)
    first = dict(agent.loop_data.params_temporary[usage.PARTS_KEY])

    history_tokens = 400
    agent.loop_data.params_temporary = {}
    usage.reset(agent)
    usage.record_prompt(agent, "system_tools", "stable tools")
    usage.capture_context(agent, loop_data)
    second = agent.loop_data.params_temporary[usage.PARTS_KEY]

    assert first["messages"] == 1_000 - first["skills"]
    assert second["messages"] == 400 - second["skills"]
    assert calls.count("stable tools") == 1
    assert len(agent.data[usage.CACHE_KEY]) == 3
