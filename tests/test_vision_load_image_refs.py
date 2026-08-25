import asyncio
import types
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers import images


class _TestResponse(SimpleNamespace):
    def __init__(self, message="", break_loop=False, additional=None, **kwargs):
        super().__init__(
            message=message,
            break_loop=break_loop,
            additional=additional,
            **kwargs,
        )


class _TestTool:
    def __init__(
        self,
        agent=None,
        name="",
        method=None,
        args=None,
        message="",
        loop_data=None,
        **kwargs,
    ):
        self.agent = agent
        self.name = name
        self.method = method
        self.args = args or {}
        self.message = message
        self.loop_data = loop_data


def _install_tool_stub(monkeypatch):
    tool_stub = types.ModuleType("helpers.tool")
    tool_stub.Response = _TestResponse
    tool_stub.Tool = _TestTool
    history_stub = types.ModuleType("helpers.history")

    class _RawMessage(dict):
        def __init__(self, raw_content, preview):
            super().__init__(raw_content=raw_content, preview=preview)

    history_stub.RawMessage = _RawMessage
    monkeypatch.setitem(sys.modules, "helpers.tool", tool_stub)
    monkeypatch.setitem(sys.modules, "helpers.history", history_stub)
    monkeypatch.delitem(sys.modules, "tools.vision_load", raising=False)


def test_prepare_content_keeps_missing_local_image_refs_strict():
    missing_path = "/tmp/a0-missing-desktop-screenshot.png"

    with pytest.raises(FileNotFoundError):
        images.prepare_content(
            [{"type": "image_url", "image_url": {"url": missing_path}}]
        )


@pytest.mark.anyio
async def test_vision_load_materializes_local_image_to_chat_artifact(monkeypatch, tmp_path):
    _install_tool_stub(monkeypatch)
    import tools.vision_load as vision_load_module

    def fake_get_abs_path(*parts):
        return str(tmp_path.joinpath(*parts))

    def fake_normalize_a0_path(path):
        return "/a0/" + str(Path(path).relative_to(tmp_path)).replace("\\", "/")

    monkeypatch.setattr(vision_load_module.chat_media.files, "get_abs_path", fake_get_abs_path)
    monkeypatch.setattr(vision_load_module.chat_media.files, "normalize_a0_path", fake_normalize_a0_path)
    monkeypatch.setattr(vision_load_module, "get_chat_model_config", lambda _agent: {"vision": True, "max_embeds": 10})
    monkeypatch.setattr(vision_load_module, "get_vision_model_config", lambda _agent: {})
    monkeypatch.setattr(vision_load_module, "use_vision_sidecar", lambda _agent: False)

    async def direct_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        vision_load_module.runtime,
        "call_development_function",
        direct_call,
    )

    image_path = tmp_path / "sample-image.png"
    image_path.write_bytes(b"png-data")

    tool_results = []
    messages = []
    updates = []
    agent = SimpleNamespace(
        context=SimpleNamespace(id="ctx-vision"),
        agent_name="Agent 0",
        hist_add_tool_result=lambda *args, **kwargs: tool_results.append((args, kwargs)),
        hist_add_message=lambda *args, **kwargs: messages.append((args, kwargs)),
    )
    tool = vision_load_module.VisionLoad(
        agent=agent,
        name="vision_load",
        method=None,
        args={"paths": [str(image_path)]},
        message="",
        loop_data=None,
    )
    tool.log = SimpleNamespace(id="vision-log", update=lambda **kwargs: updates.append(kwargs))

    response = await tool.execute(paths=[str(image_path)])
    image_path.unlink()
    await tool.after_execution(response)

    raw_message = messages[0][1]["content"]
    stored_ref = raw_message["raw_content"][0]["image_url"]["url"]
    assert stored_ref.startswith("/a0/usr/chats/ctx-vision/images/vision-load/sample-image-")
    stored_path = tmp_path / stored_ref.removeprefix("/a0/")
    assert stored_path.read_bytes() == b"png-data"
    assert updates[-1]["result"] == "1 images loaded, 0 skipped"


def test_vision_sidecar_route_matrix_prefers_main_native_vision(monkeypatch):
    from plugins._model_config.helpers import model_config

    cases = [
        ({"vision": False}, {}, False),
        ({"vision": True}, {"provider": "p", "name": "v"}, False),
        ({"vision": False}, {"provider": "p", "name": "v"}, True),
        (
            {"vision": True},
            {"provider": "p", "name": "v", "override_main": True},
            True,
        ),
    ]
    for chat, vision, expected in cases:
        monkeypatch.setattr(
            model_config,
            "get_effective_config",
            lambda _agent=None, chat=chat, vision=vision: {
                "chat_model": chat,
                "vision_model": vision,
            },
        )
        assert model_config.use_vision_sidecar() is expected


@pytest.mark.anyio
async def test_vision_sidecar_sends_multiple_images_once_and_keeps_history_text_only(
    monkeypatch,
    tmp_path,
):
    _install_tool_stub(monkeypatch)
    import tools.vision_load as vision_load_module

    async def direct_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    calls = []

    class FakeVisionModel:
        async def unified_call(self, **kwargs):
            calls.append(kwargs)
            return "The second screenshot fixes the red login error.", ""

    monkeypatch.setattr(vision_load_module.runtime, "call_development_function", direct_call)
    monkeypatch.setattr(vision_load_module, "build_vision_model", lambda _agent: FakeVisionModel())
    monkeypatch.setattr(
        vision_load_module,
        "get_chat_model_config",
        lambda _agent: {"vision": True, "max_embeds": 1},
    )
    monkeypatch.setattr(
        vision_load_module,
        "get_vision_model_config",
        lambda _agent: {"provider": "test", "name": "vision", "max_embeds": 5},
    )
    monkeypatch.setattr(vision_load_module, "use_vision_sidecar", lambda _agent: True)

    image_paths = [tmp_path / "before.png", tmp_path / "after.png"]
    for path in image_paths:
        path.write_bytes(b"png-data")

    tool_results = []
    raw_messages = []
    agent = SimpleNamespace(
        context=SimpleNamespace(id=""),
        agent_name="Agent 0",
        hist_add_tool_result=lambda *args, **kwargs: tool_results.append((args, kwargs)),
        hist_add_message=lambda *args, **kwargs: raw_messages.append((args, kwargs)),
    )
    tool = vision_load_module.VisionLoad(
        agent=agent,
        name="vision_load",
        method=None,
        args={"paths": [str(path) for path in image_paths]},
        message="",
        loop_data=None,
    )
    tool.log = SimpleNamespace(id="vision-log", update=lambda **kwargs: None)

    response = await tool.execute(
        paths=[str(path) for path in image_paths],
        query="Compare the login errors.",
    )
    response.additional = {"_responses_output_item": {"output": response.message}}
    await tool.after_execution(response)

    assert len(calls) == 1
    content = calls[0]["messages"][1].content
    assert content[0] == {"type": "text", "text": "Compare the login errors."}
    assert [item["type"] for item in content].count("image_url") == 2
    assert "max_tokens" not in calls[0]
    assert "explicit_caching" not in calls[0]
    assert "fixes the red login error" in response.message
    assert response.message != "dummy"
    assert raw_messages == []
    assert tool.loaded_paths == [str(path) for path in image_paths]
    assert tool_results[0][1]["_responses_output_item"]["output"] == response.message


@pytest.mark.anyio
async def test_parallel_worker_consumes_parent_ephemeral_image(monkeypatch, tmp_path):
    _install_tool_stub(monkeypatch)
    import tools.vision_load as vision_load_module

    def fake_get_abs_path(*parts):
        return str(tmp_path.joinpath(*parts))

    def fake_normalize_a0_path(path):
        return "/a0/" + str(Path(path).relative_to(tmp_path)).replace("\\", "/")

    monkeypatch.setattr(vision_load_module.chat_media.files, "get_abs_path", fake_get_abs_path)
    monkeypatch.setattr(vision_load_module.chat_media.files, "normalize_a0_path", fake_normalize_a0_path)
    parent_id = "parent-vision"
    parent_agent = SimpleNamespace(
        context=SimpleNamespace(id=parent_id),
        agent_name="Parent Agent",
    )
    parent_context = SimpleNamespace(agent0=parent_agent)
    agent_stub = types.ModuleType("agent")
    agent_stub.AgentContext = SimpleNamespace(
        get=lambda context_id: parent_context if context_id == parent_id else None
    )
    monkeypatch.setitem(sys.modules, "agent", agent_stub)
    config_owners = []

    def get_chat_config(owner):
        config_owners.append(owner)
        return {"vision": True, "max_embeds": 10}

    monkeypatch.setattr(
        vision_load_module,
        "get_chat_model_config",
        get_chat_config,
    )
    monkeypatch.setattr(vision_load_module, "get_vision_model_config", lambda _agent: {})
    monkeypatch.setattr(vision_load_module, "use_vision_sidecar", lambda _agent: False)

    ref = vision_load_module.ephemeral_images.put_image_bytes(
        context_id=parent_id,
        mime="image/png",
        payload=b"png-data",
        name="shot.png",
    )
    context = SimpleNamespace(
        id="parallel-worker",
        get_data=lambda key: parent_id
        if key == vision_load_module.PARALLEL_WORKER_PARENT_CONTEXT_KEY
        else None,
    )
    agent = SimpleNamespace(context=context, agent_name="Agent 0")
    tool = vision_load_module.VisionLoad(
        agent=agent,
        name="vision_load",
        method=None,
        args={"paths": [ref]},
        message="",
        loop_data=None,
    )

    await tool.execute(paths=[ref])

    assert tool._config_owner is parent_agent
    assert config_owners and all(owner is parent_agent for owner in config_owners)
    assert tool._context_id() == parent_id
    assert tool.loaded_paths == ["shot.png"]
    assert vision_load_module.ephemeral_images.get_image(ref, context_id=parent_id) is None
    stored_ref = tool.images_dict["shot.png"]
    assert stored_ref.startswith("/a0/usr/chats/parent-vision/images/vision-load/shot-")


@pytest.mark.anyio
async def test_independent_vision_sidecar_calls_can_run_concurrently(monkeypatch, tmp_path):
    _install_tool_stub(monkeypatch)
    import tools.vision_load as vision_load_module

    active = 0
    max_active = 0
    call_count = 0

    class FakeVisionModel:
        async def unified_call(self, **kwargs):
            nonlocal active, max_active, call_count
            active += 1
            call_count += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "done", ""

    async def direct_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(vision_load_module.runtime, "call_development_function", direct_call)
    monkeypatch.setattr(vision_load_module, "build_vision_model", lambda _agent: FakeVisionModel())
    monkeypatch.setattr(vision_load_module, "get_chat_model_config", lambda _agent: {"vision": False})
    monkeypatch.setattr(
        vision_load_module,
        "get_vision_model_config",
        lambda _agent: {"provider": "test", "name": "vision", "max_embeds": 10},
    )
    monkeypatch.setattr(vision_load_module, "use_vision_sidecar", lambda _agent: True)

    image_paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in image_paths:
        path.write_bytes(b"png-data")

    def make_tool(index):
        agent = SimpleNamespace(context=SimpleNamespace(id=""), agent_name=f"Agent {index}")
        return vision_load_module.VisionLoad(
            agent=agent,
            name="vision_load",
            method=None,
            args={"paths": [str(path) for path in image_paths]},
            message="",
            loop_data=None,
        )

    responses = await asyncio.gather(
        *(
            make_tool(index).execute(
                paths=[str(path) for path in image_paths],
                query=f"inspection {index}",
            )
            for index in range(4)
        )
    )

    assert call_count == 4
    assert max_active == 4
    assert all("done" in response.message for response in responses)
