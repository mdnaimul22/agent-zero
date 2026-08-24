from types import SimpleNamespace

from plugins._context_doctor.helpers.context_doctor import repair_and_minify, update_log_item
from plugins._context_doctor.extensions.python.message_loop_result._10_context_doctor import (
    ContextDoctor,
)


def test_repairs_and_minifies_tool_call():
    response = '{"tool_name":"response","tool_args":{"text":"ok",},}'

    assert repair_and_minify(response, suppress_xml=True) == (
        '{"tool_name":"response","tool_args":{"text":"ok"}}'
    )


def test_ignores_non_tool_json():
    assert repair_and_minify('{"message":"ok"}', suppress_xml=True) is None


def test_suppresses_xml_when_enabled():
    assert repair_and_minify('<tool>response</tool>', suppress_xml=True) == "{}"
    assert repair_and_minify('<tool>response</tool>', suppress_xml=False) is None


def test_extension_replaces_completed_result(monkeypatch):
    monkeypatch.setattr(
        "plugins._context_doctor.extensions.python.message_loop_result._10_context_doctor.get_plugin_config",
        lambda *args, **kwargs: {"suppress_xml": True},
    )
    llm_result = SimpleNamespace(
        response='{"tool_name":"response","tool_args":{"text":"ok",},}'
    )
    agent = SimpleNamespace(loop_data=SimpleNamespace(params_temporary={}))

    ContextDoctor(agent).execute({"llm_result": llm_result})

    assert llm_result.response == '{"tool_name":"response","tool_args":{"text":"ok"}}'


def test_updates_log_with_repaired_tool_call():
    log_item = SimpleNamespace(update=lambda **kwargs: setattr(log_item, "data", kwargs))

    update_log_item(
        SimpleNamespace(agent_name="A0"),
        log_item,
        '{"headline":"Done","tool_name":"response","tool_args":{"text":"ok"}}',
    )

    assert log_item.data["content"] == (
        '{"headline":"Done","tool_name":"response","tool_args":{"text":"ok"}}'
    )
    assert log_item.data["heading"] == "A0: Done"


def test_extension_updates_log_only_when_enabled(monkeypatch):
    llm_result = SimpleNamespace(
        response='{"tool_name":"response","tool_args":{"text":"ok",},}'
    )
    log_item = SimpleNamespace(update=lambda **kwargs: setattr(log_item, "data", kwargs))
    agent = SimpleNamespace(
        agent_name="A0",
        loop_data=SimpleNamespace(params_temporary={"log_item_generating": log_item}),
    )

    monkeypatch.setattr(
        "plugins._context_doctor.extensions.python.message_loop_result._10_context_doctor.get_plugin_config",
        lambda *args, **kwargs: {"suppress_xml": True, "update_log": False},
    )
    ContextDoctor(agent).execute({"llm_result": llm_result})
    assert not hasattr(log_item, "data")

    monkeypatch.setattr(
        "plugins._context_doctor.extensions.python.message_loop_result._10_context_doctor.get_plugin_config",
        lambda *args, **kwargs: {"suppress_xml": True, "update_log": True},
    )
    ContextDoctor(agent).execute({"llm_result": llm_result})
    assert log_item.data["content"] == llm_result.response
