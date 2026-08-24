from types import SimpleNamespace

from extensions.python.message_loop_result._20_loop_control import LoopControl


class FakeAgent:
    def __init__(self, response: str, reasoning: str = "", last_response: str = ""):
        self.loop_data = SimpleNamespace(
            last_response=last_response,
            params_temporary={},
        )
        self.context = SimpleNamespace(log=SimpleNamespace(log=lambda **kwargs: None))
        self.response = response
        self.reasoning = reasoning
        self.warnings = []
        self.history = []

    def read_prompt(self, name):
        return {
            "fw.msg_empty_response.md": "empty",
            "fw.msg_repeat.md": "repeat",
        }[name]

    def hist_add_ai_response(self, response, **kwargs):
        self.history.append(response)
        return SimpleNamespace(id="assistant")

    def _remember_llm_result_state(self, *args):
        pass

    def hist_add_warning(self, message):
        self.warnings.append(message)
        return SimpleNamespace(id="warning")


def _run(agent):
    result_data = {
        "llm_result": SimpleNamespace(response=agent.response, reasoning=agent.reasoning)
    }
    LoopControl(agent).execute(result_data)
    return result_data


def test_empty_result_skips_default_processing():
    agent = FakeAgent("")

    assert _run(agent)["skip_default_processing"] is True
    assert agent.history == [""]
    assert agent.warnings == ["empty"]


def test_repeat_skips_default_processing():
    agent = FakeAgent('{"tool_name":"response"}', last_response='{"tool_name":"response"}')

    assert _run(agent)["skip_default_processing"] is True
    assert agent.warnings == ["repeat"]


def test_result_with_reasoning_uses_default_processing():
    agent = FakeAgent("", reasoning="thinking")

    assert "skip_default_processing" not in _run(agent)
    assert agent.history == []
