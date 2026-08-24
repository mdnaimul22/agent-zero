from types import SimpleNamespace

from extensions.python.message_loop_result._20_empty_response import EmptyResponse
from extensions.python.message_loop_result._30_repeat_response import RepeatResponse


class FakeAgent:
    def __init__(self, response: str, reasoning: str = "", last_response: str = ""):
        self.loop_data = SimpleNamespace(
            last_response=last_response,
            params_temporary={},
        )
        self.logs = []
        self.context = SimpleNamespace(
            log=SimpleNamespace(log=lambda **entry: self.logs.append(entry))
        )
        self.agent_name = "A0"
        self.response = response
        self.reasoning = reasoning
        self.warnings = []
        self.history = []

    def read_prompt(self, name):
        return {
            "fw.msg_empty_response.md": "empty",
            "fw.msg_repeat.md": "repeat",
            "fw.msg_repeat_response.md": "Repeated response detected. Retrying.",
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
    EmptyResponse(agent).execute(result_data)
    RepeatResponse(agent).execute(result_data)
    return result_data


def test_empty_result_skips_default_processing():
    agent = FakeAgent("")

    assert _run(agent)["skip_default_processing"] is True
    assert agent.history == [""]
    assert agent.warnings == []
    assert agent.logs == [{"type": "warning", "content": "A0: empty"}]


def test_repeat_skips_default_processing():
    agent = FakeAgent('{"tool_name":"response"}', last_response='{"tool_name":"response"}')

    assert _run(agent)["skip_default_processing"] is True
    assert agent.warnings == ["repeat"]
    assert agent.logs == [
        {
            "type": "warning",
            "content": "A0: Repeated response detected. Retrying.",
            "id": "warning",
        }
    ]


def test_repeat_ignores_reasoning():
    response = '{"tool_name":"response"}'
    agent = FakeAgent(response, reasoning="thinking", last_response=response)

    assert _run(agent)["skip_default_processing"] is True
    assert agent.warnings == ["repeat"]


def test_result_with_reasoning_uses_default_processing():
    agent = FakeAgent("", reasoning="thinking")

    assert "skip_default_processing" not in _run(agent)
    assert agent.history == []
