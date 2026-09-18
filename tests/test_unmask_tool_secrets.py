from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = (
    PROJECT_ROOT
    / "extensions"
    / "python"
    / "tool_execute_before"
    / "_10_unmask_secrets.py"
)
SPEC = importlib.util.spec_from_file_location("test_unmask_tool_secrets", MODULE_PATH)
assert SPEC and SPEC.loader
unmask_tool_secrets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(unmask_tool_secrets)


class _FakeSecretsManager:
    def replace_placeholders(self, text: str) -> str:
        return text.replace("§§secret(TOKEN)", "resolved-secret")


class _FakeAgent:
    context = object()


@pytest.mark.asyncio
async def test_unmask_tool_secrets_replaces_nested_parallel_arguments(monkeypatch):
    monkeypatch.setattr(
        unmask_tool_secrets,
        "get_secrets_manager",
        lambda _context: _FakeSecretsManager(),
    )
    tool_args = {
        "tool_calls": [
            {
                "tool_name": "code_execution_tool",
                "tool_args": {
                    "code": "curl -H 'Authorization: Bearer §§secret(TOKEN)'",
                    "options": ("§§secret(TOKEN)", 3),
                },
            }
        ],
        "metadata": {"token": "§§secret(TOKEN)"},
    }

    extension = unmask_tool_secrets.UnmaskToolSecrets(_FakeAgent())
    await extension.execute(tool_args=tool_args)

    nested_args = tool_args["tool_calls"][0]["tool_args"]
    assert nested_args["code"] == (
        "curl -H 'Authorization: Bearer resolved-secret'"
    )
    assert nested_args["options"] == ("resolved-secret", 3)
    assert tool_args["metadata"] == {"token": "resolved-secret"}
