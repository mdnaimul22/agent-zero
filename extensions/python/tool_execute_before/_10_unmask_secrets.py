from collections.abc import Callable
from typing import Any

from helpers.extension import Extension
from helpers.secrets import get_secrets_manager


def replace_placeholders_recursive(
    value: Any,
    replace: Callable[[str], str],
) -> Any:
    if isinstance(value, str):
        return replace(value)
    if isinstance(value, dict):
        return {
            key: replace_placeholders_recursive(item, replace)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_placeholders_recursive(item, replace) for item in value]
    if isinstance(value, tuple):
        return tuple(replace_placeholders_recursive(item, replace) for item in value)
    return value


class UnmaskToolSecrets(Extension):

    async def execute(self, **kwargs):
        if not self.agent:
            return

        # Get tool args from kwargs
        tool_args = kwargs.get("tool_args")
        if not tool_args:
            return

        secrets_mgr = get_secrets_manager(self.agent.context)

        # Unmask placeholders in args for actual tool execution
        for k, v in tool_args.items():
            tool_args[k] = replace_placeholders_recursive(
                v, secrets_mgr.replace_placeholders
            )
