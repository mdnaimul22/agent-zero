"""Repair and minify model tool-call JSON before default processing."""

from __future__ import annotations

from typing import Any, override

from helpers.extension import Extension
from helpers.plugins import get_plugin_config
from plugins._context_doctor.helpers.context_doctor import repair_and_minify, update_log_item


class ContextDoctor(Extension):
    @override
    def execute(self, result_data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not self.agent or not isinstance(result_data, dict):
            return

        llm_result = result_data.get("llm_result")
        response = getattr(llm_result, "response", None)
        if not isinstance(response, str):
            return

        config = get_plugin_config("_context_doctor", agent=self.agent) or {}
        repaired = repair_and_minify(
            response, suppress_xml=config.get("suppress_xml", True)
        )
        if repaired is None:
            return

        llm_result.response = repaired
        params = getattr(getattr(self.agent, "loop_data", None), "params_temporary", None)
        log_item = params.get("log_item_generating") if isinstance(params, dict) else None
        if log_item is not None and config.get("update_log", False):
            update_log_item(self.agent, log_item, repaired)
