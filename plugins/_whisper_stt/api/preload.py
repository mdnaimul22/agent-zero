import importlib
from helpers.api import ApiHandler, Request, Response
from plugins._whisper_stt.helpers import runtime


class Preload(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        if not runtime.is_globally_enabled():
            return Response(status=409, response="Whisper STT plugin is disabled")

        if not hasattr(runtime, "get_loaded_model_type"):
            try:
                importlib.reload(runtime)
            except Exception:
                pass

        model_name = str(input.get("model_name") or "").strip() or None
        model_type_fn = getattr(runtime, "get_loaded_model_type", lambda: "whisper")
        try:
            await runtime.preload(model_name)
            return {
                "success": True,
                "loaded_model": runtime.get_loaded_model_name(),
                "model_type": model_type_fn(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
