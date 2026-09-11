import importlib
from helpers.api import ApiHandler, Request, Response
from plugins._whisper_stt.helpers import runtime


class DeleteModel(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        if not runtime.is_globally_enabled():
            return Response(status=409, response="Whisper STT plugin is disabled")

        if not hasattr(runtime, "delete_model"):
            try:
                importlib.reload(runtime)
            except Exception:
                pass

        model_name = str(input.get("model_name") or "").strip()
        if not model_name:
            return Response(status=400, response="Missing model_name")

        try:
            success = await runtime.delete_model(model_name)
            return {
                "success": success,
                "model_name": model_name,
                "downloaded_models": runtime.get_downloaded_models(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
