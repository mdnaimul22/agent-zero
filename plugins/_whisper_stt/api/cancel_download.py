import json
from helpers.api import ApiHandler, Request, Response
from plugins._whisper_stt.helpers import runtime


class CancelDownload(ApiHandler):
    @classmethod
    def get_methods(cls) -> list[str]:
        return ["POST"]

    async def process(self, input: dict, request: Request) -> Response:
        model_name = str(input.get("model_name") or "").strip()
        runtime.cancel_download(model_name)
        return Response(
            json.dumps({"success": True, "cancelled": True, "model": model_name}),
            status=200,
            mimetype="application/json",
            headers={
                "Cache-Control": "no-cache",
            },
        )
