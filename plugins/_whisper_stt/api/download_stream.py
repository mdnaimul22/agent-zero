import json
from helpers.api import ApiHandler, Request, Response
from plugins._whisper_stt.helpers import runtime


class DownloadStream(ApiHandler):
    @classmethod
    def get_methods(cls) -> list[str]:
        return ["POST"]

    async def process(self, input: dict, request: Request) -> Response:
        if not runtime.is_globally_enabled():
            return Response("Whisper STT plugin is disabled", status=409, mimetype="text/plain")

        model_name = str(input.get("model_name") or "").strip()
        if not model_name:
            err = json.dumps({"type": "error", "message": "Model name cannot be empty."}) + "\n"
            return Response(err, status=400, mimetype="application/x-ndjson")

        def generate():
            for event in runtime.stream_download_model(model_name):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return Response(
            generate(),
            status=200,
            mimetype="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
