from agent import AgentContext
from helpers.api import ApiHandler, Request, Response


class Stop(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        ctxid = input.get("context", "")
        if not isinstance(ctxid, str) or not ctxid.strip():
            return Response(
                '{"error": "context is required"}',
                status=400,
                mimetype="application/json",
            )

        context = AgentContext.use(ctxid.strip())
        if not context:
            return Response(
                '{"error": "Chat context not found"}',
                status=404,
                mimetype="application/json",
            )
        was_running = context.is_running()

        context.kill_process()
        context.paused = False
        context.log.set_progress("", active=False)

        msg = "Agent process stopped."
        context.log.log(type="info", content=msg, finished=True)

        return {
            "message": msg,
            "context": context.id,
            "stopped": was_running,
        }
