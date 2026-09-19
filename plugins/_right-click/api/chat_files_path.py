from agent import AgentContext
from helpers import files, persist_chat
from helpers.api import ApiHandler, Request, Response


class ChatFilesPath(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        ctxid = input.get("ctxid")
        if not isinstance(ctxid, str) or not ctxid:
            return Response("Missing chat ID.", 400)
        context = AgentContext.get(ctxid)
        if not context:
            return Response("Chat context not found.", 404)

        path = persist_chat.get_chat_folder_path(context.id)
        if not files.is_dir(path):
            path = files.get_abs_path(persist_chat.CHATS_FOLDER)
        return {"ok": True, "path": files.normalize_a0_path(path)}
