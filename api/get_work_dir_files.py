from helpers.api import ApiHandler, Request, Response
from helpers.file_browser import FileBrowser
from helpers import runtime, files, settings

class GetWorkDirFiles(ApiHandler):

    @classmethod
    def get_methods(cls):
        return ["GET"]

    async def process(self, input: dict, request: Request) -> dict | Response:
        current_path = request.args.get("path", "") or "$WORK_DIR"
        if current_path == "$WORK_DIR":
            workdir = settings.get_settings().get("workdir_path") or "/a0"
            current_path = files.fix_dev_path(workdir)

        result = await runtime.call_development_function(get_files, current_path)

        return {"data": result}


async def get_files(path):
    browser = FileBrowser()
    return browser.get_files(path)
