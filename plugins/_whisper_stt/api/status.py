import importlib
import importlib.metadata

from helpers.api import ApiHandler, Request, Response
from plugins._whisper_stt.helpers import migration, runtime


class Status(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        migration.ensure_config_seeded()

        if not hasattr(runtime, "get_loaded_model_type"):
            try:
                importlib.reload(runtime)
            except Exception:
                pass

        package_version = ""
        package_error = ""
        try:
            package_version = importlib.metadata.version("openai-whisper")
        except Exception as e:
            package_error = str(e)

        model_type_fn = getattr(runtime, "get_loaded_model_type", lambda: "whisper")
        downloaded_models_fn = getattr(runtime, "get_downloaded_models", lambda: [])
        progress_fn = getattr(runtime, "get_current_download_progress", lambda: {})

        return {
            "plugin": "_whisper_stt",
            "enabled": runtime.is_globally_enabled(),
            "config": runtime.get_config(),
            "downloaded_models": downloaded_models_fn(),
            "download_progress": progress_fn(),
            "model": {
                "ready": await runtime.is_downloaded(),
                "loading": await runtime.is_downloading(),
                "loaded_model": runtime.get_loaded_model_name(),
                "model_type": model_type_fn(),
            },
            "package": {
                "version": package_version,
                "error": package_error,
            },
        }
