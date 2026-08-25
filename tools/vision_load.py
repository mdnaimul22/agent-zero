import asyncio
import json
from mimetypes import guess_type

from langchain_core.messages import HumanMessage, SystemMessage

from helpers import chat_media, ephemeral_images, files, history, images, runtime
from helpers.parallel_tools import PARALLEL_WORKER_PARENT_CONTEXT_KEY, coerce_bool
from helpers.print_style import PrintStyle
from helpers.tool import Response, Tool
from plugins._model_config.helpers.model_config import (
    build_vision_model,
    get_chat_model_config,
    get_vision_model_config,
    use_vision_sidecar,
)

TOKENS_ESTIMATE = 1500
VISION_TIMEOUT_SECONDS = 300
VISION_SYSTEM_PROMPT = (
    "You are a precise vision analyst. Answer only what was asked about the images. "
    "Be concise and factual. Preserve exact visible text when asked to read it."
)
DEFAULT_VISION_QUERY = (
    "Describe the images precisely, including the key objects, visible text, and layout."
)


class VisionLoad(Tool):
    async def execute(
        self,
        paths: list[str] | str | None = None,
        query: str = "",
        raw: bool = False,
        **kwargs,
    ) -> Response:
        self.images_dict: dict[str, str] = {}
        self.loaded_paths: list[str] = []
        self.skipped_paths: list[str] = []
        self._config_owner = self._config_agent()
        self._main_has_vision = bool(
            get_chat_model_config(self._config_owner).get("vision", False)
        )
        self._delegated = use_vision_sidecar(self._config_owner) and not (
            coerce_bool(raw, False) and self._main_has_vision
        )
        self._max_embeds = self._get_max_embeds()

        normalized = self._normalize_paths(paths)
        if isinstance(normalized, str):
            self._history_result = normalized
            return Response(message=normalized, break_loop=False)

        requested = [
            (path.strip(), self._display_input_path(path.strip(), index + 1))
            for index, path in enumerate(normalized)
        ]
        limited = requested if self._max_embeds <= 0 else requested[-self._max_embeds :]
        if self._max_embeds > 0 and len(requested) > self._max_embeds:
            self.skipped_paths = [display for _, display in requested[: -self._max_embeds]]

        for index, (path, display_path) in enumerate(limited):
            if not path:
                continue
            if ephemeral_images.is_ref(path):
                image = ephemeral_images.consume_image(path, context_id=self._context_id())
                if image is None:
                    continue
                display_path = image.display_name or display_path
                stored_ref = self._store_ephemeral_image(image)
            elif self._is_data_image_url(path):
                stored_ref = self._store_data_url(
                    path, preferred_name=f"vision-load-{index + 1}.png"
                )
            elif await runtime.call_development_function(files.exists, path):
                mime_type, _ = guess_type(path)
                if not mime_type or not mime_type.startswith("image/"):
                    continue
                try:
                    stored_ref = self._store_local_image(
                        path, preferred_name=files.basename(path)
                    )
                except (FileNotFoundError, OSError, ValueError):
                    continue
            else:
                continue

            if stored_ref:
                self.images_dict[display_path] = stored_ref
                self.loaded_paths.append(display_path)

        summary = self._summary()
        if self._delegated and self.images_dict:
            try:
                capsule = await self._call_vision_model(
                    list(self.images_dict.values()), self._query(query, kwargs)
                )
                message = (
                    f"Vision Model analyzed {len(self.images_dict)} image(s)"
                    f"; {len(self.skipped_paths)} skipped.\n\n{capsule.strip()}"
                )
                self._history_result = message
                return Response(message=message, break_loop=False)
            except Exception as exc:
                message = f"Vision Model error: {str(exc)[:1000]}"
                self._history_result = f"{summary}\n\n{message}"
                return Response(message=message, break_loop=False)

        if self.images_dict and not self._main_has_vision:
            summary += (
                "\n\nImages were not injected because neither Main native vision nor "
                "a usable Vision Model is active."
            )
        self._history_result = (
            summary if self.images_dict or self.skipped_paths else "No images processed"
        )
        message = (
            "No images processed"
            if not self.images_dict and not self.skipped_paths
            else f"{len(self.images_dict)} images loaded, {len(self.skipped_paths)} skipped"
        )
        return Response(message=message, break_loop=False)

    async def after_execution(self, response: Response, **kwargs):
        log_id = str(getattr(getattr(self, "log", None), "id", "") or "")
        self.agent.hist_add_tool_result(
            self.name,
            self._history_result,
            id=log_id,
            **(response.additional or {}),
        )

        if self.images_dict and self._main_has_vision and not self._delegated:
            content = [
                {"type": "image_url", "image_url": {"url": image_path}}
                for image_path in self.images_dict.values()
            ]
            self.agent.hist_add_message(
                False,
                content=history.RawMessage(
                    raw_content=content,
                    preview="<Image attachments loaded by path>",
                ),
                tokens=TOKENS_ESTIMATE * len(content),
            )

        PrintStyle(
            font_color="#1B4F72", background_color="white", padding=True, bold=True
        ).print(f"{self.agent.agent_name}: Response from tool '{self.name}'")
        PrintStyle(font_color="#85C1E9").print(response.message)
        if getattr(self, "log", None):
            self.log.update(result=response.message)

    def _get_max_embeds(self) -> int:
        cfg = (
            get_vision_model_config(self._config_owner)
            if self._delegated
            else get_chat_model_config(self._config_owner)
        )
        try:
            return int(cfg.get("max_embeds", 10) or 0)
        except (TypeError, ValueError):
            return 10

    def _context_id(self) -> str:
        context = getattr(self.agent, "context", None)
        if not context:
            return ""
        get_data = getattr(context, "get_data", None)
        parent_id = get_data(PARALLEL_WORKER_PARENT_CONTEXT_KEY) if get_data else ""
        return str(parent_id or getattr(context, "id", "") or "").strip()

    def _config_agent(self):
        context = getattr(self.agent, "context", None)
        get_data = getattr(context, "get_data", None)
        parent_id = get_data(PARALLEL_WORKER_PARENT_CONTEXT_KEY) if get_data else ""
        if parent_id:
            from agent import AgentContext

            parent = AgentContext.get(str(parent_id))
            if parent:
                return parent.agent0
        return self.agent

    async def _call_vision_model(self, image_paths: list[str], query: str) -> str:
        model = build_vision_model(self._config_owner)
        content = [{"type": "text", "text": query or DEFAULT_VISION_QUERY}]
        content.extend(
            {"type": "image_url", "image_url": {"url": path}}
            for path in image_paths
        )
        response, _ = await asyncio.wait_for(
            model.unified_call(
                messages=[
                    SystemMessage(content=VISION_SYSTEM_PROMPT),
                    HumanMessage(content=content),
                ],
                explicit_caching=False,
                max_tokens=2000,
            ),
            timeout=VISION_TIMEOUT_SECONDS,
        )
        if not str(response or "").strip():
            raise RuntimeError("Vision Model returned an empty response.")
        return str(response)

    def _store_ephemeral_image(self, image: ephemeral_images.EphemeralImage) -> str:
        context_id = self._context_id()
        if not context_id:
            return image.data_url
        source = chat_media.infer_source(image.ref, image.display_name)
        saved = chat_media.save_image_base64(
            context_id=context_id,
            data=image.data,
            mime_type=image.mime,
            category=chat_media.category_for_source(source),
            source=source,
            preferred_name=image.display_name,
        )
        return saved.a0_path

    def _store_data_url(self, data_url: str, *, preferred_name: str = "") -> str:
        context_id = self._context_id()
        if not context_id:
            return data_url
        source = chat_media.infer_source(data_url, preferred_name)
        saved = chat_media.save_image_data_url(
            context_id=context_id,
            data_url=data_url,
            category=chat_media.category_for_source(source),
            source=source,
            preferred_name=preferred_name,
        )
        return saved.a0_path

    def _store_local_image(self, path: str, *, preferred_name: str = "") -> str:
        context_id = self._context_id()
        if not context_id:
            return images.to_data_url(path)
        return chat_media.materialize_image_ref(
            context_id=context_id,
            url=path,
            source=chat_media.infer_source(path, preferred_name),
            preferred_name=preferred_name,
        )

    def _summary(self) -> str:
        loaded = "\n".join(self.loaded_paths) if self.loaded_paths else "none"
        skipped = "\n".join(self.skipped_paths) if self.skipped_paths else "none"
        return (
            f"Loaded images ({len(self.loaded_paths)}):\n{loaded}\n\n"
            f"Skipped images ({len(self.skipped_paths)}, max {self._max_embeds}):\n{skipped}"
        )

    @staticmethod
    def _normalize_paths(paths: list[str] | str | None) -> list[str] | str:
        if isinstance(paths, str):
            try:
                decoded = json.loads(paths)
            except json.JSONDecodeError:
                decoded = paths
            paths = decoded if isinstance(decoded, list) else [paths]
        if paths is None:
            return []
        if not isinstance(paths, (list, tuple)):
            return "vision_load error: `paths` must be an array of image paths."
        return [str(path or "").strip() for path in paths]

    @staticmethod
    def _query(query: str, kwargs: dict) -> str:
        if str(query or "").strip():
            return str(query).strip()
        for key in ("prompt", "question", "instruction", "focus", "request"):
            if str(kwargs.get(key) or "").strip():
                return str(kwargs[key]).strip()
        return DEFAULT_VISION_QUERY

    @staticmethod
    def _is_data_image_url(value: str) -> bool:
        normalized = str(value or "").strip().lower()
        return normalized.startswith("data:image/") and ";base64," in normalized

    @classmethod
    def _display_input_path(cls, value: str, index: int) -> str:
        if ephemeral_images.is_ref(value):
            return ephemeral_images.display_ref(value)
        if cls._is_data_image_url(value):
            return f"{value.split(',', 1)[0]},<ephemeral-image-{index}>"
        return value
