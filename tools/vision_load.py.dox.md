# vision_load.py DOX

## Purpose

- Own the `vision_load.py` agent tool.
- This module routes images either into Main model-visible content or through the preset's optional Vision Model.
- Keep this file-level DOX profile synchronized with `vision_load.py` because this directory is intentionally flat.

## Ownership

- `vision_load.py` owns the runtime implementation.
- `vision_load.py.dox.md` owns durable notes about responsibilities, contracts, side effects, and verification for that implementation.
- Classes:
- `VisionLoad` (`Tool`)
  - `async execute(self, paths, query="", raw=False, **kwargs) -> Response`
  - `async after_execution(self, response: Response, **kwargs)`
- Notable constants/configuration names: `TOKENS_ESTIMATE`.

## Runtime Contracts

- Tool modules must define `helpers.tool.Tool` subclasses and return `helpers.tool.Response` from `execute(...)`.
- One call may contain multiple paths. The delegated route sends every selected path in one Vision Model request and returns one textual capsule.
- Main native vision wins unless the effective preset selects the sidecar route; `raw=true` returns to Main native vision only when Main supports it.
- Delegation completes during `execute(...)` so native Responses function output contains the real capsule before `after_execution(...)` persists it.
- Delegated history contains the text capsule only. Native history contains the tool result followed by one raw message holding all loaded image blocks.
- Direct parallel workers resolve ephemeral refs, model routing, and durable chat-media storage against their recorded parent context; independent vision jobs remain generic parallel jobs.
- `max_embeds` comes from the model that actually receives the images.
- Vision Model calls use the selected model's Advanced `kwargs`; this tool does not impose a separate timeout or output-token limit.
- Update this file whenever tool arguments, output shape, `break_loop` behavior, intervention handling, prompt instructions, or side effects change.
- `VisionLoad` is a `Tool`.
- `VisionLoad` defines `execute(...)`.
- Observed side-effect areas: filesystem writes, model calls, plugin state, settings/state persistence, secret handling.
- Imported dependency areas include: `helpers`, `helpers.print_style`, `helpers.tool`, `mimetypes`.

## Key Concepts

- Important called helpers/classes observed in the source: `build_vision_model`, `use_vision_sidecar`, `self._get_max_embeds`, `Response`, `self._context_id`, `chat_media.save_image_base64`, `chat_media.save_image_data_url`, `chat_media.materialize_image_ref`, `ephemeral_images.consume_image`, `images.to_data_url`, `self.agent.hist_add_tool_result`, `history.RawMessage`, `model.unified_call`.
- Keep request/response, tool, or helper semantics documented here at the same time as source changes.

## Work Guidance

- Keep tool output concise, model-readable, and safe for history persistence.
- Coordinate argument or behavior changes with prompt tool instructions and skill guidance.
- Respect intervention flow for long-running, external, or user-visible operations.

## Verification

- Run targeted tool and prompt-contract tests for changed behavior; smoke-test agent execution when no focused test exists.
- Related tests observed by source search:
  - `tests/test_browser_agent_regressions.py`
  - `tests/test_host_browser_connector.py`
  - `tests/test_office_desktop_state.py`
  - `tests/test_vision_load_image_refs.py`

## Child DOX Index

No child DOX files.
