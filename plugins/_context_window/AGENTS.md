# Context Window Plugin DOX

## Purpose

- Own context-window token accounting, the usage API, the composer indicator,
  its popover, and its Interface visibility row.

## Ownership

- `helpers/usage.py` owns per-prompt bucket measurement and reconciliation.
- `extensions/python/` records prompt parts at their source extension points.
- `api/context_window.py` exposes the active chat's token usage and effective
  model limit without returning prompt content.
- `webui/` and `extensions/webui/` own the Alpine store, indicator, popover,
  model-override refresh, and Interface visibility row.

## Local Contracts

- The six used-token buckets are `messages`, `system_tools`, `skills`,
  `mcp_tools`, `system_prompt`, and `extras`.
- Tools, MCP tools, and the available-skills catalog are measured from their
  extensible prompt builders, never inferred from rendered headings.
- Loaded skill instructions are removed from Messages and added to Skills.
- Protocol and prompt extras are reported together as Extras.
- Messages reuse the history record token ledger; independently rendered
  fragments use a bounded, content-addressed, runtime-only cache.
- Bucket totals reconcile to the already-stored prompt token total; the
  unclaimed remainder belongs to System prompt.
- Older chats without a stored breakdown show the explanatory empty state.
- `_model_config` supplies the effective model limit and the
  `model-context-strip-end` WebUI slot; it does not own this feature's state.
- The `contextWindowUsage` Interface setting defaults to visible on mobile and
  desktop.

## Work Guidance

- Keep prompt accounting out of rendered-text heuristics.
- Keep the API response limited to counts needed by the UI.
- Preserve the upward, right-aligned popover geometry used beside the model and
  profile selectors.

## Verification

- Run `conda run -n a0 pytest plugins/_context_window/tests`.
- Smoke-test the indicator, popover, chat switching, post-run refresh, and
  mobile/desktop visibility against the live WebUI.

## Child DOX Index

No child DOX files.
