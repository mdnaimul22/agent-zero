# Context Doctor Plugin DOX

## Purpose

- Repair malformed Agent Zero tool-call JSON and preserve raw output as compact thoughts JSON when repair cannot produce a tool call.

## Ownership

- `helpers/context_doctor.py` transforms output and refreshes log fields.
- `extensions/python/message_loop_result/` normalizes completed model output before default processing.
- `webui/config.html` exposes XML suppression and log-detail settings.

## Local Contracts

- Repaired and fallback JSON is always minified.
- Nonempty non-tool output becomes `{"thoughts":[raw]}`; XML-like output becomes `{}` only when suppression is enabled.
- Log kvps and heading always reflect transformed output; `update_log` controls only View Details content.
- A repaired `response` tool call refreshes the response log item when streaming did not create it.

## Work Guidance

- Keep repair scoped to complete tool-call JSON.
- Use framework-installed `json_repair`; apply plugin-local parser patch before repair. Do not vendor dependencies.

## Verification

- Run `pytest plugins/_context_doctor/tests`.

## Child DOX Index

No child DOX files.
