# Context Doctor Plugin DOX

## Purpose

- Repair malformed Agent Zero tool-call JSON and persist compact repaired output.

## Ownership

- `helpers/context_doctor.py` validates and repairs tool-call JSON.
- `extensions/python/message_loop_result/` normalizes completed model output before default processing.
- `webui/config.html` exposes XML-output suppression only.

## Local Contracts

- Repaired tool-call JSON is always minified.
- Invalid non-tool output is unchanged; native processing retains ownership.
- Do not write settings that alter repair mode or log content.

## Work Guidance

- Keep repair scoped to complete tool-call JSON.
- Use framework-installed `json_repair`; apply the plugin-local parser patch before repair. Do not vendor dependencies.

## Verification

- Run `pytest plugins/_context_doctor/tests`.

## Child DOX Index

No child DOX files.
