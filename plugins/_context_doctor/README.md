# Context Doctor

Repairs model tool call JSON before sending it to history and tool processing.

## Behavior

- Uses `json_repair` after native parsing has completed.
- Accepts only complete Agent Zero tool calls (`tool_name` and object `tool_args`).
- Stores and displays repaired tool calls as compact JSON.
- Optionally replaces XML-like output with `{}` in cases where model uses native XML tool calls.
