## multimodal vision tools

### vision_load
{{if sidecar}}analyze images with the preset's separate Vision Model and return a text result{{endif}}
{{if not sidecar}}load images into Main for visual reasoning{{endif}}
args: `paths` list of absolute image paths or ephemeral image refs{{if sidecar}}, `query` optional focused instruction, `raw` optional boolean{{endif}}
{{if sidecar}}
Input schema for tool_args:
```json
{
  "type": "object",
  "properties": {
    "paths": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Absolute image paths or ephemeral image refs."
    },
    "query": {
      "type": "string",
      "description": "What the Vision Model should inspect, compare, locate, or read."
    },
    "raw": {
      "type": "boolean",
      "description": "Use Main's native vision instead of the separate Vision Model, when Main supports vision."
    }
  },
  "required": ["paths"],
  "additionalProperties": false
}
```
{{endif}}
rules:
- put all images needed for one comparison or visual task in the same `paths` array
- use when the task depends on screenshots, diagrams, scanned documents, charts, or photos
{{if sidecar}}
- when the separate Vision Model is active, use a focused `query`; the result is a text capsule and Main does not receive raw images
- use `raw=true` only when Main supports vision and must inspect the pixels itself
{{endif}}
- only bitmaps are supported; convert other formats first if needed
- the tool result includes loaded/skipped image totals and the corresponding path lists
example:
```json
{
  "thoughts": [
    "I need to inspect the screenshot before answering."
  ],
  "headline": "Comparing screenshots",
  "tool_name": "vision_load",
  "tool_args": {
    "paths": ["/path/to/before.png", "/path/to/after.png"]{{if sidecar}},
    "query": "Compare the error banners and describe what changed."
    {{endif}}
  }
}
```
