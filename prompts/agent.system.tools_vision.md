## multimodal vision tools

### vision_load
load or analyze images for visual reasoning
args: `paths` list of absolute image paths or ephemeral image refs, `query` optional focused instruction
Input schema for tool_args:
```json
{
  "type": "object",
  "properties": {
    "paths": {
      "type": "array",
      "items": {"type": "string"}
    },
    "query": {"type": "string"}
  },
  "required": ["paths"],
  "additionalProperties": false
}
```
rules:
- put all images needed for one comparison or visual task in the same `paths` array
- use when the task depends on screenshots, diagrams, scanned documents, charts, or photos
- use a focused `query` when asking the configured Vision Model to inspect, compare, locate, or read something
- only bitmaps are supported; convert other formats first if needed
- the tool result reports loaded and skipped image counts
example:
```json
{
  "thoughts": [
    "I need to inspect the screenshot before answering."
  ],
  "headline": "Comparing screenshots",
  "tool_name": "vision_load",
  "tool_args": {
    "paths": ["/path/to/before.png", "/path/to/after.png"],
    "query": "Compare the error banners and describe what changed."
  }
}
```
