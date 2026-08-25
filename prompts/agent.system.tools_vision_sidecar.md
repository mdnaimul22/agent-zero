## multimodal vision tools

### vision_load
analyze images with the separate Vision Model and return a text result
args: `paths` list of absolute image paths or ephemeral image refs, `query` optional focused instruction, `raw` optional boolean
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
      "description": "Use the Main model's native vision instead, when Main supports vision."
    }
  },
  "required": ["paths"],
  "additionalProperties": false
}
```
rules:
- put all images needed for one comparison or visual task in the same `paths` array; they are sent in one Vision Model call
- use a focused `query`; if omitted, the Vision Model returns a concise general description
- the result is a text capsule; the Main model does not receive the raw images
- use `raw=true` only when Main supports vision and must inspect the pixels itself
- only bitmaps are supported
example:
```json
{
  "thoughts": [
    "I need to compare both screenshots before answering."
  ],
  "headline": "Comparing screenshots",
  "tool_name": "vision_load",
  "tool_args": {
    "paths": ["/path/to/before.png", "/path/to/after.png"],
    "query": "Compare the error banners and describe what changed."
  }
}
```
