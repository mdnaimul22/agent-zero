# Plugin WebUI

Sources: `/a0/webui/AGENTS.md`, `/a0/webui/js/AGENTS.md`, `/a0/webui/components/AGENTS.md`, `/a0/plugins/AGENTS.md`. Match existing component geometry and use framework UI surfaces.

## Mandatory Frontend Patterns

### 1. The "Store Gate" Template
To avoid race conditions and undefined errors, every component must use this wrapper:
```html
<div x-data>
  <template x-if="$store.myPluginStore">
    <div x-init="$store.myPluginStore.onOpen()" x-destroy="$store.myPluginStore.cleanup()">
       <!-- Content goes here -->
    </div>
  </template>
</div>
```

### 2. Separate Store Module
Place store logic in a separate .js file. Do NOT use alpine:init listeners inside HTML.
```javascript
// webui/my-store.js
import { createStore } from "/js/AlpineStore.js";
export const store = createStore("myPluginStore", {
    status: 'idle',
    init() {},
    onOpen() {},
    cleanup() {}
});
```
Import it in the HTML <head>:
```html
<head>
  <script type="module" src="/plugins/<plugin_name>/webui/my-store.js"></script>
</head>
```

### 3. User Feedback: A0 Notifications Only
Do **not** show errors or success via inline boxes (e.g. a red `<div>` bound to `store.error`). Use the project notification system so toasts and history stay consistent.

- **Errors**: `toastFrontendError(message, "My Plugin")` (or `$store.notificationStore.frontendError(...)`)
- **Success**: `toastFrontendSuccess(message, "My Plugin")`
- **Warnings/Info**: `toastFrontendWarning`, `toastFrontendInfo` from `/components/notifications/notification-store.js`

Import and call from your store; do not render a dedicated error/success block in the template. See [Notifications](/a0/docs/developer/notifications.md) for the full API.

---

## Plugin Settings

If your plugin needs user-configurable settings, add `webui/config.html`. The system detects it automatically and shows a Settings button in the relevant tabs (per `settings_sections` in `plugin.yaml`).

### Settings modal contract

The modal provides Project + Agent profile context selectors. The plugin settings wrapper instantiates a local modal context from `$store.pluginSettingsPrototype`. Inside `config.html`, bind plugin fields to `config.*` and use `context.*` for modal-level state and actions:

```html
<html>
<head>
  <title>My Plugin Settings</title>
  <script type="module">
    import { store } from "/components/plugins/plugin-settings-store.js";
  </script>
</head>
<body>
  <div x-data>
    <input x-model="config.my_key" />
    <input type="checkbox" x-model="config.feature_enabled" />
  </div>
</body>
</html>
```

The modal's Save button persists `config` to `config.json` in the correct scope (project/agent/global).

### Sidebar Button (sidebar entry point)
- Extension point: `sidebar-quick-actions-main-start`
- Class: `class="config-button"`
- Placement: `x-move-after=".config-button#dashboard"`
- Action: `@click="openModal('/plugins/<plugin_name>/webui/my-modal.html')"`

---


## Custom Message Handlers

Prefer the existing `tool` log type and `get_tool_message_handler` hook when only a tool's presentation needs customization. For a distinct log type, add `extensions/webui/get_message_handler/my-handler.js` inside the plugin:

```javascript
import { drawProcessStep } from "/js/messages.js";

export default function (extData) {
  if (extData.type !== "my_plugin_step") return;
  extData.handler = function (log) {
    return drawProcessStep({
      id: log.id,
      title: log.heading,
      code: "MY",
      content: log.content,
      kvps: log.kvps,
      log,
    });
  };
}
```

- The backend emits the matching `type="my_plugin_step"` through the normal log API (for a tool, override `get_log_object()`). Use plugin-specific type names and stable IDs for streaming updates.
- The default extension function assigns `extData.handler` only for its own types. A handler receives `{ id, no, type, heading, content, kvps, timestamp, agentno, ... }` and returns `{ element, ... }`, or a promise of that result. Return the object from `drawProcessStep`, not just its DOM element.
- Register every custom type rendered as a process step using the hook below. The UI calls it before grouping raw logs for paging and hidden utility messages. No core type list or plugin manifest edit is needed. Keep a type's process/standalone role consistent across its handlers.
- Pass the original handler argument as `log`; copying or reconstructing it can lose internal rendering metadata. Standalone handlers can use `drawMessageDefault(log)` and do not register a process type. Without a custom handler, types retain the generic tool-step fallback, including disabled plugins' old logs. The dispatcher recognizes that fallback before grouping; standalone rendering must come from an explicit handler.

Add `extensions/webui/get_process_step_types/my-types.js` in the same plugin:

```javascript
export default function (context) {
  context.processStepTypes.add("my_plugin_step");
}
```

The context contains a fresh `Set` seeded with core types. Add only your process types, preserving existing entries; registration must be repeatable and independent of individual records. Use the normal plugin enable/disable flow and reload the page when prompted. The existing extension loader owns caching.

Examples: `/a0/plugins/_code_execution/extensions/webui/get_message_handler/code-exe-handler.js` and `/a0/plugins/_text_editor/extensions/webui/get_message_handler/_10_text_editor_handler.js`. Verify live updates, replay, hidden/visible utilities, root-response completion, paging beyond 50 steps, and rendering old logs with the plugin disabled.

## Verification

Check the browser console, store loading, settings persistence in the intended scope, and the rendered feature in the target instance. Verify desktop and narrow layouts for UI changes. Close only test surfaces you opened.
