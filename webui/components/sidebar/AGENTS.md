# Sidebar Components DOX

## Purpose

- Own left sidebar layout, chat/task lists, top actions, and bottom preferences components.

## Ownership

- `left-sidebar.html` and `sidebar-store.js` own sidebar shell and shared state.
- `top-section/` owns header and quick actions.
- `chats/` owns chat list UI and state.
- `tasks/` owns task list UI and state.
- `bottom/` owns lower sidebar controls and preferences panel.

## Local Contracts

- Preserve responsive sidebar behavior and collapsed/expanded state.
- Keep chat and task list updates compatible with WebSocket state sync.
- Contexts with `parent_context_id` render as indented children beneath their parent chat; they must remain selectable while hidden from the top-level chat list.
- Chat tree expand/collapse controls use a parent-only leading slot and must not consume normal chat row text margin.
- A restored selected parent chat with children auto-expands once during context hydration unless the user has already toggled it.
- The Tasks list is reserved for scheduler-backed task contexts and must not be used for chat-bound parallel children.
- Running parent and child chats share the chat-list working-bubble animation; keep it scoped away from task and connection-status indicators.
- Chat and task lists reclaim the same part of the sidebar's left content inset so their project bubbles align, while their section headers retain the standard sidebar inset.
- Chat-row action buttons consume layout width only while a pointer row is hovered or while that row is selected on a touch device.
- Bottom version information shows its commit timestamp in UTC without a timezone suffix and remains on one line.
- Avoid text or controls overflowing fixed sidebar widths.
- Instance-level interface visibility preferences own independent mobile and desktop states for the chat-top controls and right canvas rail; mobile uses the shared 768px breakpoint.

## Work Guidance

- Coordinate navigation and state changes with WebSocket sync and chat/project stores.

## Verification

- Smoke-test sidebar collapse, chat list, task list, quick actions, and preferences after changes.

## Child DOX Index

No child DOX files.
