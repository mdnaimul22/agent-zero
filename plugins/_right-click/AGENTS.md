# Right Click Plugin DOX

## Purpose

- Own optional right-click actions for sidebar chats/tasks and Files entries.

## Ownership

- `plugin.yaml` owns the toggleable, configuration-free bundled plugin.
- `api/chat_files_path.py` resolves a known chat's saved folder through `persist_chat`; the endpoint retains default authentication/CSRF and never creates or selects a context.
- `webui/right-click-store.js` owns gesture routing, pointer placement, dismissal, and keyboard focus.
- `extensions/webui/sidebar-start/` owns page-scoped listeners and their Alpine teardown.
- `extensions/webui/sidebar-row-actions-menu/` adds Open in Files and Save/Load only to chat context menus.
- `extensions/webui/file-browser-actions-menu/` adds directory Open, file Download, and inline-confirmed Delete only to file context menus.
- `webui/thumbnail.webp` owns the generated 256 × 256 plugin thumbnail, kept below 20,000 bytes.

## Local Contracts

- Keep `always_enabled: false`; activation uses the standard plugin toggle and frontend reload.
- Invoke the existing row's overflow trigger, preserving plugin actions and Files host ownership; never clone menus or patch stores.
- Reuse Files dropdown positioning and the shared dropdown styles.
- Right-click must not select a chat, navigate a folder, open a file, or change checkbox selection.
- Save passes the clicked chat ID to `chats.saveChat(ctxid)` without switching the active chat.
- Open in Files resolves the clicked chat through this plugin's `chat_files_path` endpoint, then uses `chatInput.browseFiles(path)` for standard canvas/modal routing. Always open `usr/chats/<id>`, independent of project membership; use `usr/chats` if the chat has no saved folder yet. Reject missing/unknown IDs rather than selecting or creating a context. Do not infer the path from the selected chat or fall back to another chat on errors.
- Disable Open in Files while Files is loading, renaming, or doing bulk work so its navigation guards preserve pending operations.
- File actions keep backend-owned remote permission checks. Reuse existing Edit (Editor), Open in Browser/Desktop, and folder Download ZIP entries without duplication.
- Delete uses `$confirmClick` and stops the click from reaching the menu's automatic close handler; close only after confirmation. Closing the menu discards the plugin entry and its armed state.
- Preserve native menus outside supported rows, in text inputs and file pickers, and with Shift held.
- Close on outside click, surrounding scroll, resize, Escape, action, and unmount; scrolling the menu itself stays usable.

## Work Guidance

- Keep gesture behavior plugin-local and both existing three-dot menus intact.
- Cover parent/child chats, task rows, list/icon entries, and modal/canvas Files hosts.

## Verification

- Run `pytest plugins/_right-click/tests tests/test_sidebar_row_actions.py tests/test_editor_files.py`.
- Smoke-test plugin disable/re-enable with page reload, keyboard navigation, and viewport-edge placement.

## Child DOX Index

No child DOX files.
