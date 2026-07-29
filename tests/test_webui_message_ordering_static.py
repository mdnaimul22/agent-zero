import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read(*parts: str) -> str:
    return (PROJECT_ROOT / Path(*parts)).read_text(encoding="utf-8")


def test_full_log_replays_replace_existing_message_dom():
    index_js = read("webui", "index.js")
    messages_js = read("webui", "js", "messages.js")

    assert "snapshot.logs?.[0]?.no === 0" in index_js
    assert "msgs.resetMessageRenderState();" in index_js
    assert "export function resetMessageRenderState" in messages_js
    assert "normalized.sort(" in messages_js


def test_message_ordering_uses_a_bounded_tail_first_renderer_cache():
    messages_js = read("webui", "js", "messages.js")
    message_window_js = read("webui", "js", "message-window.js")

    assert 'from "./message-window.js"' in messages_js
    assert "_messageWindow.compactTailIfNeeded()" in messages_js
    assert "_messageWindow.visibleMessages()" in messages_js
    assert "class MessageWindow" in message_window_js
    assert "showTail()" in message_window_js
    assert "shiftOlder()" in message_window_js
    assert "shiftNewer()" in message_window_js
    assert "_messageWindowFollowTail" in messages_js
    assert "_messageWindow.showTail()" in messages_js


def test_virtual_paging_uses_passive_loaders_and_cancels_stale_scrolling():
    messages_js = read("webui", "js", "messages.js")
    scroller_js = read("webui", "js", "scroller.js")
    messages_css = read("webui", "css", "messages.css")

    assert "createMessageWindowIndicator" in messages_js
    assert 'indicator.setAttribute("role", "status")' in messages_js
    assert "message-window-loader-bubble" in messages_css
    assert "@keyframes message-window-loader-dot" in messages_css
    assert "Load ${Math.min" not in messages_js
    assert "export function cancelPendingScroll" in scroller_js
    assert "cancelPendingScroll(history)" in messages_js


def test_message_actions_put_copy_before_speak():
    sources = [
        read("webui", "js", "messages.js"),
        read(
            "plugins",
            "_browser",
            "extensions",
            "webui",
            "get_tool_message_handler",
            "browser-tool-handler.js",
        ),
    ]

    for source in sources:
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if 'createActionButton("speak"' not in line:
                continue
            preceding_actions = [
                match.group(1)
                for candidate in lines[max(0, index - 12) : index]
                if (match := re.search(r'createActionButton\("(detail|copy|speak)"', candidate))
            ]
            assert preceding_actions and preceding_actions[-1] == "copy"
