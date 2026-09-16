from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = PROJECT_ROOT / "webui/components/modals/file-browser/file-browser-store.js"


def test_open_file_link_navigates_live_browser_in_place():
    """Path-link clicks must reuse a live file browser instead of stacking a second window."""
    source = STORE_PATH.read_text(encoding="utf-8")
    assert "window.openFileLink" in source
    assert "hasLiveBrowser" in source
    assert "navigateToFolder(resp.abs_path)" in source


def test_open_file_link_keeps_modal_fallback():
    """Without a live browser the path link still opens the browser with the clicked path."""
    source = STORE_PATH.read_text(encoding="utf-8")
    assert "await store.open(resp.abs_path)" in source
