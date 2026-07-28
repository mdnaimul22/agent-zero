from pathlib import Path
from html.parser import HTMLParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self.scripts.append(dict(attrs))


def test_bootstrap_is_local_and_deferred() -> None:
    index_html = (PROJECT_ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert "cdn.jsdelivr.net/npm/bootstrap" not in index_html
    assert '<script defer src="vendor/bootstrap/bootstrap.bundle.min.js"></script>' in index_html
    assert (PROJECT_ROOT / "webui" / "vendor" / "bootstrap" / "bootstrap.bundle.min.js").is_file()


def test_classic_startup_scripts_are_deferred() -> None:
    index_html = (PROJECT_ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    parser = ScriptParser()
    parser.feed(index_html)

    blocking_scripts = [
        script["src"]
        for script in parser.scripts
        if script.get("src")
        and script.get("type") != "module"
        and "defer" not in script
        and "async" not in script
    ]

    assert blocking_scripts == []


def test_ui_asset_bundle_fetch_precedes_frontend_assets() -> None:
    index_html = (PROJECT_ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    ui_server = (PROJECT_ROOT / "helpers" / "ui_server.py").read_text(encoding="utf-8")

    bundle_index = index_html.index('fetch("/ui/asset-bundle"')
    assert bundle_index < index_html.index('<link rel="preload" as="style" href="index.css"')
    assert bundle_index < index_html.index('<script type="module" src="index.js"></script>')
    assert 'id="ui-asset-bundle"' not in index_html
    assert '"/ui/asset-bundle"' in ui_server
    assert "handlers.serve_ui_asset_bundle" in ui_server
    assert 'response.headers["Content-Encoding"] = "gzip"' in ui_server
    assert 'response.set_etag(bundle["version"], weak=True)' in ui_server


def test_initial_styles_load_without_blocking_splash_paint() -> None:
    index_html = (PROJECT_ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert '<link rel="stylesheet"' not in index_html
    assert index_html.count('rel="preload" as="style"') == 18
    assert index_html.count("onload=\"this.onload=null;this.rel='stylesheet'\"") == 18
    assert 'new Promise((resolve) => addEventListener("load", resolve' in index_html
    assert 'document.addEventListener("webui-bundle-loaded"' in index_html


def test_startup_splash_is_inline_and_waits_for_extension_readiness() -> None:
    index_html = (PROJECT_ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    extensions_js = (PROJECT_ROOT / "webui" / "js" / "extensions.js").read_text(
        encoding="utf-8"
    )

    assert index_html.index("#startup-splash") < index_html.index('fetch("/ui/asset-bundle"')
    assert index_html.index('id="startup-splash"') < index_html.index('<div class="container">')
    assert 'data-splash-theme="dark"' in index_html
    assert 'localStorage.getItem("darkMode") === "false"' in index_html
    assert 'src="/public/a0-fullDark.svg"' in index_html
    assert "width: clamp(12rem, 34vw, 23rem)" in index_html
    assert 'document.addEventListener("webui-extensions-loaded"' in index_html
    assert 'export let initialHtmlExtensionsLoaded = false' in extensions_js
    assert 'const LOADING_SELECTOR = "x-component > .loading:empty, x-extension.loading"' in extensions_js
    assert 'targetElement.classList.add("loading")' in extensions_js
    assert 'targetElement.classList.remove("loading")' in extensions_js
    assert 'document.dispatchEvent(new Event("webui-extensions-loaded"))' in extensions_js
    assert "globalThis.Alpine.nextTick" in extensions_js
    assert "pendingHtmlImports" not in extensions_js
    assert "data-extension-loaded" not in extensions_js
