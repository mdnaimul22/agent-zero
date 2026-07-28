from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read(*parts: str) -> str:
    return PROJECT_ROOT.joinpath(*parts).read_text(encoding="utf-8")


def test_root_service_worker_caches_only_small_text_assets() -> None:
    worker = read("webui", "sw.js")

    assert 'event.data?.type !== "preload-ui-bundle"' in worker
    assert "await cache.put(request, response);" in worker
    assert "activeBundleEntries.get(event.request.url)" in worker
    assert "responseFromEntry(bundledEntry)" in worker
    assert "const cached = await cache.match(event.request);" in worker
    assert "const MAX_CACHEABLE_FILE_BYTES = 40 * 1024;" in worker
    assert "CACHEABLE_FILE_PATTERN" in worker
    assert "cacheRuntimeResponse(" in worker
    assert 'entry[1] === "text"' in worker
    assert "decodeBase64" not in worker
    assert "body.byteLength > MAX_CACHEABLE_FILE_BYTES" in worker
    assert "cacheMarkerRequest(targetCacheName)" in worker
    assert "cleanupCaches(activeCacheName)" in worker
    assert 'url.pathname.startsWith("/api/")' in worker
    assert 'request.mode === "navigate"' in worker


def test_index_registers_root_cache_without_frontend_loader_hooks() -> None:
    index = read("webui", "index.html")
    components = read("webui", "js", "components.js")
    extensions = read("webui", "js", "extensions.js")
    init_fw = read("webui", "js", "initFw.js")

    assert 'fetch("/ui/asset-bundle"' in index
    assert 'id="ui-asset-bundle"' not in index
    assert 'navigator.serviceWorker.register(' in index
    assert '{ scope: "/", updateViaCache: "none" }' in index
    assert "preload-ui-bundle" in index
    assert 'new Event("webui-bundle-loaded")' in index
    assert "webuiComponentCache" not in components
    assert "preload-ui-bundle" not in components
    assert "preload-ui-bundle" not in extensions
    assert "preload-ui-bundle" not in init_fw
