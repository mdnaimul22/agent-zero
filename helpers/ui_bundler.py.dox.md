# ui_bundler.py DOX

## Purpose

- Build the versioned static-asset payload used to prefill the WebUI service-worker cache.
- Keep component and extension loaders independent from transport-level caching.

## Ownership

- `ui_bundler.py` owns same-origin text-asset discovery, recursive reference scanning, safe URL-to-file resolution, bundle hashing, and JSON serialization.
- `ui_bundler.py.dox.md` owns the durable contracts for that behavior.
- Top-level functions:
  - `get_ui_asset_bundle(agent: Agent | None = None) -> dict`
  - `serialize_ui_asset_bundle(bundle: dict) -> str`

## Runtime Contracts

- The bundle seeds core component HTML, eligible HTML/CSS/JavaScript under enabled extension roots, top-level enabled-plugin HTML entry pages, and recursively referenced HTML/CSS/JavaScript.
- Asset URLs resolve only inside the WebUI static root or an enabled WebUI extension/plugin root; external URLs, traversal, and symlink escapes are excluded.
- Only valid UTF-8 HTML, CSS, and JavaScript files no larger than 40 KiB are embedded. Oversized eligible text is still scanned for referenced dependencies, while images, audio, video, fonts, manifests, and other file types remain under normal browser HTTP caching.
- Every entry carries the response content type required by browsers and native ES modules.
- Bundle versions combine the cache-policy version, bundled content, and an HTML/CSS/JavaScript path/mtime/size inventory. Policy changes and changes to both preloaded and runtime-cached eligible files therefore advance the service-worker cache version, and activation removes caches created under the old policy. The helper cache area also participates in normal extension/plugin invalidation.
- Serialized JSON is returned by the authenticated `/ui/asset-bundle` endpoint rather than inserted into `index.html`.

## Work Guidance

- Add reference extractors here rather than teaching component or extension loaders about preloading.
- Keep API, WebSocket, navigation, authentication, and other dynamic responses outside the asset bundle.
- Preserve URL identity so native module imports, component fetches, and stylesheet resolution use normal browser semantics.

## Verification

- Run `tests/test_ui_bundler.py` and the WebUI service-worker/startup tests.
- Smoke-test a cold first load and a controlled reload; verify ordinary asset URLs remain visible while their responses come from the service-worker cache after activation.

## Child DOX Index

No child DOX files.
