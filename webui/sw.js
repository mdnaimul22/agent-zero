const CACHE_PREFIX = "agent-zero-ui-assets-";
const SCRIPT_VERSION = new URL(self.location.href).searchParams.get("version") || "runtime";
const MAX_CACHEABLE_FILE_BYTES = 40 * 1024;
const CACHEABLE_FILE_PATTERN = /\.(?:css|html?|xhtml|m?js)$/i;

let activeCacheName = cacheName(SCRIPT_VERSION);
let activeBundleEntries = new Map();
let cachePopulation = { name: "", promise: Promise.resolve() };

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      cleanupCaches(activeCacheName),
      self.clients.claim(),
    ]),
  );
});

self.addEventListener("message", (event) => {
  if (event.data?.type !== "preload-ui-bundle") return;
  const bundle = event.data.bundle;
  if (!bundle?.version || !bundle.files || typeof bundle.files !== "object") return;

  activeCacheName = cacheName(bundle.version);
  activeBundleEntries = bundleEntries(bundle.files);

  if (cachePopulation.name !== activeCacheName) {
    const targetCacheName = activeCacheName;
    const promise = preloadBundle(bundle, targetCacheName).catch((error) => {
      if (cachePopulation.promise === promise) {
        cachePopulation = { name: "", promise: Promise.resolve() };
      }
      throw error;
    });
    cachePopulation = { name: targetCacheName, promise };
  }

  event.waitUntil(cachePopulation.promise);
});

self.addEventListener("fetch", (event) => {
  if (!isCacheableRequest(event.request)) return;

  const bundledEntry = activeBundleEntries.get(event.request.url);
  if (bundledEntry) {
    event.respondWith(Promise.resolve(responseFromEntry(bundledEntry)));
    return;
  }

  let cacheWrite = Promise.resolve();
  const response = (async () => {
    const cache = await caches.open(activeCacheName);
    const cached = await cache.match(event.request);
    if (cached) return cached;

    const networkResponse = await fetch(event.request);
    if (isCacheableResponse(networkResponse)) {
      cacheWrite = cacheRuntimeResponse(
        cache,
        event.request,
        networkResponse.clone(),
      );
    }
    return networkResponse;
  })();

  event.respondWith(response);
  event.waitUntil(response.then(() => cacheWrite).catch(() => undefined));
});

function cacheName(version) {
  const safeVersion = String(version).replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 64);
  return `${CACHE_PREFIX}${safeVersion || "runtime"}`;
}

async function preloadBundle(bundle, targetCacheName) {
  await cleanupCaches(targetCacheName);
  const cache = await caches.open(targetCacheName);
  const marker = cacheMarkerRequest(targetCacheName);
  if (await cache.match(marker)) return;

  await Promise.all(
    Object.entries(bundle.files).map(async ([url, entry]) => {
      const response = responseFromEntry(entry);
      if (!response) return;
      const request = new Request(new URL(url, self.location.origin), {
        credentials: "same-origin",
      });
      await cache.put(request, response);
    }),
  );
  await cache.put(marker, new Response(bundle.version));
}

async function cleanupCaches(targetCacheName) {
  const names = await caches.keys();
  await Promise.all(
    names
      .filter((name) => name.startsWith(CACHE_PREFIX) && name !== targetCacheName)
      .map((name) => caches.delete(name)),
  );
}

function bundleEntries(files) {
  const entries = new Map();
  for (const [url, entry] of Object.entries(files)) {
    try {
      const absoluteUrl = new URL(url, self.location.origin);
      if (absoluteUrl.origin === self.location.origin && isBundleEntry(entry)) {
        entries.set(absoluteUrl.href, entry);
      }
    } catch (_error) {
      continue;
    }
  }
  return entries;
}

function isBundleEntry(entry) {
  return (
    Array.isArray(entry) &&
    entry.length === 3 &&
    entry[1] === "text" &&
    typeof entry[2] === "string"
  );
}

function responseFromEntry(entry) {
  if (!isBundleEntry(entry)) return null;
  const [contentType, _encoding, content] = entry;
  return new Response(content, {
    headers: {
      "Content-Type": contentType || "application/octet-stream",
      "X-Agent-Zero-Cache": "preloaded",
    },
  });
}

function cacheMarkerRequest(targetCacheName) {
  return new Request(
    new URL(`/.agent-zero-cache/${encodeURIComponent(targetCacheName)}`, self.location.origin),
  );
}

function isCacheableRequest(request) {
  if (request.method !== "GET" || request.headers.has("range")) return false;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || request.mode === "navigate") return false;
  if (
    url.pathname === "/" ||
    url.pathname === "/login" ||
    url.pathname === "/logout" ||
    url.pathname.startsWith("/api/") ||
    url.pathname.startsWith("/ws") ||
    url.pathname.startsWith("/socket.io/") ||
    url.pathname.startsWith("/mcp/") ||
    url.pathname.startsWith("/a2a/")
  ) {
    return false;
  }
  return CACHEABLE_FILE_PATTERN.test(url.pathname);
}

function isCacheableResponse(response) {
  return response.ok && (response.type === "basic" || response.type === "default");
}

async function cacheRuntimeResponse(cache, request, response) {
  const contentLength = response.headers.get("content-length");
  if (contentLength !== null) {
    const size = Number(contentLength);
    if (!Number.isFinite(size) || size > MAX_CACHEABLE_FILE_BYTES) return;
  } else {
    const body = await response.clone().arrayBuffer();
    if (body.byteLength > MAX_CACHEABLE_FILE_BYTES) return;
  }
  await cache.put(request, response);
}
