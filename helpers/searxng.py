import asyncio
import shutil
import aiohttp
from helpers import runtime
from helpers.print_style import PrintStyle

URL = "http://localhost:55510/search"
CONTAINER_NAME = "a0-searxng"
_deploy_lock = asyncio.Lock()


async def search(query: str):
    return await runtime.call_development_function(_search, query=query)


async def _check_searxng_alive() -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            async with session.get("http://localhost:55510/") as response:
                return response.status in (200, 302, 404, 405)
    except Exception:
        return False


async def _ensure_searxng_deployed() -> bool:
    async with _deploy_lock:
        if await _check_searxng_alive():
            return True

        docker_bin = shutil.which("docker")
        if not docker_bin:
            PrintStyle.warning("Docker CLI not found on host. Cannot self-deploy SearXNG.")
            return False

        PrintStyle.standard("SearXNG is not running on localhost:55510. Auto-deploying SearXNG via Docker...")

        try:
            # Check if container already exists
            check_cmd = [
                docker_bin, "ps", "-a",
                "--filter", f"name={CONTAINER_NAME}",
                "--format", "{{.Names}} {{.Status}}"
            ]
            proc = await asyncio.create_subprocess_exec(
                *check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode().strip()

            if CONTAINER_NAME in output:
                if "Up" not in output:
                    PrintStyle.standard(f"Starting existing stopped container '{CONTAINER_NAME}'...")
                    start_proc = await asyncio.create_subprocess_exec(
                        docker_bin, "start", CONTAINER_NAME,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await start_proc.communicate()
            else:
                PrintStyle.standard(f"Creating and launching new SearXNG container '{CONTAINER_NAME}' on port 55510...")
                run_cmd = [
                    docker_bin, "run", "-d",
                    "--name", CONTAINER_NAME,
                    "--restart", "unless-stopped",
                    "-p", "55510:8080",
                    "-e", "SEARXNG_BASE_URL=http://localhost:55510/",
                    "-e", "SEARXNG_SECRET_KEY=agent0_searxng_secret_2026",
                    "-e", "SEARXNG_ENABLE_METRICS=false",
                    "-e", "SEARXNG_LIM_ENABLED=false",
                    "searxng/searxng:latest"
                ]
                run_proc = await asyncio.create_subprocess_exec(
                    *run_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                run_stdout, run_stderr = await run_proc.communicate()
                if run_proc.returncode != 0:
                    PrintStyle.warning(f"Failed to launch SearXNG container: {run_stderr.decode().strip()}")
                    return False

            # Wait for SearXNG to become responsive
            for _ in range(15):
                await asyncio.sleep(1)
                if await _check_searxng_alive():
                    PrintStyle.success("SearXNG container is up and running!")
                    return True

        except Exception as e:
            PrintStyle.warning(f"SearXNG self-deployment encountered an error: {e}")

        return False


def _fallback_ddg(query: str) -> dict:
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        items = ddgs.text(query, max_results=10)
        results = []
        for item in items:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", item.get("url", "")),
                "content": item.get("body", item.get("content", ""))
            })
        return {"query": query, "results": results}
    except Exception as e:
        PrintStyle.warning(f"DuckDuckGo fallback search error: {e}")
        return {"query": query, "results": []}


async def _search(query: str):
    # 1. Try querying SearXNG directly
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
            async with session.post(URL, data={"q": query, "format": "json"}) as response:
                if response.status == 200:
                    return await response.json()
    except Exception:
        pass

    # 2. If failed, attempt self-deployment
    deployed = await _ensure_searxng_deployed()
    if deployed:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                async with session.post(URL, data={"q": query, "format": "json"}) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            PrintStyle.warning(f"SearXNG query failed after deployment: {e}")

    # 3. Seamless fallback to DuckDuckGo
    PrintStyle.standard("SearXNG unreachable; falling back to DuckDuckGo search.")
    return await asyncio.to_thread(_fallback_ddg, query)

