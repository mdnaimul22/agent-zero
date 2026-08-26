import asyncio
import os
import aiohttp
from helpers import runtime, files

URL = "http://localhost:55510/search"
CONTAINER_NAME = "a0-searxng"


async def search(query: str):
    return await runtime.call_development_function(_search, query=query)


async def _ensure_service():
    """Starts or runs the SearXNG Docker container if not running."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Status}}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    status = stdout.decode().strip()

    if status:
        if "Up" not in status:
            start_proc = await asyncio.create_subprocess_exec("docker", "start", CONTAINER_NAME)
            await start_proc.wait()
    else:
        settings_file = files.get_abs_path("docker/run/fs/etc/searxng/settings.yml")
        cmd = [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "--restart", "unless-stopped",
            "-p", "55510:8080",
            "-e", "SEARXNG_BASE_URL=http://localhost:55510/",
        ]
        if os.path.exists(settings_file):
            cmd.extend(["-v", f"{settings_file}:/etc/searxng/settings.yml:ro"])
        cmd.append("searxng/searxng:latest")

        run_proc = await asyncio.create_subprocess_exec(*cmd)
        await run_proc.wait()

    # Wait for service readiness
    for _ in range(10):
        await asyncio.sleep(1)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
                async with session.get(URL) as resp:
                    if resp.status in (200, 400, 405):
                        return
        except Exception:
            pass


async def _search(query: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(URL, data={"q": query, "format": "json"}) as response:
                return await response.json()
    except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError):
        await _ensure_service()
        async with aiohttp.ClientSession() as session:
            async with session.post(URL, data={"q": query, "format": "json"}) as response:
                return await response.json()

