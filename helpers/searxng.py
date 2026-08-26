import asyncio
import os
import shutil
import aiohttp
from helpers import runtime, files
from helpers.print_style import PrintStyle

URL = "http://localhost:55510/search"
CONTAINER_NAME = "a0-searxng"
_deploy_lock = asyncio.Lock()


async def search(query: str):
    return await runtime.call_development_function(_search, query=query)


async def _check_searxng_alive() -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            async with session.get(URL) as response:
                return response.status in (200, 400, 405)
    except Exception:
        return False


async def _ensure_searxng_running() -> bool:
    async with _deploy_lock:
        if await _check_searxng_alive():
            return True

        docker_bin = shutil.which("docker")
        if not docker_bin:
            PrintStyle.error("Docker is not available to start SearXNG.")
            return False

        PrintStyle.standard("SearXNG is not running on localhost:55510. Auto-starting SearXNG container...")

        try:
            # Check if container exists
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
            status_output = stdout.decode().strip()

            if CONTAINER_NAME in status_output:
                if "Up" not in status_output:
                    PrintStyle.standard(f"Starting existing container '{CONTAINER_NAME}'...")
                    start_proc = await asyncio.create_subprocess_exec(
                        docker_bin, "start", CONTAINER_NAME,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await start_proc.communicate()
            else:
                settings_file = files.get_abs_path("docker/run/fs/etc/searxng/settings.yml")
                volume_arg = f"{settings_file}:/etc/searxng/settings.yml:ro" if os.path.exists(settings_file) else None

                run_cmd = [
                    docker_bin, "run", "-d",
                    "--name", CONTAINER_NAME,
                    "--restart", "unless-stopped",
                    "-p", "55510:8080",
                    "-e", "SEARXNG_BASE_URL=http://localhost:55510/",
                ]
                if volume_arg:
                    run_cmd.extend(["-v", volume_arg])
                run_cmd.append("searxng/searxng:latest")

                PrintStyle.standard(f"Deploying new SearXNG container '{CONTAINER_NAME}' on port 55510...")
                run_proc = await asyncio.create_subprocess_exec(
                    *run_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                _, run_err = await run_proc.communicate()
                if run_proc.returncode != 0:
                    PrintStyle.error(f"Failed to deploy SearXNG container: {run_err.decode().strip()}")
                    return False

            # Wait for SearXNG to become ready
            for _ in range(15):
                await asyncio.sleep(1)
                if await _check_searxng_alive():
                    PrintStyle.success("SearXNG is ready on localhost:55510.")
                    return True

        except Exception as e:
            PrintStyle.error(f"Error while starting SearXNG: {e}")

        return False


async def _search(query: str):
    # 1. Query SearXNG directly
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(URL, data={"q": query, "format": "json"}) as response:
                if response.status == 200:
                    return await response.json()
    except Exception:
        pass

    # 2. If SearXNG is down, auto-deploy it and retry
    started = await _ensure_searxng_running()
    if started:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.post(URL, data={"q": query, "format": "json"}) as response:
                return await response.json()

    raise ConnectionError("SearXNG is not running on localhost:55510 and could not be auto-started.")

