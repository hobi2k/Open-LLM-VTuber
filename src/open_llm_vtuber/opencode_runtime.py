"""Discover and manage a local OpenCode server for the VTuber runtime."""

import asyncio
import atexit
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config_manager.stateless_llm import OpenCodeConfig
from .executable_utils import executable_environment


_LISTEN_URL = re.compile(r"https?://(?:127\.0\.0\.1|localhost|\[::1\]):\d+")
_managed_process: asyncio.subprocess.Process | None = None
_managed_output_task: asyncio.Task | None = None
_managed_url: str | None = None
_managed_lock = asyncio.Lock()


async def discover_or_start_opencode(
    config: OpenCodeConfig,
    executable: str | None,
    *,
    auto_start: bool,
) -> dict:
    """Find an accessible OpenCode server, starting a local companion if needed."""
    configured = config.base_url.rstrip("/")
    configured_health = await _health(configured, config)
    if configured_health:
        return _connection(configured, configured_health, "configured", False)

    if _is_loopback_url(configured):
        for base_url in await _listener_urls():
            if base_url == configured:
                continue
            health = await _health(base_url, config, include_config_auth=False)
            if health:
                return _connection(base_url, health, "detected", False)

    if not auto_start or not executable or not _is_loopback_url(configured):
        return {
            "connected": False,
            "base_url": configured,
            "source": None,
            "managed": False,
            "version": None,
            "error": "No accessible OpenCode server was found",
        }

    try:
        base_url = await _ensure_managed_server(
            executable,
            config.workspace_directory,
        )
        health = await _health(base_url, config, include_config_auth=False)
        if not health:
            raise RuntimeError("Managed OpenCode server did not pass its health check")
        return _connection(base_url, health, "managed", True)
    except (OSError, RuntimeError, asyncio.TimeoutError) as error:
        return {
            "connected": False,
            "base_url": configured,
            "source": None,
            "managed": False,
            "version": None,
            "error": str(error),
        }


async def _health(
    base_url: str,
    config: OpenCodeConfig,
    *,
    include_config_auth: bool = True,
) -> dict | None:
    auth = None
    if include_config_auth and config.server_username and config.server_password:
        auth = (config.server_username, config.server_password)
    try:
        async with httpx.AsyncClient(auth=auth, timeout=min(config.timeout, 3)) as client:
            response = await client.get(
                f"{base_url}/global/health",
                params={"directory": config.workspace_directory},
            )
            response.raise_for_status()
            payload = response.json()
            return payload if payload.get("healthy") is True else None
    except (httpx.HTTPError, ValueError):
        return None


def _connection(base_url: str, health: dict, source: str, managed: bool) -> dict:
    return {
        "connected": True,
        "base_url": base_url,
        "source": source,
        "managed": managed,
        "version": health.get("version"),
        "error": None,
    }


def _is_loopback_url(base_url: str) -> bool:
    try:
        return urlparse(base_url).hostname in {"127.0.0.1", "localhost", "::1"}
    except ValueError:
        return False


async def _listener_urls() -> list[str]:
    lsof = shutil.which("lsof") or "/usr/sbin/lsof"
    if not Path(lsof).is_file():
        return []
    try:
        process = await asyncio.create_subprocess_exec(
            lsof,
            "-nP",
            "-iTCP",
            "-sTCP:LISTEN",
            "-Fpcn",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3)
    except (OSError, asyncio.TimeoutError):
        if "process" in locals() and process.returncode is None:
            process.kill()
            await process.wait()
        return []
    if process.returncode != 0:
        return []
    return _parse_listener_urls(stdout.decode("utf-8", errors="replace"))


def _parse_listener_urls(output: str) -> list[str]:
    command = ""
    urls = []
    for line in output.splitlines():
        if line.startswith("p"):
            command = ""
            continue
        if line.startswith("c"):
            command = line[1:].lower()
            continue
        if not line.startswith("n") or "opencode" not in command:
            continue
        address = line[1:]
        match = re.search(r":(\d+)$", address)
        if not match:
            continue
        host = address[: match.start()]
        if host not in {"127.0.0.1", "localhost", "::1", "[::1]", "*", "0.0.0.0"}:
            continue
        urls.append(f"http://127.0.0.1:{match.group(1)}")
    return list(dict.fromkeys(urls))


async def _ensure_managed_server(executable: str, workspace_directory: str) -> str:
    global _managed_process, _managed_output_task, _managed_url

    async with _managed_lock:
        if (
            _managed_process
            and _managed_process.returncode is None
            and _managed_url
        ):
            return _managed_url

        workspace = Path(workspace_directory).expanduser()
        environment = executable_environment()
        environment.pop("OPENCODE_SERVER_PASSWORD", None)
        _managed_process = await asyncio.create_subprocess_exec(
            executable,
            "serve",
            "--hostname",
            "127.0.0.1",
            "--port",
            "0",
            cwd=str(workspace if workspace.is_dir() else Path.home()),
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        _managed_url = await _read_server_url(_managed_process)
        _managed_output_task = asyncio.create_task(
            _drain_output(_managed_process.stdout)
        )
        return _managed_url


async def stop_managed_opencode() -> None:
    global _managed_process, _managed_output_task, _managed_url

    async with _managed_lock:
        process = _managed_process
        output_task = _managed_output_task
        _managed_process = None
        _managed_output_task = None
        _managed_url = None
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if output_task:
            output_task.cancel()


async def _read_server_url(process: asyncio.subprocess.Process) -> str:
    if not process.stdout:
        raise RuntimeError("Managed OpenCode server has no output stream")
    deadline = asyncio.get_running_loop().time() + 15
    output = []
    while asyncio.get_running_loop().time() < deadline:
        remaining = deadline - asyncio.get_running_loop().time()
        line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
        if not line:
            break
        text = line.decode("utf-8", errors="replace").strip()
        output.append(text)
        match = _LISTEN_URL.search(text)
        if match:
            return match.group(0).replace("localhost", "127.0.0.1")
    if process.returncode is None:
        process.terminate()
        await process.wait()
    detail = "\n".join(output[-5:])
    raise RuntimeError(detail or "Managed OpenCode server did not report its address")


async def _drain_output(stream: asyncio.StreamReader | None) -> None:
    if not stream:
        return
    while await stream.readline():
        pass


def _stop_managed_server() -> None:
    if _managed_process and _managed_process.returncode is None:
        _managed_process.terminate()


atexit.register(_stop_managed_server)
