import asyncio
import ipaddress
from pathlib import Path
from typing import Iterable, Literal

import httpx
import yaml
from pydantic import BaseModel, Field

from .config_manager import validate_config
from .config_manager.stateless_llm import OpenCodeConfig
from .executable_utils import (
    executable_environment,
    executable_version,
    resolve_executable,
)
from .opencode_runtime import discover_or_start_opencode
from .service_context import ServiceContext


class OpenCodeSettingsUpdate(BaseModel):
    executable: str = Field(default="auto", min_length=1)
    base_url: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    agent: str = Field(default="vtuber", min_length=1)
    interaction_mode: Literal["character", "coding"] = "character"
    launch_mode: Literal["direct", "omlx"] = "direct"
    session_id: str = ""
    workspace_directory: str = Field(default=".", min_length=1)
    timeout: float = Field(default=300, gt=0)
    keep_sessions: bool = False
    allow_tools: bool = False
    show_reasoning: bool = False


def require_loopback_client(host: str | None) -> None:
    if host == "localhost":
        return
    if host is None or not ipaddress.ip_address(host).is_loopback:
        raise PermissionError("OpenCode settings are only available from this computer")


def get_opencode_config(context: ServiceContext) -> OpenCodeConfig:
    config = context.character_config.agent_config.llm_configs.opencode_llm
    if config is None:
        raise ValueError("OpenCode is not configured")
    return config


def settings_payload(context: ServiceContext) -> dict:
    config = get_opencode_config(context)
    provider = context.character_config.agent_config.agent_settings.basic_memory_agent
    return {
        "enabled": provider.llm_provider == "opencode_llm",
        "executable": config.executable,
        "base_url": config.base_url,
        "provider_id": config.provider_id,
        "model": config.model,
        "agent": config.agent,
        "interaction_mode": config.interaction_mode,
        "launch_mode": config.launch_mode,
        "session_id": config.session_id,
        "workspace_directory": config.workspace_directory,
        "timeout": config.timeout,
        "keep_sessions": config.keep_sessions,
        "allow_tools": config.allow_tools,
        "show_reasoning": config.show_reasoning,
        "has_server_password": bool(config.server_password),
    }


async def connection_payload(config: OpenCodeConfig) -> dict:
    executable = await opencode_executable_payload(config)
    auth = None
    if config.server_username and config.server_password:
        auth = (config.server_username, config.server_password)

    try:
        async with httpx.AsyncClient(
            auth=auth,
            timeout=min(config.timeout, 5),
        ) as client:
            response = await client.get(
                f"{config.base_url.rstrip('/')}/global/health",
                params={"directory": config.workspace_directory},
            )
            response.raise_for_status()
            payload = response.json()
            return {
                "connected": payload.get("healthy") is True,
                "base_url": config.base_url.rstrip("/"),
                "source": "configured",
                "managed": False,
                "version": payload.get("version"),
                "path": executable["path"],
                "executable_available": executable["available"],
                "executable_version": executable["version"],
                "executable_error": executable["error"],
                "error": None,
            }
    except (httpx.HTTPError, ValueError) as error:
        return {
            "connected": False,
            "base_url": config.base_url.rstrip("/"),
            "source": None,
            "managed": False,
            "version": None,
            "path": executable["path"],
            "executable_available": executable["available"],
            "executable_version": executable["version"],
            "executable_error": executable["error"],
            "error": str(error),
        }


async def discover_connection_payload(
    config: OpenCodeConfig,
    *,
    auto_start: bool,
) -> dict:
    executable = await opencode_executable_payload(config)
    runtime = await discover_or_start_opencode(
        config,
        executable["path"],
        auto_start=auto_start,
    )
    return {
        **runtime,
        "path": executable["path"],
        "executable_available": executable["available"],
        "executable_version": executable["version"],
        "executable_error": executable["error"],
    }


async def opencode_executable_payload(config: OpenCodeConfig) -> dict:
    resolved = resolve_executable(config.executable, "opencode")
    if not resolved:
        return {
            "available": False,
            "path": None,
            "version": None,
            "error": "Executable not found",
        }

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            resolved,
            "--version",
            cwd=str(Path(config.workspace_directory).expanduser()),
            env=executable_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        output = stdout.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            return {
                "available": False,
                "path": resolved,
                "version": None,
                "error": output or "Version check failed",
            }
        return {
            "available": True,
            "path": resolved,
            "version": executable_version(output),
            "error": None,
        }
    except (OSError, asyncio.TimeoutError) as error:
        if process and process.returncode is None:
            process.kill()
            await process.wait()
        return {
            "available": False,
            "path": resolved,
            "version": None,
            "error": str(error),
        }


async def apply_opencode_settings(
    default_context: ServiceContext,
    client_contexts: Iterable[ServiceContext],
    settings: OpenCodeSettingsUpdate,
    config_path: str | Path = "conf.yaml",
) -> None:
    current = get_opencode_config(default_context)
    opencode_config = OpenCodeConfig(
        **settings.model_dump(),
        interrupt_method=current.interrupt_method,
        server_username=current.server_username,
        server_password=current.server_password,
    )
    contexts = [*client_contexts, default_context]
    previous_engines = {
        id(context.agent_engine): context.agent_engine for context in contexts
    }

    for context in contexts:
        agent_config = context.character_config.agent_config.model_copy(deep=True)
        agent_config.agent_settings.basic_memory_agent.llm_provider = "opencode_llm"
        agent_config.llm_configs.opencode_llm = opencode_config.model_copy(deep=True)
        await context.init_agent(agent_config, context.character_config.persona_prompt)
        context.character_config.agent_config = agent_config
        context.config.character_config.agent_config = agent_config

    active_engines = {id(context.agent_engine) for context in contexts}
    for engine_id, engine in previous_engines.items():
        if (
            engine_id in active_engines
            or engine is None
            or not hasattr(engine, "close")
        ):
            continue
        await engine.close()

    persist_opencode_settings(settings, config_path)


def persist_opencode_settings(
    settings: OpenCodeSettingsUpdate,
    config_path: str | Path,
) -> None:
    path = Path(config_path)
    config_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    agent_config = config_data["character_config"]["agent_config"]
    agent_config["agent_settings"]["basic_memory_agent"]["llm_provider"] = (
        "opencode_llm"
    )
    opencode_config = agent_config["llm_configs"].setdefault("opencode_llm", {})
    opencode_config.update(settings.model_dump())
    validate_config(config_data)

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)
