import ipaddress
from pathlib import Path
from typing import Iterable

import httpx
import yaml
from pydantic import BaseModel, Field

from .config_manager import validate_config
from .config_manager.stateless_llm import OpenCodeConfig
from .service_context import ServiceContext


class OpenCodeSettingsUpdate(BaseModel):
    base_url: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    agent: str = Field(default="vtuber", min_length=1)
    workspace_directory: str = Field(default=".", min_length=1)
    timeout: float = Field(default=300, gt=0)
    keep_sessions: bool = False
    allow_tools: bool = False


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
        "base_url": config.base_url,
        "provider_id": config.provider_id,
        "model": config.model,
        "agent": config.agent,
        "workspace_directory": config.workspace_directory,
        "timeout": config.timeout,
        "keep_sessions": config.keep_sessions,
        "allow_tools": config.allow_tools,
        "has_server_password": bool(config.server_password),
    }


async def connection_payload(config: OpenCodeConfig) -> dict:
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
                "version": payload.get("version"),
                "error": None,
            }
    except (httpx.HTTPError, ValueError) as error:
        return {
            "connected": False,
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
    previous_engines = {id(context.agent_engine): context.agent_engine for context in contexts}

    for context in contexts:
        agent_config = context.character_config.agent_config.model_copy(deep=True)
        agent_config.agent_settings.basic_memory_agent.llm_provider = "opencode_llm"
        agent_config.llm_configs.opencode_llm = opencode_config.model_copy(deep=True)
        await context.init_agent(agent_config, context.character_config.persona_prompt)
        context.character_config.agent_config = agent_config
        context.config.character_config.agent_config = agent_config

    active_engines = {id(context.agent_engine) for context in contexts}
    for engine_id, engine in previous_engines.items():
        if engine_id in active_engines or engine is None or not hasattr(engine, "close"):
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
