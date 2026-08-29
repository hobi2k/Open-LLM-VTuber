import asyncio
import shutil
from pathlib import Path
from typing import Iterable, Literal

import yaml
from pydantic import BaseModel, Field

from .config_manager import validate_config
from .config_manager.stateless_llm import CLIAgentConfig, OpenCodeConfig
from .opencode_settings import (
    OpenCodeSettingsUpdate,
    connection_payload as opencode_connection_payload,
    get_opencode_config,
)
from .service_context import ServiceContext


RuntimeProvider = Literal[
    "opencode_llm",
    "claude_code_llm",
    "codex_cli_llm",
    "hermes_cli_llm",
]


class CLISettingsUpdate(BaseModel):
    executable: str = Field(default="auto", min_length=1)
    launch_mode: Literal["direct", "omlx"] = "direct"
    session_id: str = ""
    model: str = ""
    provider: str = ""
    workspace_directory: str = Field(default=".", min_length=1)
    timeout: float = Field(default=300, gt=0)


class AgentRuntimeSettingsUpdate(BaseModel):
    provider: RuntimeProvider
    opencode: OpenCodeSettingsUpdate
    claude_code: CLISettingsUpdate
    codex: CLISettingsUpdate
    hermes: CLISettingsUpdate


def _cli_config(context: ServiceContext, name: RuntimeProvider) -> CLIAgentConfig:
    config = getattr(context.character_config.agent_config.llm_configs, name)
    if config is None:
        raise ValueError(f"{name} is not configured")
    return config


async def _cli_connection_payload(config: CLIAgentConfig, runtime: str) -> dict:
    configured = str(config.executable or "auto").strip()
    executable = str(Path(configured).expanduser())
    resolved = None if configured in {"", "auto"} else shutil.which(executable)
    if configured not in {"", "auto"} and not resolved and Path(executable).is_file():
        resolved = executable
    if configured in {"", "auto"}:
        resolved = shutil.which(
            {"claude_code": "claude", "codex": "codex", "hermes": "hermes"}[runtime]
        )
    workspace = Path(config.workspace_directory).expanduser()
    if not resolved:
        return {
            "available": False,
            "path": None,
            "version": None,
            "error": "Executable not found",
        }
    if not workspace.is_dir():
        return {
            "available": False,
            "path": resolved,
            "version": None,
            "error": "Workspace directory not found",
        }

    try:
        process = await asyncio.create_subprocess_exec(
            resolved,
            "--version",
            cwd=str(workspace),
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
            "version": output.splitlines()[0] if output else None,
            "error": None,
        }
    except (OSError, asyncio.TimeoutError) as error:
        if "process" in locals() and process.returncode is None:
            process.kill()
            await process.wait()
        return {
            "available": False,
            "path": resolved,
            "version": None,
            "error": str(error),
        }


def _cli_payload(config: CLIAgentConfig) -> dict:
    return {
        "executable": config.executable,
        "launch_mode": config.launch_mode,
        "session_id": config.session_id,
        "model": config.model,
        "provider": config.provider,
        "workspace_directory": config.workspace_directory,
        "timeout": config.timeout,
    }


def _active_session_id(context: ServiceContext) -> str:
    llm = getattr(context.agent_engine, "_llm", None)
    session_id = getattr(llm, "session_id", "")
    return session_id if isinstance(session_id, str) else ""


async def runtime_settings_payload(context: ServiceContext) -> dict:
    opencode = get_opencode_config(context)
    claude_code = _cli_config(context, "claude_code_llm")
    codex = _cli_config(context, "codex_cli_llm")
    hermes = _cli_config(context, "hermes_cli_llm")
    opencode_status, claude_status, codex_status, hermes_status = await asyncio.gather(
        opencode_connection_payload(opencode),
        _cli_connection_payload(claude_code, "claude_code"),
        _cli_connection_payload(codex, "codex"),
        _cli_connection_payload(hermes, "hermes"),
    )
    active = context.character_config.agent_config.agent_settings.basic_memory_agent.llm_provider
    active_session_id = _active_session_id(context)
    return {
        "provider": active,
        "opencode": {
            "base_url": opencode.base_url,
            "provider_id": opencode.provider_id,
            "model": opencode.model,
            "agent": opencode.agent,
            "launch_mode": opencode.launch_mode,
            "session_id": (
                active_session_id if active == "opencode_llm" else opencode.session_id
            ),
            "workspace_directory": opencode.workspace_directory,
            "timeout": opencode.timeout,
            "keep_sessions": opencode.keep_sessions,
            "allow_tools": opencode.allow_tools,
            "has_server_password": bool(opencode.server_password),
            "connection": opencode_status,
        },
        "claude_code": {
            **_cli_payload(claude_code),
            "session_id": (
                active_session_id
                if active == "claude_code_llm"
                else claude_code.session_id
            ),
            "connection": claude_status,
        },
        "codex": {
            **_cli_payload(codex),
            "session_id": (
                active_session_id if active == "codex_cli_llm" else codex.session_id
            ),
            "connection": codex_status,
        },
        "hermes": {
            **_cli_payload(hermes),
            "session_id": (
                active_session_id if active == "hermes_cli_llm" else hermes.session_id
            ),
            "connection": hermes_status,
        },
    }


async def runtime_connection_payload(
    context: ServiceContext,
    settings: AgentRuntimeSettingsUpdate,
) -> dict:
    current_opencode = get_opencode_config(context)
    opencode = OpenCodeConfig(
        **settings.opencode.model_dump(),
        interrupt_method=current_opencode.interrupt_method,
        server_username=current_opencode.server_username,
        server_password=current_opencode.server_password,
    )
    statuses = await asyncio.gather(
        opencode_connection_payload(opencode),
        _cli_connection_payload(
            CLIAgentConfig(**settings.claude_code.model_dump()), "claude_code"
        ),
        _cli_connection_payload(CLIAgentConfig(**settings.codex.model_dump()), "codex"),
        _cli_connection_payload(
            CLIAgentConfig(**settings.hermes.model_dump()), "hermes"
        ),
    )
    return dict(
        zip(("opencode", "claude_code", "codex", "hermes"), statuses, strict=True)
    )


async def apply_runtime_settings(
    default_context: ServiceContext,
    client_contexts: Iterable[ServiceContext],
    settings: AgentRuntimeSettingsUpdate,
    config_path: str | Path = "conf.yaml",
) -> None:
    current_opencode = get_opencode_config(default_context)
    opencode = OpenCodeConfig(
        **settings.opencode.model_dump(),
        interrupt_method=current_opencode.interrupt_method,
        server_username=current_opencode.server_username,
        server_password=current_opencode.server_password,
    )
    cli_configs = {
        "claude_code_llm": CLIAgentConfig(**settings.claude_code.model_dump()),
        "codex_cli_llm": CLIAgentConfig(**settings.codex.model_dump()),
        "hermes_cli_llm": CLIAgentConfig(**settings.hermes.model_dump()),
    }
    contexts = [*client_contexts, default_context]
    previous_engines = {
        id(context.agent_engine): context.agent_engine for context in contexts
    }

    for context in contexts:
        agent_config = context.character_config.agent_config.model_copy(deep=True)
        agent_config.agent_settings.basic_memory_agent.llm_provider = settings.provider
        agent_config.llm_configs.opencode_llm = opencode.model_copy(deep=True)
        for name, config in cli_configs.items():
            setattr(agent_config.llm_configs, name, config.model_copy(deep=True))
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

    persist_runtime_settings(settings, config_path)


def persist_runtime_settings(
    settings: AgentRuntimeSettingsUpdate, config_path: str | Path
) -> None:
    path = Path(config_path)
    config_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    agent_config = config_data["character_config"]["agent_config"]
    agent_config["agent_settings"]["basic_memory_agent"]["llm_provider"] = (
        settings.provider
    )
    llm_configs = agent_config["llm_configs"]
    llm_configs.setdefault("opencode_llm", {}).update(settings.opencode.model_dump())
    llm_configs["claude_code_llm"] = settings.claude_code.model_dump()
    llm_configs["codex_cli_llm"] = settings.codex.model_dump()
    llm_configs["hermes_cli_llm"] = settings.hermes.model_dump()
    validate_config(config_data)

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)
