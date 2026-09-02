import asyncio
from pathlib import Path
from typing import Iterable, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .config_manager import validate_config
from .config_manager.stateless_llm import CLIAgentConfig, OpenCodeConfig
from .executable_utils import (
    executable_environment,
    executable_version,
    resolve_executable,
)
from .opencode_settings import (
    OpenCodeSettingsUpdate,
    discover_connection_payload as opencode_connection_payload,
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
    interaction_mode: Literal["character", "coding"] = "character"
    session_id: str = ""
    new_session_title: str = Field(default="", max_length=120)
    model: str = ""
    provider: str = ""
    workspace_directory: str = Field(default=".", min_length=1)
    timeout: float = Field(default=300, gt=0)
    show_reasoning: bool = False
    reasoning_effort: Literal[
        "default", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
    ] = "default"
    allow_tools: bool = False
    permission_mode: Literal["disabled", "manual", "auto", "plan"] | None = None

    @model_validator(mode="after")
    def resolve_permission_mode(self):
        if self.permission_mode is None:
            self.permission_mode = "auto" if self.allow_tools else "disabled"
        self.allow_tools = self.permission_mode != "disabled"
        return self


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
    resolved = resolve_executable(
        config.executable,
        {"claude_code": "claude", "codex": "codex", "hermes": "hermes"}[runtime],
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
        "interaction_mode": config.interaction_mode,
        "session_id": config.session_id,
        "new_session_title": config.new_session_title,
        "model": config.model,
        "provider": config.provider,
        "workspace_directory": config.workspace_directory,
        "timeout": config.timeout,
        "show_reasoning": config.show_reasoning,
        "reasoning_effort": config.reasoning_effort,
        "allow_tools": config.allow_tools,
        "permission_mode": config.permission_mode,
    }


def _unchecked_opencode_connection() -> dict:
    return {
        "connected": False,
        "base_url": None,
        "source": None,
        "managed": False,
        "version": None,
        "path": None,
        "executable_available": False,
        "executable_version": None,
        "executable_error": None,
        "error": None,
    }


def _unchecked_cli_connection() -> dict:
    return {
        "available": False,
        "path": None,
        "version": None,
        "error": None,
    }


def _active_session_id(context: ServiceContext) -> str:
    llm = getattr(context.agent_engine, "_llm", None)
    session_id = getattr(llm, "session_id", "")
    return session_id if isinstance(session_id, str) else ""


def _active_new_session_title(context: ServiceContext) -> str:
    llm = getattr(context.agent_engine, "_llm", None)
    title = getattr(llm, "new_session_title", "")
    return title if isinstance(title, str) else ""


async def runtime_settings_payload(context: ServiceContext) -> dict:
    opencode = get_opencode_config(context)
    claude_code = _cli_config(context, "claude_code_llm")
    codex = _cli_config(context, "codex_cli_llm")
    hermes = _cli_config(context, "hermes_cli_llm")
    active = context.character_config.agent_config.agent_settings.basic_memory_agent.llm_provider
    active_session_id = _active_session_id(context)
    active_new_session_title = _active_new_session_title(context)
    return {
        "provider": active,
        "opencode": {
            "executable": opencode.executable,
            "base_url": opencode.base_url,
            "provider_id": opencode.provider_id,
            "model": opencode.model,
            "agent": opencode.agent,
            "interaction_mode": opencode.interaction_mode,
            "launch_mode": opencode.launch_mode,
            "session_id": (
                active_session_id if active == "opencode_llm" else opencode.session_id
            ),
            "new_session_title": (
                active_new_session_title
                if active == "opencode_llm"
                else opencode.new_session_title
            ),
            "workspace_directory": opencode.workspace_directory,
            "timeout": opencode.timeout,
            "keep_sessions": opencode.keep_sessions,
            "allow_tools": opencode.allow_tools,
            "permission_mode": opencode.permission_mode,
            "show_reasoning": opencode.show_reasoning,
            "has_server_password": bool(opencode.server_password),
            "connection": _unchecked_opencode_connection(),
        },
        "claude_code": {
            **_cli_payload(claude_code),
            "session_id": (
                active_session_id
                if active == "claude_code_llm"
                else claude_code.session_id
            ),
            "new_session_title": (
                active_new_session_title
                if active == "claude_code_llm"
                else claude_code.new_session_title
            ),
            "connection": _unchecked_cli_connection(),
        },
        "codex": {
            **_cli_payload(codex),
            "session_id": (
                active_session_id if active == "codex_cli_llm" else codex.session_id
            ),
            "new_session_title": (
                active_new_session_title
                if active == "codex_cli_llm"
                else codex.new_session_title
            ),
            "connection": _unchecked_cli_connection(),
        },
        "hermes": {
            **_cli_payload(hermes),
            "session_id": (
                active_session_id if active == "hermes_cli_llm" else hermes.session_id
            ),
            "new_session_title": (
                active_new_session_title
                if active == "hermes_cli_llm"
                else hermes.new_session_title
            ),
            "connection": _unchecked_cli_connection(),
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
        opencode_connection_payload(
            opencode,
            auto_start=settings.provider == "opencode_llm",
        ),
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
