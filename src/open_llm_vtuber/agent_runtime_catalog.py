import asyncio
import json
import shutil
import sqlite3
from contextlib import suppress
from pathlib import Path

import httpx

from .agent_runtime_settings import _cli_config, _cli_connection_payload
from .config_manager.stateless_llm import OpenCodeConfig
from .opencode_settings import get_opencode_config, opencode_executable_payload
from .service_context import ServiceContext


async def runtime_catalog_payload(context: ServiceContext) -> dict:
    opencode = get_opencode_config(context)
    claude_code = _cli_config(context, "claude_code_llm")
    codex = _cli_config(context, "codex_cli_llm")
    hermes = _cli_config(context, "hermes_cli_llm")
    (
        opencode_catalog,
        omlx,
        local,
        opencode_status,
        claude_status,
        codex_status,
        hermes_status,
    ) = await asyncio.gather(
        _opencode_catalog(opencode),
        _omlx_catalog(),
        asyncio.to_thread(_local_catalog),
        opencode_executable_payload(opencode),
        _cli_connection_payload(claude_code, "claude_code"),
        _cli_connection_payload(codex, "codex"),
        _cli_connection_payload(hermes, "hermes"),
    )

    opencode_models = opencode_catalog["models"]
    if not opencode_models:
        opencode_models = [{**model, "provider": "omlx"} for model in omlx["models"]]
    local["models"]["opencode"] = opencode_models
    local["models"]["hermes"] = _merge_models(
        local["models"]["hermes"],
        [{**model, "provider": "omlx"} for model in omlx["models"] if omlx["base_url"]],
    )
    local["sessions"]["opencode"] = opencode_catalog["sessions"]

    configured_projects = [
        _project(config.workspace_directory, "VTuber")
        for config in (opencode, claude_code, codex, hermes)
    ]
    return {
        "executables": {
            "opencode": opencode_status,
            "claude_code": claude_status,
            "codex": codex_status,
            "hermes": hermes_status,
            "omlx": {
                "available": omlx["available"],
                "path": omlx["path"],
                "version": omlx["version"],
                "error": omlx["error"],
            },
        },
        "omlx": omlx,
        "models": local["models"],
        "projects": _merge_projects(
            opencode_catalog["projects"],
            local["projects"],
            configured_projects,
        ),
        "sessions": local["sessions"],
    }


async def _opencode_catalog(config: OpenCodeConfig) -> dict:
    result = {"models": [], "projects": [], "sessions": []}
    auth = None
    if config.server_username and config.server_password:
        auth = (config.server_username, config.server_password)
    try:
        async with httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            auth=auth,
            timeout=min(config.timeout, 3),
        ) as client:
            provider_response, project_response = await asyncio.gather(
                client.get(
                    "/provider",
                    params={"directory": config.workspace_directory},
                ),
                client.get(
                    "/project",
                    params={"directory": config.workspace_directory},
                ),
            )
            provider_response.raise_for_status()
            project_response.raise_for_status()
            projects = project_response.json()
            directories = list(
                dict.fromkeys(
                    [
                        config.workspace_directory,
                        *(
                            str(project.get("worktree") or project.get("directory"))
                            for project in projects
                            if project.get("worktree") or project.get("directory")
                        ),
                    ]
                )
            )
            session_groups = await asyncio.gather(
                *(_opencode_sessions(client, directory) for directory in directories)
            )
    except (httpx.HTTPError, ValueError):
        return result

    providers = provider_response.json()
    connected = set(providers.get("connected", []))
    for provider in providers.get("all", []):
        provider_id = str(provider.get("id", ""))
        if not provider_id or (connected and provider_id not in connected):
            continue
        provider_models = provider.get("models", {})
        items = (
            provider_models.items()
            if isinstance(provider_models, dict)
            else ((model.get("id"), model) for model in provider_models)
        )
        for model_id, model in items:
            if not model_id:
                continue
            result["models"].append(
                {
                    "id": str(model_id),
                    "label": str(model.get("name") or model_id),
                    "provider": provider_id,
                }
            )

    for project in projects:
        path = project.get("worktree") or project.get("directory")
        if path:
            item = _project(str(path), "OpenCode")
            if project.get("name"):
                item["name"] = str(project["name"])
            result["projects"].append(item)

    sessions = {}
    for directory, session_group in zip(directories, session_groups):
        for session in session_group:
            session_id = str(session.get("id", ""))
            if not session_id:
                continue
            time = session.get("time", {})
            sessions[session_id] = {
                "id": session_id,
                "title": _session_title(session.get("title")),
                "workspace": str(session.get("directory") or directory),
                "updated_at": time.get("updated") or time.get("created"),
                "source": "opencode",
            }
    result["sessions"] = sorted(
        sessions.values(),
        key=lambda session: session.get("updated_at") or 0,
        reverse=True,
    )
    return result


async def _opencode_sessions(client: httpx.AsyncClient, directory: str) -> list:
    try:
        response = await client.get(
            "/session",
            params={"directory": directory, "limit": 10000},
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []
    except (httpx.HTTPError, ValueError):
        return []


async def _omlx_catalog() -> dict:
    executable = shutil.which("omlx")
    headers = {}
    candidates = ["http://127.0.0.1:8005/v1", "http://127.0.0.1:8000/v1"]
    settings_path = Path.home() / ".omlx/settings.json"
    with suppress(OSError, json.JSONDecodeError, TypeError):
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        server = settings.get("server", {})
        host = str(server.get("host") or "127.0.0.1")
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        port = int(server.get("port") or 8000)
        candidates.insert(0, f"http://{host}:{port}/v1")
        api_key = str(settings.get("auth", {}).get("api_key") or "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    result = {
        "available": bool(executable),
        "path": executable,
        "version": None,
        "base_url": None,
        "models": [],
        "error": None if executable else "oMLX is not installed",
    }
    if executable:
        with suppress(OSError, asyncio.TimeoutError):
            process = await asyncio.create_subprocess_exec(
                executable,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3)
            output = stdout.decode("utf-8", errors="replace").strip()
            result["version"] = output.splitlines()[0] if output else None

    async def probe(base_url: str) -> tuple[str, list] | None:
        try:
            async with httpx.AsyncClient(timeout=1, headers=headers) as client:
                response = await client.get(f"{base_url}/models")
                response.raise_for_status()
                return (
                    base_url,
                    [
                        {
                            "id": str(item["id"]),
                            "label": str(item["id"]),
                            "provider": "omlx",
                        }
                        for item in response.json().get("data", [])
                        if item.get("id")
                    ],
                )
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    probes = await asyncio.gather(*(probe(url) for url in dict.fromkeys(candidates)))
    found = next((item for item in probes if item), None)
    if found:
        result["base_url"], result["models"] = found
        result["available"] = True
        result["error"] = None
    elif executable:
        result["error"] = "oMLX is installed, but its model server is not running"
    return result


def _local_catalog() -> dict:
    codex_sessions = _codex_sessions()
    hermes_sessions = _hermes_sessions()
    claude_sessions = _claude_sessions()
    return {
        "models": {
            "claude_code": [
                {"id": "", "label": "Claude default", "provider": "anthropic"},
                {"id": "sonnet", "label": "Claude Sonnet", "provider": "anthropic"},
                {"id": "opus", "label": "Claude Opus", "provider": "anthropic"},
                {"id": "haiku", "label": "Claude Haiku", "provider": "anthropic"},
            ],
            "codex": _codex_models(),
            "hermes": _hermes_models(),
        },
        "projects": _merge_projects(
            [
                _project(session["workspace"], "Recent")
                for session in [
                    *codex_sessions,
                    *hermes_sessions,
                    *claude_sessions,
                ]
                if session["workspace"]
            ]
        ),
        "sessions": {
            "claude_code": claude_sessions,
            "codex": codex_sessions,
            "hermes": hermes_sessions,
        },
    }


def _codex_models() -> list[dict]:
    path = Path.home() / ".codex/models_cache.json"
    with suppress(OSError, json.JSONDecodeError, TypeError):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            {
                "id": str(model["slug"]),
                "label": str(model.get("display_name") or model["slug"]),
                "provider": "openai",
            }
            for model in payload.get("models", [])
            if model.get("slug") and model.get("visibility") != "hide"
        ]
    return []


def _hermes_models() -> list[dict]:
    path = Path.home() / ".hermes/provider_models_cache.json"
    models = []
    with suppress(OSError, json.JSONDecodeError, TypeError):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for provider, data in payload.items():
            models.extend(
                {
                    "id": str(model),
                    "label": str(model),
                    "provider": str(provider),
                }
                for model in data.get("models", [])
            )
    return models


def _codex_sessions(home: Path | None = None) -> list[dict]:
    path = (home or Path.home()) / ".codex/state_5.sqlite"
    return _sqlite_sessions(
        path,
        "SELECT id, COALESCE(NULLIF(title, ''), NULLIF(name, ''), "
        "NULLIF(first_user_message, ''), 'Untitled conversation'), cwd, "
        "updated_at, 'codex' FROM threads WHERE archived = 0 "
        "ORDER BY updated_at DESC",
    )


def _hermes_sessions(home: Path | None = None) -> list[dict]:
    path = (home or Path.home()) / ".hermes/state.db"
    return _sqlite_sessions(
        path,
        "SELECT sessions.id, COALESCE(NULLIF(title, ''), "
        "NULLIF(display_name, ''), NULLIF(SUBSTR((SELECT content FROM messages "
        "WHERE messages.session_id = sessions.id AND role = 'user' AND active = 1 "
        "AND NULLIF(content, '') IS NOT NULL ORDER BY timestamp LIMIT 1), 1, 100), ''), "
        "'Untitled conversation'), COALESCE(NULLIF(cwd, ''), "
        "NULLIF(git_repo_root, ''), ''), COALESCE(last_activity_at, started_at), "
        "source FROM sessions WHERE archived = 0 AND hidden = 0 "
        "AND message_count > 0 AND source != 'subagent' ORDER BY "
        "COALESCE(last_activity_at, started_at) DESC",
    )


def _sqlite_sessions(path: Path, query: str) -> list[dict]:
    if not path.is_file():
        return []
    with suppress(sqlite3.Error):
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return [
                {
                    "id": row[0],
                    "title": _session_title(row[1]),
                    "workspace": row[2],
                    "updated_at": row[3],
                    "source": row[4] if len(row) > 4 else "local",
                }
                for row in connection.execute(query).fetchall()
            ]
        finally:
            connection.close()
    return []


def _claude_sessions(home: Path | None = None) -> list[dict]:
    root = (home or Path.home()) / ".claude/projects"
    if not root.is_dir():
        return []
    paths = list(root.glob("*/*.jsonl"))
    paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    sessions = []
    for path in paths:
        title = "Untitled conversation"
        workspace = ""
        session_id = path.stem
        with suppress(OSError, json.JSONDecodeError):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    session_id = str(event.get("sessionId") or session_id)
                    workspace = str(event.get("cwd") or workspace)
                    message = event.get("message", {})
                    if event.get("type") != "user" or not isinstance(message, dict):
                        continue
                    content = message.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            str(item.get("text", ""))
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                    if str(content).strip():
                        title = str(content).strip().replace("\n", " ")[:100]
                        break
        sessions.append(
            {
                "id": session_id,
                "title": title,
                "workspace": workspace,
                "updated_at": path.stat().st_mtime,
                "source": "claude_code",
            }
        )
    return sessions


def _project(path: str, source: str) -> dict:
    resolved = str(Path(path).expanduser().resolve())
    return {
        "name": Path(resolved).name or resolved,
        "path": resolved,
        "source": source,
    }


def _session_title(value) -> str:
    return " ".join(str(value or "").split())[:120] or "Untitled conversation"


def _merge_models(*groups: list[dict]) -> list[dict]:
    result = {}
    for model in (item for group in groups for item in group):
        result[(model.get("provider", ""), model.get("id", ""))] = model
    return list(result.values())


def _merge_projects(*groups: list[dict]) -> list[dict]:
    result = {}
    for project in (item for group in groups for item in group):
        path = str(project.get("path", ""))
        if path:
            result[path] = project
    return list(result.values())
