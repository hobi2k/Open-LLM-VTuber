import asyncio
import json
import sqlite3
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, field_validator

from .agent_runtime_settings import _cli_config
from .executable_utils import executable_environment, resolve_executable
from .opencode_settings import get_opencode_config
from .service_context import ServiceContext


RuntimeCatalogKey = Literal["opencode", "claude_code", "codex", "hermes"]


class SessionRenameRequest(BaseModel):
    runtime: RuntimeCatalogKey
    session_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=120)
    workspace: str = ""

    @field_validator("session_id", "title")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Value must not be empty")
        return normalized

    @field_validator("workspace")
    @classmethod
    def strip_workspace(cls, value: str) -> str:
        return value.strip()


async def rename_runtime_session(
    context: ServiceContext,
    request: SessionRenameRequest,
) -> dict:
    if request.runtime == "opencode":
        await _rename_opencode(context, request)
    elif request.runtime == "claude_code":
        renamed = await asyncio.to_thread(
            rename_local_runtime_session,
            request.runtime,
            request.session_id,
            request.title,
        )
        if not renamed:
            raise FileNotFoundError("Claude Code session not found")
    elif request.runtime == "codex":
        renamed = await asyncio.to_thread(
            rename_local_runtime_session,
            request.runtime,
            request.session_id,
            request.title,
        )
        if not renamed:
            raise FileNotFoundError("Codex session not found")
    else:
        await _rename_hermes(context, request)
    return {
        "id": request.session_id,
        "title": request.title,
        "runtime": request.runtime,
    }


def rename_local_runtime_session(
    runtime: Literal["claude_code", "codex", "hermes"],
    session_id: str,
    title: str,
) -> bool:
    rename = {
        "claude_code": _rename_claude_local,
        "codex": _rename_codex_local,
        "hermes": _rename_hermes_local,
    }[runtime]
    return rename(session_id, title)


async def _rename_opencode(
    context: ServiceContext,
    request: SessionRenameRequest,
) -> None:
    config = get_opencode_config(context)
    auth = None
    if config.server_username and config.server_password:
        auth = (config.server_username, config.server_password)
    error: Exception | None = None
    try:
        async with httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            auth=auth,
            timeout=min(config.timeout, 5),
        ) as client:
            response = await client.patch(
                f"/session/{quote(request.session_id, safe='')}",
                params={
                    "directory": request.workspace or config.workspace_directory,
                },
                json={"title": request.title},
            )
            response.raise_for_status()
            return
    except (httpx.HTTPError, ValueError) as caught:
        error = caught

    if await asyncio.to_thread(
        _rename_opencode_local,
        request.session_id,
        request.title,
    ):
        return
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 404:
        raise FileNotFoundError("OpenCode session not found") from error
    raise RuntimeError(f"OpenCode session rename failed: {error}") from error


async def _rename_hermes(
    context: ServiceContext,
    request: SessionRenameRequest,
) -> None:
    config = _cli_config(context, "hermes_cli_llm")
    executable = resolve_executable(config.executable, "hermes")
    if executable:
        workspace = Path(request.workspace or config.workspace_directory).expanduser()
        cwd = workspace if workspace.is_dir() else Path.home()
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "sessions",
                "rename",
                request.session_id,
                request.title,
                cwd=str(cwd),
                env=executable_environment(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
            output = stdout.decode("utf-8", errors="replace").strip()
            if "not found" in output.lower():
                raise FileNotFoundError("Hermes session not found")
            if process.returncode == 0:
                return
            raise RuntimeError(output or "Hermes session rename failed")
        except FileNotFoundError:
            raise
        except (OSError, asyncio.TimeoutError):
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

    if await asyncio.to_thread(
        _rename_hermes_local,
        request.session_id,
        request.title,
    ):
        return
    raise FileNotFoundError("Hermes session not found")


def _rename_opencode_local(
    session_id: str,
    title: str,
    home: Path | None = None,
) -> bool:
    root = (home or Path.home()) / ".local/share/opencode"
    renamed = False
    for path in root.glob("opencode*.db"):
        with suppress(sqlite3.Error):
            connection = sqlite3.connect(path, timeout=3)
            try:
                cursor = connection.execute(
                    "UPDATE session SET title = ?, time_updated = ? WHERE id = ?",
                    (title, int(time.time() * 1000), session_id),
                )
                connection.commit()
                renamed = cursor.rowcount > 0 or renamed
            finally:
                connection.close()
    return renamed


def _rename_codex_local(
    session_id: str,
    title: str,
    home: Path | None = None,
) -> bool:
    root = (home or Path.home()) / ".codex"
    renamed = False
    path = root / "state_5.sqlite"
    if path.is_file():
        with suppress(sqlite3.Error):
            connection = sqlite3.connect(path, timeout=3)
            try:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(threads)")
                }
                field = "name" if "name" in columns else "title"
                cursor = connection.execute(
                    f"UPDATE threads SET {field} = ? WHERE id = ?",
                    (title, session_id),
                )
                connection.commit()
                renamed = cursor.rowcount > 0
            finally:
                connection.close()

    for catalog_path in (root / "sqlite").glob("codex*.db"):
        with suppress(sqlite3.Error):
            connection = sqlite3.connect(catalog_path, timeout=3)
            try:
                cursor = connection.execute(
                    "UPDATE local_thread_catalog SET display_title = ?, "
                    "pending_observed_title = 0 WHERE thread_id = ?",
                    (title, session_id),
                )
                connection.commit()
                renamed = cursor.rowcount > 0 or renamed
            finally:
                connection.close()
    return renamed


def _rename_claude_local(
    session_id: str,
    title: str,
    home: Path | None = None,
) -> bool:
    root = (home or Path.home()) / ".claude/projects"
    if not root.is_dir():
        return False
    paths = list(root.glob(f"*/{session_id}.jsonl"))
    if not paths:
        paths = [
            path
            for path in root.glob("*/*.jsonl")
            if _claude_session_id(path) == session_id
        ]
    if not paths:
        return False

    sidecar = paths[0].with_suffix("") / "custom-title.json"
    payload = {}
    with suppress(OSError, json.JSONDecodeError, TypeError):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["customTitle"] = title
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(sidecar)
    return True


def _claude_session_id(path: Path) -> str:
    with suppress(OSError, json.JSONDecodeError):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                session_id = json.loads(line).get("sessionId")
                if session_id:
                    return str(session_id)
    return path.stem


def _rename_hermes_local(
    session_id: str,
    title: str,
    home: Path | None = None,
) -> bool:
    path = (home or Path.home()) / ".hermes/state.db"
    if not path.is_file():
        return False
    with suppress(sqlite3.Error):
        connection = sqlite3.connect(path, timeout=3)
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sessions)")
            }
            query = "UPDATE sessions SET title = ? WHERE id = ?"
            values = (title, session_id)
            if "title_source" in columns:
                query = (
                    "UPDATE sessions SET title = ?, title_source = 'manual' "
                    "WHERE id = ?"
                )
            cursor = connection.execute(query, values)
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()
    return False
