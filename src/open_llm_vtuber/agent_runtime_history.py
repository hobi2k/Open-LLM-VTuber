"""Read native coding-agent sessions into the shared chat timeline."""

import asyncio
import json
import sqlite3
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, field_validator

from .agent.stateless_llm.agent_activity import tool_activity
from .agent_runtime_sessions import RuntimeCatalogKey, _claude_session_id
from .opencode_settings import get_opencode_config
from .service_context import ServiceContext


class SessionHistoryRequest(BaseModel):
    runtime: RuntimeCatalogKey
    session_id: str = Field(min_length=1, max_length=256)
    workspace: str = ""
    limit: int = Field(default=1000, ge=1, le=2000)

    @field_validator("session_id", "workspace")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


async def runtime_session_history_payload(
    context: ServiceContext,
    request: SessionHistoryRequest,
) -> dict:
    if request.runtime == "opencode":
        messages = await _opencode_history(context, request)
    else:
        loader = {
            "claude_code": _claude_history,
            "codex": _codex_history,
            "hermes": _hermes_history,
        }[request.runtime]
        messages = await asyncio.to_thread(loader, request.session_id)

    total = len(messages)
    return {
        "runtime": request.runtime,
        "session_id": request.session_id,
        "messages": messages[-request.limit :],
        "total": total,
        "truncated": total > request.limit,
    }


async def _opencode_history(
    context: ServiceContext,
    request: SessionHistoryRequest,
) -> list[dict]:
    config = get_opencode_config(context)
    auth = None
    if config.server_username and config.server_password:
        auth = (config.server_username, config.server_password)
    with suppress(httpx.HTTPError, ValueError, TypeError):
        async with httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            auth=auth,
            timeout=min(config.timeout, 8),
        ) as client:
            response = await client.get(
                f"/session/{quote(request.session_id, safe='')}/message",
                params={
                    "directory": request.workspace or config.workspace_directory,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                messages = _opencode_messages(payload)
                if messages:
                    return messages
    return await asyncio.to_thread(_opencode_local_history, request.session_id)


def _opencode_local_history(
    session_id: str,
    home: Path | None = None,
) -> list[dict]:
    root = (home or Path.home()) / ".local/share/opencode"
    candidates = []
    for path in root.glob("opencode*.db"):
        with suppress(sqlite3.Error, json.JSONDecodeError, TypeError):
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
            try:
                rows = connection.execute(
                    "SELECT id, time_created, data FROM message "
                    "WHERE session_id = ? ORDER BY time_created, id",
                    (session_id,),
                ).fetchall()
                if not rows:
                    continue
                payload = []
                for message_id, created_at, data in rows:
                    info = json.loads(data)
                    parts = [
                        json.loads(row[0])
                        for row in connection.execute(
                            "SELECT data FROM part WHERE session_id = ? "
                            "AND message_id = ? ORDER BY time_created, id",
                            (session_id, message_id),
                        ).fetchall()
                    ]
                    payload.append(
                        {
                            "info": {**info, "id": info.get("id") or message_id},
                            "parts": parts,
                            "created_at": created_at,
                        }
                    )
            finally:
                connection.close()
            messages = _opencode_messages(payload)
            candidates.append((rows[-1][1] or 0, len(messages), messages))
    if not candidates:
        return []
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _opencode_messages(payload: list[dict]) -> list[dict]:
    messages = []
    for message_index, item in enumerate(payload):
        info = item.get("info", {})
        if not isinstance(info, dict):
            continue
        role = str(info.get("role") or "")
        source_id = str(info.get("id") or f"message-{message_index}")
        time = info.get("time", {})
        timestamp = _timestamp(
            (time.get("created") if isinstance(time, dict) else None)
            or item.get("created_at")
        )
        parts = item.get("parts", [])
        parts = parts if isinstance(parts, list) else []
        if role == "user":
            text = "\n".join(
                str(part.get("text") or "").strip()
                for part in parts
                if isinstance(part, dict)
                and part.get("type") == "text"
                and str(part.get("text") or "").strip()
            )
            _append_text(messages, "human", text, source_id, timestamp)
            continue
        if role != "assistant":
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            part_id = str(part.get("id") or f"{source_id}-{part_index}")
            part_type = part.get("type")
            if part_type == "text":
                _append_text(
                    messages,
                    "ai",
                    str(part.get("text") or ""),
                    part_id,
                    timestamp,
                )
            elif part_type == "reasoning":
                _append_reasoning(
                    messages,
                    str(part.get("text") or ""),
                    part_id,
                    timestamp,
                )
            elif part_type == "tool":
                state = part.get("state", {})
                state = state if isinstance(state, dict) else {}
                _append_activity(
                    messages,
                    tool_activity(
                        part.get("callID") or part_id,
                        str(part.get("tool") or "tool"),
                        str(state.get("status") or "running"),
                        input_data=state.get("input"),
                        title=str(state.get("title") or ""),
                        output=state.get("output"),
                        error=state.get("error"),
                        metadata=state.get("metadata") or part.get("metadata"),
                    ),
                    timestamp,
                )
    return messages


def _claude_history(
    session_id: str,
    home: Path | None = None,
) -> list[dict]:
    root = (home or Path.home()) / ".claude/projects"
    paths = list(root.glob(f"*/{session_id}.jsonl"))
    if not paths:
        paths = [path for path in root.glob("*/*.jsonl") if _claude_session_id(path) == session_id]
    if not paths:
        return []

    messages = []
    activities = {}
    with suppress(OSError):
        with paths[0].open(encoding="utf-8") as stream:
            for line_index, line in enumerate(stream):
                with suppress(json.JSONDecodeError, TypeError):
                    event = json.loads(line)
                    if event.get("isSidechain") or event.get("isMeta"):
                        continue
                    message = event.get("message", {})
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content")
                    timestamp = _timestamp(event.get("timestamp"))
                    source_id = str(
                        event.get("uuid")
                        or message.get("id")
                        or f"claude-{line_index}"
                    )
                    if event.get("type") == "user":
                        _claude_user_content(
                            messages,
                            activities,
                            content,
                            source_id,
                            timestamp,
                        )
                    elif event.get("type") == "assistant":
                        _claude_assistant_content(
                            messages,
                            activities,
                            content,
                            source_id,
                            timestamp,
                        )
    return messages


def _claude_user_content(
    messages: list[dict],
    activities: dict[str, int],
    content: Any,
    source_id: str,
    timestamp: str,
) -> None:
    if isinstance(content, str):
        _append_text(messages, "human", content, source_id, timestamp)
        return
    if not isinstance(content, list):
        return
    text = "\n".join(
        str(item.get("text") or "").strip()
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and str(item.get("text") or "").strip()
    )
    _append_text(messages, "human", text, source_id, timestamp)
    for index, item in enumerate(content):
        if not isinstance(item, dict) or item.get("type") != "tool_result":
            continue
        activity_id = str(item.get("tool_use_id") or f"{source_id}-{index}")
        previous = messages[activities[activity_id]] if activity_id in activities else {}
        activity = tool_activity(
            activity_id,
            str(previous.get("tool_name") or item.get("name") or "tool"),
            "error" if item.get("is_error") else "completed",
            input_data=previous.get("permission_input"),
            output=None if item.get("is_error") else item.get("content"),
            error=item.get("content") if item.get("is_error") else None,
        )
        _upsert_activity(messages, activities, activity, timestamp)


def _claude_assistant_content(
    messages: list[dict],
    activities: dict[str, int],
    content: Any,
    source_id: str,
    timestamp: str,
) -> None:
    if not isinstance(content, list):
        return
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or f"{source_id}-{index}")
        if item.get("type") == "text":
            _append_text(messages, "ai", str(item.get("text") or ""), item_id, timestamp)
        elif item.get("type") == "thinking":
            _append_reasoning(
                messages,
                str(item.get("thinking") or item.get("text") or ""),
                item_id,
                timestamp,
            )
        elif item.get("type") == "tool_use":
            activity = tool_activity(
                item_id,
                str(item.get("name") or "tool"),
                "running",
                input_data=item.get("input"),
            )
            _upsert_activity(messages, activities, activity, timestamp)


def _codex_history(
    session_id: str,
    home: Path | None = None,
) -> list[dict]:
    root = (home or Path.home()) / ".codex"
    path = _codex_rollout_path(root, session_id)
    if path is None:
        return []
    events = []
    with suppress(OSError):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                with suppress(json.JSONDecodeError, TypeError):
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
    completed = [
        event
        for event in events
        if event.get("type") == "event_msg"
        and event.get("payload", {}).get("type") == "item_completed"
    ]
    if completed:
        return _codex_completed_history(completed)
    return _codex_legacy_history(events)


def _codex_rollout_path(root: Path, session_id: str) -> Path | None:
    database = root / "state_5.sqlite"
    if database.is_file():
        with suppress(sqlite3.Error):
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=3)
            try:
                row = connection.execute(
                    "SELECT rollout_path FROM threads WHERE id = ?",
                    (session_id,),
                ).fetchone()
            finally:
                connection.close()
            if row and row[0]:
                path = Path(str(row[0])).expanduser().resolve()
                with suppress(ValueError):
                    path.relative_to(root.resolve())
                    if path.is_file():
                        return path
    return next(
        (
            path
            for path in root.glob("sessions/**/rollout-*.jsonl")
            if session_id in path.name
        ),
        None,
    )


def _codex_completed_history(events: list[dict]) -> list[dict]:
    messages = []
    for index, event in enumerate(events):
        payload = event.get("payload", {})
        item = payload.get("item", {}) if isinstance(payload, dict) else {}
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        source_id = str(item.get("id") or f"codex-{index}")
        timestamp = _timestamp(
            payload.get("completed_at_ms")
            or payload.get("started_at_ms")
            or event.get("timestamp")
        )
        if item_type == "UserMessage":
            _append_text(messages, "human", _structured_text(item.get("content")), source_id, timestamp)
        elif item_type == "AgentMessage":
            _append_text(messages, "ai", _structured_text(item.get("content")), source_id, timestamp)
        elif item_type == "Reasoning":
            _append_reasoning(
                messages,
                _structured_text(item.get("summary_text") or item.get("raw_content")),
                source_id,
                timestamp,
            )
        else:
            activity = _codex_activity(item, source_id)
            if activity:
                _append_activity(messages, activity, timestamp)
    return messages


def _codex_legacy_history(events: list[dict]) -> list[dict]:
    messages = []
    activities = {}
    for index, event in enumerate(events):
        if event.get("type") != "response_item":
            continue
        item = event.get("payload", {})
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id") or item.get("call_id") or f"codex-{index}")
        timestamp = _timestamp(event.get("timestamp"))
        if item.get("type") == "message" and item.get("role") in {"user", "assistant"}:
            role = "human" if item.get("role") == "user" else "ai"
            _append_text(messages, role, _structured_text(item.get("content")), source_id, timestamp)
        elif item.get("type") == "reasoning":
            _append_reasoning(messages, _structured_text(item.get("summary")), source_id, timestamp)
        elif item.get("type") in {"custom_tool_call", "function_call"}:
            activity = tool_activity(
                str(item.get("call_id") or source_id),
                str(item.get("name") or "tool"),
                str(item.get("status") or "running"),
                input_data=_json_value(item.get("input") or item.get("arguments")),
            )
            _upsert_activity(messages, activities, activity, timestamp)
        elif item.get("type") in {"custom_tool_call_output", "function_call_output"}:
            activity_id = str(item.get("call_id") or source_id)
            previous = messages[activities[activity_id]] if activity_id in activities else {}
            activity = tool_activity(
                activity_id,
                str(previous.get("tool_name") or "tool"),
                "completed",
                output=item.get("output"),
            )
            _upsert_activity(messages, activities, activity, timestamp)
    return messages


def _codex_activity(item: dict, source_id: str) -> dict | None:
    item_type = str(item.get("type") or "")
    if item_type == "CommandExecution":
        return tool_activity(
            source_id,
            "command",
            str(item.get("status") or "completed"),
            input_data={"command": _structured_text(item.get("command"))},
            output=item.get("aggregated_output") or item.get("formatted_output"),
            error=item.get("stderr") if item.get("status") == "failed" else None,
            metadata={"exit_code": item.get("exit_code")},
        )
    if item_type == "FileChange":
        changes = item.get("changes", [])
        changes = changes if isinstance(changes, list) else []
        return tool_activity(
            source_id,
            "file_change",
            str(item.get("status") or "completed"),
            input_data={
                "path": "\n".join(
                    str(change.get("path"))
                    for change in changes
                    if isinstance(change, dict) and change.get("path")
                ),
                "diff": "\n\n".join(
                    str(change.get("diff") or change.get("patch"))
                    for change in changes
                    if isinstance(change, dict) and (change.get("diff") or change.get("patch"))
                ),
            },
            output=item.get("stdout"),
            error=item.get("stderr"),
        )
    if item_type not in {"McpToolCall", "ToolCall", "WebSearch"}:
        return None
    return tool_activity(
        source_id,
        str(item.get("tool") or item.get("name") or item.get("server") or item_type),
        str(item.get("status") or "completed"),
        input_data=item.get("arguments") or item.get("input"),
        output=item.get("result") or item.get("output"),
        error=item.get("error"),
    )


def _hermes_history(
    session_id: str,
    home: Path | None = None,
) -> list[dict]:
    path = (home or Path.home()) / ".hermes/state.db"
    if not path.is_file():
        return []
    with suppress(sqlite3.Error):
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(messages)")
            }
            fields = [
                field if field in columns else f"NULL AS {field}"
                for field in (
                    "id",
                    "role",
                    "content",
                    "timestamp",
                    "reasoning_content",
                    "reasoning",
                    "tool_call_id",
                    "tool_calls",
                    "tool_name",
                )
            ]
            active = " AND active = 1" if "active" in columns else ""
            rows = connection.execute(
                f"SELECT {', '.join(fields)} FROM messages "
                f"WHERE session_id = ?{active} ORDER BY timestamp, id",
                (session_id,),
            ).fetchall()
        finally:
            connection.close()
    if "rows" not in locals():
        return []

    messages = []
    activities = {}
    activity_inputs = {}
    for row in rows:
        (
            row_id,
            role,
            content,
            timestamp_value,
            reasoning_content,
            reasoning,
            tool_call_id,
            tool_calls,
            tool_name,
        ) = row
        timestamp = _timestamp(timestamp_value)
        source_id = f"hermes-{row_id}"
        if role == "user":
            _append_text(messages, "human", _structured_text(content), source_id, timestamp)
        elif role == "assistant":
            _append_reasoning(
                messages,
                _structured_text(reasoning_content or reasoning),
                f"{source_id}-reasoning",
                timestamp,
            )
            _append_text(messages, "ai", _structured_text(content), source_id, timestamp)
            calls = _json_value(tool_calls)
            calls = calls if isinstance(calls, list) else [calls]
            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                function = function if isinstance(function, dict) else {}
                activity_id = str(call.get("call_id") or call.get("id") or f"{source_id}-{index}")
                name = str(function.get("name") or call.get("name") or "tool")
                inputs = _json_value(function.get("arguments") or call.get("arguments"))
                activity_inputs[activity_id] = (name, inputs)
                _upsert_activity(
                    messages,
                    activities,
                    tool_activity(activity_id, name, "running", input_data=inputs),
                    timestamp,
                )
        elif role == "tool" and tool_call_id:
            activity_id = str(tool_call_id)
            name, inputs = activity_inputs.get(activity_id, (str(tool_name or "tool"), {}))
            result = _json_value(content)
            failed = isinstance(result, dict) and (
                result.get("success") is False
                or result.get("error") not in {None, ""}
                or result.get("exit_code") not in {None, 0}
            )
            _upsert_activity(
                messages,
                activities,
                tool_activity(
                    activity_id,
                    name,
                    "error" if failed else "completed",
                    input_data=inputs,
                    output=None if failed else result,
                    error=result.get("error") if failed and isinstance(result, dict) else None,
                    metadata=result,
                ),
                timestamp,
            )
    return messages


def _append_text(
    messages: list[dict],
    role: str,
    content: str,
    source_id: str,
    timestamp: str,
) -> None:
    text = content.strip()
    if not text:
        return
    messages.append(
        {
            "id": f"history-{source_id}-text",
            "content": text,
            "role": role,
            "type": "text",
            "timestamp": timestamp,
        }
    )


def _append_reasoning(
    messages: list[dict],
    content: str,
    source_id: str,
    timestamp: str,
) -> None:
    text = content.strip()
    if not text:
        return
    messages.append(
        {
            "id": f"history-{source_id}-reasoning",
            "reasoning_id": str(source_id),
            "content": text,
            "role": "ai",
            "type": "reasoning",
            "status": "completed",
            "timestamp": timestamp,
        }
    )


def _append_activity(messages: list[dict], event: dict, timestamp: str) -> None:
    event = {**event, "type": "agent_activity"}
    messages.append(
        {
            "id": f"history-{event['activity_id']}-activity",
            "content": str(event.get("title") or event.get("tool_name") or "Tool"),
            "role": "ai",
            "timestamp": timestamp,
            **event,
        }
    )


def _upsert_activity(
    messages: list[dict],
    positions: dict[str, int],
    event: dict,
    timestamp: str,
) -> None:
    activity_id = str(event["activity_id"])
    if activity_id not in positions:
        positions[activity_id] = len(messages)
        _append_activity(messages, event, timestamp)
        return
    current = messages[positions[activity_id]]
    messages[positions[activity_id]] = {
        **current,
        **event,
        "type": "agent_activity",
        "content": str(event.get("title") or event.get("tool_name") or current["content"]),
        "timestamp": timestamp,
    }


def _structured_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            with suppress(json.JSONDecodeError):
                parsed = json.loads(text)
                if isinstance(parsed, (dict, list)):
                    return _structured_text(parsed)
        return text
    if isinstance(value, list):
        return "\n".join(
            text
            for item in value
            if (text := _structured_text(item))
        )
    if not isinstance(value, dict):
        return ""
    for key in (
        "text",
        "content",
        "message",
        "summary",
        "summary_text",
        "reasoning_content",
        "reasoning",
        "thinking",
    ):
        text = _structured_text(value.get(key))
        if text:
            return text
    return ""


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    with suppress(json.JSONDecodeError):
        return json.loads(value)
    return value


def _timestamp(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        with suppress(ValueError):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                timezone.utc
            ).isoformat()
    with suppress(TypeError, ValueError, OSError):
        numeric = float(value)
        if numeric > 10_000_000_000:
            numeric /= 1000
        return datetime.fromtimestamp(numeric, timezone.utc).isoformat()
    return datetime.fromtimestamp(0, timezone.utc).isoformat()
