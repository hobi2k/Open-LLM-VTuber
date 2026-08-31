"""Normalize native coding-agent tool events for the chat timeline."""

import difflib
import json
from typing import Any


_MAX_TEXT = 6_000
_COMMAND_TOOLS = {"bash", "shell", "terminal", "command", "exec", "run"}
_FILE_TOOLS = {
    "edit",
    "multiedit",
    "write",
    "patch",
    "apply_patch",
    "file_change",
}


def tool_activity(
    activity_id: str,
    tool_name: str,
    status: str,
    *,
    input_data: Any = None,
    title: str = "",
    output: Any = None,
    error: Any = None,
    metadata: Any = None,
) -> dict:
    """Build one stable activity update from a native tool call or result."""
    inputs = input_data if isinstance(input_data, dict) else {}
    details = metadata if isinstance(metadata, dict) else {}
    command = _first_text(inputs, "command", "cmd")
    path = _first_text(
        inputs,
        "file_path",
        "filePath",
        "filepath",
        "path",
        "filename",
    )
    diff = _diff_text(inputs, details, path)
    kind = _activity_kind(tool_name, command, diff)
    result = error if error is not None and error != "" else output
    event = {
        "type": "agent-activity",
        "activity_id": str(activity_id),
        "activity_kind": kind,
        "tool_name": str(tool_name or "tool"),
        "title": _clip(_activity_title(title, inputs, command, path, tool_name)),
        "status": _status(status),
    }
    optional = {
        "command": command,
        "path": path,
        "input": _input_text(inputs, command, path),
        "output": _structured_text(result),
        "diff": diff,
    }
    event.update({key: _clip(value) for key, value in optional.items() if value})
    return event


def activity_signature(event: dict) -> str:
    """Return a deterministic signature used to suppress duplicate stream updates."""
    return json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)


def _activity_kind(tool_name: str, command: str, diff: str) -> str:
    normalized = tool_name.lower().replace("-", "_").strip()
    if normalized in _COMMAND_TOOLS or command:
        return "command"
    if normalized in _FILE_TOOLS or diff:
        return "file"
    return "tool"


def _status(status: str) -> str:
    if status in {"completed", "success", "succeeded"}:
        return "completed"
    if status in {"error", "failed", "failure"}:
        return "error"
    return "running"


def _diff_text(inputs: dict, metadata: dict, path: str) -> str:
    filediff = metadata.get("filediff")
    candidates = (
        metadata.get("diff"),
        filediff.get("patch") if isinstance(filediff, dict) else None,
        inputs.get("diff"),
        inputs.get("patch"),
        inputs.get("patchText"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    old = _first_value(inputs, "old_string", "oldString", "old_text", "oldText")
    new = _first_value(
        inputs,
        "new_string",
        "newString",
        "new_text",
        "newText",
        "content",
    )
    if not isinstance(old, str) or not isinstance(new, str):
        return ""
    return "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=path or "before",
            tofile=path or "after",
            lineterm="",
        )
    )


def _input_text(inputs: dict, command: str, path: str) -> str:
    if command or path:
        return ""

    values = []
    for key in (
        "query",
        "q",
        "url",
        "pattern",
        "selector",
        "target",
        "workspace",
        "workdir",
        "cwd",
        "prompt",
    ):
        value = inputs.get(key)
        if not isinstance(value, (str, int, float, bool)) or value == "":
            continue
        text = str(value).strip()
        if not text:
            continue
        values.append((key, text))

    if len(values) == 1:
        return values[0][1]
    return "\n".join(f"{key}: {value}" for key, value in values)


def _activity_title(
    title: str,
    inputs: dict,
    command: str,
    path: str,
    tool_name: str,
) -> str:
    descriptive = _first_text(inputs, "title", "description", "label")
    generic_title = not title or title.strip().lower() == str(tool_name).strip().lower()
    if descriptive and generic_title:
        return descriptive
    return title or command or path or str(tool_name or "Tool")


def _first_text(values: dict, *keys: str) -> str:
    value = _first_value(values, *keys)
    return value if isinstance(value, str) else ""


def _first_value(values: dict, *keys: str) -> Any:
    for key in keys:
        value = values.get(key)
        if value is not None and value != "":
            return value
    return None


def _structured_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        parsed = _json_value(value)
        if parsed is not None:
            return _structured_text(parsed)
        return value.strip()
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _structured_text(item)))
    if not isinstance(value, dict):
        return str(value).strip()

    for key in ("content", "text", "output", "result", "message", "summary"):
        if key not in value:
            continue
        text = _structured_text(value[key])
        if text:
            suffix = []
            if value.get("exit_code") not in {None, 0}:
                suffix.append(f"exit code: {value['exit_code']}")
            if value.get("error"):
                suffix.append(_structured_text(value["error"]))
            return "\n".join(part for part in (text, *suffix) if part)

    scalar_values = [
        f"{key}: {item}"
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool))
        and item != ""
        and key not in {"type", "id", "activity_id"}
    ]
    return "\n".join(scalar_values)


def _json_value(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _clip(value: str) -> str:
    if len(value) <= _MAX_TEXT:
        return value
    omitted = len(value) - _MAX_TEXT
    return f"{value[:_MAX_TEXT]}\n[truncated {omitted} characters]"
