"""Discover and expand slash commands for native coding runtimes."""

import asyncio
import json
import re
from contextlib import suppress
from pathlib import Path

import yaml

from .executable_utils import executable_environment, resolve_executable


_HERMES_ACP_COMMANDS = {
    "help": "Show available commands",
    "model": "Show or change current model",
    "tools": "List available tools",
    "context": "Show conversation context info",
    "reset": "Clear conversation history",
    "compress": "Compress conversation context",
    "steer": "Inject guidance into the active turn",
    "queue": "Queue a prompt for the next turn",
    "version": "Show Hermes version",
}


def local_runtime_commands(
    runtime: str,
    workspace_directory: str,
    home: Path | None = None,
) -> list[dict]:
    home = home or Path.home()
    workspace = Path(workspace_directory).expanduser()
    roots = {
        "opencode": [
            (workspace / ".opencode/command", "command"),
            (workspace / ".opencode/commands", "command"),
            (workspace / ".opencode/skill", "skill"),
            (workspace / ".opencode/skills", "skill"),
            (home / ".config/opencode/command", "command"),
            (home / ".config/opencode/commands", "command"),
            (home / ".config/opencode/skill", "skill"),
            (home / ".config/opencode/skills", "skill"),
            (workspace / ".claude/skills", "skill"),
            (home / ".claude/skills", "skill"),
            (workspace / ".agents/skills", "skill"),
            (home / ".agents/skills", "skill"),
        ],
        "claude_code": [
            (workspace / ".claude/commands", "command"),
            (workspace / ".claude/skills", "skill"),
            (home / ".claude/commands", "command"),
            (home / ".claude/skills", "skill"),
        ],
        "codex": [
            (workspace / ".codex/skills", "skill"),
            (home / ".codex/skills", "skill"),
        ],
        "hermes": [
            (workspace / ".hermes/skills", "skill"),
            (home / ".hermes/skills", "skill"),
        ],
    }.get(runtime, [])
    commands = {}
    if runtime == "hermes":
        commands.update(
            {
                name: {
                    "name": name,
                    "description": description,
                    "source": "command",
                    "runtime": runtime,
                    "invocation": f"/{name}",
                }
                for name, description in _HERMES_ACP_COMMANDS.items()
            }
        )

    for root, source in roots:
        pattern = "**/SKILL.md" if source == "skill" else "**/*.md"
        for path in root.glob(pattern):
            item = _markdown_command(path, root, runtime, source)
            if item and item["name"] not in commands:
                commands[item["name"]] = item
    return sorted(commands.values(), key=lambda item: item["name"].lower())


def expand_runtime_slash_command(
    prompt: str,
    runtime: str,
    workspace_directory: str,
) -> str:
    match = re.match(r"^/([^\s]+)(?:\s+(.*))?$", prompt.strip(), re.DOTALL)
    if not match:
        return prompt
    name = match.group(1)
    arguments = (match.group(2) or "").strip()
    command = next(
        (
            item
            for item in local_runtime_commands(runtime, workspace_directory)
            if item["name"] == name
        ),
        None,
    )
    if command is None or command["source"] == "command" and runtime == "hermes":
        return prompt
    if runtime == "codex":
        return f"${name}{f' {arguments}' if arguments else ''}"

    path = Path(command["path"])
    with suppress(OSError):
        content = _markdown_body(path.read_text(encoding="utf-8"))
        content = _substitute_arguments(content, arguments)
        label = "skill" if command["source"] == "skill" else "command"
        return (
            f"[The user invoked the {label} /{name}. Follow these instructions.]\n\n"
            f"{content}\n\n"
            f"[Base directory: {path.parent}]\n"
            f"[User instruction: {arguments or 'Apply the instructions above.'}]"
        )
    return prompt


def codex_slash_command(prompt: str) -> tuple[str, str] | None:
    match = re.match(r"^/([^\s]+)(?:\s+(.*))?$", prompt.strip(), re.DOTALL)
    if not match:
        return None
    return match.group(1), (match.group(2) or "").strip()


def codex_skills_from_response(payload: dict) -> list[dict]:
    commands = {}
    for entry in payload.get("data", []):
        if not isinstance(entry, dict):
            continue
        for skill in entry.get("skills", []):
            if not isinstance(skill, dict) or not skill.get("enabled", True):
                continue
            name = str(skill.get("name") or "").strip()
            path = str(skill.get("path") or "").strip()
            if not name or not path or name in commands:
                continue
            commands[name] = {
                "name": name,
                "description": str(
                    skill.get("description")
                    or skill.get("shortDescription")
                    or "Codex skill"
                ),
                "source": "skill",
                "runtime": "codex",
                "invocation": f"/{name}",
                "path": path,
            }
    return sorted(commands.values(), key=lambda item: item["name"].lower())


async def codex_runtime_skills(
    executable: str,
    workspace_directory: str,
    timeout: float = 5,
) -> list[dict]:
    resolved = resolve_executable(executable, "codex")
    workspace = Path(workspace_directory).expanduser()
    if not resolved or not workspace.is_dir():
        return []

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            resolved,
            "app-server",
            "--stdio",
            cwd=str(workspace),
            env=executable_environment(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        deadline = asyncio.get_running_loop().time() + timeout
        await _codex_write(
            process,
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "open-llm-vtuber",
                        "title": "Open LLM VTuber",
                        "version": "1.2.1",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            },
        )
        await _codex_response(process, 1, deadline)
        await _codex_write(process, {"method": "initialized", "params": {}})
        await _codex_write(
            process,
            {
                "id": 2,
                "method": "skills/list",
                "params": {"cwds": [str(workspace)], "forceReload": False},
            },
        )
        return codex_skills_from_response(
            await _codex_response(process, 2, deadline)
        )
    except (OSError, asyncio.TimeoutError, RuntimeError, json.JSONDecodeError):
        return []
    finally:
        if process and process.returncode is None:
            process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), 1)
            if process.returncode is None:
                process.kill()
                await process.wait()


async def _codex_write(
    process: asyncio.subprocess.Process,
    message: dict,
) -> None:
    if process.stdin is None:
        raise RuntimeError("Codex app-server stdin is unavailable")
    process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
    await process.stdin.drain()


async def _codex_response(
    process: asyncio.subprocess.Process,
    request_id: int,
    deadline: float,
) -> dict:
    if process.stdout is None:
        raise RuntimeError("Codex app-server stdout is unavailable")
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        line = await asyncio.wait_for(process.stdout.readline(), remaining)
        if not line:
            raise RuntimeError("Codex app-server closed unexpectedly")
        message = json.loads(line)
        if not isinstance(message, dict) or message.get("id") != request_id:
            continue
        if "error" in message:
            raise RuntimeError(str(message["error"]))
        result = message.get("result")
        return result if isinstance(result, dict) else {}


def _markdown_command(
    path: Path,
    root: Path,
    runtime: str,
    source: str,
) -> dict | None:
    with suppress(OSError):
        content = path.read_text(encoding="utf-8")
        metadata = _frontmatter(content)
        relative = path.relative_to(root)
        name = str(metadata.get("name") or "").strip()
        if not name:
            name = path.parent.name if path.name == "SKILL.md" else relative.with_suffix("").as_posix()
        if not name or name.startswith("."):
            return None
        description = str(metadata.get("description") or "").strip()
        if not description:
            description = _first_text_line(_markdown_body(content))
        return {
            "name": name,
            "description": description or f"{source.title()} from {path.parent.name}",
            "source": source,
            "runtime": runtime,
            "invocation": f"/{name}",
            "path": str(path),
        }
    return None


def _frontmatter(content: str) -> dict:
    if not content.startswith("---\n"):
        return {}
    end = content.find("\n---", 4)
    if end < 0:
        return {}
    with suppress(yaml.YAMLError, TypeError):
        payload = yaml.safe_load(content[4:end]) or {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _markdown_body(content: str) -> str:
    if not content.startswith("---\n"):
        return content.strip()
    end = content.find("\n---", 4)
    return content[end + 4 :].strip() if end >= 0 else content.strip()


def _first_text_line(content: str) -> str:
    for line in content.splitlines():
        text = line.lstrip("# ").strip()
        if text:
            return text
    return ""


def _substitute_arguments(content: str, arguments: str) -> str:
    values = re.findall(r'''(?:[^\s"']|"[^"]*"|'[^']*')+''', arguments)
    result = content.replace("$ARGUMENTS", arguments)
    for index, value in reversed(list(enumerate(values, 1))):
        result = result.replace(f"${index}", value.strip("\"'"))
    if "$ARGUMENTS" not in content and not re.search(r"\$\d+", content) and arguments:
        return f"{result}\n\n{arguments}"
    return result
