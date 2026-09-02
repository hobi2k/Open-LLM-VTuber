"""OpenCode session API adapter for Open-LLM-VTuber."""

import asyncio
import json
import mimetypes
import re
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

import httpx
from loguru import logger

from ...agent_runtime_commands import (
    expand_runtime_slash_command,
    local_runtime_commands,
)
from .agent_activity import tool_activity
from .permission_bridge import PermissionMode, permission_mode_from_legacy
from .stateless_llm_interface import StatelessLLMInterface


class OpenCodePromptAborted(Exception):
    """Raised when OpenCode confirms that an interrupted prompt stopped."""


class OpenCodeLLM(StatelessLLMInterface):
    """Use an OpenCode server as a stateless, streaming LLM backend."""

    def __init__(
        self,
        base_url: str,
        provider_id: str,
        model: str,
        agent: str = "vtuber",
        interaction_mode: str = "character",
        session_id: str = "",
        new_session_title: str = "",
        workspace_directory: str = ".",
        timeout: float = 300,
        keep_sessions: bool = False,
        allow_tools: bool = False,
        permission_mode: PermissionMode | None = None,
        show_reasoning: bool = False,
        server_username: str | None = None,
        server_password: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.provider_id = provider_id
        self.model = model
        self.agent = agent
        self.interaction_mode = interaction_mode
        self.session_id = session_id
        self.new_session_title = " ".join(new_session_title.split())
        self.workspace_directory = str(Path(workspace_directory).expanduser().resolve())
        self.timeout = timeout
        self.keep_sessions = keep_sessions
        self.permission_mode = permission_mode_from_legacy(
            permission_mode, allow_tools
        )
        self.allow_tools = self.permission_mode != "disabled"
        self.show_reasoning = show_reasoning
        self.server_username = server_username
        self.server_password = server_password
        self.support_tools = False
        self._pending_permissions: set[str] = set()
        self._pending_questions: dict[str, list[str]] = {}
        self._permission_rejected = False

        logger.info(
            "Initialized OpenCodeLLM at {} with {}/{} (agent: {})",
            self.base_url,
            self.provider_id,
            self.model,
            self.agent,
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        if tools:
            logger.warning(
                "OpenCodeLLM received external tools, but MCP tool forwarding is not "
                "supported. OpenCode tools are controlled by allow_tools instead."
            )

        session_id = self.session_id or None
        self._pending_permissions.clear()
        self._pending_questions.clear()
        self._permission_rejected = False
        completed = False
        command_task = None
        auth = None
        if self.server_password:
            auth = httpx.BasicAuth(
                self.server_username or "opencode", self.server_password
            )

        timeout = httpx.Timeout(None, connect=min(self.timeout, 10))
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            auth=auth,
        ) as client:
            try:
                continuing = bool(session_id)
                if not session_id:
                    session_id = await self._create_session(client)
                    self.session_id = session_id
                    self.new_session_title = ""
                else:
                    await self._configure_session(client, session_id)
                prompt_messages, slash_command = self._prepare_slash_command(messages)
                prompt_parts = self._build_prompt_parts(
                    prompt_messages,
                    continuing or self.interaction_mode == "coding",
                )

                async with client.stream(
                    "GET",
                    "/event",
                    params={"directory": self.workspace_directory},
                ) as event_response:
                    event_response.raise_for_status()
                    event_lines = event_response.aiter_lines()
                    await self._wait_until_connected(event_lines)
                    command_task = await self._start_prompt(
                        client,
                        session_id,
                        prompt_parts,
                        system if self.interaction_mode == "character" else None,
                        slash_command,
                    )

                    emitted_text = False
                    async for chunk in self._stream_text(
                        event_lines,
                        session_id,
                    ):
                        if isinstance(chunk, str):
                            emitted_text = True
                        yield chunk

                if command_task:
                    command_response = await command_task
                    command_response.raise_for_status()

                completed = True
                if not emitted_text:
                    fallback = await self._last_assistant_text(client, session_id)
                    if fallback:
                        yield fallback
                        return
                    if self._permission_rejected:
                        yield "The requested action was rejected. No changes were made."
                        return
                    raise RuntimeError(
                        "OpenCode completed without an assistant response"
                    )
            except OpenCodePromptAborted:
                completed = True
                logger.info("OpenCode prompt stopped after interruption")
                return
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "OpenCode API returned {}: {}",
                    exc.response.status_code,
                    exc.response.text,
                )
                yield "OpenCode returned an error. Check the OpenCode server and model settings."
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                logger.error("OpenCode chat request failed: {}", exc)
                yield "Could not get a response from OpenCode. Check that OpenCode is running."
            finally:
                if command_task and not command_task.done():
                    command_task.cancel()
                    await asyncio.gather(command_task, return_exceptions=True)
                if session_id and not completed:
                    with suppress(httpx.HTTPError):
                        await client.post(
                            f"/session/{session_id}/abort",
                            params={"directory": self.workspace_directory},
                        )

    async def _create_session(self, client: httpx.AsyncClient) -> str:
        payload: Dict[str, Any] = {
            "title": (
                self.new_session_title
                or (
                    "Open-LLM coding session"
                    if self.interaction_mode == "coding"
                    else "Open-LLM-VTuber conversation"
                )
            ),
            "agent": self._selected_agent(),
            "model": {"providerID": self.provider_id, "id": self.model},
        }
        payload["permission"] = self._permission_rules()

        response = await client.post(
            "/session",
            params={"directory": self.workspace_directory},
            json=payload,
        )
        response.raise_for_status()
        session_id = response.json().get("id")
        if not session_id:
            raise ValueError("OpenCode did not return a session ID")
        return session_id

    async def _configure_session(
        self,
        client: httpx.AsyncClient,
        session_id: str,
    ) -> None:
        response = await client.patch(
            f"/session/{session_id}",
            params={"directory": self.workspace_directory},
            json={"permission": self._permission_rules()},
        )
        response.raise_for_status()

    def _permission_rules(self) -> list[dict[str, str]]:
        if self.permission_mode == "disabled":
            return [{"permission": "*", "pattern": "*", "action": "deny"}]
        if self.permission_mode == "manual":
            return [{"permission": "*", "pattern": "*", "action": "ask"}]
        if self.permission_mode == "auto":
            return [{"permission": "*", "pattern": "*", "action": "allow"}]
        return [
            {"permission": "*", "pattern": "*", "action": "deny"},
            {"permission": "read", "pattern": "*", "action": "allow"},
            {"permission": "glob", "pattern": "*", "action": "allow"},
            {"permission": "grep", "pattern": "*", "action": "allow"},
            {"permission": "list", "pattern": "*", "action": "allow"},
            {"permission": "lsp", "pattern": "*", "action": "allow"},
        ]

    async def _start_prompt(
        self,
        client: httpx.AsyncClient,
        session_id: str,
        parts: List[Dict[str, Any]],
        system: str | None,
        slash_command: tuple[str, str] | None = None,
    ) -> asyncio.Task[httpx.Response] | None:
        if slash_command:
            name, arguments = slash_command
            return asyncio.create_task(
                client.post(
                    f"/session/{session_id}/command",
                    params={"directory": self.workspace_directory},
                    json={
                        "command": name,
                        "arguments": arguments,
                        "agent": self._selected_agent(),
                    },
                )
            )
        payload: Dict[str, Any] = {
            "model": {"providerID": self.provider_id, "modelID": self.model},
            "agent": self._selected_agent(),
            "parts": parts,
        }
        if system:
            payload["system"] = system

        response = await client.post(
            f"/session/{session_id}/prompt_async",
            params={"directory": self.workspace_directory},
            json=payload,
        )
        response.raise_for_status()
        return None

    @staticmethod
    def _slash_command(messages: List[Dict[str, Any]]) -> tuple[str, str] | None:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                return None
            match = re.match(r"^/([^\s]+)(?:\s+(.*))?$", content.strip(), re.DOTALL)
            if not match:
                return None
            return match.group(1), (match.group(2) or "").strip()
        return None

    def _prepare_slash_command(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], tuple[str, str] | None]:
        slash_command = self._slash_command(messages)
        if slash_command is None:
            return messages, None
        name, _ = slash_command
        command = next(
            (
                item
                for item in local_runtime_commands(
                    "opencode", self.workspace_directory
                )
                if item["name"] == name
            ),
            None,
        )
        if command is None or command["source"] != "skill":
            return messages, slash_command

        user_index = next(
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        )
        expanded = expand_runtime_slash_command(
            str(messages[user_index]["content"]),
            "opencode",
            self.workspace_directory,
        )
        return [
            {**message, "content": expanded} if index == user_index else message
            for index, message in enumerate(messages)
        ], None

    @staticmethod
    async def _wait_until_connected(event_lines) -> None:
        async for line in event_lines:
            event = OpenCodeLLM._parse_event(line)
            if event and event.get("type") == "server.connected":
                return
        raise RuntimeError("OpenCode event stream closed before connecting")

    async def _stream_text(
        self, event_lines, session_id: str
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        assistant_messages = set()
        part_types: Dict[str, str] = {}
        raw_text: Dict[str, str] = {}
        active_reasoning = set()
        output_started = False

        async for line in event_lines:
            event = self._parse_event(line)
            if not event:
                continue

            properties = event.get("properties", {})
            if properties.get("sessionID") != session_id:
                continue

            if event.get("type") == "permission.asked":
                request_id = properties.get("id")
                if not isinstance(request_id, str) or not request_id:
                    continue
                if self.permission_mode == "auto":
                    await self._reply_permission(request_id, "always")
                    continue
                if self.permission_mode in {"disabled", "plan"}:
                    await self._reply_permission(request_id, "reject")
                    continue
                self._pending_permissions.add(request_id)
                yield {
                    "type": "permission-request",
                    "request_id": request_id,
                    "runtime": "opencode",
                    "tool_name": properties.get("permission") or "tool",
                    "title": properties.get("permission") or "Permission request",
                    "description": "\n".join(properties.get("patterns") or []),
                    "input": properties.get("metadata") or {},
                    "options": [
                        {"id": "once", "label": "Allow once"},
                        {"id": "always", "label": "Allow for session"},
                        {"id": "reject", "label": "Reject"},
                    ],
                }
                continue

            if event.get("type") == "permission.replied":
                request_id = properties.get("requestID")
                if isinstance(request_id, str):
                    self._pending_permissions.discard(request_id)
                continue

            if event.get("type") == "question.asked":
                request_id = properties.get("id")
                questions = properties.get("questions")
                if (
                    not isinstance(request_id, str)
                    or not request_id
                    or not isinstance(questions, list)
                ):
                    continue
                normalized_questions = [
                    {
                        **question,
                        "id": str(question.get("id") or index),
                    }
                    for index, question in enumerate(questions)
                    if isinstance(question, dict)
                ]
                self._pending_questions[request_id] = [
                    question["id"] for question in normalized_questions
                ]
                yield {
                    "type": "permission-request",
                    "request_id": request_id,
                    "runtime": "opencode",
                    "tool_name": "user_input",
                    "title": (
                        str(normalized_questions[0].get("header") or "Question")
                        if normalized_questions
                        else "Question"
                    ),
                    "description": "\n".join(
                        str(question.get("question") or "")
                        for question in normalized_questions
                    ).strip(),
                    "input": {"questions": normalized_questions},
                    "options": [
                        {"id": "once", "label": "Submit answer"},
                        {"id": "reject", "label": "Cancel"},
                    ],
                }
                continue

            if event.get("type") in {"question.replied", "question.rejected"}:
                request_id = properties.get("requestID")
                if isinstance(request_id, str):
                    self._pending_questions.pop(request_id, None)
                continue

            if event.get("type") == "message.updated":
                info = properties.get("info", {})
                if info.get("role") == "assistant":
                    assistant_messages.add(info.get("id"))
                    error = info.get("error")
                    if (
                        isinstance(error, dict)
                        and error.get("name") == "MessageAbortedError"
                    ):
                        raise OpenCodePromptAborted
                    if error:
                        raise RuntimeError(str(error))
                continue

            if event.get("type") == "message.part.updated":
                part = properties.get("part", {})
                part_type = part.get("type")
                if part.get("messageID") not in assistant_messages:
                    continue
                if part_type == "tool" and self.interaction_mode == "coding":
                    yield self._tool_activity(part)
                    continue
                if part_type in {"text", "reasoning"}:
                    part_id = part.get("id")
                    if not part_id:
                        continue
                    part_types[part_id] = part_type
                    complete_text = part.get("text", "")
                    previous_text = raw_text.get(part_id, "")
                    if complete_text.startswith(previous_text):
                        chunk = complete_text[len(previous_text) :]
                        raw_text[part_id] = complete_text
                        if part_type == "reasoning":
                            if self.show_reasoning and part_id not in active_reasoning:
                                active_reasoning.add(part_id)
                                yield self._reasoning_event("reasoning-start", part_id)
                            if self.show_reasoning and chunk:
                                yield self._reasoning_event(
                                    "reasoning-delta", part_id, chunk
                                )
                            continue
                        chunk, output_started = self._visible_chunk(
                            chunk, output_started
                        )
                        if chunk:
                            yield chunk
                continue

            if event.get("type") == "message.part.delta":
                part_id = properties.get("partID")
                part_type = part_types.get(part_id)
                if (
                    properties.get("messageID") in assistant_messages
                    and part_type in {"text", "reasoning"}
                    and properties.get("field") == "text"
                ):
                    chunk = properties.get("delta", "")
                    raw_text[part_id] = raw_text.get(part_id, "") + chunk
                    if part_type == "reasoning":
                        if self.show_reasoning and part_id not in active_reasoning:
                            active_reasoning.add(part_id)
                            yield self._reasoning_event("reasoning-start", part_id)
                        if self.show_reasoning and chunk:
                            yield self._reasoning_event(
                                "reasoning-delta", part_id, chunk
                            )
                        continue
                    chunk, output_started = self._visible_chunk(chunk, output_started)
                    if chunk:
                        yield chunk
                continue

            if event.get("type") == "session.idle":
                for part_id in active_reasoning:
                    yield self._reasoning_event("reasoning-end", part_id)
                return

        raise RuntimeError("OpenCode event stream closed before the response completed")

    async def respond_to_permission(
        self,
        request_id: str,
        decision: str,
        message: str = "",
    ) -> bool:
        question_ids = self._pending_questions.get(request_id)
        if question_ids is not None:
            if decision not in {"once", "reject"}:
                return False
            await self._reply_question(request_id, decision, message, question_ids)
            self._pending_questions.pop(request_id, None)
            return True
        if request_id not in self._pending_permissions:
            return False
        if decision not in {"once", "always", "reject"}:
            return False
        await self._reply_permission(request_id, decision, message)
        if decision == "reject":
            self._permission_rejected = True
        self._pending_permissions.discard(request_id)
        return True

    async def _reply_question(
        self,
        request_id: str,
        decision: str,
        message: str,
        question_ids: list[str],
    ) -> None:
        auth = None
        if self.server_password:
            auth = httpx.BasicAuth(
                self.server_username or "opencode", self.server_password
            )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=min(self.timeout, 30),
            auth=auth,
        ) as client:
            if decision == "reject":
                response = await client.post(
                    f"/question/{request_id}/reject",
                    params={"directory": self.workspace_directory},
                )
            else:
                response = await client.post(
                    f"/question/{request_id}/reply",
                    params={"directory": self.workspace_directory},
                    json={"answers": self._question_answers(message, question_ids)},
                )
            response.raise_for_status()

    @staticmethod
    def _question_answers(message: str, question_ids: list[str]) -> list[list[str]]:
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return [
                OpenCodeLLM._answer_values(parsed.get(question_id))
                for question_id in question_ids
            ]
        if isinstance(parsed, list):
            return [
                OpenCodeLLM._answer_values(answer)
                for answer in parsed[: len(question_ids)]
            ] + [[] for _ in question_ids[len(parsed) :]]
        return [
            [message.strip()] if index == 0 and message.strip() else []
            for index, _ in enumerate(question_ids)
        ]

    @staticmethod
    def _answer_values(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None or value == "":
            return []
        return [str(value)]

    async def _reply_permission(
        self,
        request_id: str,
        decision: str,
        message: str = "",
    ) -> None:
        auth = None
        if self.server_password:
            auth = httpx.BasicAuth(
                self.server_username or "opencode", self.server_password
            )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=min(self.timeout, 30),
            auth=auth,
        ) as client:
            response = await client.post(
                f"/permission/{request_id}/reply",
                params={"directory": self.workspace_directory},
                json={"reply": decision, "message": message or None},
            )
            response.raise_for_status()

    def _selected_agent(self) -> str:
        if self.permission_mode == "plan":
            return "plan"
        if self.interaction_mode == "coding" and self.agent == "vtuber":
            return "build"
        return self.agent

    @staticmethod
    def _visible_chunk(chunk: str, output_started: bool) -> tuple[str, bool]:
        if output_started:
            return chunk, True
        visible = chunk.lstrip()
        return visible, bool(visible)

    @staticmethod
    def _reasoning_event(event_type: str, part_id: str, text: str = "") -> dict:
        return {
            "type": event_type,
            "reasoning_id": part_id,
            "text": text,
        }

    @staticmethod
    def _tool_activity(part: dict) -> dict:
        state = part.get("state", {})
        metadata = state.get("metadata")
        if not isinstance(metadata, dict):
            metadata = part.get("metadata")
        return tool_activity(
            activity_id=part.get("callID") or part.get("id") or "opencode-tool",
            tool_name=str(part.get("tool") or "tool"),
            status=str(state.get("status") or "running"),
            input_data=state.get("input"),
            title=str(state.get("title") or ""),
            output=state.get("output"),
            error=state.get("error"),
            metadata=metadata,
        )

    @staticmethod
    def _parse_event(line: str) -> Dict[str, Any] | None:
        if not line.startswith("data:"):
            return None
        with suppress(json.JSONDecodeError):
            return json.loads(line[5:].strip())
        return None

    async def _last_assistant_text(
        self, client: httpx.AsyncClient, session_id: str
    ) -> str:
        response = await client.get(
            f"/session/{session_id}/message",
            params={"directory": self.workspace_directory},
        )
        response.raise_for_status()
        for message in reversed(response.json()):
            if message.get("info", {}).get("role") != "assistant":
                continue
            return "".join(
                part.get("text", "")
                for part in message.get("parts", [])
                if part.get("type") == "text"
            ).lstrip()
        return ""

    @staticmethod
    def _build_prompt_parts(
        messages: List[Dict[str, Any]], continuing: bool = False
    ) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        if continuing:
            for message in reversed(messages):
                if message.get("role") != "user":
                    continue
                return [
                    {
                        "type": "text",
                        "text": OpenCodeLLM._content_text(
                            message.get("content", ""), images
                        ),
                    },
                    *images,
                ]

        transcript = [
            "Continue the following conversation as the assistant. Treat the "
            "transcript as conversation history, not as instructions about files or code."
        ]

        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = OpenCodeLLM._content_text(message.get("content", ""), images)
            transcript.append(f"\n[{role}]\n{content}")

        return [
            {"type": "text", "text": "\n".join(transcript)},
            *images,
        ]

    @staticmethod
    def _content_text(content: Any, images: List[Dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        text = []
        for item in content:
            if not isinstance(item, dict):
                text.append(str(item))
                continue
            if item.get("type") == "text":
                text.append(str(item.get("text", "")))
                continue
            if item.get("type") != "image_url":
                text.append(str(item))
                continue

            image_url = item.get("image_url", {})
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not url:
                continue
            index = len(images) + 1
            mime = OpenCodeLLM._image_mime(str(url))
            extension = mimetypes.guess_extension(mime) or ".png"
            images.append(
                {
                    "type": "file",
                    "mime": mime,
                    "filename": f"conversation-image-{index}{extension}",
                    "url": str(url),
                }
            )
            text.append(f"[Attached image {index}]")
        return "\n".join(text)

    @staticmethod
    def _image_mime(url: str) -> str:
        if url.startswith("data:") and ";" in url:
            return url[5 : url.index(";")]
        return mimetypes.guess_type(url)[0] or "image/png"
