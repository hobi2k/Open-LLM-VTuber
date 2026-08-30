"""OpenCode session API adapter for Open-LLM-VTuber."""

import json
import mimetypes
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import httpx
from loguru import logger

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
        session_id: str = "",
        workspace_directory: str = ".",
        timeout: float = 300,
        keep_sessions: bool = False,
        allow_tools: bool = False,
        server_username: str | None = None,
        server_password: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.provider_id = provider_id
        self.model = model
        self.agent = agent
        self.session_id = session_id
        self.workspace_directory = str(Path(workspace_directory).expanduser().resolve())
        self.timeout = timeout
        self.keep_sessions = keep_sessions
        self.allow_tools = allow_tools
        self.server_username = server_username
        self.server_password = server_password
        self.support_tools = False

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
    ) -> AsyncIterator[str]:
        if tools:
            logger.warning(
                "OpenCodeLLM received external tools, but MCP tool forwarding is not "
                "supported. OpenCode tools are controlled by allow_tools instead."
            )

        session_id = self.session_id or None
        completed = False
        auth = None
        if self.server_password:
            auth = httpx.BasicAuth(
                self.server_username or "opencode", self.server_password
            )

        timeout = httpx.Timeout(self.timeout, connect=min(self.timeout, 10))
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
                prompt_parts = self._build_prompt_parts(messages, continuing)

                async with client.stream(
                    "GET",
                    "/event",
                    params={"directory": self.workspace_directory},
                ) as event_response:
                    event_response.raise_for_status()
                    event_lines = event_response.aiter_lines()
                    await self._wait_until_connected(event_lines)
                    await self._start_prompt(
                        client,
                        session_id,
                        prompt_parts,
                        system,
                    )

                    emitted = False
                    async for chunk in self._stream_text(
                        event_lines,
                        session_id,
                    ):
                        emitted = True
                        yield chunk

                completed = True
                if not emitted:
                    fallback = await self._last_assistant_text(client, session_id)
                    if fallback:
                        yield fallback
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
                if session_id and not completed:
                    with suppress(httpx.HTTPError):
                        await client.post(
                            f"/session/{session_id}/abort",
                            params={"directory": self.workspace_directory},
                        )

    async def _create_session(self, client: httpx.AsyncClient) -> str:
        payload: Dict[str, Any] = {
            "title": "Open-LLM-VTuber conversation",
            "agent": self.agent,
            "model": {"providerID": self.provider_id, "id": self.model},
        }
        if not self.allow_tools:
            payload["permission"] = [
                {"permission": "*", "pattern": "*", "action": "deny"}
            ]

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

    async def _start_prompt(
        self,
        client: httpx.AsyncClient,
        session_id: str,
        parts: List[Dict[str, Any]],
        system: str | None,
    ) -> None:
        payload: Dict[str, Any] = {
            "model": {"providerID": self.provider_id, "modelID": self.model},
            "agent": self.agent,
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

    @staticmethod
    async def _wait_until_connected(event_lines) -> None:
        async for line in event_lines:
            event = OpenCodeLLM._parse_event(line)
            if event and event.get("type") == "server.connected":
                return
        raise RuntimeError("OpenCode event stream closed before connecting")

    @staticmethod
    async def _stream_text(event_lines, session_id: str) -> AsyncIterator[str]:
        assistant_messages = set()
        text_parts = set()
        raw_text: Dict[str, str] = {}
        output_started = False

        async for line in event_lines:
            event = OpenCodeLLM._parse_event(line)
            if not event:
                continue

            properties = event.get("properties", {})
            if properties.get("sessionID") != session_id:
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
                if (
                    part.get("type") == "text"
                    and part.get("messageID") in assistant_messages
                ):
                    part_id = part.get("id")
                    if not part_id:
                        continue
                    text_parts.add(part_id)
                    complete_text = part.get("text", "")
                    previous_text = raw_text.get(part_id, "")
                    if complete_text.startswith(previous_text):
                        chunk = complete_text[len(previous_text) :]
                        raw_text[part_id] = complete_text
                        chunk, output_started = OpenCodeLLM._visible_chunk(
                            chunk, output_started
                        )
                        if chunk:
                            yield chunk
                continue

            if event.get("type") == "message.part.delta":
                part_id = properties.get("partID")
                if (
                    properties.get("messageID") in assistant_messages
                    and part_id in text_parts
                    and properties.get("field") == "text"
                ):
                    chunk = properties.get("delta", "")
                    raw_text[part_id] = raw_text.get(part_id, "") + chunk
                    chunk, output_started = OpenCodeLLM._visible_chunk(
                        chunk, output_started
                    )
                    if chunk:
                        yield chunk
                continue

            if event.get("type") == "session.idle":
                return

        raise RuntimeError("OpenCode event stream closed before the response completed")

    @staticmethod
    def _visible_chunk(chunk: str, output_started: bool) -> tuple[str, bool]:
        if output_started:
            return chunk, True
        visible = chunk.lstrip()
        return visible, bool(visible)

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
