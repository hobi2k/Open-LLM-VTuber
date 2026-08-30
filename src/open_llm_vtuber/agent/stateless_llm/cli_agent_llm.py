"""Adapters for using installed agent CLIs as stateless chat backends."""

import asyncio
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union
from uuid import uuid4

from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface


class CLIAgentLLM(StatelessLLMInterface):
    """Run Claude Code, Codex, or Hermes in a constrained one-shot mode."""

    def __init__(
        self,
        runtime: str,
        executable: str,
        model: str = "",
        provider: str = "",
        launch_mode: str = "direct",
        interaction_mode: str = "character",
        session_id: str = "",
        workspace_directory: str = ".",
        timeout: float = 300,
        show_reasoning: bool = False,
        allow_tools: bool = False,
    ):
        self.runtime = runtime
        self.executable = self._resolve_executable(executable)
        self.model = model
        self.provider = provider
        self.launch_mode = launch_mode
        self.interaction_mode = interaction_mode
        self.session_id = session_id
        self.workspace_directory = str(Path(workspace_directory).expanduser().resolve())
        self.timeout = timeout
        self.show_reasoning = show_reasoning
        self.allow_tools = allow_tools
        self.support_tools = False

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        if tools:
            logger.warning("{} does not forward VTuber tools", self.runtime)

        prompt = (
            self._latest_user_text(messages)
            if self.session_id or self.interaction_mode == "coding"
            else self._build_prompt(messages, system)
        )
        command, stdin = self._command(prompt)
        process = None
        stderr_task = None
        reasoning_id = f"{self.runtime}-{uuid4().hex}"
        reasoning_started = False

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.workspace_directory,
                stdin=asyncio.subprocess.PIPE
                if stdin is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stderr_task = asyncio.create_task(process.stderr.read())
            if stdin is not None:
                process.stdin.write(stdin.encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()

            deadline = asyncio.get_running_loop().time() + self.timeout
            stdout_lines = []
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                line = await asyncio.wait_for(process.stdout.readline(), remaining)
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                stdout_lines.append(text)
                reasoning = self._reasoning_delta(text) if self.show_reasoning else ""
                if reasoning and self.runtime != "hermes":
                    if not reasoning_started:
                        reasoning_started = True
                        yield self._reasoning_event("reasoning-start", reasoning_id)
                    yield self._reasoning_event(
                        "reasoning-delta", reasoning_id, reasoning
                    )

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(process.wait(), remaining)
            stderr = await stderr_task
            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    detail or f"process exited with status {process.returncode}"
                )

            output = "".join(stdout_lines)
            error_output = stderr.decode("utf-8", errors="replace")
            response = self._response_text(output)
            self._capture_session(output, error_output)
            reasoning = (
                self._reasoning_text(output)
                if self.show_reasoning and not reasoning_started
                else ""
            )
            if self.runtime == "hermes":
                native_response, native_reasoning = await asyncio.to_thread(
                    self._hermes_message_text
                )
                if native_response:
                    response = native_response
                if self.show_reasoning:
                    reasoning = native_reasoning
            if reasoning:
                yield self._reasoning_event("reasoning-start", reasoning_id)
                yield self._reasoning_event(
                    "reasoning-delta", reasoning_id, reasoning.strip()
                )
                reasoning_started = True
            if reasoning_started:
                yield self._reasoning_event("reasoning-end", reasoning_id)
            if not response:
                raise RuntimeError("the CLI returned an empty response")
            yield response.lstrip()
        except asyncio.TimeoutError:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            if stderr_task:
                await stderr_task
            logger.error("{} timed out after {} seconds", self.runtime, self.timeout)
            if reasoning_started:
                yield self._reasoning_event("reasoning-end", reasoning_id)
            yield f"{self._display_name()} timed out. Check the runtime settings."
        except asyncio.CancelledError:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            if stderr_task:
                await stderr_task
            raise
        except (
            FileNotFoundError,
            NotADirectoryError,
            PermissionError,
            RuntimeError,
        ) as error:
            logger.error("{} request failed: {}", self.runtime, error)
            if reasoning_started:
                yield self._reasoning_event("reasoning-end", reasoning_id)
            yield f"Could not get a response from {self._display_name()}. Check the runtime settings."

    def _command(self, prompt: str) -> tuple[list[str], str | None]:
        if self.runtime == "claude_code":
            command = [
                self.executable,
                "-p",
                "--output-format",
                "stream-json" if self.show_reasoning else "json",
            ]
            if self.allow_tools:
                command.extend(
                    ["--tools", "default", "--permission-mode", "acceptEdits"]
                )
            else:
                command.extend(["--tools", "", "--permission-mode", "dontAsk"])
            if self.show_reasoning:
                command.extend(["--include-partial-messages", "--verbose"])
            if self.session_id:
                command.extend(["--resume", self.session_id])
            else:
                self.session_id = str(uuid4())
                command.extend(["--session-id", self.session_id])
            if self.model:
                command.extend(["--model", self.model])
            return command, prompt

        if self.runtime == "codex":
            if self.session_id:
                command = [
                    self.executable,
                    "exec",
                    "resume",
                    "--json",
                    "--skip-git-repo-check",
                    "-c",
                    'sandbox_mode="workspace-write"'
                    if self.allow_tools
                    else 'sandbox_mode="read-only"',
                ]
                if self.interaction_mode != "coding":
                    command.append("--ignore-rules")
            else:
                command = [
                    self.executable,
                    "exec",
                    "--json",
                    "--color",
                    "never",
                    "--sandbox",
                    "workspace-write" if self.allow_tools else "read-only",
                    "--skip-git-repo-check",
                ]
                if self.interaction_mode != "coding":
                    command.append("--ignore-rules")
            if self.model:
                command.extend(["--model", self.model])
            if self.session_id:
                command.append(self.session_id)
            command.append("-")
            return command, prompt

        if self.runtime == "hermes":
            command = [
                self.executable,
                "chat",
                "--query",
                prompt,
                "--quiet",
                "--source",
                "cli" if self.interaction_mode == "coding" else "tool",
            ]
            if self.allow_tools:
                command.extend(["--max-turns", "50"])
            else:
                if self.interaction_mode != "coding":
                    command.append("--ignore-rules")
                command.extend(["--toolsets", "", "--max-turns", "1"])
            if self.session_id:
                command.extend(["--resume", self.session_id, "--no-restore-cwd"])
            if self.model:
                command.extend(["--model", self.model])
            provider = "omlx" if self.launch_mode == "omlx" else self.provider
            if provider:
                command.extend(["--provider", provider])
            return command, None

        raise RuntimeError(f"unsupported CLI runtime: {self.runtime}")

    def _response_text(self, output: str) -> str:
        if self.runtime == "hermes":
            if "Reasoning" not in output:
                return output.strip()
            lines = [
                line.strip()
                for line in output.replace("\r", "").splitlines()
                if line.strip() and not line.strip().lower().startswith("session_id:")
            ]
            return lines[-1] if lines else ""

        if self.runtime == "claude_code":
            payloads = self._json_payloads(output)
            results = [
                payload.get("result")
                for payload in payloads
                if isinstance(payload.get("result"), str)
            ]
            if results:
                return results[-1].strip()
            assistant_text = ""
            for payload in payloads:
                if payload.get("type") != "assistant":
                    continue
                blocks = payload.get("message", {}).get("content", [])
                text = "".join(
                    block.get("text", "")
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if text:
                    assistant_text = text
            return assistant_text.strip()

        messages = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item", {})
            if (
                event.get("type") == "item.completed"
                and item.get("type") == "agent_message"
            ):
                text = item.get("text")
                if isinstance(text, str):
                    messages.append(text)
        return messages[-1].strip() if messages else ""

    def _capture_session(self, output: str, error_output: str) -> None:
        if self.runtime == "claude_code":
            for payload in reversed(self._json_payloads(output)):
                session_id = payload.get("session_id")
                if isinstance(session_id, str) and session_id:
                    self.session_id = session_id
                    return
            return

        if self.runtime == "codex":
            for line in output.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                session_id = event.get("thread_id")
                if event.get("type") == "thread.started" and isinstance(
                    session_id, str
                ):
                    self.session_id = session_id
                    return
            return

        if self.runtime == "hermes":
            for line in reversed(error_output.splitlines()):
                if line.strip().lower().startswith("session_id:"):
                    self.session_id = line.split(":", 1)[1].strip()
                    return

    def _reasoning_text(self, output: str) -> str:
        if self.runtime == "claude_code":
            deltas = []
            complete = ""
            for payload in self._json_payloads(output):
                event = payload.get("event", {})
                delta = event.get("delta", {}) if isinstance(event, dict) else {}
                if delta.get("type") == "thinking_delta":
                    text = delta.get("thinking")
                    if isinstance(text, str):
                        deltas.append(text)
                if payload.get("type") != "assistant":
                    continue
                blocks = payload.get("message", {}).get("content", [])
                thinking = "".join(
                    block.get("thinking", "")
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "thinking"
                )
                if thinking:
                    complete = thinking
            return "".join(deltas) or complete

        if self.runtime == "codex":
            reasoning = []
            for payload in self._json_payloads(output):
                item = payload.get("item", {})
                if (
                    payload.get("type") == "item.completed"
                    and isinstance(item, dict)
                    and item.get("type") == "reasoning"
                ):
                    text = self._structured_text(item)
                    if text:
                        reasoning.append(text)
            return "\n\n".join(reasoning)

        return ""

    def _reasoning_delta(self, output: str) -> str:
        if self.runtime == "claude_code":
            deltas = []
            for payload in self._json_payloads(output):
                event = payload.get("event", {})
                delta = event.get("delta", {}) if isinstance(event, dict) else {}
                if delta.get("type") == "thinking_delta" and isinstance(
                    delta.get("thinking"), str
                ):
                    deltas.append(delta["thinking"])
            return "".join(deltas)
        return self._reasoning_text(output)

    def _hermes_message_text(self) -> tuple[str, str]:
        if not self.session_id:
            return "", ""
        database = (
            Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "state.db"
        )
        if not database.is_file():
            return "", ""
        try:
            connection = sqlite3.connect(
                f"file:{database}?mode=ro",
                uri=True,
                timeout=1,
            )
            row = connection.execute(
                """
                SELECT content, reasoning_content, reasoning, reasoning_details,
                       codex_reasoning_items
                FROM messages
                WHERE session_id = ? AND role = 'assistant' AND active = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (self.session_id,),
            ).fetchone()
            connection.close()
        except sqlite3.Error as error:
            logger.warning("Could not read Hermes reasoning: {}", error)
            return "", ""
        if not row:
            return "", ""
        response_value = row[0]
        response_json = (
            self._json_value(response_value)
            if isinstance(response_value, str)
            else None
        )
        response = self._structured_text(
            response_json if response_json is not None else response_value
        )
        for value in row[1:]:
            if not isinstance(value, str) or not value.strip():
                continue
            with_json = self._json_value(value)
            text = self._structured_text(with_json if with_json is not None else value)
            if text:
                return response, text
        return response, ""

    @staticmethod
    def _json_payloads(output: str) -> List[Dict[str, Any]]:
        payloads = []
        for line in output.splitlines() or [output]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    @staticmethod
    def _json_value(value: str) -> Any | None:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _structured_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(
                text for item in value if (text := CLIAgentLLM._structured_text(item))
            )
        if not isinstance(value, dict):
            return ""
        for key in (
            "text",
            "summary",
            "reasoning_content",
            "reasoning",
            "thinking",
            "content",
        ):
            text = CLIAgentLLM._structured_text(value.get(key))
            if text:
                return text
        return ""

    @staticmethod
    def _reasoning_event(event_type: str, reasoning_id: str, text: str = "") -> dict:
        return {
            "type": event_type,
            "reasoning_id": reasoning_id,
            "text": text,
        }

    @staticmethod
    def _build_prompt(messages: List[Dict[str, Any]], system: str | None) -> str:
        transcript = [
            "You are the conversational response engine for a virtual character. "
            "Reply only to the latest user message. Do not inspect files or use tools."
        ]
        if system:
            transcript.append(f"\n[SYSTEM]\n{system}")

        for message in messages:
            role = str(message.get("role", "user")).upper()
            transcript.append(
                f"\n[{role}]\n{CLIAgentLLM._content_text(message.get('content', ''))}"
            )
        return "\n".join(transcript)

    @staticmethod
    def _latest_user_text(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return CLIAgentLLM._content_text(message.get("content", ""))
        if messages:
            return CLIAgentLLM._content_text(messages[-1].get("content", ""))
        return ""

    def _resolve_executable(self, executable: str) -> str:
        configured = str(executable or "auto").strip()
        if configured not in {"", "auto"}:
            expanded = str(Path(configured).expanduser())
            return shutil.which(expanded) or expanded

        command = {
            "claude_code": "claude",
            "codex": "codex",
            "hermes": "hermes",
        }.get(self.runtime, self.runtime)
        return shutil.which(command) or command

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
                continue
            if isinstance(item, dict) and item.get("type") == "image_url":
                parts.append(
                    "[An image was attached, but this CLI text adapter cannot inspect it.]"
                )
                continue
            parts.append(str(item))
        return "\n".join(parts)

    def _display_name(self) -> str:
        return {
            "claude_code": "Claude Code",
            "codex": "Codex",
            "hermes": "Hermes",
        }.get(self.runtime, self.runtime)
