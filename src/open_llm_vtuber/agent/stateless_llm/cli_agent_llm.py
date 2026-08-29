"""Adapters for using installed agent CLIs as stateless chat backends."""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
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
        session_id: str = "",
        workspace_directory: str = ".",
        timeout: float = 300,
    ):
        self.runtime = runtime
        self.executable = self._resolve_executable(executable)
        self.model = model
        self.provider = provider
        self.launch_mode = launch_mode
        self.session_id = session_id
        self.workspace_directory = str(Path(workspace_directory).expanduser().resolve())
        self.timeout = timeout
        self.support_tools = False

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        if tools:
            logger.warning("{} does not forward VTuber tools", self.runtime)

        prompt = (
            self._latest_user_text(messages)
            if self.session_id
            else self._build_prompt(messages, system)
        )
        command, stdin = self._command(prompt)
        process = None

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
            stdout, stderr = await asyncio.wait_for(
                process.communicate(
                    stdin.encode("utf-8") if stdin is not None else None
                ),
                timeout=self.timeout,
            )
            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    detail or f"process exited with status {process.returncode}"
                )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            response = self._response_text(output)
            self._capture_session(output, error_output)
            if not response:
                raise RuntimeError("the CLI returned an empty response")
            yield response.lstrip()
        except asyncio.TimeoutError:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            logger.error("{} timed out after {} seconds", self.runtime, self.timeout)
            yield f"{self._display_name()} timed out. Check the runtime settings."
        except asyncio.CancelledError:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except (
            FileNotFoundError,
            NotADirectoryError,
            PermissionError,
            RuntimeError,
        ) as error:
            logger.error("{} request failed: {}", self.runtime, error)
            yield f"Could not get a response from {self._display_name()}. Check the runtime settings."

    def _command(self, prompt: str) -> tuple[list[str], str | None]:
        if self.runtime == "claude_code":
            command = [
                self.executable,
                "-p",
                "--output-format",
                "json",
                "--tools",
                "",
                "--permission-mode",
                "dontAsk",
            ]
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
                    "--ignore-rules",
                    "-c",
                    'sandbox_mode="read-only"',
                ]
            else:
                command = [
                    self.executable,
                    "exec",
                    "--json",
                    "--color",
                    "never",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ignore-rules",
                ]
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
                "--ignore-rules",
                "--toolsets",
                "",
                "--reasoning",
                "none",
                "--source",
                "tool",
                "--max-turns",
                "1",
            ]
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
            return output.strip()

        if self.runtime == "claude_code":
            try:
                payload = json.loads(output)
            except json.JSONDecodeError as error:
                raise RuntimeError("Claude Code returned invalid JSON") from error
            result = payload.get("result")
            return result.strip() if isinstance(result, str) else ""

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
            try:
                session_id = json.loads(output).get("session_id")
            except json.JSONDecodeError:
                return
            if isinstance(session_id, str) and session_id:
                self.session_id = session_id
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
