"""Adapters for using installed agent CLIs as stateless chat backends."""

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

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
        workspace_directory: str = ".",
        timeout: float = 300,
    ):
        self.runtime = runtime
        self.executable = (
            str(Path(executable).expanduser()) if "/" in executable else executable
        )
        self.model = model
        self.provider = provider
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

        prompt = self._build_prompt(messages, system)
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

            response = self._response_text(stdout.decode("utf-8", errors="replace"))
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
                "--no-session-persistence",
            ]
            if self.model:
                command.extend(["--model", self.model])
            return command, prompt

        if self.runtime == "codex":
            command = [
                self.executable,
                "exec",
                "--json",
                "--color",
                "never",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append("-")
            return command, prompt

        if self.runtime == "hermes":
            command = [
                self.executable,
                "--oneshot",
                prompt,
                "--safe-mode",
            ]
            if self.model:
                command.extend(["--model", self.model])
            if self.provider:
                command.extend(["--provider", self.provider])
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
