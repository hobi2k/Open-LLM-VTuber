"""Claude Agent SDK adapter with native interactive approvals."""

import asyncio
import json
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, List, Union
from uuid import uuid4

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    UserMessage,
)
from loguru import logger

from ...agent_runtime_commands import expand_runtime_slash_command
from .cli_agent_llm import CLIAgentLLM


class ClaudeAgentSDKLLM(CLIAgentLLM):
    """Run Claude Code through its SDK instead of a closed one-shot process."""

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        if tools:
            logger.warning("Claude Code does not forward VTuber tools")
        if error := self._configuration_error():
            yield error
            return

        prompt = (
            self._latest_user_text(messages)
            if self.session_id or self.interaction_mode == "coding"
            else self._build_prompt(messages, system)
        )
        prompt = expand_runtime_slash_command(
            prompt, "claude_code", self.workspace_directory
        )
        sdk_prompt = self._sdk_prompt(
            prompt,
            self._content_images(self._latest_user_content(messages)),
        )
        output: asyncio.Queue[Any] = asyncio.Queue()
        finished = object()

        async def forward_permissions() -> None:
            while True:
                await output.put(await self._permission_bridge.events.get())

        async def run() -> None:
            text_streamed = False
            reasoning_started = False
            reasoning_id = f"claude-{uuid4().hex}"
            try:
                options = ClaudeAgentOptions(
                    tools=[] if self.permission_mode == "disabled" else None,
                    system_prompt=(
                        system if self.interaction_mode == "character" else None
                    ),
                    permission_mode=self._sdk_permission_mode(),
                    can_use_tool=(
                        self._can_use_tool
                        if self.permission_mode != "disabled"
                        else None
                    ),
                    cwd=self.workspace_directory,
                    cli_path=self.executable,
                    model=self.model or None,
                    resume=self.session_id or None,
                    include_partial_messages=True,
                    effort=(
                        self.reasoning_effort
                        if self.reasoning_effort
                        in {"low", "medium", "high", "xhigh", "max"}
                        else None
                    ),
                    max_turns=self._max_turns(),
                )
                async with ClaudeSDKClient(options) as client:
                    await client.query(sdk_prompt)
                    async for message in client.receive_response():
                        if isinstance(message, ResultMessage):
                            self.session_id = message.session_id or self.session_id
                            if message.is_error:
                                raise RuntimeError(
                                    "\n".join(message.errors or [])
                                    or message.result
                                    or message.subtype
                                )
                            if not text_streamed and message.result:
                                await output.put(message.result.lstrip())
                                text_streamed = True
                            continue

                        if isinstance(message, StreamEvent):
                            event = message.event
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "text_delta":
                                text = delta.get("text")
                                if isinstance(text, str) and text:
                                    text_streamed = True
                                    await output.put(text)
                            if (
                                self.show_reasoning
                                and delta_type == "thinking_delta"
                                and isinstance(delta.get("thinking"), str)
                            ):
                                if not reasoning_started:
                                    reasoning_started = True
                                    await output.put(
                                        self._reasoning_event(
                                            "reasoning-start", reasoning_id
                                        )
                                    )
                                await output.put(
                                    self._reasoning_event(
                                        "reasoning-delta",
                                        reasoning_id,
                                        delta["thinking"],
                                    )
                                )
                            for activity in self._claude_activity_events(
                                {"type": "stream_event", "event": event}
                            ):
                                await output.put(activity)
                            continue

                        if not isinstance(message, (AssistantMessage, UserMessage)):
                            continue
                        payload = self._claude_message_payload(message)
                        for activity in self._claude_activity_events(payload):
                            await output.put(activity)
                        if isinstance(message, UserMessage):
                            continue
                        if text_streamed:
                            continue
                        for block in message.content:
                            if isinstance(block, TextBlock) and block.text:
                                text_streamed = True
                                await output.put(block.text.lstrip())
                            if (
                                self.show_reasoning
                                and isinstance(block, ThinkingBlock)
                                and block.thinking
                            ):
                                if not reasoning_started:
                                    reasoning_started = True
                                    await output.put(
                                        self._reasoning_event(
                                            "reasoning-start", reasoning_id
                                        )
                                    )
                                await output.put(
                                    self._reasoning_event(
                                        "reasoning-delta",
                                        reasoning_id,
                                        block.thinking,
                                    )
                                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("Claude Agent SDK request failed: {}", error)
                await output.put(self._runtime_error_text(error))
            finally:
                if reasoning_started:
                    await output.put(
                        self._reasoning_event("reasoning-end", reasoning_id)
                    )
                await output.put(finished)

        permission_task = asyncio.create_task(forward_permissions())
        runtime_task = asyncio.create_task(run())
        try:
            while True:
                item = await output.get()
                if item is finished:
                    return
                yield item
        finally:
            self._permission_bridge.cancel_all()
            permission_task.cancel()
            if not runtime_task.done():
                runtime_task.cancel()
            await asyncio.gather(
                permission_task,
                runtime_task,
                return_exceptions=True,
            )

    @staticmethod
    def _sdk_prompt(
        prompt: str, images: List[Dict[str, str]]
    ) -> str | AsyncIterator[Dict[str, Any]]:
        if not images:
            return prompt

        async def stream() -> AsyncIterator[Dict[str, Any]]:
            yield {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image["mime_type"],
                                    "data": image["data"],
                                },
                            }
                            for image in images
                        ],
                    ],
                },
                "parent_tool_use_id": None,
            }

        return stream()

    def _max_turns(self) -> int | None:
        if self.permission_mode == "disabled":
            return 1
        return None

    @staticmethod
    def _runtime_error_text(error: Exception) -> str:
        detail = " ".join(str(error).split()).strip()
        if not detail:
            return "Claude Code stopped without an error message."
        return f"Claude Code stopped: {detail}"

    async def _can_use_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        reply = await self._permission_bridge.request(
            tool_name=tool_name,
            input_data=input_data,
            title=self._permission_title(tool_name, input_data),
            description=self._permission_description(tool_name, input_data),
            force_manual=tool_name == "AskUserQuestion",
        )
        if reply.decision == "reject":
            return PermissionResultDeny(
                message=reply.message or "User rejected this action"
            )
        if tool_name == "AskUserQuestion":
            answers = self._question_answers(input_data, reply.message)
            if not answers:
                return PermissionResultDeny(
                    message="The user did not provide an answer."
                )
            return PermissionResultAllow(
                updated_input={
                    "questions": input_data.get("questions", []),
                    "answers": answers,
                }
            )
        updates = None
        if reply.decision == "always":
            updates = [
                replace(suggestion, destination="session")
                for suggestion in (context.suggestions or [])
            ]
        return PermissionResultAllow(
            updated_input=input_data,
            updated_permissions=updates,
        )

    def _sdk_permission_mode(self) -> str:
        return {
            "disabled": "dontAsk",
            "manual": "default",
            "auto": "default",
            "plan": "plan",
        }[self.permission_mode]

    @classmethod
    def _claude_message_payload(
        cls,
        message: AssistantMessage | UserMessage,
    ) -> dict[str, Any]:
        return {
            "type": "assistant" if isinstance(message, AssistantMessage) else "user",
            "message": {
                "content": (
                    [cls._claude_block(block) for block in message.content]
                    if isinstance(message.content, list)
                    else [{"type": "text", "text": message.content}]
                )
            },
        }

    @staticmethod
    def _permission_title(tool_name: str, input_data: dict[str, Any]) -> str:
        return str(
            input_data.get("description")
            or input_data.get("command")
            or input_data.get("file_path")
            or tool_name
        )

    @staticmethod
    def _permission_description(tool_name: str, input_data: dict[str, Any]) -> str:
        if tool_name == "AskUserQuestion":
            return "\n".join(
                str(question.get("question") or "")
                for question in input_data.get("questions", [])
                if isinstance(question, dict)
            )
        return str(input_data.get("description") or "")

    @staticmethod
    def _question_answers(
        input_data: dict[str, Any],
        message: str,
    ) -> dict[str, str]:
        questions = [
            question
            for question in input_data.get("questions", [])
            if isinstance(question, dict)
        ]
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            parsed = None
        answers = {}
        for index, question in enumerate(questions):
            key = str(question.get("id") or index)
            value = parsed.get(key) if isinstance(parsed, dict) else None
            values = value if isinstance(value, list) else [value]
            text = ", ".join(
                str(item) for item in values if item is not None and item != ""
            )
            if not text and index == 0 and not isinstance(parsed, dict):
                text = message.strip()
            if text:
                answers[str(question.get("question") or key)] = text
        return answers

    @staticmethod
    def _claude_block(block: Any) -> dict[str, Any]:
        names = {
            "TextBlock": "text",
            "ThinkingBlock": "thinking",
            "ToolUseBlock": "tool_use",
            "ToolResultBlock": "tool_result",
            "ServerToolUseBlock": "server_tool_use",
            "ServerToolResultBlock": "server_tool_result",
        }
        if hasattr(block, "__dict__"):
            return {
                "type": names.get(block.__class__.__name__, "unknown"),
                **block.__dict__,
            }
        return {}
