"""Codex app-server adapter with native approvals and Plan mode."""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Union
from uuid import uuid4

from loguru import logger

from ...agent_runtime_commands import (
    codex_skills_from_response,
    codex_slash_command,
)
from ...executable_utils import executable_environment
from .agent_activity import tool_activity
from .cli_agent_llm import CLIAgentLLM, _SUBPROCESS_STREAM_LIMIT


class CodexAppServerLLM(CLIAgentLLM):
    """Run Codex through its bidirectional JSON-RPC app-server protocol."""

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        if tools:
            logger.warning("Codex does not forward VTuber tools")
        if error := self._configuration_error():
            yield error
            return
        prompt = (
            self._latest_user_text(messages)
            if self.session_id or self.interaction_mode == "coding"
            else self._build_prompt(messages, system)
        )
        images = self._content_images(self._latest_user_content(messages))
        slash_command = codex_slash_command(prompt)
        process = None
        stderr_task = None
        reasoning_id = f"codex-{uuid4().hex}"
        reasoning_started = False
        text_started = False
        backlog: list[dict[str, Any]] = []

        try:
            process = await asyncio.create_subprocess_exec(
                self.executable,
                "app-server",
                "--stdio",
                cwd=self.workspace_directory,
                env=executable_environment(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_SUBPROCESS_STREAM_LIMIT,
            )
            stderr_task = asyncio.create_task(process.stderr.read())
            await self._request(
                process,
                1,
                "initialize",
                {
                    "clientInfo": {
                        "name": "open-llm-vtuber",
                        "title": "Open LLM VTuber",
                        "version": "1.2.1",
                    },
                    "capabilities": {"experimentalApi": True},
                },
                backlog,
            )
            await self._write_json(process, {"method": "initialized", "params": {}})

            request_id = 2
            turn_input = [
                {"type": "text", "text": prompt},
                *self._image_turn_input(images),
            ]
            if slash_command:
                skill_result = await self._request(
                    process,
                    request_id,
                    "skills/list",
                    {"cwds": [self.workspace_directory], "forceReload": False},
                    backlog,
                )
                request_id += 1
                turn_input = [
                    *self._skill_turn_input(prompt, skill_result),
                    *self._image_turn_input(images),
                ]

            thread_params = {
                "cwd": self.workspace_directory,
                "model": self.model or None,
                "approvalPolicy": self._approval_policy(),
                "approvalsReviewer": "user",
                "sandbox": self._sandbox_mode(),
            }
            if self.permission_mode == "disabled":
                thread_params["developerInstructions"] = (
                    "Do not call tools, run commands, inspect files, browse, or modify "
                    "the workspace. Answer only from the conversation text."
                )
            if self.session_id:
                thread_params["threadId"] = self.session_id
                thread_result = await self._request(
                    process,
                    request_id,
                    "thread/resume",
                    thread_params,
                    backlog,
                )
            else:
                thread_result = await self._request(
                    process,
                    request_id,
                    "thread/start",
                    thread_params,
                    backlog,
                )
            thread = thread_result.get("thread", {})
            thread_id = thread.get("id")
            if not isinstance(thread_id, str) or not thread_id:
                raise RuntimeError("Codex app-server did not return a thread ID")
            self.session_id = thread_id
            await self._apply_new_session_title()
            request_id += 1

            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": turn_input,
                "approvalPolicy": self._approval_policy(),
                "approvalsReviewer": "user",
                "cwd": self.workspace_directory,
            }
            if self.model:
                turn_params["model"] = self.model
            if self.reasoning_effort != "default":
                turn_params["effort"] = self.reasoning_effort
            if self.permission_mode == "plan":
                active_model = str(thread_result.get("model") or self.model)
                if not active_model:
                    raise RuntimeError("Codex did not report an active model")
                turn_params["collaborationMode"] = {
                    "mode": "plan",
                    "settings": {
                        "model": active_model,
                        "reasoning_effort": (
                            None
                            if self.reasoning_effort == "default"
                            else self.reasoning_effort
                        ),
                        "developer_instructions": None,
                    },
                }
            await self._write_json(
                process,
                {"id": request_id, "method": "turn/start", "params": turn_params},
            )

            while True:
                message = (
                    backlog.pop(0)
                    if backlog
                    else await self._read_json(process)
                )
                method = message.get("method")
                params = message.get("params", {})

                if "id" in message and isinstance(method, str):
                    async for event in self._handle_server_request(process, message):
                        yield event
                    continue

                if method == "item/agentMessage/delta":
                    text = params.get("delta")
                    if isinstance(text, str) and text:
                        text_started = True
                        yield text
                    continue

                if method in {
                    "item/reasoning/textDelta",
                    "item/reasoning/summaryTextDelta",
                }:
                    text = params.get("delta")
                    if self.show_reasoning and isinstance(text, str) and text:
                        if not reasoning_started:
                            reasoning_started = True
                            yield self._reasoning_event(
                                "reasoning-start", reasoning_id
                            )
                        yield self._reasoning_event(
                            "reasoning-delta", reasoning_id, text
                        )
                    continue

                if method in {"item/started", "item/completed"}:
                    item = params.get("item", {})
                    activity = self._activity_event(
                        item,
                        completed=method == "item/completed",
                    )
                    if activity and self.interaction_mode == "coding":
                        yield activity
                    if (
                        not text_started
                        and method == "item/completed"
                        and item.get("type") == "agentMessage"
                        and isinstance(item.get("text"), str)
                        and item["text"]
                    ):
                        text_started = True
                        yield item["text"].lstrip()
                    continue

                if method == "turn/completed":
                    turn = params.get("turn", {})
                    status = str(turn.get("status") or "completed")
                    if status in {"failed", "error"}:
                        error = turn.get("error")
                        raise RuntimeError(str(error or "Codex turn failed"))
                    break

            if reasoning_started:
                yield self._reasoning_event("reasoning-end", reasoning_id)
            if not text_started:
                raise RuntimeError("Codex completed without an assistant response")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Codex app-server request failed: {}", error)
            if reasoning_started:
                yield self._reasoning_event("reasoning-end", reasoning_id)
            yield "Could not get a response from Codex. Check the runtime settings."
        finally:
            self._permission_bridge.cancel_all()
            if process and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            if stderr_task:
                stderr = await stderr_task
                if stderr:
                    logger.debug(
                        "Codex app-server stderr: {}",
                        stderr.decode("utf-8", errors="replace").strip(),
                    )

    async def _handle_server_request(
        self,
        process: asyncio.subprocess.Process,
        message: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        method = str(message.get("method") or "")
        params = message.get("params", {})
        logger.debug("Codex app-server request {}: {}", method, params)
        if method == "item/tool/requestUserInput":
            async for event in self._handle_user_input_request(
                process,
                message,
            ):
                yield event
            return
        tool_name = {
            "item/commandExecution/requestApproval": "command",
            "item/fileChange/requestApproval": "file_change",
            "item/permissions/requestApproval": "permissions",
        }.get(method)
        if not tool_name:
            logger.warning("Unsupported Codex app-server request: {}", method)
            await self._write_json(
                process,
                {
                    "id": message["id"],
                    "error": {"code": -32601, "message": "Unsupported request"},
                },
            )
            return

        title = str(
            params.get("command")
            or params.get("reason")
            or params.get("itemId")
            or tool_name
        )
        reply_task = asyncio.create_task(
            self._permission_bridge.request(
                tool_name=tool_name,
                input_data=params,
                title=title,
                description=str(params.get("reason") or ""),
            )
        )
        if self.permission_mode == "manual":
            yield await self._permission_bridge.events.get()
        reply = await reply_task

        if method == "item/permissions/requestApproval":
            result = {
                "permissions": (
                    params.get("permissions", {})
                    if reply.decision != "reject"
                    else {}
                ),
                "scope": "session" if reply.decision == "always" else "turn",
            }
        else:
            result = {
                "decision": {
                    "once": "accept",
                    "always": "acceptForSession",
                    "reject": "decline",
                }[reply.decision]
            }
        await self._write_json(
            process,
            {"id": message["id"], "result": result},
        )

    async def _handle_user_input_request(
        self,
        process: asyncio.subprocess.Process,
        message: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        params = message.get("params", {})
        questions = params.get("questions", [])
        question_list = [item for item in questions if isinstance(item, dict)]
        title = str(
            question_list[0].get("header")
            if question_list
            else "Codex needs your input"
        )
        description = "\n".join(
            str(item.get("question") or "") for item in question_list
        ).strip()
        reply_task = asyncio.create_task(
            self._permission_bridge.request(
                tool_name="user_input",
                input_data={"questions": question_list},
                title=title,
                description=description,
                options=[
                    {"id": "once", "label": "Submit answer"},
                    {"id": "reject", "label": "Cancel"},
                ],
                force_manual=True,
            )
        )
        yield await self._permission_bridge.events.get()
        reply = await reply_task

        answer_values: dict[str, list[str]] = {}
        if reply.decision != "reject" and reply.message.strip():
            try:
                parsed = json.loads(reply.message)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                answer_values = {
                    str(key): (
                        [str(item) for item in value]
                        if isinstance(value, list)
                        else [str(value)]
                    )
                    for key, value in parsed.items()
                }
            elif question_list:
                answer_values[str(question_list[0].get("id") or "answer")] = [
                    reply.message.strip()
                ]

        await self._write_json(
            process,
            {
                "id": message["id"],
                "result": {
                    "answers": {
                        question_id: {"answers": answers}
                        for question_id, answers in answer_values.items()
                    }
                },
            },
        )

    async def _request(
        self,
        process: asyncio.subprocess.Process,
        request_id: int,
        method: str,
        params: dict[str, Any],
        backlog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        await self._write_json(
            process,
            {"id": request_id, "method": method, "params": params},
        )
        while True:
            message = await self._read_json(process)
            if message.get("id") != request_id or "method" in message:
                backlog.append(message)
                continue
            if "error" in message:
                raise RuntimeError(str(message["error"]))
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    @staticmethod
    async def _write_json(
        process: asyncio.subprocess.Process,
        message: dict[str, Any],
    ) -> None:
        if process.stdin is None:
            raise RuntimeError("Codex app-server stdin is unavailable")
        process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await process.stdin.drain()

    @staticmethod
    async def _read_json(process: asyncio.subprocess.Process) -> dict[str, Any]:
        if process.stdout is None:
            raise RuntimeError("Codex app-server stdout is unavailable")
        line = await process.stdout.readline()
        if not line:
            raise RuntimeError("Codex app-server closed unexpectedly")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid Codex app-server message: {line!r}") from error
        if not isinstance(message, dict):
            raise RuntimeError("Invalid Codex app-server response")
        return message

    def _approval_policy(self) -> str:
        if self.permission_mode in {"auto", "disabled"}:
            return "never"
        return "untrusted"

    def _sandbox_mode(self) -> str:
        if self.permission_mode in {"manual", "auto"}:
            return "workspace-write"
        return "read-only"

    @staticmethod
    def _image_turn_input(images: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"type": "image", "url": image["url"]} for image in images]

    @staticmethod
    def _skill_turn_input(prompt: str, skill_result: dict) -> list[dict[str, str]]:
        slash_command = codex_slash_command(prompt)
        if slash_command is None:
            return [{"type": "text", "text": prompt}]
        skill = next(
            (
                item
                for item in codex_skills_from_response(skill_result)
                if item["name"] == slash_command[0]
            ),
            None,
        )
        if skill is None:
            return [{"type": "text", "text": prompt}]
        result = [
            {
                "type": "skill",
                "name": skill["name"],
                "path": skill["path"],
            }
        ]
        if slash_command[1]:
            result.append({"type": "text", "text": slash_command[1]})
        return result

    @staticmethod
    def _activity_event(item: dict[str, Any], *, completed: bool) -> dict | None:
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        status = "completed" if completed else "running"
        if str(item.get("status") or "").lower() in {"failed", "error", "declined"}:
            status = "error"
        activity_id = str(item.get("id") or f"codex-{uuid4().hex}")

        if item_type == "commandExecution":
            command = str(item.get("command") or "")
            return tool_activity(
                activity_id,
                "command",
                status,
                input_data={"command": command},
                title=command,
                output=item.get("aggregatedOutput"),
                metadata={"exit_code": item.get("exitCode")},
            )
        if item_type == "fileChange":
            changes = item.get("changes") or []
            paths = [
                str(change.get("path"))
                for change in changes
                if isinstance(change, dict) and change.get("path")
            ]
            return tool_activity(
                activity_id,
                "file_change",
                status,
                input_data={"path": "\n".join(paths)},
                title="\n".join(paths) or "File changes",
                output=changes if completed else None,
            )
        if item_type != "mcpToolCall":
            return None
        name = str(item.get("tool") or "mcp_tool")
        return tool_activity(
            activity_id,
            name,
            status,
            input_data=(
                item.get("arguments")
                if isinstance(item.get("arguments"), dict)
                else {"arguments": item.get("arguments")}
            ),
            title=name,
            output=item.get("result"),
            error=item.get("error"),
        )
