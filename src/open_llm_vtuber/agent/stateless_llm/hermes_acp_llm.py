"""Hermes ACP adapter with native approval callbacks."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Union
from uuid import uuid4

import acp
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
    ToolCallProgress,
    ToolCallStart,
)
from loguru import logger

from ...agent_runtime_commands import expand_runtime_slash_command
from ...executable_utils import executable_environment
from .agent_activity import tool_activity
from .cli_agent_llm import CLIAgentLLM, _SUBPROCESS_STREAM_LIMIT


class _HermesACPClient:
    def __init__(self, llm: "HermesACPLLM"):
        self.llm = llm
        self.events: asyncio.Queue[Union[str, Dict[str, Any]]] = asyncio.Queue()
        self.connection = None

    def on_connect(self, connection) -> None:
        self.connection = connection

    async def request_permission(
        self,
        options,
        session_id: str,
        tool_call,
        **_: Any,
    ) -> RequestPermissionResponse:
        normalized = [
            {
                "id": str(option.option_id),
                "label": str(option.name),
                "kind": str(option.kind),
            }
            for option in options
        ]
        raw_input = getattr(tool_call, "raw_input", None)
        input_data = raw_input if raw_input is not None else {}
        reply = await self.llm._permission_bridge.request(
            tool_name=str(getattr(tool_call, "kind", None) or "tool"),
            input_data=input_data,
            title=str(getattr(tool_call, "title", None) or "Permission request"),
            options=[
                {"id": self._decision(option), "label": option["label"]}
                for option in normalized
            ],
        )
        option = self._select_option(normalized, reply.decision)
        if option is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=option["id"])
        )

    async def session_update(self, session_id: str, update, **_: Any) -> None:
        if isinstance(update, AgentMessageChunk):
            text = getattr(update.content, "text", None)
            if isinstance(text, str) and text:
                await self.events.put(text)
            return
        if isinstance(update, AgentThoughtChunk):
            if not self.llm.show_reasoning:
                return
            text = getattr(update.content, "text", None)
            if isinstance(text, str) and text:
                await self.events.put(
                    {
                        "type": "hermes-reasoning-delta",
                        "text": text,
                    }
                )
            return
        if not isinstance(update, (ToolCallStart, ToolCallProgress)):
            return
        await self.events.put(self._tool_activity(update))

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"Unsupported Hermes ACP client method: {method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("Ignoring Hermes ACP notification {}", method)

    @staticmethod
    def _decision(option: dict[str, str]) -> str:
        if option["kind"] == "allow_once":
            return "once"
        if option["kind"] == "allow_always":
            return "always"
        return "reject"

    @staticmethod
    def _select_option(
        options: list[dict[str, str]],
        decision: str,
    ) -> dict[str, str] | None:
        if decision == "once":
            return next(
                (option for option in options if option["kind"] == "allow_once"),
                None,
            )
        if decision == "always":
            return next(
                (
                    option
                    for option in options
                    if "session" in option["id"]
                    or option["kind"] == "allow_always"
                ),
                None,
            )
        return next(
            (option for option in options if option["kind"].startswith("reject")),
            None,
        )

    @staticmethod
    def _tool_activity(update) -> dict[str, Any]:
        kind = str(getattr(update, "kind", None) or "other")
        status_value = str(getattr(update, "status", None) or "in_progress")
        status = {
            "pending": "running",
            "in_progress": "running",
            "completed": "completed",
            "failed": "error",
        }.get(status_value, "running")
        raw_input = getattr(update, "raw_input", None)
        raw_output = getattr(update, "raw_output", None)
        input_data = raw_input if isinstance(raw_input, dict) else {}
        return tool_activity(
            activity_id=str(
                getattr(update, "tool_call_id", None) or f"hermes-{uuid4().hex}"
            ),
            tool_name=(
                "command"
                if kind == "execute"
                else "file_change"
                if kind in {"edit", "delete", "move"}
                else kind
            ),
            status=status,
            input_data=input_data,
            title=str(getattr(update, "title", None) or kind),
            output=raw_output if status == "completed" else None,
            error=raw_output if status == "error" else None,
        )


class HermesACPLLM(CLIAgentLLM):
    """Run Hermes through Agent Client Protocol instead of single-query mode."""

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        if tools:
            logger.warning("Hermes does not forward VTuber tools")
        if error := self._configuration_error():
            yield error
            return
        prompt = (
            self._latest_user_text(messages)
            if self.session_id or self.interaction_mode == "coding"
            else self._build_prompt(messages, system)
        )
        prompt = expand_runtime_slash_command(prompt, "hermes", self.workspace_directory)
        if self.permission_mode == "plan":
            prompt = (
                "Plan mode is active. Inspect and reason about the project, but do not "
                "modify files or run state-changing commands. Produce a concrete plan.\n\n"
                f"{prompt}"
            )
        elif self.permission_mode == "manual":
            prompt = (
                "Interactive approval mode is active. A denied permission is an "
                "authoritative user decision. Do not retry the same action through a "
                "different tool, shell command, script, or API after it is denied. "
                "Explain that the action was not performed instead.\n\n"
                f"{prompt}"
            )

        client = _HermesACPClient(self)
        reasoning_id = f"hermes-{uuid4().hex}"
        reasoning_started = False
        text_started = False
        arguments = self._acp_arguments()
        environment = self._acp_environment()
        deadline = asyncio.get_running_loop().time() + self.timeout
        stderr_task = None
        prompt_task = None
        event_task = None
        permission_task = None

        try:
            async with acp.spawn_agent_process(
                client,
                self.executable,
                *arguments,
                cwd=self.workspace_directory,
                env=environment,
                transport_kwargs={"limit": _SUBPROCESS_STREAM_LIMIT},
                use_unstable_protocol=True,
            ) as (connection, process):
                if process.stderr:
                    stderr_task = asyncio.create_task(process.stderr.read())
                await connection.initialize(
                    protocol_version=acp.PROTOCOL_VERSION,
                    client_capabilities=None,
                    client_info={
                        "name": "open-llm-vtuber",
                        "title": "Open LLM VTuber",
                        "version": "1.2.1",
                    },
                )
                if self.session_id:
                    session = await connection.load_session(
                        cwd=self.workspace_directory,
                        session_id=self.session_id,
                        mcp_servers=[],
                    )
                else:
                    session = await connection.new_session(
                        cwd=self.workspace_directory,
                        mcp_servers=[],
                    )
                if session is None:
                    raise RuntimeError("Hermes ACP could not open the selected session")
                self.session_id = str(session.session_id)
                await self._configure_session(connection)

                prompt_task = asyncio.create_task(
                    connection.prompt(
                        prompt=[acp.text_block(prompt)],
                        session_id=self.session_id,
                    )
                )
                event_task = asyncio.create_task(client.events.get())
                permission_task = asyncio.create_task(
                    self._permission_bridge.events.get()
                )
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    done, _ = await asyncio.wait(
                        {prompt_task, event_task, permission_task},
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        raise asyncio.TimeoutError
                    if event_task in done:
                        event = event_task.result()
                        event_task = asyncio.create_task(client.events.get())
                        if (
                            isinstance(event, dict)
                            and event.get("type") == "hermes-reasoning-delta"
                        ):
                            if not reasoning_started:
                                reasoning_started = True
                                yield self._reasoning_event(
                                    "reasoning-start", reasoning_id
                                )
                            yield self._reasoning_event(
                                "reasoning-delta",
                                reasoning_id,
                                str(event.get("text") or ""),
                            )
                        else:
                            if isinstance(event, str):
                                text_started = text_started or bool(event.strip())
                            yield event
                    if permission_task in done:
                        yield permission_task.result()
                        permission_task = asyncio.create_task(
                            self._permission_bridge.events.get()
                        )
                    if prompt_task in done:
                        await prompt_task
                        if client.events.empty():
                            event_task.cancel()
                            permission_task.cancel()
                            await asyncio.gather(
                                event_task,
                                permission_task,
                                return_exceptions=True,
                            )
                            break

                if reasoning_started:
                    yield self._reasoning_event("reasoning-end", reasoning_id)
                if not text_started:
                    raise RuntimeError("Hermes completed without an assistant response")
        except asyncio.TimeoutError:
            logger.error("Hermes ACP timed out after {} seconds", self.timeout)
            if reasoning_started:
                yield self._reasoning_event("reasoning-end", reasoning_id)
            yield "Hermes timed out. Check the runtime settings."
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Hermes ACP request failed: {}", error)
            if reasoning_started:
                yield self._reasoning_event("reasoning-end", reasoning_id)
            yield "Could not get a response from Hermes. Check the runtime settings."
        finally:
            self._permission_bridge.cancel_all()
            pending = [
                task
                for task in (prompt_task, event_task, permission_task)
                if task and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if stderr_task:
                stderr = await stderr_task
                if stderr:
                    logger.debug(
                        "Hermes ACP stderr: {}",
                        stderr.decode("utf-8", errors="replace").strip(),
                    )

    def _acp_arguments(self) -> list[str]:
        arguments = []
        if self.permission_mode == "auto":
            arguments.append("--yolo")
        arguments.append("acp")
        return arguments

    async def _configure_session(self, connection) -> None:
        if self.session_id is None:
            raise RuntimeError("Hermes ACP session is unavailable")
        if self.model:
            await connection.set_session_model(
                model_id=self._model_id(),
                session_id=self.session_id,
            )
        await connection.set_session_mode(
            mode_id=self._session_mode(),
            session_id=self.session_id,
        )

    def _model_id(self) -> str:
        # The ACP process is launched with the requested provider. A bare model ID
        # preserves named custom providers such as oMLX, which Hermes normalizes
        # internally to its "custom" transport.
        return self.model

    def _acp_environment(self) -> dict[str, str]:
        environment = executable_environment()
        provider = "omlx" if self.launch_mode == "omlx" else self.provider
        if provider:
            environment["HERMES_INFERENCE_PROVIDER"] = provider
        return environment

    def _session_mode(self) -> str:
        if self.permission_mode == "auto":
            return "dont_ask"
        return "default"
