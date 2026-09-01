"""Shared approval bridge for interactive coding runtimes."""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4


PermissionMode = Literal["disabled", "manual", "auto", "plan"]
PermissionDecision = Literal["once", "always", "reject"]


@dataclass
class PermissionReply:
    decision: PermissionDecision
    message: str = ""


class PermissionBridge:
    """Pause a runtime callback until the connected UI returns a decision."""

    def __init__(self, runtime: str, mode: PermissionMode):
        self.runtime = runtime
        self.mode = mode
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[PermissionReply]] = {}

    async def request(
        self,
        *,
        tool_name: str,
        input_data: Any,
        title: str = "",
        description: str = "",
        options: list[dict[str, str]] | None = None,
        force_manual: bool = False,
    ) -> PermissionReply:
        if self.mode == "auto" and not force_manual:
            return PermissionReply("always")
        if self.mode in {"disabled", "plan"} and not force_manual:
            return PermissionReply("reject", "This runtime is read-only.")

        request_id = f"permission-{uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self.events.put(
            {
                "type": "permission-request",
                "request_id": request_id,
                "runtime": self.runtime,
                "tool_name": tool_name,
                "title": title or tool_name,
                "description": description,
                "input": input_data,
                "options": options
                or [
                    {"id": "once", "label": "Allow once"},
                    {"id": "always", "label": "Allow for session"},
                    {"id": "reject", "label": "Reject"},
                ],
            }
        )
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def respond(
        self,
        request_id: str,
        decision: str,
        message: str = "",
    ) -> bool:
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        if decision not in {"once", "always", "reject"}:
            return False
        future.set_result(PermissionReply(decision, message))
        return True

    def cancel_all(self, message: str = "The request was cancelled.") -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_result(PermissionReply("reject", message))


def permission_mode_from_legacy(
    permission_mode: str | None,
    allow_tools: bool,
) -> PermissionMode:
    if permission_mode in {"disabled", "manual", "auto", "plan"}:
        return permission_mode
    return "auto" if allow_tools else "disabled"
