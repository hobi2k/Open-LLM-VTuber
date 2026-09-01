import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from open_llm_vtuber.agent.stateless_llm.opencode_llm import OpenCodeLLM
from open_llm_vtuber.config_manager.stateless_llm import OpenCodeConfig


class OpenCodeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    prompt_started = threading.Event()
    requests = []
    use_deltas = True
    assistant_error = None
    questions = None
    question_answered = threading.Event()

    def do_POST(self):
        body = self._body()
        self.requests.append(("POST", urlparse(self.path).path, body))

        if urlparse(self.path).path == "/session":
            self._json(200, {"id": "ses_test"})
            return
        if urlparse(self.path).path.endswith("/prompt_async"):
            self.prompt_started.set()
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if urlparse(self.path).path.endswith("/command"):
            self.prompt_started.set()
            self._json(200, {"info": {"id": "msg_command"}, "parts": []})
            return
        if urlparse(self.path).path.endswith("/abort"):
            self._json(200, True)
            return
        if urlparse(self.path).path in {
            "/question/que_test/reply",
            "/question/que_test/reject",
        }:
            self.question_answered.set()
            self._json(200, True)
            return
        self._json(404, {"error": "not found"})

    def do_GET(self):
        self.requests.append(("GET", urlparse(self.path).path, None))
        if urlparse(self.path).path == "/event":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self._event("server.connected", {})
            self.prompt_started.wait(timeout=2)
            self._event(
                "message.updated",
                {
                    "sessionID": "ses_test",
                    "info": {
                        "id": "msg_assistant",
                        "role": "assistant",
                        **(
                            {"error": self.assistant_error}
                            if self.assistant_error
                            else {}
                        ),
                    },
                },
            )
            if self.assistant_error:
                self._event("session.idle", {"sessionID": "ses_test"})
                self.close_connection = True
                return
            if self.questions:
                self._event(
                    "question.asked",
                    {
                        "id": "que_test",
                        "sessionID": "ses_test",
                        "questions": self.questions,
                    },
                )
                self.question_answered.wait(timeout=2)
            self._event(
                "message.part.updated",
                {
                    "sessionID": "ses_test",
                    "part": {
                        "id": "reasoning",
                        "messageID": "msg_assistant",
                        "type": "reasoning",
                        "text": "private reasoning",
                    },
                },
            )
            self._event(
                "message.part.updated",
                {
                    "sessionID": "ses_test",
                    "part": {
                        "id": "tool-command",
                        "callID": "call-command",
                        "messageID": "msg_assistant",
                        "type": "tool",
                        "tool": "bash",
                        "state": {
                            "status": "running",
                            "input": {"command": "pwd"},
                            "title": "pwd",
                        },
                    },
                },
            )
            self._event(
                "message.part.updated",
                {
                    "sessionID": "ses_test",
                    "part": {
                        "id": "tool-command",
                        "callID": "call-command",
                        "messageID": "msg_assistant",
                        "type": "tool",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "pwd"},
                            "title": "pwd",
                            "output": "/tmp/project",
                            "metadata": {},
                        },
                    },
                },
            )
            self._event(
                "message.part.updated",
                {
                    "sessionID": "ses_test",
                    "part": {
                        "id": "tool-edit",
                        "callID": "call-edit",
                        "messageID": "msg_assistant",
                        "type": "tool",
                        "tool": "edit",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "src/app.ts"},
                            "title": "src/app.ts",
                            "output": "Edit applied successfully.",
                            "metadata": {
                                "diff": "@@ -1 +1 @@\n-old\n+new"
                            },
                        },
                    },
                },
            )
            self._event(
                "message.part.updated",
                {
                    "sessionID": "ses_test",
                    "part": {
                        "id": "text",
                        "messageID": "msg_assistant",
                        "type": "text",
                        "text": "" if self.use_deltas else "\n\n안녕하세요",
                    },
                },
            )
            if self.use_deltas:
                self._event(
                    "message.part.delta",
                    {
                        "sessionID": "ses_test",
                        "messageID": "msg_assistant",
                        "partID": "text",
                        "field": "text",
                        "delta": "\n\n안녕",
                    },
                )
                self._event(
                    "message.part.delta",
                    {
                        "sessionID": "ses_test",
                        "messageID": "msg_assistant",
                        "partID": "text",
                        "field": "text",
                        "delta": "하세요",
                    },
                )
                self._event(
                    "message.part.updated",
                    {
                        "sessionID": "ses_test",
                        "part": {
                            "id": "text",
                            "messageID": "msg_assistant",
                            "type": "text",
                            "text": "\n\n안녕하세요",
                        },
                    },
                )
            self._event("session.idle", {"sessionID": "ses_test"})
            self.close_connection = True
            return
        self._json(404, {"error": "not found"})

    def do_PATCH(self):
        body = self._body()
        self.requests.append(("PATCH", urlparse(self.path).path, body))
        if urlparse(self.path).path == "/session/ses_test":
            self._json(200, {"id": "ses_test", **(body or {})})
            return
        self._json(404, {"error": "not found"})

    def do_DELETE(self):
        self.requests.append(("DELETE", urlparse(self.path).path, None))
        self._json(200, True)

    def log_message(self, *_args):
        return

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return None
        return json.loads(self.rfile.read(length))

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _event(self, event_type, properties):
        data = json.dumps({"type": event_type, "properties": properties}).encode()
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()


class OpenCodeLLMTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        OpenCodeHandler.prompt_started = threading.Event()
        OpenCodeHandler.requests = []
        OpenCodeHandler.use_deltas = True
        OpenCodeHandler.assistant_error = None
        OpenCodeHandler.questions = None
        OpenCodeHandler.question_answered = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), OpenCodeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    async def test_streams_only_assistant_text_and_keeps_session(self):
        llm = self._llm()
        chunks = [
            chunk
            async for chunk in llm.chat_completion(
                [
                    {"role": "user", "content": "처음 질문"},
                    {"role": "assistant", "content": "처음 답변"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "이 이미지를 봐줘"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                        ],
                    },
                ],
                system="캐릭터 지시문",
            )
        ]

        self.assertEqual("".join(chunks), "안녕하세요")
        session_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/session"
        )
        self.assertEqual(
            session_request["model"],
            {"providerID": "omlx", "id": "local-model"},
        )
        self.assertEqual(
            session_request["permission"],
            [{"permission": "*", "pattern": "*", "action": "deny"}],
        )
        prompt_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path.endswith("/prompt_async")
        )
        self.assertEqual(prompt_request["system"], "캐릭터 지시문")
        self.assertIn("[ASSISTANT]\n처음 답변", prompt_request["parts"][0]["text"])
        self.assertEqual(prompt_request["parts"][1]["mime"], "image/png")
        self.assertNotIn(
            ("DELETE", "/session/ses_test", None), OpenCodeHandler.requests
        )

    async def test_slash_command_uses_native_opencode_command_endpoint(self):
        llm = self._llm()

        chunks = [
            chunk
            async for chunk in llm.chat_completion(
                [{"role": "user", "content": "/review staged changes"}]
            )
        ]

        self.assertEqual("".join(chunks), "안녕하세요")
        request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path.endswith("/command")
        )
        self.assertEqual(request["command"], "review")
        self.assertEqual(request["arguments"], "staged changes")

    def test_local_opencode_skill_expands_before_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            skill = workspace / ".opencode/skills/release/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(
                "---\nname: release\n---\nPrepare $ARGUMENTS",
                encoding="utf-8",
            )
            messages, slash_command = self._llm(
                workspace_directory=str(workspace)
            )._prepare_slash_command(
                [{"role": "user", "content": "/release frontend"}]
            )

        self.assertIsNone(slash_command)
        self.assertIn("Prepare frontend", messages[0]["content"])

    async def test_reuses_session_and_sends_only_latest_user_message(self):
        llm = self._llm()
        for _ in range(2):
            chunks = [
                chunk
                async for chunk in llm.chat_completion(
                    [
                        {"role": "assistant", "content": "Previous answer"},
                        {"role": "user", "content": "Latest question"},
                    ],
                    system="Character prompt",
                )
            ]
            self.assertEqual("".join(chunks), "안녕하세요")

        session_posts = [
            request
            for request in OpenCodeHandler.requests
            if request[0] == "POST" and request[1] == "/session"
        ]
        self.assertEqual(len(session_posts), 1)
        prompt_posts = [
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path.endswith("/prompt_async")
        ]
        self.assertIn("[ASSISTANT]", prompt_posts[0]["parts"][0]["text"])
        self.assertEqual(prompt_posts[1]["parts"][0]["text"], "Latest question")
        permission_updates = [
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "PATCH" and path == "/session/ses_test"
        ]
        self.assertEqual(
            permission_updates,
            [
                {
                    "permission": [
                        {"permission": "*", "pattern": "*", "action": "deny"}
                    ]
                }
            ],
        )

    async def test_selected_native_session_is_resumed_without_creating_a_copy(self):
        llm = self._llm(session_id="ses_test")

        chunks = [
            chunk
            async for chunk in llm.chat_completion(
                [
                    {"role": "assistant", "content": "Native session history"},
                    {"role": "user", "content": "Continue this session"},
                ]
            )
        ]

        self.assertEqual("".join(chunks), "안녕하세요")
        self.assertEqual(llm.session_id, "ses_test")
        self.assertFalse(
            any(
                method == "POST" and path == "/session"
                for method, path, _ in OpenCodeHandler.requests
            )
        )
        prompt = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/session/ses_test/prompt_async"
        )
        self.assertEqual(prompt["parts"][0]["text"], "Continue this session")

    async def test_uses_final_text_when_provider_sends_no_deltas(self):
        OpenCodeHandler.use_deltas = False
        chunks = [
            chunk
            async for chunk in self._llm().chat_completion(
                [{"role": "user", "content": "안녕"}]
            )
        ]
        self.assertEqual("".join(chunks), "안녕하세요")

    async def test_streams_reasoning_separately_when_enabled(self):
        chunks = [
            chunk
            async for chunk in self._llm(show_reasoning=True).chat_completion(
                [{"role": "user", "content": "안녕"}]
            )
        ]

        reasoning = [chunk for chunk in chunks if isinstance(chunk, dict)]
        self.assertEqual(
            [chunk["type"] for chunk in reasoning],
            ["reasoning-start", "reasoning-delta", "reasoning-end"],
        )
        self.assertEqual(reasoning[1]["text"], "private reasoning")
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "안녕하세요",
        )

    async def test_allow_tools_explicitly_auto_approves_session_tools(self):
        llm = self._llm(allow_tools=True)
        _ = [
            chunk
            async for chunk in llm.chat_completion(
                [{"role": "user", "content": "안녕"}]
            )
        ]
        session_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/session"
        )
        self.assertEqual(
            session_request["permission"],
            [{"permission": "*", "pattern": "*", "action": "allow"}],
        )

    async def test_coding_mode_uses_build_agent_without_character_prompt(self):
        llm = self._llm(interaction_mode="coding", allow_tools=True)
        _ = [
            chunk
            async for chunk in llm.chat_completion(
                [
                    {"role": "assistant", "content": "Character reply"},
                    {"role": "user", "content": "Fix this project"},
                ],
                system="Character prompt",
            )
        ]

        session_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/session"
        )
        self.assertEqual(session_request["agent"], "build")
        self.assertEqual(
            session_request["permission"],
            [{"permission": "*", "pattern": "*", "action": "allow"}],
        )
        prompt_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path.endswith("/prompt_async")
        )
        self.assertEqual(prompt_request["agent"], "build")
        self.assertNotIn("system", prompt_request)
        self.assertEqual(prompt_request["parts"][0]["text"], "Fix this project")

    async def test_coding_mode_can_keep_tools_disabled(self):
        llm = self._llm(interaction_mode="coding", allow_tools=False)
        _ = [
            chunk
            async for chunk in llm.chat_completion(
                [{"role": "user", "content": "Review this project"}]
            )
        ]

        session_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/session"
        )
        self.assertEqual(
            session_request["permission"],
            [{"permission": "*", "pattern": "*", "action": "deny"}],
        )

    async def test_manual_mode_asks_before_every_tool(self):
        llm = self._llm(interaction_mode="coding", permission_mode="manual")
        _ = [
            chunk
            async for chunk in llm.chat_completion(
                [{"role": "user", "content": "Review this project"}]
            )
        ]
        session_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/session"
        )
        self.assertEqual(
            session_request["permission"],
            [{"permission": "*", "pattern": "*", "action": "ask"}],
        )

    async def test_question_request_waits_for_ui_and_replies_in_native_format(self):
        OpenCodeHandler.questions = [
            {
                "header": "Scope",
                "question": "Which scope should be changed?",
                "options": [{"label": "Workspace"}, {"label": "Global"}],
                "multiple": False,
                "custom": True,
            },
            {
                "header": "Checks",
                "question": "Which checks should run?",
                "options": [{"label": "Tests"}, {"label": "Lint"}],
                "multiple": True,
                "custom": True,
            },
        ]
        llm = self._llm(interaction_mode="coding", permission_mode="manual")
        stream = llm.chat_completion(
            [{"role": "user", "content": "Ask before choosing scope"}]
        )

        question = await anext(stream)
        self.assertEqual(question["type"], "permission-request")
        self.assertEqual(question["tool_name"], "user_input")
        self.assertEqual(
            [item["id"] for item in question["input"]["questions"]],
            ["0", "1"],
        )
        self.assertTrue(
            await llm.respond_to_permission(
                "que_test",
                "once",
                json.dumps({"0": ["Workspace"], "1": ["Tests", "Lint"]}),
            )
        )
        remaining = [item async for item in stream]

        self.assertEqual(
            "".join(item for item in remaining if isinstance(item, str)),
            "안녕하세요",
        )
        reply = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/question/que_test/reply"
        )
        self.assertEqual(reply["answers"], [["Workspace"], ["Tests", "Lint"]])

    async def test_plan_mode_uses_plan_agent_and_read_only_tools(self):
        llm = self._llm(interaction_mode="coding", permission_mode="plan")
        _ = [
            chunk
            async for chunk in llm.chat_completion(
                [{"role": "user", "content": "Review this project"}]
            )
        ]
        session_request = next(
            body
            for method, path, body in OpenCodeHandler.requests
            if method == "POST" and path == "/session"
        )
        self.assertEqual(session_request["agent"], "plan")
        permission = session_request["permission"]
        self.assertEqual(permission[0]["action"], "deny")
        self.assertEqual(
            {rule["permission"] for rule in permission[1:]},
            {"read", "glob", "grep", "list", "lsp"},
        )

    async def test_coding_mode_streams_command_and_file_activity(self):
        chunks = [
            chunk
            async for chunk in self._llm(
                interaction_mode="coding",
                allow_tools=True,
            ).chat_completion([{"role": "user", "content": "Fix the project"}])
        ]

        activities = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("type") == "agent-activity"
        ]
        self.assertEqual(
            [(item["activity_kind"], item["status"]) for item in activities],
            [
                ("command", "running"),
                ("command", "completed"),
                ("file", "completed"),
            ],
        )
        self.assertEqual(activities[1]["output"], "/tmp/project")
        self.assertEqual(activities[2]["path"], "src/app.ts")
        self.assertIn("+new", activities[2]["diff"])

    async def test_interrupted_prompt_finishes_without_error_or_second_abort(self):
        OpenCodeHandler.assistant_error = {
            "name": "MessageAbortedError",
            "data": {"message": "Aborted"},
        }

        chunks = [
            chunk
            async for chunk in self._llm().chat_completion(
                [{"role": "user", "content": "중단될 요청"}]
            )
        ]

        self.assertEqual(chunks, [])
        self.assertNotIn(
            ("POST", "/session/ses_test/abort", None), OpenCodeHandler.requests
        )

    def test_config_defaults_are_safe(self):
        config = OpenCodeConfig(provider_id="omlx", model="local-model")
        self.assertFalse(config.allow_tools)
        self.assertFalse(config.keep_sessions)
        self.assertEqual(config.agent, "vtuber")
        self.assertEqual(config.executable, "auto")
        self.assertEqual(config.interaction_mode, "character")

    def _llm(
        self,
        allow_tools=False,
        permission_mode=None,
        show_reasoning=False,
        interaction_mode="character",
        workspace_directory=".",
        session_id="",
    ):
        return OpenCodeLLM(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            provider_id="omlx",
            model="local-model",
            agent="vtuber",
            workspace_directory=workspace_directory,
            timeout=5,
            allow_tools=allow_tools,
            permission_mode=permission_mode,
            show_reasoning=show_reasoning,
            interaction_mode=interaction_mode,
            session_id=session_id,
        )


if __name__ == "__main__":
    unittest.main()
