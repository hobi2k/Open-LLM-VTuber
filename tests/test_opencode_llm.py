import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from open_llm_vtuber.agent.stateless_llm.opencode_llm import OpenCodeLLM
from open_llm_vtuber.config_manager.stateless_llm import OpenCodeConfig


class OpenCodeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    prompt_started = threading.Event()
    requests = []
    use_deltas = True
    assistant_error = None

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
        if urlparse(self.path).path.endswith("/abort"):
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

    async def test_allow_tools_omits_session_permission_override(self):
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
        self.assertNotIn("permission", session_request)

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
        self.assertNotIn("permission", session_request)
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
        show_reasoning=False,
        interaction_mode="character",
    ):
        return OpenCodeLLM(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            provider_id="omlx",
            model="local-model",
            agent="vtuber",
            workspace_directory=".",
            timeout=5,
            allow_tools=allow_tools,
            show_reasoning=show_reasoning,
            interaction_mode=interaction_mode,
        )


if __name__ == "__main__":
    unittest.main()
