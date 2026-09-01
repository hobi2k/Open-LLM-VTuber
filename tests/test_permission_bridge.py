import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from open_llm_vtuber.agent.stateless_llm.claude_agent_sdk_llm import (
    ClaudeAgentSDKLLM,
)
from open_llm_vtuber.agent.stateless_llm.codex_app_server_llm import (
    CodexAppServerLLM,
)
from open_llm_vtuber.agent.stateless_llm.hermes_acp_llm import HermesACPLLM
from open_llm_vtuber.agent.stateless_llm.permission_bridge import PermissionBridge
from open_llm_vtuber.agent.stateless_llm_factory import LLMFactory
from open_llm_vtuber.websocket_handler import WebSocketHandler
from claude_agent_sdk.types import (
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    ToolPermissionContext,
    ToolResultBlock,
    UserMessage,
)


class PermissionBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_manual_mode_waits_for_matching_response(self):
        bridge = PermissionBridge("codex", "manual")
        request = asyncio.create_task(
            bridge.request(
                tool_name="command",
                input_data={"command": "printf test"},
                title="printf test",
            )
        )

        event = await asyncio.wait_for(bridge.events.get(), 1)
        self.assertEqual(event["type"], "permission-request")
        self.assertEqual(event["runtime"], "codex")
        self.assertEqual(event["input"]["command"], "printf test")
        self.assertFalse(request.done())
        self.assertFalse(await bridge.respond("unknown", "once"))
        self.assertTrue(await bridge.respond(event["request_id"], "always", "ok"))

        reply = await asyncio.wait_for(request, 1)
        self.assertEqual(reply.decision, "always")
        self.assertEqual(reply.message, "ok")

    async def test_noninteractive_modes_never_wait_for_ui(self):
        auto = await PermissionBridge("claude_code", "auto").request(
            tool_name="Bash", input_data={}
        )
        disabled = await PermissionBridge("claude_code", "disabled").request(
            tool_name="Bash", input_data={}
        )
        plan = await PermissionBridge("claude_code", "plan").request(
            tool_name="Bash", input_data={}
        )

        self.assertEqual(auto.decision, "always")
        self.assertEqual(disabled.decision, "reject")
        self.assertEqual(plan.decision, "reject")

    async def test_cancel_all_releases_a_waiting_runtime(self):
        bridge = PermissionBridge("hermes", "manual")
        request = asyncio.create_task(
            bridge.request(tool_name="execute", input_data={})
        )
        await asyncio.wait_for(bridge.events.get(), 1)

        bridge.cancel_all()

        self.assertEqual((await asyncio.wait_for(request, 1)).decision, "reject")

    async def test_forced_user_input_waits_even_in_auto_mode(self):
        bridge = PermissionBridge("codex", "auto")
        request = asyncio.create_task(
            bridge.request(
                tool_name="user_input",
                input_data={"questions": []},
                force_manual=True,
            )
        )

        event = await asyncio.wait_for(bridge.events.get(), 1)
        self.assertFalse(request.done())
        self.assertTrue(
            await bridge.respond(event["request_id"], "once", "user answer")
        )
        self.assertEqual(
            (await asyncio.wait_for(request, 1)).message,
            "user answer",
        )


class NativeAdapterFactoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create(self, provider: str):
        return LLMFactory.create_llm(
            provider,
            executable="/usr/bin/true",
            workspace_directory=str(self.workspace),
            permission_mode="manual",
        )

    def test_factory_uses_bidirectional_native_adapters(self):
        self.assertIsInstance(self._create("claude_code_llm"), ClaudeAgentSDKLLM)
        self.assertIsInstance(self._create("codex_cli_llm"), CodexAppServerLLM)
        self.assertIsInstance(self._create("hermes_cli_llm"), HermesACPLLM)


class NativeSessionResumeTest(unittest.IsolatedAsyncioTestCase):
    async def test_claude_resumes_selected_session_without_replaying_history(self):
        class Client:
            options = None
            prompt = None

            def __init__(self, options):
                Client.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def query(self, prompt):
                Client.prompt = prompt

            async def receive_response(self):
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=1,
                    session_id="native-claude",
                    result="continued",
                )

        llm = ClaudeAgentSDKLLM(
            runtime="claude_code",
            executable="/usr/bin/true",
            workspace_directory=tempfile.gettempdir(),
            permission_mode="disabled",
            session_id="native-claude",
        )
        with patch(
            "open_llm_vtuber.agent.stateless_llm.claude_agent_sdk_llm.ClaudeSDKClient",
            Client,
        ):
            chunks = [
                chunk
                async for chunk in llm.chat_completion(
                    [
                        {"role": "assistant", "content": "Native history"},
                        {"role": "user", "content": "Continue here"},
                    ]
                )
            ]

        self.assertEqual(chunks, ["continued"])
        self.assertEqual(Client.options.resume, "native-claude")
        self.assertEqual(Client.prompt, "Continue here")
        self.assertEqual(llm.session_id, "native-claude")

    async def test_codex_resumes_selected_thread_without_starting_a_copy(self):
        process = SimpleNamespace(
            returncode=0,
            stderr=SimpleNamespace(read=AsyncMock(return_value=b"")),
        )
        llm = CodexAppServerLLM(
            runtime="codex",
            executable="/usr/bin/true",
            workspace_directory=tempfile.gettempdir(),
            permission_mode="manual",
            session_id="native-codex",
        )
        methods = []

        async def request(_process, _request_id, method, params, _backlog):
            methods.append((method, params))
            if method == "initialize":
                return {}
            if method == "thread/resume":
                return {"thread": {"id": "native-codex"}}
            raise AssertionError(f"Unexpected Codex method: {method}")

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "continued"}},
            },
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)),
            patch.object(llm, "_request", side_effect=request),
            patch.object(llm, "_write_json", AsyncMock()),
            patch.object(llm, "_read_json", AsyncMock(side_effect=messages)),
        ):
            chunks = [
                chunk
                async for chunk in llm.chat_completion(
                    [
                        {"role": "assistant", "content": "Native history"},
                        {"role": "user", "content": "Continue here"},
                    ]
                )
            ]

        self.assertEqual(chunks, ["continued"])
        self.assertNotIn("thread/start", [method for method, _ in methods])
        resume = next(params for method, params in methods if method == "thread/resume")
        self.assertEqual(resume["threadId"], "native-codex")
        self.assertEqual(llm.session_id, "native-codex")

    async def test_hermes_loads_selected_session_without_creating_a_copy(self):
        connection = SimpleNamespace(
            initialize=AsyncMock(),
            load_session=AsyncMock(
                return_value=SimpleNamespace(session_id="native-hermes")
            ),
            new_session=AsyncMock(),
            set_session_mode=AsyncMock(),
            prompt=None,
        )
        process = SimpleNamespace(stderr=None)

        class Spawned:
            def __init__(self, client):
                async def prompt(**_kwargs):
                    await client.events.put("continued")

                connection.prompt = prompt

            async def __aenter__(self):
                return connection, process

            async def __aexit__(self, *_args):
                return None

        llm = HermesACPLLM(
            runtime="hermes",
            executable="/usr/bin/true",
            workspace_directory=tempfile.gettempdir(),
            permission_mode="manual",
            session_id="native-hermes",
        )
        with patch(
            "open_llm_vtuber.agent.stateless_llm.hermes_acp_llm.acp.spawn_agent_process",
            side_effect=lambda client, *_args, **_kwargs: Spawned(client),
        ):
            chunks = [
                chunk
                async for chunk in llm.chat_completion(
                    [
                        {"role": "assistant", "content": "Native history"},
                        {"role": "user", "content": "Continue here"},
                    ]
                )
            ]

        self.assertEqual(chunks, ["continued"])
        connection.load_session.assert_awaited_once_with(
            cwd=llm.workspace_directory,
            session_id="native-hermes",
            mcp_servers=[],
        )
        connection.new_session.assert_not_awaited()
        self.assertEqual(llm.session_id, "native-hermes")


class NativeImageInputTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.content = [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ]
        self.images = ClaudeAgentSDKLLM._content_images(self.content)

    def test_data_url_is_parsed_once_for_native_runtimes(self):
        self.assertEqual(
            self.images,
            [
                {
                    "url": "data:image/png;base64,AAAA",
                    "mime_type": "image/png",
                    "data": "AAAA",
                }
            ],
        )
        self.assertEqual(
            ClaudeAgentSDKLLM._content_text(self.content),
            "Describe this image\n[Attached image]",
        )

    async def test_claude_uses_structured_multimodal_user_message(self):
        prompt = ClaudeAgentSDKLLM._sdk_prompt("Describe this image", self.images)

        self.assertFalse(isinstance(prompt, str))
        message = await anext(prompt)
        self.assertEqual(message["message"]["content"][0]["type"], "text")
        self.assertEqual(
            message["message"]["content"][1],
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "AAAA",
                },
            },
        )

    def test_codex_uses_official_image_turn_input(self):
        self.assertEqual(
            CodexAppServerLLM._image_turn_input(self.images),
            [{"type": "image", "url": "data:image/png;base64,AAAA"}],
        )

    def test_hermes_uses_acp_image_content_block(self):
        blocks = HermesACPLLM._acp_prompt("Describe this image", self.images)

        self.assertEqual(blocks[0].type, "text")
        self.assertEqual(blocks[1].type, "image")
        self.assertEqual(blocks[1].mime_type, "image/png")
        self.assertEqual(blocks[1].data, "AAAA")


class HermesACPConfigurationTest(unittest.IsolatedAsyncioTestCase):
    def _llm(self, permission_mode="manual", launch_mode="direct"):
        return HermesACPLLM(
            runtime="hermes",
            executable="/usr/bin/true",
            workspace_directory=tempfile.gettempdir(),
            permission_mode=permission_mode,
            launch_mode=launch_mode,
            provider="openrouter",
            model="anthropic/claude-sonnet-4",
            session_id="session-1",
        )

    async def test_session_model_and_permission_mode_use_acp_protocol(self):
        connection = SimpleNamespace(
            set_session_model=AsyncMock(),
            set_session_mode=AsyncMock(),
        )

        await self._llm()._configure_session(connection)

        connection.set_session_model.assert_awaited_once_with(
            model_id="anthropic/claude-sonnet-4",
            session_id="session-1",
        )
        connection.set_session_mode.assert_awaited_once_with(
            mode_id="default",
            session_id="session-1",
        )

    def test_auto_mode_uses_yolo_and_dont_ask(self):
        llm = self._llm(permission_mode="auto")

        self.assertEqual(llm._acp_arguments(), ["--yolo", "acp"])
        self.assertEqual(llm._session_mode(), "dont_ask")

    def test_omlx_model_selection_keeps_provider_context(self):
        llm = self._llm(launch_mode="omlx")

        self.assertEqual(llm._model_id(), "anthropic/claude-sonnet-4")
        self.assertEqual(llm._acp_environment()["HERMES_INFERENCE_PROVIDER"], "omlx")


class ClaudeUserInputTest(unittest.TestCase):
    def test_question_answers_use_native_question_text_keys(self):
        self.assertEqual(
            ClaudeAgentSDKLLM._question_answers(
                {
                    "questions": [
                        {"question": "Which scope?"},
                        {"question": "Which checks?", "multiSelect": True},
                    ]
                },
                json.dumps({"0": ["Workspace"], "1": ["Tests", "Lint"]}),
            ),
            {
                "Which scope?": "Workspace",
                "Which checks?": "Tests, Lint",
            },
        )

    def test_user_tool_result_becomes_completed_activity(self):
        llm = ClaudeAgentSDKLLM(
            runtime="claude_code",
            executable="/usr/bin/true",
            workspace_directory=tempfile.gettempdir(),
            permission_mode="manual",
        )
        llm._activity_inputs["tool-1"] = ("Bash", {"command": "pwd"})

        activities = llm._claude_activity_events(
            llm._claude_message_payload(
                UserMessage(
                    content=[
                        ToolResultBlock(
                            tool_use_id="tool-1",
                            content="/tmp",
                            is_error=False,
                        )
                    ]
                )
            )
        )

        self.assertEqual(activities[0]["status"], "completed")
        self.assertEqual(activities[0]["output"], "/tmp")


class ClaudePermissionModeTest(unittest.IsolatedAsyncioTestCase):
    def _llm(self, mode):
        return ClaudeAgentSDKLLM(
            runtime="claude_code",
            executable="/usr/bin/true",
            workspace_directory=tempfile.gettempdir(),
            permission_mode=mode,
        )

    def test_tool_enabled_modes_do_not_cap_claude_turns(self):
        self.assertIsNone(self._llm("manual")._max_turns())
        self.assertIsNone(self._llm("auto")._max_turns())
        self.assertIsNone(self._llm("plan")._max_turns())
        self.assertEqual(self._llm("disabled")._max_turns(), 1)

    def test_runtime_error_preserves_claude_detail(self):
        detail = "You've hit your session limit · resets 7:40pm (Asia/Seoul)"

        self.assertEqual(
            ClaudeAgentSDKLLM._runtime_error_text(RuntimeError(detail)),
            f"Claude Code stopped: {detail}",
        )
        self.assertEqual(
            ClaudeAgentSDKLLM._runtime_error_text(RuntimeError()),
            "Claude Code stopped without an error message.",
        )

    async def test_sdk_error_is_streamed_with_its_actual_reason(self):
        detail = "You've hit your session limit · resets 7:40pm (Asia/Seoul)"

        class Client:
            options = None

            def __init__(self, options):
                Client.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def query(self, _prompt):
                return None

            async def receive_response(self):
                yield ResultMessage(
                    subtype="error_during_execution",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=True,
                    num_turns=1,
                    session_id="claude-session",
                    errors=[detail],
                )

        with patch(
            "open_llm_vtuber.agent.stateless_llm.claude_agent_sdk_llm.ClaudeSDKClient",
            Client,
        ):
            chunks = [
                chunk
                async for chunk in self._llm("auto").chat_completion(
                    [{"role": "user", "content": "continue"}]
                )
            ]

        self.assertEqual(chunks, [f"Claude Code stopped: {detail}"])
        self.assertIsNone(Client.options.max_turns)

    async def test_auto_mode_asks_ui_for_user_question_only(self):
        llm = self._llm("auto")
        question = asyncio.create_task(
            llm._can_use_tool(
                "AskUserQuestion",
                {"questions": [{"question": "Which scope?"}]},
                ToolPermissionContext(),
            )
        )

        event = await asyncio.wait_for(llm._permission_bridge.events.get(), 1)
        await llm.respond_to_permission(
            event["request_id"],
            "once",
            json.dumps({"0": "workspace"}),
        )

        result = await asyncio.wait_for(question, 1)
        self.assertIsInstance(result, PermissionResultAllow)
        self.assertEqual(result.updated_input["answers"], {"Which scope?": "workspace"})
        ordinary = await llm._can_use_tool(
            "Bash",
            {"command": "pwd"},
            ToolPermissionContext(),
        )
        self.assertIsInstance(ordinary, PermissionResultAllow)

    async def test_plan_mode_rejects_tools_but_keeps_questions_interactive(self):
        llm = self._llm("plan")

        result = await llm._can_use_tool(
            "Bash",
            {"command": "touch forbidden"},
            ToolPermissionContext(),
        )

        self.assertIsInstance(result, PermissionResultDeny)
        self.assertEqual(llm._sdk_permission_mode(), "plan")


class CodexUserInputTest(unittest.IsolatedAsyncioTestCase):
    def test_codex_permission_modes_use_expected_sandbox(self):
        def llm(mode):
            return CodexAppServerLLM(
                runtime="codex",
                executable="/usr/bin/true",
                workspace_directory=tempfile.gettempdir(),
                permission_mode=mode,
            )

        self.assertEqual(llm("manual")._sandbox_mode(), "workspace-write")
        self.assertEqual(llm("auto")._sandbox_mode(), "workspace-write")
        self.assertEqual(llm("plan")._sandbox_mode(), "read-only")
        self.assertEqual(llm("disabled")._sandbox_mode(), "read-only")
        self.assertEqual(llm("manual")._approval_policy(), "untrusted")
        self.assertEqual(llm("auto")._approval_policy(), "never")

    async def test_request_user_input_returns_answers_by_question_id(self):
        class Stdin:
            def __init__(self):
                self.messages = []

            def write(self, value):
                self.messages.append(json.loads(value))

            async def drain(self):
                return None

        class Process:
            stdin = Stdin()

        llm = CodexAppServerLLM(
            runtime="codex",
            executable="/usr/bin/true",
            workspace_directory=tempfile.gettempdir(),
            permission_mode="auto",
        )
        handler = llm._handle_server_request(
            Process(),
            {
                "id": 7,
                "method": "item/tool/requestUserInput",
                "params": {
                    "questions": [
                        {
                            "id": "scope",
                            "header": "Scope",
                            "question": "Which scope?",
                            "options": [],
                        }
                    ]
                },
            },
        )

        event = await anext(handler)
        self.assertEqual(event["tool_name"], "user_input")
        await llm.respond_to_permission(
            event["request_id"],
            "once",
            json.dumps({"scope": "workspace"}),
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(handler)

        self.assertEqual(
            Process.stdin.messages[-1],
            {
                "id": 7,
                "result": {
                    "answers": {"scope": {"answers": ["workspace"]}}
                },
            },
        )

    async def test_codex_read_waits_without_an_internal_timeout(self):
        class Stdout:
            async def readline(self):
                await asyncio.sleep(0.02)
                return b'{"id": 1, "result": {}}\n'

        class Process:
            stdout = Stdout()

        self.assertEqual(
            await asyncio.wait_for(CodexAppServerLLM._read_json(Process()), 1),
            {"id": 1, "result": {}},
        )

    def test_codex_skill_uses_structured_app_server_input(self):
        self.assertEqual(
            CodexAppServerLLM._skill_turn_input(
                "/browser:control-in-app-browser inspect the page",
                {
                    "data": [
                        {
                            "cwd": "/tmp",
                            "errors": [],
                            "skills": [
                                {
                                    "name": "browser:control-in-app-browser",
                                    "description": "Control the browser",
                                    "enabled": True,
                                    "path": "/skills/browser/SKILL.md",
                                    "scope": "user",
                                }
                            ],
                        }
                    ]
                },
            ),
            [
                {
                    "type": "skill",
                    "name": "browser:control-in-app-browser",
                    "path": "/skills/browser/SKILL.md",
                },
                {"type": "text", "text": "inspect the page"},
            ],
        )


class PermissionWebSocketTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_websocket_initializes_an_independent_agent(self):
        copyable = MagicMock()
        copyable.model_copy.side_effect = lambda **_: copyable
        character = SimpleNamespace(agent_config=object(), persona_prompt="persona")
        default_context = SimpleNamespace(
            config=copyable,
            system_config=copyable,
            character_config=copyable,
            live2d_model=object(),
            asr_engine=object(),
            tts_engine=object(),
            vad_engine=object(),
            agent_engine=object(),
            translate_engine=None,
            mcp_server_registery=None,
            tool_adapter=None,
        )
        context = SimpleNamespace(
            load_cache=AsyncMock(),
            init_agent=AsyncMock(),
            character_config=character,
        )

        with patch(
            "open_llm_vtuber.websocket_handler.ServiceContext",
            return_value=context,
        ):
            result = await WebSocketHandler(default_context)._init_service_context(
                AsyncMock(), "client"
            )

        self.assertIs(result, context)
        self.assertIsNone(context.load_cache.await_args.kwargs["agent_engine"])
        context.init_agent.assert_awaited_once_with(
            character.agent_config,
            character.persona_prompt,
        )

    async def test_permission_response_is_routed_to_active_runtime(self):
        class LLM:
            async def respond_to_permission(self, request_id, decision, message):
                self.response = (request_id, decision, message)
                return True

        class WebSocket:
            async def send_json(self, payload):
                self.payload = payload

        llm = LLM()
        websocket = WebSocket()
        handler = WebSocketHandler(None)
        handler.client_contexts["client"] = type(
            "Context",
            (),
            {"agent_engine": type("Agent", (), {"_llm": llm})()},
        )()

        await handler._handle_permission_response(
            websocket,
            "client",
            {
                "request_id": "permission-1",
                "decision": "once",
                "message": "approved",
            },
        )

        self.assertEqual(llm.response, ("permission-1", "once", "approved"))
        self.assertEqual(
            websocket.payload,
            {
                "type": "permission-resolved",
                "request_id": "permission-1",
                "decision": "once",
                "success": True,
            },
        )
