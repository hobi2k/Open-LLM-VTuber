import json
import os
import sqlite3
import tempfile
import textwrap
import unittest
from pathlib import Path

from open_llm_vtuber.agent.stateless_llm.cli_agent_llm import CLIAgentLLM


class CLIAgentLLMTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.executable = self.directory / "fake-agent"
        self.arguments = self.directory / "arguments.json"
        self.stdin = self.directory / "stdin.txt"
        self.executable.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json
                import pathlib
                import sys

                pathlib.Path({str(self.arguments)!r}).write_text(json.dumps(sys.argv[1:]))
                content = sys.stdin.read()
                pathlib.Path({str(self.stdin)!r}).write_text(content)
                if "-p" in sys.argv:
                    if "stream-json" in sys.argv:
                        print(json.dumps({{
                            "type": "system",
                            "session_id": "11111111-1111-1111-1111-111111111111",
                        }}))
                        print(json.dumps({{
                            "type": "stream_event",
                            "event": {{
                                "type": "content_block_delta",
                                "delta": {{"type": "thinking_delta", "thinking": "Claude reasoning"}},
                            }},
                        }}))
                        print(json.dumps({{
                            "type": "result",
                            "result": "Claude response",
                            "session_id": "11111111-1111-1111-1111-111111111111",
                        }}))
                    else:
                        print(json.dumps({{
                            "result": "Claude response",
                            "session_id": "11111111-1111-1111-1111-111111111111",
                        }}))
                elif "exec" in sys.argv:
                    print(json.dumps({{"type": "thread.started", "thread_id": "test"}}))
                    print(json.dumps({{
                        "type": "item.completed",
                        "item": {{"type": "reasoning", "text": "Codex reasoning"}},
                    }}))
                    print(json.dumps({{
                        "type": "item.completed",
                        "item": {{"type": "agent_message", "text": "Codex response"}},
                    }}))
                elif "chat" in sys.argv:
                    print("session_id: hermes-test", file=sys.stderr)
                    print("Hermes response")
                else:
                    print("fake-agent 1.0")
                """
            ),
            encoding="utf-8",
        )
        os.chmod(self.executable, 0o755)

    def tearDown(self):
        self.temporary_directory.cleanup()

    async def test_claude_code_disables_tools_and_parses_json(self):
        response = await self._complete("claude_code", model="sonnet")

        self.assertEqual(response, "Claude response")
        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertIn("--tools", arguments)
        self.assertEqual(arguments[arguments.index("--tools") + 1], "")
        self.assertIn("sonnet", arguments)
        self.assertIn(
            "[SYSTEM]\nCharacter prompt", self.stdin.read_text(encoding="utf-8")
        )

    async def test_codex_uses_read_only_sandbox_and_parses_jsonl(self):
        response = await self._complete("codex")

        self.assertEqual(response, "Codex response")
        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertEqual(arguments[arguments.index("--sandbox") + 1], "read-only")
        self.assertNotIn("--ephemeral", arguments)
        self.assertIn("[USER]\nHello", self.stdin.read_text(encoding="utf-8"))

    async def test_hermes_keeps_provider_config_and_parses_plain_text(self):
        response = await self._complete(
            "hermes", model="test-model", provider="test-provider"
        )

        self.assertEqual(response, "Hermes response")
        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertNotIn("--safe-mode", arguments)
        self.assertIn("--ignore-rules", arguments)
        self.assertEqual(arguments[arguments.index("--toolsets") + 1], "")
        self.assertNotIn("--reasoning", arguments)
        self.assertEqual(arguments[arguments.index("--model") + 1], "test-model")
        self.assertEqual(arguments[arguments.index("--provider") + 1], "test-provider")
        prompt = arguments[arguments.index("--query") + 1]
        self.assertIn("[ASSISTANT]\nPrevious answer", prompt)

    async def test_hermes_omlx_mode_uses_named_provider(self):
        llm = self._llm("hermes", model="test-model", provider="stale-provider")
        llm.launch_mode = "omlx"

        await self._complete_with(llm)

        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertEqual(arguments[arguments.index("--provider") + 1], "omlx")

    async def test_claude_resumes_persisted_session_with_latest_message_only(self):
        llm = self._llm("claude_code")
        await self._complete_with(llm)
        await self._complete_with(llm)

        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertEqual(
            arguments[arguments.index("--resume") + 1],
            "11111111-1111-1111-1111-111111111111",
        )
        self.assertEqual(self.stdin.read_text(encoding="utf-8"), "Hello")

    async def test_claude_exposes_reasoning_without_changing_effort(self):
        chunks = await self._chunks(self._llm("claude_code", show_reasoning=True))

        self.assertEqual(
            [chunk["type"] for chunk in chunks if isinstance(chunk, dict)],
            ["reasoning-start", "reasoning-delta", "reasoning-end"],
        )
        self.assertEqual(
            next(
                chunk["text"]
                for chunk in chunks
                if isinstance(chunk, dict) and chunk["type"] == "reasoning-delta"
            ),
            "Claude reasoning",
        )
        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertIn("--include-partial-messages", arguments)
        self.assertIn("--verbose", arguments)
        self.assertNotIn("--effort", arguments)

    async def test_codex_exposes_reasoning_without_overriding_config(self):
        chunks = await self._chunks(self._llm("codex", show_reasoning=True))

        self.assertEqual(
            next(
                chunk["text"]
                for chunk in chunks
                if isinstance(chunk, dict) and chunk["type"] == "reasoning-delta"
            ),
            "Codex reasoning",
        )
        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertNotIn("model_reasoning_effort", " ".join(arguments))

    async def test_hermes_reads_reasoning_from_its_native_session(self):
        database = self.directory / "state.db"
        connection = sqlite3.connect(database)
        connection.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                reasoning_content TEXT,
                reasoning TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                active INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO messages (
                session_id, role, content, reasoning_content, active
            ) VALUES (
                'hermes-test', 'assistant', 'Hermes response',
                'Hermes reasoning', 1
            )
            """
        )
        connection.commit()
        connection.close()
        previous_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.directory)
        try:
            chunks = await self._chunks(self._llm("hermes", show_reasoning=True))
        finally:
            if previous_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous_home

        self.assertEqual(
            next(
                chunk["text"]
                for chunk in chunks
                if isinstance(chunk, dict) and chunk["type"] == "reasoning-delta"
            ),
            "Hermes reasoning",
        )
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "Hermes response",
        )

    def test_hermes_reasoning_panel_does_not_leak_into_response(self):
        llm = self._llm("hermes")

        response = llm._response_text(
            "\r\n┌─ Reasoning ─────────┐\r\n\r\nInternal notes\r\n\r\nOK\n"
        )

        self.assertEqual(response, "OK")

    async def test_codex_resumes_created_thread(self):
        llm = self._llm("codex")
        await self._complete_with(llm)
        await self._complete_with(llm)

        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertEqual(arguments[:3], ["exec", "resume", "--json"])
        self.assertIn("test", arguments)
        self.assertEqual(self.stdin.read_text(encoding="utf-8"), "Hello")

    async def test_hermes_resumes_created_session(self):
        llm = self._llm("hermes", model="test-model", provider="test-provider")
        await self._complete_with(llm)
        await self._complete_with(llm)

        arguments = json.loads(self.arguments.read_text(encoding="utf-8"))
        self.assertEqual(arguments[arguments.index("--resume") + 1], "hermes-test")
        self.assertEqual(arguments[arguments.index("--query") + 1], "Hello")

    async def _complete(self, runtime, model="", provider=""):
        return await self._complete_with(self._llm(runtime, model, provider))

    def _llm(self, runtime, model="", provider="", show_reasoning=False):
        return CLIAgentLLM(
            runtime=runtime,
            executable=str(self.executable),
            model=model,
            provider=provider,
            workspace_directory=str(self.directory),
            timeout=5,
            show_reasoning=show_reasoning,
        )

    async def _complete_with(self, llm):
        chunks = await self._chunks(llm)
        return "".join(chunk for chunk in chunks if isinstance(chunk, str))

    async def _chunks(self, llm):
        return [
            chunk
            async for chunk in llm.chat_completion(
                [
                    {"role": "assistant", "content": "Previous answer"},
                    {"role": "user", "content": "Hello"},
                ],
                system="Character prompt",
            )
        ]


if __name__ == "__main__":
    unittest.main()
