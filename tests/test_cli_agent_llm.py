import json
import os
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
                    print(json.dumps({{
                        "result": "Claude response",
                        "session_id": "11111111-1111-1111-1111-111111111111",
                    }}))
                elif "exec" in sys.argv:
                    print(json.dumps({{"type": "thread.started", "thread_id": "test"}}))
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
        self.assertEqual(arguments[arguments.index("--reasoning") + 1], "none")
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

    def _llm(self, runtime, model="", provider=""):
        return CLIAgentLLM(
            runtime=runtime,
            executable=str(self.executable),
            model=model,
            provider=provider,
            workspace_directory=str(self.directory),
            timeout=5,
        )

    async def _complete_with(self, llm):
        chunks = [
            chunk
            async for chunk in llm.chat_completion(
                [
                    {"role": "assistant", "content": "Previous answer"},
                    {"role": "user", "content": "Hello"},
                ],
                system="Character prompt",
            )
        ]
        return "".join(chunks)


if __name__ == "__main__":
    unittest.main()
