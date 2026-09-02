import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from open_llm_vtuber.agent_runtime_history import (
    SessionHistoryRequest,
    _claude_history,
    _codex_history,
    _hermes_history,
    _opencode_local_history,
    _structured_text,
)


class AgentRuntimeHistoryTest(unittest.TestCase):
    def test_plain_numeric_message_is_not_treated_as_json_metadata(self):
        self.assertEqual(_structured_text("123"), "123")

    def test_history_request_validates_session_and_limit(self):
        request = SessionHistoryRequest(
            runtime="codex",
            session_id=" session-id ",
            workspace=" /workspace ",
            limit=250,
        )

        self.assertEqual(request.session_id, "session-id")
        self.assertEqual(request.workspace, "/workspace")
        with self.assertRaises(ValidationError):
            SessionHistoryRequest(runtime="codex", session_id="", limit=1)
        with self.assertRaises(ValidationError):
            SessionHistoryRequest(runtime="codex", session_id="session", limit=2001)

    def test_opencode_history_reads_text_reasoning_and_tool_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = home / ".local/share/opencode/opencode.db"
            path.parent.mkdir(parents=True)
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE message (id TEXT, session_id TEXT, "
                "time_created INTEGER, data TEXT)"
            )
            connection.execute(
                "CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, "
                "time_created INTEGER, data TEXT)"
            )
            connection.executemany(
                "INSERT INTO message VALUES (?, ?, ?, ?)",
                [
                    (
                        "user-message",
                        "session",
                        1000,
                        json.dumps({"role": "user", "time": {"created": 1000}}),
                    ),
                    (
                        "assistant-message",
                        "session",
                        2000,
                        json.dumps(
                            {"role": "assistant", "time": {"created": 2000}}
                        ),
                    ),
                ],
            )
            connection.executemany(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        "user-text",
                        "user-message",
                        "session",
                        1000,
                        json.dumps({"type": "text", "text": "Hello"}),
                    ),
                    (
                        "reasoning",
                        "assistant-message",
                        "session",
                        2000,
                        json.dumps({"type": "reasoning", "text": "Inspect first"}),
                    ),
                    (
                        "tool",
                        "assistant-message",
                        "session",
                        2001,
                        json.dumps(
                            {
                                "type": "tool",
                                "tool": "bash",
                                "callID": "call-1",
                                "state": {
                                    "status": "completed",
                                    "input": {"command": "pwd"},
                                    "output": "/workspace",
                                },
                            }
                        ),
                    ),
                    (
                        "assistant-text",
                        "assistant-message",
                        "session",
                        2002,
                        json.dumps({"type": "text", "text": "Done"}),
                    ),
                ],
            )
            connection.commit()
            connection.close()

            messages = _opencode_local_history("session", home)

        self.assertEqual(
            [message["type"] for message in messages],
            ["text", "reasoning", "agent_activity", "text"],
        )
        self.assertEqual(messages[0]["role"], "human")
        self.assertEqual(messages[2]["activity_kind"], "command")
        self.assertEqual(messages[2]["status"], "completed")

    def test_claude_history_ignores_tool_results_as_human_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / ".claude/projects/workspace"
            root.mkdir(parents=True)
            events = [
                {
                    "type": "user",
                    "sessionId": "session",
                    "uuid": "user-1",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {"content": "Fix the file"},
                },
                {
                    "type": "assistant",
                    "sessionId": "session",
                    "uuid": "assistant-1",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "Inspect it"},
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "Read",
                                "input": {"file_path": "/workspace/file.ts"},
                            },
                        ]
                    },
                },
                {
                    "type": "user",
                    "sessionId": "session",
                    "uuid": "tool-result",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "file contents",
                            }
                        ]
                    },
                },
                {
                    "type": "assistant",
                    "sessionId": "session",
                    "uuid": "assistant-2",
                    "timestamp": "2026-01-01T00:00:03Z",
                    "message": {"content": [{"type": "text", "text": "Fixed"}]},
                },
            ]
            (root / "session.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            messages = _claude_history("session", home)

        self.assertEqual(
            [message["type"] for message in messages],
            ["text", "reasoning", "agent_activity", "text"],
        )
        self.assertEqual(messages[2]["status"], "completed")
        self.assertEqual(messages[2]["output"], "file contents")
        self.assertEqual(
            [message["content"] for message in messages if message["role"] == "human"],
            ["Fix the file"],
        )

    def test_codex_history_uses_completed_timeline_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / ".codex"
            rollout = root / "sessions/2026/01/01/rollout-session.jsonl"
            rollout.parent.mkdir(parents=True)
            items = [
                {"type": "UserMessage", "id": "user", "content": "Build it"},
                {"type": "Reasoning", "id": "reason", "summary_text": "Plan"},
                {
                    "type": "CommandExecution",
                    "id": "command",
                    "command": "bun test",
                    "status": "completed",
                    "aggregated_output": "ok",
                    "exit_code": 0,
                },
                {"type": "AgentMessage", "id": "agent", "content": "Complete"},
            ]
            events = [
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "payload": {"type": "item_completed", "item": item},
                }
                for item in items
            ]
            events.append(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Complete"}],
                    },
                }
            )
            rollout.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            database = root / "state_5.sqlite"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT)")
            connection.execute(
                "INSERT INTO threads VALUES (?, ?)",
                ("session", str(rollout)),
            )
            connection.commit()
            connection.close()

            messages = _codex_history("session", home)

        self.assertEqual(
            [message["type"] for message in messages],
            ["text", "reasoning", "agent_activity", "text"],
        )
        self.assertEqual(messages[-1]["content"], "Complete")
        self.assertEqual(messages[2]["command"], "bun test")

    def test_hermes_history_reads_reasoning_and_finishes_tool_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = home / ".hermes/state.db"
            path.parent.mkdir()
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, "
                "content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, "
                "timestamp REAL, reasoning TEXT, reasoning_content TEXT, active INTEGER)"
            )
            connection.executemany(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                [
                    (1, "session", "user", "Run it", None, None, None, 1, None, None),
                    (
                        2,
                        "session",
                        "assistant",
                        "",
                        None,
                        json.dumps(
                            [
                                {
                                    "id": "tool-1",
                                    "function": {
                                        "name": "shell",
                                        "arguments": json.dumps({"command": "pwd"}),
                                    },
                                }
                            ]
                        ),
                        None,
                        2,
                        None,
                        "Check workspace",
                    ),
                    (
                        3,
                        "session",
                        "tool",
                        json.dumps({"success": True, "output": "/workspace"}),
                        "tool-1",
                        None,
                        "shell",
                        3,
                        None,
                        None,
                    ),
                    (4, "session", "assistant", "Done", None, None, None, 4, None, None),
                ],
            )
            connection.commit()
            connection.close()

            messages = _hermes_history("session", home)

        self.assertEqual(
            [message["type"] for message in messages],
            ["text", "reasoning", "agent_activity", "text"],
        )
        self.assertEqual(messages[2]["status"], "completed")
        self.assertEqual(messages[-1]["content"], "Done")


if __name__ == "__main__":
    unittest.main()
