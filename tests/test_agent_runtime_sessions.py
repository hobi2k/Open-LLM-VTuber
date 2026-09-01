import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from open_llm_vtuber.agent_runtime_catalog import (
    _claude_sessions,
    _codex_sessions,
    _hermes_sessions,
    _opencode_local_sessions,
)
from open_llm_vtuber.agent_runtime_sessions import (
    SessionRenameRequest,
    _rename_claude_local,
    _rename_codex_local,
    _rename_hermes_local,
    _rename_opencode_local,
)


class AgentRuntimeSessionRenameTest(unittest.TestCase):
    def test_rename_request_normalizes_and_validates_title(self):
        request = SessionRenameRequest(
            runtime="codex",
            session_id=" codex-session ",
            title="  FANZA\n development  ",
        )

        self.assertEqual(request.session_id, "codex-session")
        self.assertEqual(request.title, "FANZA development")
        with self.assertRaises(ValidationError):
            SessionRenameRequest(
                runtime="codex",
                session_id="codex-session",
                title="   ",
            )

    def test_opencode_rename_updates_native_database(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = home / ".local/share/opencode/opencode-test.db"
            path.parent.mkdir(parents=True)
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE session (id TEXT, title TEXT, directory TEXT, "
                "time_updated INTEGER, time_archived INTEGER, parent_id TEXT)"
            )
            connection.execute(
                "INSERT INTO session VALUES (?, ?, ?, ?, NULL, NULL)",
                ("opencode-session", "Old title", "/workspace", 1),
            )
            connection.commit()
            connection.close()

            self.assertTrue(
                _rename_opencode_local(
                    "opencode-session",
                    "New OpenCode title",
                    home,
                )
            )
            sessions = _opencode_local_sessions(home)

        self.assertEqual(sessions[0]["title"], "New OpenCode title")

    def test_codex_rename_updates_custom_name(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = home / ".codex/state_5.sqlite"
            path.parent.mkdir()
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE threads (id TEXT, title TEXT, name TEXT, "
                "first_user_message TEXT, cwd TEXT, updated_at INTEGER, "
                "archived INTEGER)"
            )
            connection.execute(
                "INSERT INTO threads VALUES (?, ?, '', '', ?, ?, 0)",
                ("codex-session", "Automatic title", "/workspace", 1),
            )
            connection.commit()
            connection.close()

            self.assertTrue(
                _rename_codex_local(
                    "codex-session",
                    "New Codex title",
                    home,
                )
            )
            sessions = _codex_sessions(home)

        self.assertEqual(sessions[0]["title"], "New Codex title")

    def test_claude_rename_writes_native_custom_title(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / ".claude/projects/test"
            root.mkdir(parents=True)
            session_id = "claude-session"
            (root / f"{session_id}.jsonl").write_text(
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": session_id,
                        "cwd": "/workspace",
                        "message": {"content": "Automatic title"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertTrue(
                _rename_claude_local(
                    session_id,
                    "New Claude title",
                    home,
                )
            )
            metadata = json.loads(
                (root / session_id / "custom-title.json").read_text(encoding="utf-8")
            )
            sessions = _claude_sessions(home)

        self.assertEqual(metadata["customTitle"], "New Claude title")
        self.assertEqual(sessions[0]["title"], "New Claude title")

    def test_hermes_rename_updates_manual_native_title(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = home / ".hermes/state.db"
            path.parent.mkdir()
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE sessions (id TEXT, title TEXT, title_source TEXT, "
                "display_name TEXT, cwd TEXT, git_repo_root TEXT, "
                "last_activity_at INTEGER, started_at INTEGER, archived INTEGER, "
                "hidden INTEGER, source TEXT, message_count INTEGER)"
            )
            connection.execute(
                "CREATE TABLE messages (session_id TEXT, role TEXT, active INTEGER, "
                "content TEXT, timestamp INTEGER)"
            )
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, '', ?, '', ?, ?, 0, 0, "
                "'tui', 1)",
                (
                    "hermes-session",
                    "Automatic title",
                    "auto",
                    "/workspace",
                    1,
                    1,
                ),
            )
            connection.commit()
            connection.close()

            self.assertTrue(
                _rename_hermes_local(
                    "hermes-session",
                    "New Hermes title",
                    home,
                )
            )
            connection = sqlite3.connect(path)
            source = connection.execute(
                "SELECT title_source FROM sessions WHERE id = ?",
                ("hermes-session",),
            ).fetchone()[0]
            connection.close()
            sessions = _hermes_sessions(home)

        self.assertEqual(source, "manual")
        self.assertEqual(sessions[0]["title"], "New Hermes title")


if __name__ == "__main__":
    unittest.main()
