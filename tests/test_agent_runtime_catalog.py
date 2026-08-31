import json
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from open_llm_vtuber.agent_runtime_catalog import (
    _claude_sessions,
    _codex_models,
    _codex_sessions,
    _hermes_sessions,
    _merge_models,
    _merge_sessions,
    _opencode_local_sessions,
    _opencode_catalog,
    _project,
    _runtime_configs,
)
from open_llm_vtuber.agent_runtime_settings import (
    AgentRuntimeSettingsUpdate,
    CLISettingsUpdate,
)
from open_llm_vtuber.config_manager import read_yaml, validate_config
from open_llm_vtuber.config_manager.stateless_llm import OpenCodeConfig
from open_llm_vtuber.opencode_settings import opencode_executable_payload
from open_llm_vtuber.opencode_settings import OpenCodeSettingsUpdate


class AgentRuntimeCatalogTest(unittest.TestCase):
    def test_root_project_keeps_a_visible_name(self):
        self.assertEqual(_project("/", "OpenCode")["name"], "/")

    def test_model_merge_keeps_distinct_providers(self):
        models = _merge_models(
            [{"id": "local", "label": "Local", "provider": "omlx"}],
            [{"id": "local", "label": "Remote", "provider": "openai"}],
        )

        self.assertEqual(len(models), 2)

    def test_codex_models_include_supported_reasoning_efforts(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = home / ".codex/models_cache.json"
            cache.parent.mkdir()
            cache.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "slug": "test-model",
                                "display_name": "Test model",
                                "visibility": "list",
                                "supported_reasoning_levels": [
                                    {"effort": "low"},
                                    {"effort": "high"},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "open_llm_vtuber.agent_runtime_catalog.Path.home",
                return_value=home,
            ):
                models = _codex_models()

        self.assertEqual(models[0]["reasoning_efforts"], ["low", "high"])

    def test_local_catalogs_return_more_than_fifty_native_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._create_codex_sessions(home, 61)
            self._create_hermes_sessions(home, 62)
            self._create_claude_sessions(home, 63)

            self.assertEqual(len(_codex_sessions(home)), 61)
            self.assertEqual(len(_hermes_sessions(home)), 62)
            self.assertEqual(len(_claude_sessions(home)), 63)

    def test_opencode_sessions_load_without_a_running_server(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = home / ".local/share/opencode/opencode-dev.db"
            path.parent.mkdir(parents=True)
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE session (id TEXT, title TEXT, directory TEXT, "
                "time_updated INTEGER, time_archived INTEGER, parent_id TEXT)"
            )
            connection.executemany(
                "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("main", "FANZA development", "/workspace/fanza", 2, None, None),
                    ("child", "Subagent", "/workspace/fanza", 3, None, "main"),
                    ("archived", "Old", "/workspace", 1, 4, None),
                ],
            )
            connection.commit()
            connection.close()

            sessions = _opencode_local_sessions(home)

        self.assertEqual([session["id"] for session in sessions], ["main"])
        self.assertEqual(sessions[0]["workspace"], "/workspace/fanza")

    def test_session_merge_keeps_latest_copy_and_sort_order(self):
        sessions = _merge_sessions(
            [{"id": "same", "updated_at": 1}, {"id": "old", "updated_at": 2}],
            [{"id": "same", "updated_at": 3}],
        )

        self.assertEqual([session["id"] for session in sessions], ["same", "old"])

    def test_catalog_uses_current_unsaved_settings(self):
        config = validate_config(read_yaml("conf.yaml"))
        context = type("Context", (), {"character_config": config.character_config})()
        cli = CLISettingsUpdate(workspace_directory="/current/cli")
        settings = AgentRuntimeSettingsUpdate(
            provider="codex_cli_llm",
            opencode=OpenCodeSettingsUpdate(
                base_url="http://127.0.0.1:4096",
                provider_id="test",
                model="test",
                workspace_directory="/current/opencode",
            ),
            claude_code=cli,
            codex=cli.model_copy(update={"workspace_directory": "/current/codex"}),
            hermes=cli.model_copy(update={"workspace_directory": "/current/hermes"}),
        )

        opencode, claude, codex, hermes = _runtime_configs(context, settings)

        self.assertEqual(opencode.workspace_directory, "/current/opencode")
        self.assertEqual(claude.workspace_directory, "/current/cli")
        self.assertEqual(codex.workspace_directory, "/current/codex")
        self.assertEqual(hermes.workspace_directory, "/current/hermes")

    @staticmethod
    def _create_codex_sessions(home: Path, count: int):
        path = home / ".codex/state_5.sqlite"
        path.parent.mkdir()
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE threads (id TEXT, title TEXT, name TEXT, "
            "first_user_message TEXT, cwd TEXT, updated_at INTEGER, archived INTEGER)"
        )
        connection.executemany(
            "INSERT INTO threads VALUES (?, ?, '', '', ?, ?, 0)",
            (
                (f"codex-{index}", f"Codex {index}", "/codex", index)
                for index in range(count)
            ),
        )
        connection.commit()
        connection.close()

    @staticmethod
    def _create_hermes_sessions(home: Path, count: int):
        path = home / ".hermes/state.db"
        path.parent.mkdir()
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE sessions (id TEXT, title TEXT, display_name TEXT, cwd TEXT, "
            "git_repo_root TEXT, last_activity_at INTEGER, started_at INTEGER, "
            "archived INTEGER, hidden INTEGER, source TEXT, message_count INTEGER)"
        )
        connection.execute(
            "CREATE TABLE messages (session_id TEXT, role TEXT, active INTEGER, "
            "content TEXT, timestamp INTEGER)"
        )
        connection.executemany(
            "INSERT INTO sessions VALUES (?, ?, '', ?, '', ?, ?, 0, 0, 'tui', 1)",
            (
                (f"hermes-{index}", f"Hermes {index}", "/hermes", index, index)
                for index in range(count)
            ),
        )
        connection.commit()
        connection.close()

    @staticmethod
    def _create_claude_sessions(home: Path, count: int):
        root = home / ".claude/projects/test"
        root.mkdir(parents=True)
        for index in range(count):
            (root / f"claude-{index}.jsonl").write_text(
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": f"claude-{index}",
                        "cwd": "/claude",
                        "message": {"content": f"Claude {index}"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )


class OpenCodeCatalogHandler(BaseHTTPRequestHandler):
    directories = []

    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/provider":
            self._json({"connected": [], "all": []})
            return
        if request.path == "/project":
            self._json(
                [
                    {"name": "One", "worktree": "/workspace/one"},
                    {"name": "Two", "worktree": "/workspace/two"},
                ]
            )
            return
        if request.path == "/session":
            directory = parse_qs(request.query)["directory"][0]
            self.directories.append(directory)
            self._json(
                [
                    {
                        "id": f"session-{Path(directory).name}",
                        "title": Path(directory).name,
                        "directory": directory,
                        "time": {"updated": len(self.directories)},
                    }
                ]
            )
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args):
        return

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class OpenCodeCatalogTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        OpenCodeCatalogHandler.directories = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), OpenCodeCatalogHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    async def test_sessions_are_collected_from_every_opencode_project(self):
        result = await _opencode_catalog(
            OpenCodeConfig(
                base_url=f"http://127.0.0.1:{self.server.server_port}",
                workspace_directory="/workspace/current",
                provider_id="test",
                model="test",
            )
        )

        self.assertEqual(len(result["sessions"]), 3)
        self.assertEqual(
            set(OpenCodeCatalogHandler.directories),
            {"/workspace/current", "/workspace/one", "/workspace/two"},
        )


class OpenCodeExecutableTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_custom_executable_is_reported(self):
        config = OpenCodeConfig(
            executable="/missing/custom-opencode",
            provider_id="test",
            model="test",
        )
        with patch(
            "open_llm_vtuber.opencode_settings.resolve_executable",
            return_value=None,
        ):
            result = await opencode_executable_payload(config)

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "Executable not found")


if __name__ == "__main__":
    unittest.main()
