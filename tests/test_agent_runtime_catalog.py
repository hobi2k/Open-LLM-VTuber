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
    _merge_commands,
    _merge_models,
    _merge_sessions,
    _opencode_local_sessions,
    _opencode_catalog,
    _project,
    _runtime_configs,
)
from open_llm_vtuber.agent_runtime_commands import (
    expand_runtime_slash_command,
    local_runtime_commands,
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

    def test_codex_sessions_prefer_custom_name_over_automatic_title(self):
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
                "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, 0)",
                (
                    "named-session",
                    "Initial prompt used as automatic title",
                    "FANZA development",
                    "Initial prompt",
                    "/workspace/fanza",
                    1,
                ),
            )
            connection.commit()
            connection.close()

            sessions = _codex_sessions(home)

        self.assertEqual(sessions[0]["title"], "FANZA development")

    def test_codex_sessions_use_desktop_catalog_title_before_initial_prompt(self):
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
                "INSERT INTO threads VALUES (?, ?, '', ?, ?, ?, 0)",
                (
                    "catalog-session",
                    "A very long initial prompt used as the legacy title",
                    "A very long initial prompt",
                    "/workspace/fanza",
                    1,
                ),
            )
            connection.commit()
            connection.close()
            catalog = home / ".codex/sqlite/codex-dev.db"
            catalog.parent.mkdir()
            connection = sqlite3.connect(catalog)
            connection.execute(
                "CREATE TABLE local_thread_catalog (thread_id TEXT, "
                "display_title TEXT, source_updated_at REAL, "
                "missing_candidate INTEGER)"
            )
            connection.execute(
                "INSERT INTO local_thread_catalog VALUES (?, ?, ?, 0)",
                ("catalog-session", "Toptoon FANZA development", 1),
            )
            connection.commit()
            connection.close()

            sessions = _codex_sessions(home)

        self.assertEqual(sessions[0]["title"], "Toptoon FANZA development")

    def test_claude_sessions_prefer_custom_title_over_first_prompt(self):
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
                        "cwd": "/workspace/dlsite",
                        "message": {"content": "Initial prompt"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metadata = root / session_id
            metadata.mkdir()
            (metadata / "custom-title.json").write_text(
                json.dumps({"customTitle": "DLsite development"}),
                encoding="utf-8",
            )

            sessions = _claude_sessions(home)

        self.assertEqual(sessions[0]["title"], "DLsite development")

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

    def test_runtime_commands_include_workspace_and_user_skills(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            skill = workspace / ".claude/skills/release/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(
                "---\nname: release\ndescription: Prepare a release\n---\n"
                "Verify $ARGUMENTS",
                encoding="utf-8",
            )

            commands = local_runtime_commands(
                "claude_code", str(workspace), root / "home"
            )
            expanded = expand_runtime_slash_command(
                "/release frontend", "claude_code", str(workspace)
            )

        self.assertEqual(commands[0]["name"], "release")
        self.assertEqual(commands[0]["source"], "skill")
        self.assertIn("Verify frontend", expanded)

    def test_opencode_commands_and_skills_load_without_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            command = workspace / ".opencode/commands/review.md"
            command.parent.mkdir(parents=True)
            command.write_text(
                "---\ndescription: Review changes\n---\nReview $ARGUMENTS",
                encoding="utf-8",
            )
            skill = root / "home/.config/opencode/skills/release/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(
                "---\nname: release\ndescription: Prepare release\n---\nShip $ARGUMENTS",
                encoding="utf-8",
            )

            commands = local_runtime_commands("opencode", str(workspace), root / "home")

        self.assertEqual([item["name"] for item in commands], ["release", "review"])
        self.assertEqual(commands[0]["source"], "skill")
        self.assertEqual(commands[1]["source"], "command")

    def test_native_opencode_command_wins_over_local_fallback(self):
        commands = _merge_commands(
            [{"name": "review", "description": "Native"}],
            [
                {"name": "review", "description": "Local"},
                {"name": "release", "description": "Local release"},
            ],
        )

        self.assertEqual([item["name"] for item in commands], ["release", "review"])
        self.assertEqual(commands[1]["description"], "Native")

    def test_codex_skill_invocation_uses_native_dollar_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            skill = workspace / ".codex/skills/review/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("---\nname: review\n---\nReview changes", encoding="utf-8")

            expanded = expand_runtime_slash_command(
                "/review staged", "codex", str(workspace)
            )

        self.assertEqual(expanded, "$review staged")

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
        if request.path == "/command":
            self._json(
                [
                    {
                        "name": "review",
                        "description": "Review changes",
                        "source": "command",
                        "hints": ["$ARGUMENTS"],
                    }
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
        self.assertEqual(result["commands"][0]["name"], "review")
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
