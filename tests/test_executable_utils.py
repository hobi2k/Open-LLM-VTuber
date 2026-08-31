import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_llm_vtuber.executable_utils import (
    executable_environment,
    executable_version,
    resolve_executable,
)


class ExecutableUtilsTest(unittest.TestCase):
    def test_auto_detects_common_cli_locations_with_an_empty_path(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            executables = {
                command: home / ".local/bin" / command
                for command in ("opencode", "claude", "codex", "hermes", "omlx")
            }
            for executable in executables.values():
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_text("#!/bin/sh\n", encoding="utf-8")
                executable.chmod(0o755)

            with patch("open_llm_vtuber.executable_utils.Path.home", return_value=home):
                with patch(
                    "open_llm_vtuber.executable_utils.os.access",
                    side_effect=lambda path, _mode: str(path).startswith(str(home)),
                ):
                    with patch.dict(os.environ, {"PATH": ""}):
                        resolved = {
                            command: resolve_executable("auto", command)
                            for command in executables
                        }

        self.assertEqual(
            resolved,
            {command: str(executable) for command, executable in executables.items()},
        )

    def test_custom_missing_binary_is_not_reported_as_available(self):
        self.assertIsNone(resolve_executable("/missing/codex", "codex"))

    def test_subprocess_environment_includes_cli_dependency_paths(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            path = executable_environment()["PATH"].split(os.pathsep)

        self.assertIn("/opt/homebrew/bin", path)
        self.assertIn(str(Path.home() / ".local/bin"), path)
        self.assertIn("/usr/bin", path)
        self.assertIn("/bin", path)

    def test_version_ignores_launcher_warning(self):
        self.assertEqual(
            executable_version("WARNING: launcher warning\ncodex-cli 1.0\n"),
            "codex-cli 1.0",
        )
