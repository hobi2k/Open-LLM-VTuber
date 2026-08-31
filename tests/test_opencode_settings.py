import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml

from open_llm_vtuber.config_manager import read_yaml, validate_config
from open_llm_vtuber.opencode_settings import (
    OpenCodeSettingsUpdate,
    persist_opencode_settings,
    require_loopback_client,
    settings_payload,
)
from open_llm_vtuber.opencode_runtime import (
    _parse_listener_urls,
    discover_or_start_opencode,
)
from open_llm_vtuber.config_manager.stateless_llm import OpenCodeConfig
from open_llm_vtuber.agent_runtime_settings import (
    AgentRuntimeSettingsUpdate,
    CLISettingsUpdate,
    persist_runtime_settings,
    runtime_settings_payload,
)


class OpenCodeSettingsTest(unittest.TestCase):
    def test_accepts_only_local_clients(self):
        require_loopback_client("127.0.0.1")
        require_loopback_client("::1")
        require_loopback_client("localhost")
        with self.assertRaises(PermissionError):
            require_loopback_client("192.0.2.1")

    def test_settings_payload_never_returns_server_password(self):
        config = validate_config(read_yaml("conf.yaml"))
        opencode_config = config.character_config.agent_config.llm_configs.opencode_llm
        opencode_config.server_password = "secret"
        context = SimpleNamespace(character_config=config.character_config)

        payload = settings_payload(context)

        self.assertNotIn("server_password", payload)
        self.assertTrue(payload["has_server_password"])
        self.assertEqual(payload["executable"], "auto")

    def test_persists_provider_and_open_code_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conf.yaml"
            shutil.copyfile("conf.yaml", path)
            settings = OpenCodeSettingsUpdate(
                executable="/tmp/custom-opencode",
                base_url="http://127.0.0.1:4999",
                provider_id="test-provider",
                model="test-model",
                agent="test-agent",
                interaction_mode="coding",
                workspace_directory="/tmp/test-workspace",
                timeout=42,
                keep_sessions=True,
                allow_tools=True,
                show_reasoning=True,
            )

            persist_opencode_settings(settings, path)

            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
            agent_config = saved["character_config"]["agent_config"]
            self.assertEqual(
                agent_config["agent_settings"]["basic_memory_agent"]["llm_provider"],
                "opencode_llm",
            )
            self.assertEqual(
                agent_config["llm_configs"]["opencode_llm"]["model"],
                "test-model",
            )
            self.assertEqual(
                agent_config["llm_configs"]["opencode_llm"]["executable"],
                "/tmp/custom-opencode",
            )
            validate_config(saved)
            self.assertTrue(
                agent_config["llm_configs"]["opencode_llm"]["show_reasoning"]
            )
            self.assertEqual(
                agent_config["llm_configs"]["opencode_llm"]["interaction_mode"],
                "coding",
            )

    def test_persists_each_cli_runtime_and_selected_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conf.yaml"
            shutil.copyfile("conf.yaml", path)
            cli = CLISettingsUpdate(
                executable="/tmp/fake-cli",
                model="test-model",
                workspace_directory="/tmp",
                timeout=45,
                interaction_mode="coding",
                reasoning_effort="high",
                allow_tools=True,
            )
            settings = AgentRuntimeSettingsUpdate(
                provider="hermes_cli_llm",
                opencode=OpenCodeSettingsUpdate(
                    base_url="http://127.0.0.1:4096",
                    provider_id="test-provider",
                    model="test-model",
                ),
                claude_code=cli,
                codex=cli.model_copy(update={"executable": "/tmp/fake-codex"}),
                hermes=cli.model_copy(
                    update={
                        "executable": "/tmp/fake-hermes",
                        "provider": "test-provider",
                    }
                ),
            )

            persist_runtime_settings(settings, path)

            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
            agent_config = saved["character_config"]["agent_config"]
            self.assertEqual(
                agent_config["agent_settings"]["basic_memory_agent"]["llm_provider"],
                "hermes_cli_llm",
            )
            self.assertEqual(
                agent_config["llm_configs"]["claude_code_llm"]["executable"],
                "/tmp/fake-cli",
            )
            self.assertEqual(
                agent_config["llm_configs"]["codex_cli_llm"]["executable"],
                "/tmp/fake-codex",
            )
            self.assertEqual(
                agent_config["llm_configs"]["hermes_cli_llm"]["provider"],
                "test-provider",
            )
            self.assertEqual(
                agent_config["llm_configs"]["codex_cli_llm"]["interaction_mode"],
                "coding",
            )
            self.assertTrue(
                agent_config["llm_configs"]["claude_code_llm"]["allow_tools"]
            )
            self.assertEqual(
                agent_config["llm_configs"]["claude_code_llm"]["reasoning_effort"],
                "high",
            )
            validate_config(saved)

    def test_all_character_presets_reference_installed_live2d_models(self):
        models = {
            model["name"]: model
            for model in json.loads(Path("model_dict.json").read_text(encoding="utf-8"))
        }
        base = yaml.safe_load(Path("conf.yaml").read_text(encoding="utf-8"))
        base_model = base["character_config"]["live2d_model_name"]

        for path in Path("characters").glob("*.yaml"):
            preset = yaml.safe_load(path.read_text(encoding="utf-8"))
            model_name = preset["character_config"].get(
                "live2d_model_name",
                base_model,
            )
            self.assertIn(model_name, models, path.name)
            model_path = Path(models[model_name]["url"].lstrip("/"))
            self.assertTrue(model_path.is_file(), f"{path.name}: {model_path}")

    def test_default_templates_configure_all_agent_runtimes(self):
        for path in (
            Path("config_templates/conf.default.yaml"),
            Path("config_templates/conf.ZH.default.yaml"),
        ):
            config = validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))
            llm_configs = config.character_config.agent_config.llm_configs
            self.assertIsNotNone(llm_configs.opencode_llm, path)
            self.assertIsNotNone(llm_configs.claude_code_llm, path)
            self.assertIsNotNone(llm_configs.codex_cli_llm, path)
            self.assertIsNotNone(llm_configs.hermes_cli_llm, path)


class OpenCodeRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def test_parses_random_opencode_listener_ports(self):
        output = "\n".join(
            [
                "p100",
                "cOpenCode",
                "n127.0.0.1:54329",
                "p200",
                "cpython3",
                "n127.0.0.1:8005",
                "p300",
                "copencode",
                "n*:4096",
            ]
        )

        self.assertEqual(
            _parse_listener_urls(output),
            ["http://127.0.0.1:54329", "http://127.0.0.1:4096"],
        )

    async def test_discovers_running_server_without_a_configured_port(self):
        detected = "http://127.0.0.1:54329"
        config = OpenCodeConfig(
            base_url="http://127.0.0.1:1",
            workspace_directory="/tmp",
            provider_id="test",
            model="test",
        )
        with (
            patch(
                "open_llm_vtuber.opencode_runtime._listener_urls",
                new=AsyncMock(return_value=[detected]),
            ),
            patch(
                "open_llm_vtuber.opencode_runtime._health",
                new=AsyncMock(
                    side_effect=[None, {"healthy": True, "version": "test"}]
                ),
            ),
        ):
            result = await discover_or_start_opencode(
                config,
                executable=None,
                auto_start=False,
            )

        self.assertTrue(result["connected"])
        self.assertEqual(result["base_url"], detected)
        self.assertEqual(result["source"], "detected")
        self.assertFalse(result["managed"])

class AgentRuntimeSettingsPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_loading_settings_does_not_probe_runtimes(self):
        config = validate_config(read_yaml("conf.yaml"))
        context = SimpleNamespace(
            character_config=config.character_config,
            agent_engine=SimpleNamespace(),
        )

        with (
            patch(
                "open_llm_vtuber.agent_runtime_settings.opencode_connection_payload",
                new=AsyncMock(),
            ) as opencode_probe,
            patch(
                "open_llm_vtuber.agent_runtime_settings._cli_connection_payload",
                new=AsyncMock(),
            ) as cli_probe,
        ):
            payload = await runtime_settings_payload(context)

        opencode_probe.assert_not_awaited()
        cli_probe.assert_not_awaited()
        self.assertFalse(payload["opencode"]["connection"]["connected"])
        self.assertIsNone(payload["opencode"]["connection"]["error"])
        self.assertFalse(payload["codex"]["connection"]["available"])
        self.assertIsNone(payload["codex"]["connection"]["error"])


if __name__ == "__main__":
    unittest.main()
