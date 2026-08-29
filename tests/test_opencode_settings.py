import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from open_llm_vtuber.config_manager import read_yaml, validate_config
from open_llm_vtuber.opencode_settings import (
    OpenCodeSettingsUpdate,
    persist_opencode_settings,
    require_loopback_client,
    settings_payload,
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

    def test_persists_provider_and_open_code_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conf.yaml"
            shutil.copyfile("conf.yaml", path)
            settings = OpenCodeSettingsUpdate(
                base_url="http://127.0.0.1:4999",
                provider_id="test-provider",
                model="test-model",
                agent="test-agent",
                workspace_directory="/tmp/test-workspace",
                timeout=42,
                keep_sessions=True,
                allow_tools=True,
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


if __name__ == "__main__":
    unittest.main()
