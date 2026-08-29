import unittest
from unittest.mock import patch

from open_llm_vtuber.agent_runtime_catalog import _merge_models, _project
from open_llm_vtuber.config_manager.stateless_llm import OpenCodeConfig
from open_llm_vtuber.opencode_settings import opencode_executable_payload


class AgentRuntimeCatalogTest(unittest.TestCase):
    def test_root_project_keeps_a_visible_name(self):
        self.assertEqual(_project("/", "OpenCode")["name"], "/")

    def test_model_merge_keeps_distinct_providers(self):
        models = _merge_models(
            [{"id": "local", "label": "Local", "provider": "omlx"}],
            [{"id": "local", "label": "Remote", "provider": "openai"}],
        )

        self.assertEqual(len(models), 2)


class OpenCodeExecutableTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_custom_executable_is_reported(self):
        config = OpenCodeConfig(
            executable="/missing/custom-opencode",
            provider_id="test",
            model="test",
        )
        with patch("open_llm_vtuber.opencode_settings.shutil.which", return_value=None):
            result = await opencode_executable_payload(config)

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "Executable not found")


if __name__ == "__main__":
    unittest.main()
