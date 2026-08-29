import unittest

from open_llm_vtuber.agent_runtime_catalog import _merge_models, _project


class AgentRuntimeCatalogTest(unittest.TestCase):
    def test_root_project_keeps_a_visible_name(self):
        self.assertEqual(_project("/", "OpenCode")["name"], "/")

    def test_model_merge_keeps_distinct_providers(self):
        models = _merge_models(
            [{"id": "local", "label": "Local", "provider": "omlx"}],
            [{"id": "local", "label": "Remote", "provider": "openai"}],
        )

        self.assertEqual(len(models), 2)


if __name__ == "__main__":
    unittest.main()
