import unittest

from open_llm_vtuber.agent.stateless_llm.agent_activity import tool_activity


class AgentActivityTest(unittest.TestCase):
    def test_browser_tool_uses_human_title_without_exposing_code_json(self):
        event = tool_activity(
            "browser-call",
            "js",
            "completed",
            input_data={
                "code": "await browser.tabs.list()",
                "timeout_ms": 30_000,
                "title": "Inspect the active browser tab",
            },
            title="js",
            output={
                "content": [{"type": "text", "text": "Browser connected"}],
                "images": [],
            },
        )

        self.assertEqual(event["title"], "Inspect the active browser tab")
        self.assertNotIn("input", event)
        self.assertEqual(event["output"], "Browser connected")
        self.assertNotIn("{", event["output"])

    def test_unknown_structured_tool_is_rendered_as_readable_lines(self):
        event = tool_activity(
            "search-call",
            "search",
            "completed",
            input_data={"query": "native Codex tool UI", "limit": 10},
            output={"matches": 4, "status": "ok"},
        )

        self.assertEqual(event["input"], "native Codex tool UI")
        self.assertEqual(event["output"], "matches: 4\nstatus: ok")
        self.assertNotIn('"matches"', event["output"])


if __name__ == "__main__":
    unittest.main()
