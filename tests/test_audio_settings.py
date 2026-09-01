import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from open_llm_vtuber.audio_settings import (
    TTSSettingsUpdate,
    apply_tts_settings,
    _persist_tts_settings,
    _updated_tts_config,
    audio_settings_payload,
)
from open_llm_vtuber.agent.output_types import DisplayText
from open_llm_vtuber.config_manager import read_yaml, validate_config
from open_llm_vtuber.conversations.conversation_utils import process_user_input
from open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from open_llm_vtuber.service_context import ServiceContext


class AudioSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = validate_config(read_yaml("config_templates/conf.default.yaml"))

    def test_payload_reports_active_audio_engines(self):
        context = SimpleNamespace(
            character_config=self.config.character_config,
            asr_engine=object(),
            tts_engine=object(),
        )

        payload = audio_settings_payload(context)

        self.assertTrue(payload["tts"]["enabled"])
        self.assertTrue(payload["tts"]["loaded"])
        self.assertEqual(payload["tts"]["engine"], "edge_tts")
        self.assertEqual(payload["tts"]["voice"], "en-US-AvaMultilingualNeural")
        self.assertEqual(
            payload["tts"]["engines"]["edge_tts"]["voice"],
            "en-US-AvaMultilingualNeural",
        )
        self.assertEqual(payload["asr"]["engine"], "sherpa_onnx_asr")
        self.assertEqual(payload["asr"]["model_type"], "sense_voice")
        self.assertTrue(payload["asr"]["enabled"])
        self.assertTrue(payload["asr"]["loaded"])

    def test_existing_config_defaults_audio_engines_to_enabled(self):
        self.assertTrue(self.config.character_config.asr_config.enabled)
        self.assertTrue(self.config.character_config.tts_config.enabled)

    def test_update_changes_edge_voice_without_mutating_original(self):
        current = self.config.character_config.tts_config

        updated = _updated_tts_config(
            current,
            TTSSettingsUpdate(engine="edge_tts", voice="ja-JP-NanamiNeural"),
        )

        self.assertEqual(updated.edge_tts.voice, "ja-JP-NanamiNeural")
        self.assertEqual(current.edge_tts.voice, "en-US-AvaMultilingualNeural")

    def test_persist_updates_only_active_engine_settings(self):
        updated = _updated_tts_config(
            self.config.character_config.tts_config,
            TTSSettingsUpdate(engine="edge_tts", voice="ja-JP-NanamiNeural"),
        )

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "conf.yaml"
            config_path.write_text(
                Path("conf.yaml").read_text(encoding="utf-8"), encoding="utf-8"
            )
            original = read_yaml(config_path)

            _persist_tts_settings(updated, config_path)
            persisted = read_yaml(config_path)

        self.assertEqual(
            persisted["character_config"]["tts_config"]["edge_tts"]["voice"],
            "ja-JP-NanamiNeural",
        )
        self.assertEqual(
            persisted["character_config"]["tts_config"]["bark_tts"],
            original["character_config"]["tts_config"]["bark_tts"],
        )

    def test_disabling_audio_releases_all_shared_engine_references(self):
        default_context = self._context(self.config.model_copy(deep=True))
        client_context = self._context(self.config.model_copy(deep=True))

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "conf.yaml"
            config_path.write_text(
                Path("conf.yaml").read_text(encoding="utf-8"), encoding="utf-8"
            )
            apply_tts_settings(
                default_context,
                [client_context],
                TTSSettingsUpdate(
                    engine="edge_tts",
                    asr_enabled=False,
                    tts_enabled=False,
                ),
                config_path,
            )
            persisted = read_yaml(config_path)

        self.assertIsNone(default_context.asr_engine)
        self.assertIsNone(default_context.tts_engine)
        self.assertIsNone(client_context.asr_engine)
        self.assertIsNone(client_context.tts_engine)
        self.assertFalse(default_context.character_config.asr_config.enabled)
        self.assertFalse(default_context.character_config.tts_config.enabled)
        self.assertFalse(persisted["character_config"]["asr_config"]["enabled"])
        self.assertFalse(persisted["character_config"]["tts_config"]["enabled"])

    def test_enabling_audio_initializes_once_and_shares_engines(self):
        default_config = self.config.model_copy(deep=True)
        default_config.character_config.asr_config.enabled = False
        default_config.character_config.tts_config.enabled = False
        client_config = default_config.model_copy(deep=True)
        default_context = self._context(default_config, loaded=False)
        client_context = self._context(client_config, loaded=False)
        asr_engine = object()
        tts_engine = object()

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "conf.yaml"
            config_path.write_text(
                Path("conf.yaml").read_text(encoding="utf-8"), encoding="utf-8"
            )
            with (
                patch(
                    "open_llm_vtuber.service_context.ASRFactory.get_asr_system",
                    return_value=asr_engine,
                ) as create_asr,
                patch(
                    "open_llm_vtuber.service_context.TTSFactory.get_tts_engine",
                    return_value=tts_engine,
                ) as create_tts,
            ):
                apply_tts_settings(
                    default_context,
                    [client_context],
                    TTSSettingsUpdate(
                        engine="edge_tts",
                        asr_enabled=True,
                        tts_enabled=True,
                    ),
                    config_path,
                )

        create_asr.assert_called_once()
        create_tts.assert_called_once()
        self.assertIs(default_context.asr_engine, asr_engine)
        self.assertIs(client_context.asr_engine, asr_engine)
        self.assertIs(default_context.tts_engine, tts_engine)
        self.assertIs(client_context.tts_engine, tts_engine)

    @staticmethod
    def _context(config, loaded=True):
        context = ServiceContext()
        context.config = config
        context.system_config = config.system_config
        context.character_config = config.character_config
        context.asr_engine = object() if loaded else None
        context.tts_engine = object() if loaded else None
        return context


class DisabledAudioBehaviorTest(unittest.IsolatedAsyncioTestCase):
    async def test_audio_input_fails_cleanly_when_asr_is_disabled(self):
        async def send_message(_message):
            return None

        with self.assertRaisesRegex(RuntimeError, "ASR is disabled"):
            await process_user_input(
                np.zeros(160, dtype=np.float32), None, send_message
            )

    async def test_tts_off_sends_display_text_without_synthesis(self):
        messages = []

        async def send_message(message):
            messages.append(json.loads(message))

        manager = TTSTaskManager()
        await manager.speak(
            "Text response",
            DisplayText(text="Text response"),
            None,
            SimpleNamespace(),
            None,
            send_message,
        )
        await asyncio.wait_for(manager._payload_queue.join(), timeout=1)
        manager.clear()

        self.assertEqual(manager.task_list, [])
        self.assertEqual(messages[0]["display_text"]["text"], "Text response")
        self.assertIsNone(messages[0]["audio"])


if __name__ == "__main__":
    unittest.main()
