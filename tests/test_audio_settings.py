import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from open_llm_vtuber.audio_settings import (
    TTSSettingsUpdate,
    _persist_tts_settings,
    _updated_tts_config,
    audio_settings_payload,
)
from open_llm_vtuber.config_manager import read_yaml, validate_config


class AudioSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = validate_config(read_yaml("conf.yaml"))

    def test_payload_reports_active_audio_engines(self):
        context = SimpleNamespace(character_config=self.config.character_config)

        payload = audio_settings_payload(context)

        self.assertEqual(payload["tts"]["engine"], "edge_tts")
        self.assertEqual(payload["tts"]["voice"], "en-US-AvaMultilingualNeural")
        self.assertEqual(
            payload["tts"]["engines"]["edge_tts"]["voice"],
            "en-US-AvaMultilingualNeural",
        )
        self.assertEqual(payload["asr"]["engine"], "sherpa_onnx_asr")
        self.assertEqual(payload["asr"]["model_type"], "sense_voice")

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


if __name__ == "__main__":
    unittest.main()
