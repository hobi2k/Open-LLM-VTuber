import gc
import sys
from pathlib import Path
from typing import Iterable

import yaml
from pydantic import BaseModel, Field

from .config_manager import ASRConfig, TTSConfig, validate_config
from .service_context import ServiceContext


class TTSSettingsUpdate(BaseModel):
    engine: str = Field(min_length=1)
    voice: str | None = None
    asr_enabled: bool | None = None
    tts_enabled: bool | None = None


def audio_settings_payload(context: ServiceContext) -> dict:
    tts_config = context.character_config.tts_config
    asr_config = context.character_config.asr_config
    tts_engine = tts_config.tts_model
    asr_engine = asr_config.asr_model
    active_tts = getattr(tts_config, tts_engine)
    active_asr = getattr(asr_config, asr_engine)
    tts_values = active_tts.model_dump(exclude_none=True)
    asr_values = active_asr.model_dump(exclude_none=True)
    voice_field = next(
        (
            key
            for key in ("voice", "default_voice", "voice_id", "speaker")
            if key in tts_values
        ),
        None,
    )
    model = next(
        (
            str(asr_values[key])
            for key in (
                "sense_voice",
                "model",
                "model_path",
                "model_name",
                "model_size",
            )
            if asr_values.get(key)
        ),
        asr_values.get("model_type", asr_engine),
    )
    available_engines = {
        key: _tts_engine_payload(getattr(tts_config, key))
        for key, value in tts_config.model_dump(exclude_none=True).items()
        if key != "tts_model" and isinstance(value, dict)
    }
    return {
        "tts": {
            "enabled": tts_config.enabled,
            "loaded": context.tts_engine is not None,
            "engine": tts_engine,
            "available_engines": list(available_engines),
            "engines": available_engines,
            "voice": str(tts_values[voice_field]) if voice_field else None,
            "voice_field": voice_field,
            "model": next(
                (
                    str(tts_values[key])
                    for key in ("model", "model_name", "model_id")
                    if tts_values.get(key)
                ),
                None,
            ),
        },
        "asr": {
            "enabled": asr_config.enabled,
            "loaded": context.asr_engine is not None,
            "engine": asr_engine,
            "model": model,
            "model_type": asr_values.get("model_type"),
            "device": asr_values.get("device"),
        },
    }


def _tts_engine_payload(engine_config: BaseModel) -> dict:
    values = engine_config.model_dump(exclude_none=True)
    voice_field = next(
        (
            key
            for key in ("voice", "default_voice", "voice_id", "speaker")
            if key in values
        ),
        None,
    )
    return {
        "voice": str(values[voice_field]) if voice_field else None,
        "voice_field": voice_field,
        "model": next(
            (
                str(values[key])
                for key in ("model", "model_name", "model_id")
                if values.get(key)
            ),
            None,
        ),
    }


def apply_tts_settings(
    default_context: ServiceContext,
    client_contexts: Iterable[ServiceContext],
    settings: TTSSettingsUpdate,
    config_path: str | Path = "conf.yaml",
) -> None:
    contexts = [default_context, *client_contexts]
    tts_config = _updated_tts_config(
        default_context.character_config.tts_config, settings
    )
    asr_config = default_context.character_config.asr_config.model_copy(deep=True)
    if settings.tts_enabled is not None:
        tts_config.enabled = settings.tts_enabled
    if settings.asr_enabled is not None:
        asr_config.enabled = settings.asr_enabled

    default_context.init_asr(asr_config)
    default_context.init_tts(tts_config)

    for context in contexts:
        context_asr_config = asr_config.model_copy(deep=True)
        context_tts_config = tts_config.model_copy(deep=True)
        context.asr_engine = default_context.asr_engine
        context.tts_engine = default_context.tts_engine
        context.character_config.asr_config = context_asr_config
        context.character_config.tts_config = context_tts_config
        context.config.character_config.asr_config = context_asr_config
        context.config.character_config.tts_config = context_tts_config

    _persist_audio_settings(tts_config, asr_config, config_path)
    _release_model_memory()


def _updated_tts_config(current: TTSConfig, settings: TTSSettingsUpdate) -> TTSConfig:
    tts_config = current.model_copy(deep=True)
    if not hasattr(tts_config, settings.engine):
        raise ValueError(f"Unsupported TTS engine: {settings.engine}")
    engine_config = getattr(tts_config, settings.engine)
    if engine_config is None:
        raise ValueError(f"TTS engine is not configured: {settings.engine}")

    values = engine_config.model_dump(exclude_none=True)
    voice_field = next(
        (
            key
            for key in ("voice", "default_voice", "voice_id", "speaker")
            if key in values
        ),
        None,
    )
    if settings.voice is not None and voice_field is None:
        raise ValueError(
            f"TTS engine does not expose a voice setting: {settings.engine}"
        )
    if settings.voice is not None:
        setattr(engine_config, voice_field, settings.voice)
    tts_config.tts_model = settings.engine
    return tts_config


def _persist_tts_settings(tts_config: TTSConfig, config_path: str | Path) -> None:
    _persist_audio_settings(tts_config, None, config_path)


def _persist_audio_settings(
    tts_config: TTSConfig,
    asr_config: ASRConfig | None,
    config_path: str | Path,
) -> None:
    path = Path(config_path)
    config_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    persisted_tts = config_data["character_config"]["tts_config"]
    persisted_tts["enabled"] = tts_config.enabled
    persisted_tts["tts_model"] = tts_config.tts_model
    if asr_config is not None:
        config_data["character_config"]["asr_config"]["enabled"] = (
            asr_config.enabled
        )

    engine_config = getattr(tts_config, tts_config.tts_model)
    engine_values = engine_config.model_dump(exclude_none=True)
    voice_field = next(
        (
            key
            for key in ("voice", "default_voice", "voice_id", "speaker")
            if key in engine_values
        ),
        None,
    )
    if voice_field:
        persisted_tts.setdefault(tts_config.tts_model, {})[voice_field] = engine_values[
            voice_field
        ]
    validate_config(config_data)

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _release_model_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()
