"""Unit tests for application configuration."""

from historyverse.config.settings import Settings
from historyverse.core.enums import DeviceType, LLMProvider


def test_settings_default_values():
    settings = Settings()
    assert settings.api_port == 8000
    assert settings.llm_provider in list(LLMProvider)
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.drift_confidence_threshold > 0.0
    assert settings.voice_enabled is True
    assert settings.vision_enabled is True
    assert settings.video_enabled is True


def test_settings_device_types():
    settings_cpu = Settings(llm_device_type=DeviceType.CPU)
    assert settings_cpu.llm_device_type == DeviceType.CPU

    settings_cuda = Settings(llm_device_type=DeviceType.CUDA)
    assert settings_cuda.llm_device_type == DeviceType.CUDA

    settings_auto = Settings(llm_device_type=DeviceType.AUTO)
    assert settings_auto.llm_device_type == DeviceType.AUTO


def test_secret_masking():
    settings = Settings(google_api_key="sensitive-key-12345")
    assert "sensitive-key-12345" not in repr(settings.google_api_key)
    assert settings.google_api_key.get_secret_value() == "sensitive-key-12345"


def test_settings_agent_persona_name():
    settings_default = Settings()
    assert settings_default.agent_persona_name == "Mora"

    settings_custom = Settings(agent_persona_name="Clio")
    assert settings_custom.agent_persona_name == "Clio"
