"""Unit tests for prompt rendering fallbacks, LLMFactory registration, and provider edge cases."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.config.settings import Settings
from agent.core.base import BaseLLMProvider
from agent.core.enums import LLMProvider
from agent.core.exceptions import ConfigurationError
from agent.services.langfuse.prompts import PromptManager
from agent.services.llm.factory import LLMFactory
from agent.services.llm.providers.local import LocalProvider
from agent.services.llm.providers.openai_provider import OpenAIProvider


def test_prompt_manager_yaml_read_error(tmp_path):
    """Verify PromptManager returns fallback template when reading yaml fails."""
    bad_yaml = tmp_path / "custom.yaml"
    bad_yaml.write_text("template: valid")

    pm = PromptManager(settings=Settings(prompts_dir=str(tmp_path)))
    with patch("builtins.open", side_effect=PermissionError("File access denied")):
        tpl = pm._load_from_yaml("custom")
        assert "Your name is {assistant_name}" in tpl


def test_prompt_manager_render_key_error_fallback(tmp_path):
    """Verify PromptManager handles missing format keys gracefully using replacement fallback."""
    yaml_file = tmp_path / "missing_keys.yaml"
    yaml_file.write_text("template: 'Hello {assistant_name}, tell me about {unknown_key}'\n")

    pm = PromptManager(settings=Settings(prompts_dir=str(tmp_path)))
    # render with assistant_name provided, but unknown_key omitted
    rendered = pm.render("missing_keys", assistant_name="Thoth")
    assert "Hello Thoth" in rendered
    assert "{unknown_key}" in rendered


def test_llm_factory_register_and_create():
    """Verify LLMFactory can register custom providers dynamically."""
    class CustomMockProvider(BaseLLMProvider):
        def __init__(self, settings):
            self._settings = settings

        @property
        def model_name(self):
            return "custom-provider-model"

        async def generate(self, prompt, **kwargs):
            return "custom response"

        async def is_available(self):
            return True

    # Register custom enum and class
    LLMFactory.register(LLMProvider.LOCAL, CustomMockProvider)
    settings = Settings(llm_provider=LLMProvider.LOCAL)
    provider = LLMFactory.create(settings)
    assert isinstance(provider, CustomMockProvider)
    assert provider.model_name == "custom-provider-model"

    # Restore standard LocalProvider
    LLMFactory.register(LLMProvider.LOCAL, LocalProvider)


def test_llm_factory_unknown_provider_raises():
    """Verify LLMFactory raises ConfigurationError when provider is unregistered."""
    settings = Settings()
    # Force unknown provider enum
    settings.llm_provider = "UNREGISTERED_PROVIDER"
    with pytest.raises(ConfigurationError, match="Unsupported or unregistered LLM provider"):
        LLMFactory.create(settings)


def test_openai_provider_model_name():
    """Verify OpenAIProvider model_name property."""
    settings = Settings(llm_model_name="gpt-4o")
    provider = OpenAIProvider(settings)
    assert provider.model_name == "gpt-4o"


def test_local_provider_gguf_download_success(monkeypatch):
    """Verify LocalProvider GGUF download logs success when hf_hub_download succeeds."""
    mock_hf = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hf)

    settings = Settings(
        llm_model_name="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        local_model_path="/tmp/test_gguf_models",
        llm_device_type="cpu",
    )
    provider = LocalProvider(settings)
    assert provider is not None
    assert mock_hf.hf_hub_download.call_count >= 1


def test_local_provider_gguf_download_importerror(monkeypatch):
    """Verify LocalProvider handles missing huggingface_hub gracefully."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    settings = Settings(
        llm_model_name="TheBloke/Model-GGUF/model.gguf",
        local_model_path="/tmp/test_gguf_models",
    )
    provider = LocalProvider(settings)
    # Should not raise exception
    provider._ensure_gguf_available()
