"""Unit tests for the LLMFactory."""

from historyverse.config.settings import Settings
from historyverse.core.enums import LLMProvider
from historyverse.services.llm.factory import LLMFactory
from historyverse.services.llm.providers.gemini import GeminiProvider
from historyverse.services.llm.providers.local import LocalProvider
from historyverse.services.llm.providers.ollama_provider import OllamaProvider
from historyverse.services.llm.providers.openai_provider import OpenAIProvider


def test_factory_creates_gemini():
    settings = Settings(llm_provider=LLMProvider.GEMINI, google_api_key="test-key")
    provider = LLMFactory.create(settings)
    assert isinstance(provider, GeminiProvider)


def test_factory_creates_openai():
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    provider = LLMFactory.create(settings)
    assert isinstance(provider, OpenAIProvider)


def test_factory_creates_local():
    settings = Settings(llm_provider=LLMProvider.LOCAL, llm_auto_download=False)
    provider = LLMFactory.create(settings)
    assert isinstance(provider, LocalProvider)


def test_factory_creates_ollama():
    settings = Settings(llm_provider=LLMProvider.OLLAMA, llm_auto_download=False)
    provider = LLMFactory.create(settings)
    assert isinstance(provider, OllamaProvider)


def test_local_provider_unified_routing():
    # LocalProvider with Ollama model
    settings_ollama = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="llama3.2",
        llm_auto_download=False,
    )
    provider_ollama = LocalProvider(settings_ollama)
    assert provider_ollama._use_ollama is True

    # LocalProvider with standalone GGUF model
    settings_gguf = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="TheBloke/Llama-2-7b-Chat-GGUF",
        llm_model_basename="llama-2-7b-chat.Q4_K_M.gguf",
        llm_auto_download=False,
    )
    provider_gguf = LocalProvider(settings_gguf)
    assert provider_gguf._use_ollama is False
