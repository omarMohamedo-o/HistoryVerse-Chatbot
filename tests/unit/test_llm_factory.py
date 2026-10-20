"""Unit tests for the LLMFactory."""

from historyverse.config.settings import Settings
from historyverse.core.enums import LLMProvider
from historyverse.services.llm.factory import LLMFactory
from historyverse.services.llm.providers.gemini import GeminiProvider
from historyverse.services.llm.providers.local import LocalProvider
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
