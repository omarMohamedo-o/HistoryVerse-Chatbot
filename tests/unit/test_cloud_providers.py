"""Unit tests for Gemini and OpenAI cloud LLM providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from agentverse.config.settings import Settings
from agentverse.core.enums import LLMProvider
from agentverse.core.exceptions import LLMProviderError
from agentverse.services.llm.providers.gemini import GeminiProvider
from agentverse.services.llm.providers.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_gemini_provider_availability():
    # Without key
    settings_no_key = Settings(google_api_key=SecretStr(""))
    provider_no_key = GeminiProvider(settings_no_key)
    assert await provider_no_key.is_available() is False
    assert provider_no_key.supports_image_generation is False
    assert await provider_no_key.generate_image("Prompt") is None

    # With key
    settings_with_key = Settings(google_api_key=SecretStr("test-google-key"))
    provider_with_key = GeminiProvider(settings_with_key)
    assert await provider_with_key.is_available() is True
    assert provider_with_key.supports_image_generation is True


@pytest.mark.asyncio
async def test_gemini_provider_generate_image_and_text():
    settings = Settings(google_api_key=SecretStr("test-key"))
    provider = GeminiProvider(settings)

    mock_img_resp = MagicMock()
    mock_img_resp.data = [MagicMock(url="https://gemini.test/imagen.png")]

    mock_chat_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Historical answer from Gemini"
    mock_chat_resp.choices = [mock_choice]

    with (
        patch("litellm.aimage_generation", AsyncMock(return_value=mock_img_resp)),
        patch("litellm.acompletion", AsyncMock(return_value=mock_chat_resp)),
    ):
        img_url = await provider.generate_image("Ancient Nile scene")
        assert img_url == "https://gemini.test/imagen.png"

        text = await provider.generate("Who was Ptolemy I?")
        assert "Historical answer from Gemini" in text


@pytest.mark.asyncio
async def test_gemini_provider_error_handling():
    settings = Settings(google_api_key=SecretStr("test-key"))
    provider = GeminiProvider(settings)

    with (
        patch("litellm.acompletion", AsyncMock(side_effect=Exception("API Quota exceeded"))),
        pytest.raises(LLMProviderError, match=r"\[gemini\] API Quota exceeded"),
    ):
        await provider.generate("Any query")


@pytest.mark.asyncio
async def test_openai_provider_availability_and_generation():
    # Without key
    settings_no_key = Settings(openai_api_key=SecretStr(""))
    provider_no_key = OpenAIProvider(settings_no_key)
    assert await provider_no_key.is_available() is False
    assert provider_no_key.supports_image_generation is False
    assert await provider_no_key.generate_image("Prompt") is None

    # With key
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key=SecretStr("sk-test-openai"))
    provider = OpenAIProvider(settings)
    assert await provider.is_available() is True
    assert provider.supports_image_generation is True

    mock_img_resp = MagicMock()
    mock_img_resp.data = [MagicMock(url="https://openai.test/dall-e.png")]

    mock_chat_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI historical insight"
    mock_chat_resp.choices = [mock_choice]

    with (
        patch("litellm.aimage_generation", AsyncMock(return_value=mock_img_resp)),
        patch("litellm.acompletion", AsyncMock(return_value=mock_chat_resp)),
    ):
        img_url = await provider.generate_image("Roman forum reconstruction")
        assert img_url == "https://openai.test/dall-e.png"

        text = await provider.generate("Tell me about the Roman Senate")
        assert "OpenAI historical insight" in text


@pytest.mark.asyncio
async def test_openai_provider_error_handling():
    settings = Settings(openai_api_key=SecretStr("sk-test-openai"))
    provider = OpenAIProvider(settings)

    with (
        patch(
            "litellm.acompletion", AsyncMock(side_effect=Exception("OpenAI service unavailable"))
        ),
        pytest.raises(LLMProviderError, match=r"\[openai\] OpenAI service unavailable"),
    ):
        await provider.generate("Any query")
