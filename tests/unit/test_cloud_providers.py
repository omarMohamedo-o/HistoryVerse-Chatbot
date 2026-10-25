"""Unit tests for Gemini and OpenAI cloud LLM providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from agent.config.settings import Settings
from agent.core.enums import LLMProvider
from agent.core.exceptions import LLMProviderError
from agent.services.llm.providers.gemini import GeminiProvider
from agent.services.llm.providers.openai_provider import OpenAIProvider


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


@pytest.mark.asyncio
async def test_cloud_providers_streaming_and_multimodal():
    from PIL import Image

    # 1. Gemini with image and streaming
    g_prov = GeminiProvider(Settings(google_api_key=SecretStr("test-key")))
    img = Image.new("RGB", (10, 10), color="blue")

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Token 1 "))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="Token 2"))]

    async def mock_chunks():
        yield mock_chunk1
        yield mock_chunk2

    with patch("litellm.acompletion", AsyncMock(return_value=mock_chunks())):
        tokens = [t async for t in g_prov.stream("Prompt", images=[img])]
        assert tokens == ["Token 1 ", "Token 2"]

    # Gemini missing key error on generate and stream
    g_no_key = GeminiProvider(Settings(google_api_key=SecretStr("")))
    with pytest.raises(LLMProviderError, match="Google API key"):
        await g_no_key.generate("Prompt")
    with pytest.raises(LLMProviderError, match="Google API key"):
        async for _ in g_no_key.stream("Prompt"):
            pass

    # Gemini stream exception and image generation fallback
    with (
        patch("litellm.acompletion", AsyncMock(side_effect=RuntimeError("Stream broke"))),
        pytest.raises(LLMProviderError, match=r"\[gemini\] Stream broke"),
    ):
        async for _ in g_prov.stream("Prompt"):
            pass

    with patch("litellm.aimage_generation", AsyncMock(side_effect=RuntimeError("DALL-E broke"))):
        assert await g_prov.generate_image("Prompt") is None

    # Gemini generate with image
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Image analyzed"))]
    with patch("litellm.acompletion", AsyncMock(return_value=mock_resp)):
        ans_g = await g_prov.generate("Prompt", images=[img])
        assert ans_g == "Image analyzed"

    # 2. OpenAI with image and streaming
    o_prov = OpenAIProvider(Settings(openai_api_key=SecretStr("sk-test")))
    with patch("litellm.acompletion", AsyncMock(return_value=mock_chunks())):
        tokens_o = [t async for t in o_prov.stream("Prompt", images=[img])]
        assert tokens_o == ["Token 1 ", "Token 2"]

    # OpenAI missing key error on generate and stream
    o_no_key = OpenAIProvider(Settings(openai_api_key=SecretStr("")))
    with pytest.raises(LLMProviderError, match="OpenAI API key"):
        await o_no_key.generate("Prompt")
    with pytest.raises(LLMProviderError, match="OpenAI API key"):
        async for _ in o_no_key.stream("Prompt"):
            pass

    # OpenAI stream exception and image generation fallback
    with (
        patch("litellm.acompletion", AsyncMock(side_effect=RuntimeError("OpenAI stream broke"))),
        pytest.raises(LLMProviderError, match=r"\[openai\] OpenAI stream broke"),
    ):
        async for _ in o_prov.stream("Prompt"):
            pass

    with patch(
        "litellm.aimage_generation", AsyncMock(side_effect=RuntimeError("OpenAI image broke"))
    ):
        assert await o_prov.generate_image("Prompt") is None

    # OpenAI generate with image
    with patch("litellm.acompletion", AsyncMock(return_value=mock_resp)):
        ans_o = await o_prov.generate("Prompt", images=[img])
        assert ans_o == "Image analyzed"
