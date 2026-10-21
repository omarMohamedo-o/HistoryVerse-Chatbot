"""Unit tests for OllamaProvider and LLMFactory integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agent.config.settings import Settings
from agent.core.enums import LLMProvider
from agent.services.llm.factory import LLMFactory
from agent.services.llm.providers.ollama_provider import OllamaProvider


def test_ollama_factory_creation():
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="llama3",
        ollama_base_url="http://localhost:11434",
        llm_auto_download=False,
    )
    provider = LLMFactory.create(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "llama3"


def test_ollama_model_name_normalization():
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="ollama/mistral:7b",
        llm_auto_download=False,
    )
    provider = OllamaProvider(settings)
    assert provider.model_name == "mistral:7b"


@pytest.mark.asyncio
async def test_ollama_is_available_success():
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="llama3",
        llm_auto_download=False,
    )
    provider = OllamaProvider(settings)

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        available = await provider.is_available()
        assert available is True


@pytest.mark.asyncio
async def test_ollama_is_available_failure():
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="llama3",
        llm_auto_download=False,
    )
    provider = OllamaProvider(settings)

    with patch("httpx.AsyncClient.get", side_effect=Exception("Connection refused")):
        available = await provider.is_available()
        assert available is False


@pytest.mark.asyncio
async def test_ollama_generate_litellm():
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="llama3",
        llm_auto_download=False,
    )
    provider = OllamaProvider(settings)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content="Alexander founded Alexandria in 331 BC."))
    ]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = mock_resp
        output = await provider.generate("Tell me about Alexandria.")
        assert "Alexandria" in output
        assert mock_litellm.called


@pytest.mark.asyncio
async def test_ollama_generate_rest_fallback():
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model_name="llava",
        llm_auto_download=False,
    )
    provider = OllamaProvider(settings)

    # Simulate LiteLLM failure and REST API fallback
    with (
        patch("litellm.acompletion", side_effect=Exception("LiteLLM mock error")),
        patch("httpx.AsyncClient.post") as mock_post,
    ):
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"response": "This is a Greco-Roman column."}
        mock_post.return_value = mock_res

        img = Image.new("RGB", (32, 32), color="red")
        output = await provider.generate("What is this image?", images=[img])
        assert "Greco-Roman column" in output
