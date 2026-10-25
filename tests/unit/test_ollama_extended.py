"""Unit tests for OllamaProvider covering tag probing, auto-pulling, and REST fallback recovery."""

import json
from unittest.mock import MagicMock, patch

import pytest
import respx

from agent.config.settings import Settings
from agent.core.exceptions import LLMProviderError
from agent.services.llm.providers.ollama_provider import OllamaProvider


def test_ollama_model_tags_probe_found():
    """Verify OllamaProvider detects already cached models via /api/tags without pulling."""
    settings = Settings(
        llm_model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        ollama_auto_download=True,
    )

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        provider = OllamaProvider(settings)
        assert provider.model_name == "llama3.2:latest"
        mock_urlopen.assert_called_once()


def test_ollama_model_auto_pull_api_success():
    """Verify OllamaProvider automatically pulls missing model via /api/pull when not cached."""
    settings = Settings(
        llm_model_name="mistral:latest",
        ollama_base_url="http://localhost:11434",
        ollama_auto_download=True,
    )

    # First probe /api/tags returns empty models
    probe_resp = MagicMock()
    probe_resp.read.return_value = json.dumps({"models": []}).encode("utf-8")
    probe_resp.__enter__.return_value = probe_resp

    # Second call to /api/pull returns HTTP 200
    pull_resp = MagicMock()
    pull_resp.status = 200
    pull_resp.__enter__.return_value = pull_resp

    with patch("urllib.request.urlopen", side_effect=[probe_resp, pull_resp]):
        provider = OllamaProvider(settings)
        assert provider.model_name == "mistral:latest"


@pytest.mark.asyncio
@respx.mock
async def test_ollama_litellm_failure_falls_back_to_rest():
    """Verify that when litellm call fails, provider falls back to Ollama REST /api/generate."""
    settings = Settings(
        llm_model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        ollama_auto_download=False,
    )
    provider = OllamaProvider(settings)

    # Mock REST API generate endpoint
    respx.post("http://localhost:11434/api/generate").respond(
        status_code=200,
        json={"response": "REST fallback historical response"},
    )

    with patch("litellm.acompletion", side_effect=RuntimeError("LiteLLM connection error")):
        response = await provider.generate("Who was Hatshepsut?")
        assert response == "REST fallback historical response"


@pytest.mark.asyncio
@respx.mock
async def test_ollama_direct_rest_failure_raises_llm_provider_error():
    """Verify that when both LiteLLM and direct REST fail, LLMProviderError is raised."""
    settings = Settings(
        llm_model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        ollama_auto_download=False,
    )
    provider = OllamaProvider(settings)

    # Mock REST API generate endpoint returning 500
    respx.post("http://localhost:11434/api/generate").respond(
        status_code=500,
        text="Internal Server Error",
    )

    with (
        patch("litellm.acompletion", side_effect=RuntimeError("LiteLLM failed")),
        pytest.raises(LLMProviderError, match="Ollama generation failed"),
    ):
        await provider.generate("Tell me about Ramses II")
