"""Unit tests for LocalProvider GGUF routing and execution paths."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentverse.config.settings import Settings
from agentverse.core.enums import DeviceType, LLMProvider
from agentverse.core.exceptions import LLMProviderError, ModelNotFoundError
from agentverse.services.llm.providers.local import LocalProvider


def test_local_provider_gguf_routing(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    fake_gguf = model_dir / "llama-2-7b.gguf"
    fake_gguf.write_text("GGUF_HEADER")

    settings = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="llama-2-7b-custom.gguf",
        llm_local_model_path=str(model_dir),
        llm_model_basename="llama-2-7b.gguf",
        llm_auto_download=False,
    )

    provider = LocalProvider(settings)
    assert provider._use_ollama is False
    assert provider.model_name == "llama-2-7b-custom.gguf"


@pytest.mark.asyncio
async def test_local_provider_is_available_gguf(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    fake_gguf = model_dir / "llama-2-7b.gguf"
    fake_gguf.write_text("GGUF_HEADER")

    settings = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="llama-2-7b-custom.gguf",
        llm_local_model_path=str(model_dir),
        llm_model_basename="llama-2-7b.gguf",
        llm_auto_download=False,
    )
    provider = LocalProvider(settings)
    assert await provider.is_available() is True

    # If file removed
    os.remove(fake_gguf)
    assert await provider.is_available() is False


@pytest.mark.asyncio
async def test_local_provider_missing_model_raises(tmp_path):
    settings = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="missing.gguf",
        llm_local_model_path=str(tmp_path),
        llm_model_basename="missing.gguf",
        llm_auto_download=False,
    )
    provider = LocalProvider(settings)
    with pytest.raises(ModelNotFoundError):
        await provider.generate("Hello?")


@pytest.mark.asyncio
async def test_local_provider_llama_cpp_generation(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    fake_gguf = model_dir / "llama-2-7b.gguf"
    fake_gguf.write_text("GGUF_HEADER")

    settings = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="llama-2-7b.gguf",
        llm_local_model_path=str(model_dir),
        llm_model_basename="llama-2-7b.gguf",
        llm_device_type=DeviceType.CUDA,
        llm_auto_download=False,
    )
    provider = LocalProvider(settings)

    mock_llama_inst = MagicMock()
    mock_llama_inst.return_value = {"choices": [{"text": "Ptolemaic dynasty ruled from 305 BC."}]}

    mock_llama_module = MagicMock()
    mock_llama_module.Llama.return_value = mock_llama_inst

    with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
        resp = await provider.generate("Tell me about Ptolemaic dynasty")
        assert "Ptolemaic dynasty" in resp


@pytest.mark.asyncio
async def test_local_provider_litellm_fallback(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    fake_gguf = model_dir / "model.gguf"
    fake_gguf.write_text("GGUF_HEADER")

    settings = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="custom-model.gguf",
        llm_local_model_path=str(model_dir),
        llm_model_basename="model.gguf",
        llm_auto_download=False,
    )
    provider = LocalProvider(settings)

    mock_choice = MagicMock()
    mock_choice.message.content = "Fallback litellm insight"
    mock_resp = MagicMock(choices=[mock_choice])

    with (
        patch.dict("sys.modules", {"llama_cpp": None}),
        patch("litellm.acompletion", AsyncMock(return_value=mock_resp)),
    ):
        resp = await provider.generate("What is the Lighthouse?")
        assert "Fallback litellm insight" in resp


@pytest.mark.asyncio
async def test_local_provider_litellm_fallback_failure(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    fake_gguf = model_dir / "model.gguf"
    fake_gguf.write_text("GGUF_HEADER")

    settings = Settings(
        llm_provider=LLMProvider.LOCAL,
        llm_model_name="custom-model.gguf",
        llm_local_model_path=str(model_dir),
        llm_model_basename="model.gguf",
        llm_auto_download=False,
    )
    provider = LocalProvider(settings)

    with (
        patch.dict("sys.modules", {"llama_cpp": None}),
        patch("litellm.acompletion", AsyncMock(side_effect=Exception("LiteLLM crashed"))),
        pytest.raises(LLMProviderError, match="llama-cpp-python is not installed"),
    ):
        await provider.generate("Query")
