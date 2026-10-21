"""Unit tests for LocalProvider GGUF routing and execution paths."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config.settings import Settings
from agent.core.enums import DeviceType, LLMProvider
from agent.core.exceptions import LLMProviderError, ModelNotFoundError
from agent.services.llm.providers.local import LocalProvider


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


@pytest.mark.asyncio
async def test_local_provider_ollama_and_mps_branches(tmp_path):
    # 1. Ollama mode detection via LLMProvider setting
    local_ollama = LocalProvider(
        settings=Settings(llm_provider=LLMProvider.OLLAMA, llm_model_name="llama3.2")
    )
    assert local_ollama._use_ollama is True

    # 2. Ollama mode via model prefix
    local_prefix = LocalProvider(
        settings=Settings(llm_provider=LLMProvider.LOCAL, llm_model_name="ollama/mistral")
    )
    assert local_prefix._use_ollama is True

    assert local_ollama._format_llama_prompt("[INST] Context [/INST]") == "[INST] Context [/INST]"

    # 3. Ollama provider delegation for is_available and generate
    mock_ollama_sub = MagicMock()
    mock_ollama_sub.is_available = AsyncMock(return_value=True)
    mock_ollama_sub.generate = AsyncMock(return_value="Ollama generated text")
    local_ollama._ollama_provider = mock_ollama_sub
    assert await local_ollama.is_available() is True
    assert await local_ollama.generate("Tell me a story") == "Ollama generated text"

    # 4. Ensure GGUF download exception handling
    with patch.dict(
        "sys.modules",
        {
            "huggingface_hub": MagicMock(
                hf_hub_download=MagicMock(side_effect=Exception("Download failed"))
            )
        },
    ):
        local_gguf = LocalProvider(
            settings=Settings(
                llm_provider=LLMProvider.LOCAL,
                llm_model_name="TheBloke/Llama-2-7b-Chat-GGUF",
                llm_model_basename="llama-2-7b-chat.Q4_K_M.gguf",
                llm_local_model_path=str(tmp_path),
                llm_auto_download=True,
            )
        )
        local_gguf._ensure_gguf_available()

    # 5. MPS device and llama_cpp exception
    model_weight_file = tmp_path / "custom.gguf"
    model_weight_file.write_text("dummy gguf weights")
    local_mps = LocalProvider(
        settings=Settings(
            llm_provider=LLMProvider.LOCAL,
            llm_model_name="custom.gguf",
            llm_model_basename="custom.gguf",
            llm_local_model_path=str(tmp_path),
            llm_device_type=DeviceType.MPS,
            llm_auto_download=False,
        )
    )
    mock_llama_inst = MagicMock(return_value={"choices": [{"text": "Generated response text"}]})
    with patch.dict(
        "sys.modules",
        {"llama_cpp": MagicMock(Llama=MagicMock(return_value=mock_llama_inst))},
    ):
        ans = await local_mps.generate("Prompt")
        assert ans == "Generated response text"

    with (
        patch.dict(
            "sys.modules",
            {"llama_cpp": MagicMock(Llama=MagicMock(side_effect=RuntimeError("Inference fail")))},
        ),
        pytest.raises(LLMProviderError),
    ):
        await local_mps.generate("Prompt")
