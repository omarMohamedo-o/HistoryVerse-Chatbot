"""Local model provider with automatic downloading and quantization support."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any
from PIL import Image

import litellm
from historyverse.config.settings import Settings
from historyverse.core.base import BaseLLMProvider
from historyverse.core.exceptions import LLMProviderError, ModelNotFoundError

logger = logging.getLogger(__name__)

__all__ = ["LocalProvider"]


class LocalProvider(BaseLLMProvider):
    """Provider for locally executed models (Ollama or local GGUF/HF files)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_name = settings.llm_model_name
        self._local_path = settings.llm_local_model_path
        self._model_basename = settings.llm_model_basename
        self._auto_download = settings.llm_auto_download
        self._ensure_model_available()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model_available(self) -> None:
        os.makedirs(self._local_path, exist_ok=True)

        if self._model_name.startswith("ollama/"):
            ollama_model = self._model_name.replace("ollama/", "")
            if self._auto_download:
                try:
                    logger.info("Verifying Ollama model availability: %s", ollama_model)
                    subprocess.run(
                        ["ollama", "pull", ollama_model],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                except Exception as exc:
                    logger.warning("Could not auto-pull Ollama model: %s", exc)
            return

        expected_file = os.path.join(self._local_path, self._model_basename)
        if os.path.exists(expected_file):
            logger.info("Found local model weights: %s", expected_file)
            return

        if self._auto_download and "/" in self._model_name:
            try:
                from huggingface_hub import hf_hub_download

                logger.info(
                    "Downloading model %s (%s) from HuggingFace Hub...",
                    self._model_name,
                    self._model_basename,
                )
                hf_hub_download(
                    repo_id=self._model_name,
                    filename=self._model_basename,
                    cache_dir=self._local_path,
                    resume_download=True,
                )
                logger.info("Successfully downloaded local model weights.")
            except ImportError:
                logger.warning("huggingface-hub not installed; skipping automatic download.")
            except Exception as exc:
                logger.error("Failed to download local model weights: %s", exc)

    async def is_available(self) -> bool:
        if self._model_name.startswith("ollama/"):
            return True
        expected_file = os.path.join(self._local_path, self._model_basename)
        return os.path.exists(expected_file)

    async def generate(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        temp = temperature if temperature is not None else self._settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._settings.llm_max_tokens

        messages = [{"role": "user", "content": prompt}]

        if self._model_name.startswith("ollama/"):
            try:
                response = await litellm.acompletion(
                    model=self._model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    api_base="http://localhost:11434",
                    **kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.error("Ollama generation failed: %s", exc)
                raise LLMProviderError(str(exc), provider="ollama") from exc

        model_file = os.path.join(self._local_path, self._model_basename)
        if not os.path.exists(model_file):
            raise ModelNotFoundError(model_file, provider="local")

        try:
            from llama_cpp import Llama

            llm = Llama(
                model_path=model_file,
                n_ctx=self._settings.llm_context_window,
                n_batch=self._settings.llm_n_batch,
                n_gpu_layers=self._settings.llm_n_gpu_layers,
                verbose=False,
            )
            output = llm(
                prompt,
                max_tokens=tokens,
                temperature=temp,
                stop=["</s>", "[INST]"],
            )
            return output["choices"][0]["text"].strip()
        except ImportError:
            try:
                response = await litellm.acompletion(
                    model=f"custom/{self._model_name}",
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    **kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                raise LLMProviderError(
                    "llama-cpp-python is not installed and fallback generation failed.",
                    provider="local",
                ) from exc
        except Exception as exc:
            logger.error("Local model generation error: %s", exc)
            raise LLMProviderError(str(exc), provider="local") from exc
