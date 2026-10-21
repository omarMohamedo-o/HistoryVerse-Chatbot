"""Unified local model provider orchestrating Ollama and standalone LLaMA GGUF/HF engines."""

from __future__ import annotations

import logging
import os
from typing import Any

import litellm
from PIL import Image

from agent.config.settings import Settings
from agent.core.base import BaseLLMProvider
from agent.core.enums import DeviceType, LLMProvider
from agent.core.exceptions import LLMProviderError, ModelNotFoundError
from agent.services.llm.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

__all__ = ["LocalProvider"]


class LocalProvider(BaseLLMProvider):
    """Unified local provider covering both Ollama daemon models and standalone LLaMA GGUF weights."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_name = settings.llm_model_name
        self._local_path = settings.llm_local_model_path
        self._model_basename = settings.llm_model_basename
        self._auto_download = settings.llm_auto_download

        # Check whether this model should be serviced by Ollama
        self._use_ollama = self._detect_ollama_mode()
        self._ollama_provider: OllamaProvider | None = None

        if self._use_ollama:
            logger.info("LocalProvider routing to Ollama engine for model=%s", self._model_name)
            self._ollama_provider = OllamaProvider(settings)
        else:
            logger.info(
                "LocalProvider routing to LLaMA GGUF engine for file=%s", self._model_basename
            )
            self._ensure_gguf_available()

    def _detect_ollama_mode(self) -> bool:
        """Determine whether to route request to Ollama or standalone GGUF."""
        if self._settings.llm_provider == LLMProvider.OLLAMA:
            return True
        if self._model_name.startswith("ollama/"):
            return True
        # If model_name is a single tag (e.g. 'llama3.2', 'mistral', 'phi3') without '/' and not ending in .gguf
        clean_name = self._model_name.lower().strip()
        return not clean_name.endswith(".gguf") and "/" not in clean_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_gguf_available(self) -> None:
        """Verify standalone GGUF model weights exist; download from HuggingFace Hub if missing."""
        os.makedirs(self._local_path, exist_ok=True)

        expected_file = os.path.join(self._local_path, self._model_basename)
        if os.path.exists(expected_file):
            logger.info("Found local GGUF model weights: %s", expected_file)
            return

        if self._auto_download and (
            "/" in self._model_name or "llama" in self._model_basename.lower()
        ):
            repo_id = (
                self._model_name if "/" in self._model_name else "TheBloke/Llama-2-7b-Chat-GGUF"
            )
            try:
                from huggingface_hub import hf_hub_download

                logger.info(
                    "Downloading GGUF model %s (%s) from HuggingFace Hub into %s...",
                    repo_id,
                    self._model_basename,
                    self._local_path,
                )
                hf_hub_download(
                    repo_id=repo_id,
                    filename=self._model_basename,
                    cache_dir=self._local_path,
                    resume_download=True,
                )
                logger.info("Successfully downloaded local GGUF model weights.")
            except ImportError:
                logger.warning("huggingface-hub not installed; skipping automatic GGUF download.")
            except Exception as exc:
                logger.error("Failed to download local GGUF model weights: %s", exc)

    async def is_available(self) -> bool:
        """Check availability via Ollama probe or local GGUF file existence."""
        if self._use_ollama and self._ollama_provider:
            return await self._ollama_provider.is_available()

        expected_file = os.path.join(self._local_path, self._model_basename)
        return os.path.exists(expected_file)

    def _format_llama_prompt(self, prompt: str) -> str:
        """Apply LLaMA-2 instruction template if not already wrapped."""
        if "[INST]" in prompt:
            return prompt

        name = self._settings.agent_persona_name
        system_prompt = (
            f"You are {name}, an expert historical AI. Read the given context and answer user questions "
            "accurately with historical precision, focusing on Egyptian and world history."
        )
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

    async def generate(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate response via Ollama or standalone LLaMA GGUF engine."""
        # Route to Ollama if configured
        if self._use_ollama and self._ollama_provider:
            return await self._ollama_provider.generate(
                prompt,
                images=images,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        # Route to standalone GGUF / llama-cpp engine
        temp = temperature if temperature is not None else self._settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._settings.llm_max_tokens

        model_file = os.path.join(self._local_path, self._model_basename)
        if not os.path.exists(model_file):
            raise ModelNotFoundError(model_file, provider="local")

        n_gpu_layers = 0
        if self._settings.llm_device_type == DeviceType.CUDA:
            n_gpu_layers = self._settings.llm_n_gpu_layers or 35
        elif self._settings.llm_device_type == DeviceType.MPS:
            n_gpu_layers = 1

        formatted_prompt = (
            self._format_llama_prompt(prompt) if "llama" in self._model_basename.lower() else prompt
        )

        try:
            from llama_cpp import Llama

            llm = Llama(
                model_path=model_file,
                n_ctx=self._settings.llm_context_window,
                n_batch=self._settings.llm_n_batch,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            output = llm(
                formatted_prompt,
                max_tokens=tokens,
                temperature=temp,
                stop=["</s>", "[INST]"],
            )
            return output["choices"][0]["text"].strip()
        except ImportError:
            try:
                messages = [{"role": "user", "content": prompt}]
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
