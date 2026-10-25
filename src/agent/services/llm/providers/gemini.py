"""Google Gemini model provider implementation."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import litellm
from PIL import Image

from agent.config.settings import Settings
from agent.core.base import BaseLLMProvider
from agent.core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)

__all__ = ["GeminiProvider"]


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models via LiteLLM."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        model = settings.llm_model_name
        model_lower = model.lower()
        # Auto-resolve Flash Lite and 3.5 aliases to official Gemini Flash Lite endpoint
        if "3.5" in model_lower or any(alias in model_lower for alias in ["flash-lite", "flash_lite", "flash lite"]):
            model = "gemini/gemini-2.0-flash-lite"
        elif not model.startswith("gemini/"):
            model = f"gemini/{model}"
        self._model_name = model
        self._api_key = settings.google_api_key.get_secret_value()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def is_available(self) -> bool:
        return bool(self._api_key)

    @property
    def supports_image_generation(self) -> bool:
        return bool(self._api_key)

    async def generate_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs: Any,
    ) -> str | None:
        if not self._api_key:
            return None

        try:
            logger.info("Using Gemini Imagen integration for direct image creation.")
            img_model = kwargs.pop("model", self._settings.imagen_model)
            response = await litellm.aimage_generation(
                prompt=prompt,
                model=img_model,
                api_key=self._api_key,
                **kwargs,
            )
            return response.data[0].url
        except Exception as exc:
            logger.warning("Gemini native image generation fell back: %s", exc)
            return None

    async def generate(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._api_key:
            raise LLMProviderError("Google API key is not configured.", provider="gemini")

        temp = temperature if temperature is not None else self._settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._settings.llm_max_tokens

        messages: list[dict[str, Any]] = []

        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            response = await litellm.acompletion(
                model=self._model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                api_key=self._api_key,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc, exc_info=True)
            raise LLMProviderError(str(exc), provider="gemini") from exc

    async def stream(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        if not self._api_key:
            raise LLMProviderError("Google API key is not configured.", provider="gemini")

        temp = temperature if temperature is not None else self._settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._settings.llm_max_tokens

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        try:
            response = await litellm.acompletion(
                model=self._model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                api_key=self._api_key,
                stream=True,
                **kwargs,
            )
            async for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", None) or ""
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("Gemini stream failed: %s", exc)
            raise LLMProviderError(str(exc), provider="gemini") from exc
