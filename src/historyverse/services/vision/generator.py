"""Historical image generation service using LiteLLM."""

from __future__ import annotations

import base64
import logging
import os
import uuid
import httpx

import litellm
from historyverse.config.settings import Settings, get_settings
from historyverse.core.base import BaseLLMProvider
from historyverse.core.exceptions import VisionError
from historyverse.core.schemas import ImageGenerationResponse
from historyverse.services.langfuse.prompts import PromptManager

logger = logging.getLogger(__name__)

__all__ = ["ImageGenerator"]


class ImageGenerator:
    """Generates authentic historical scenes and artifact reconstructions."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        prompt_manager: PromptManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm_provider
        self._prompts = prompt_manager or PromptManager(self._settings)
        self._model = self._settings.image_generation_model
        self._outputs_dir = "outputs"

    async def generate_historical_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        enhance: bool = True,
    ) -> ImageGenerationResponse:
        """Create a historical reconstruction image from text description."""
        if not self._settings.image_generation_enabled:
            raise VisionError("Image creation is disabled in application settings.")

        enhanced_prompt = prompt
        if enhance:
            enhancement_template = self._prompts.get_prompt("image_generation")
            rendered = enhancement_template.replace("{prompt}", prompt)
            try:
                enhanced_prompt = await self._llm.generate(rendered)
                logger.info("Enhanced historical image prompt: %s", enhanced_prompt[:120])
            except Exception as exc:
                logger.warning("Prompt enhancement failed, using original prompt: %s", exc)

        api_key = (
            self._settings.openai_api_key.get_secret_value()
            or self._settings.google_api_key.get_secret_value()
        )

        os.makedirs(self._outputs_dir, exist_ok=True)
        filename = f"hist_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self._outputs_dir, filename)

        try:
            response = await litellm.aimage_generation(
                prompt=enhanced_prompt,
                model=self._model,
                size=size,
                quality=quality,
                api_key=api_key,
            )
            image_url = response.data[0].url

            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(image_url)
                if res.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(res.content)
                    image_url = filepath

            return ImageGenerationResponse(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                image_url=image_url,
                model_used=self._model,
            )
        except Exception as exc:
            logger.error("Image generation failed: %s", exc)
            raise VisionError(f"Image generation failed: {exc}") from exc
