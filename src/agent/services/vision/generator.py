"""Historical image generation service with direct provider and fallback support."""

from __future__ import annotations

import logging
import os
import uuid

import httpx
import litellm

from agent.config.settings import Settings, get_settings
from agent.core.base import BaseLLMProvider
from agent.core.exceptions import VisionError
from agent.core.schemas import ImageGenerationResponse
from agent.services.langfuse.prompts import PromptManager

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
        self._outputs_dir = self._settings.outputs_dir

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

        # 1. Global Sanitization & Domain Boundary Check
        from agent.services.nlp.arabic import clean_arabic_text
        from agent.services.nlp.colloquial import expand_egyptian_colloquial_query
        from agent.services.rag.refusal import DomainRefusalGate
        from agent.services.security.pii import redact_egyptian_pii

        sanitized_input = redact_egyptian_pii(prompt)
        cleaned_input = clean_arabic_text(sanitized_input)
        enriched_input = expand_egyptian_colloquial_query(cleaned_input)

        refusal_gate = DomainRefusalGate(domain="history", strict_mode=True)
        decision = refusal_gate.check_query(cleaned_input)
        if decision.is_refused:
            raise VisionError(
                f"Image generation request refused: {decision.refusal_message or 'The prompt is outside the historical scope.'}"
            )

        enhanced_prompt = enriched_input
        if enhance:
            enhancement_template = self._prompts.get_prompt("image_generation")
            rendered = enhancement_template.replace("{prompt}", enriched_input)
            try:
                enhanced_prompt = await self._llm.generate(rendered)
                logger.info("Enhanced historical image prompt: %s", enhanced_prompt[:120])
            except Exception as exc:
                logger.warning("Prompt enhancement failed, using original prompt: %s", exc)

        os.makedirs(self._outputs_dir, exist_ok=True)
        filename = f"hist_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self._outputs_dir, filename)

        if self._llm.supports_image_generation:
            logger.info(
                "Active provider (%s) natively supports image generation; invoking directly.",
                self._llm.model_name,
            )
            direct_url = await self._llm.generate_image(
                enhanced_prompt,
                size=size,
                quality=quality,
            )
            if direct_url:
                saved_path = await self._download_image(direct_url, filepath)
                return ImageGenerationResponse(
                    original_prompt=prompt,
                    enhanced_prompt=enhanced_prompt,
                    image_url=saved_path or direct_url,
                    model_used=self._llm.model_name,
                )

        logger.info(
            "Provider does not support direct image creation; using decoupled generator (%s).",
            self._model,
        )
        api_key = (
            self._settings.openai_api_key.get_secret_value()
            or self._settings.google_api_key.get_secret_value()
        )

        try:
            response = await litellm.aimage_generation(
                prompt=enhanced_prompt,
                model=self._model,
                size=size,
                quality=quality,
                api_key=api_key,
            )
            image_url = response.data[0].url
            saved_path = await self._download_image(image_url, filepath)

            return ImageGenerationResponse(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                image_url=saved_path or image_url,
                model_used=self._model,
            )
        except Exception as exc:
            logger.error("Image generation failed: %s", exc)
            raise VisionError(f"Image generation failed: {exc}") from exc

    async def _download_image(self, url: str, target_path: str) -> str | None:
        if not url.startswith(("http://", "https://")):
            return url
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    with open(target_path, "wb") as f:
                        f.write(res.content)
                    logger.info("Saved generated image to %s", target_path)
                    return target_path
        except Exception as exc:
            logger.warning("Failed to persist image locally, returning remote URL: %s", exc)
        return url
