"""Factory for instantiating model-agnostic LLM providers."""

from __future__ import annotations

import logging
from typing import ClassVar

from agent.config.settings import Settings, get_settings
from agent.core.base import BaseLLMProvider
from agent.core.enums import LLMProvider
from agent.core.exceptions import ConfigurationError
from agent.services.llm.providers.gemini import GeminiProvider
from agent.services.llm.providers.local import LocalProvider
from agent.services.llm.providers.ollama_provider import OllamaProvider
from agent.services.llm.providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

__all__ = ["LLMFactory"]


class LLMFactory:
    """Factory creating LLM providers according to application settings."""

    _registry: ClassVar[dict[LLMProvider, type[BaseLLMProvider]]] = {
        LLMProvider.GEMINI: GeminiProvider,
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.LOCAL: LocalProvider,
        LLMProvider.OLLAMA: OllamaProvider,
    }

    @classmethod
    def register(cls, provider_type: LLMProvider, provider_cls: type[BaseLLMProvider]) -> None:
        """Register a new LLM provider class dynamically."""
        cls._registry[provider_type] = provider_cls
        logger.info("Registered LLM provider: %s", provider_type)

    @classmethod
    def create(cls, settings: Settings | None = None) -> BaseLLMProvider:
        """Instantiate and return the configured LLM provider."""
        cfg = settings or get_settings()
        provider_cls = cls._registry.get(cfg.llm_provider)

        if not provider_cls:
            raise ConfigurationError(
                f"Unsupported or unregistered LLM provider: {cfg.llm_provider}"
            )

        logger.info(
            "Instantiating provider=%s with model=%s",
            cfg.llm_provider,
            cfg.llm_model_name,
        )
        return provider_cls(cfg)
