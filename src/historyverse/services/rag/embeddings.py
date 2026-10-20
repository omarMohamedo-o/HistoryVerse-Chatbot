"""Multilingual and Arabic embedding model registry and language compatibility profiler."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingModelProfile",
    "EmbeddingModelRegistry",
]


@dataclass(frozen=True)
class EmbeddingModelProfile:
    """Capability and language support profile for an embedding model."""

    model_id: str
    provider: str
    is_multilingual: bool
    supports_arabic: bool
    dimensions: int
    context_length: int
    supported_languages: list[str]
    notes: str


class EmbeddingModelRegistry:
    """Registry documenting language capabilities and cross-lingual performance across embedding models."""

    _PROFILES: ClassVar[dict[str, EmbeddingModelProfile]] = {
        # SOTA Multilingual & Arabic Leaders
        "BAAI/bge-m3": EmbeddingModelProfile(
            model_id="BAAI/bge-m3",
            provider="huggingface",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=1024,
            context_length=8192,
            supported_languages=["ar", "en", "fr", "de", "es", "zh", "ru", "el", "la", "100+"],
            notes="State-of-the-art multilingual & cross-lingual dense+sparse model. Premier choice for Arabic RAG.",
        ),
        "intfloat/multilingual-e5-large": EmbeddingModelProfile(
            model_id="intfloat/multilingual-e5-large",
            provider="huggingface",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=1024,
            context_length=512,
            supported_languages=["ar", "en", "fr", "es", "de", "zh", "100+"],
            notes="High accuracy multilingual model with strong Arabic-English cross-lingual alignment.",
        ),
        "intfloat/multilingual-e5-base": EmbeddingModelProfile(
            model_id="intfloat/multilingual-e5-base",
            provider="huggingface",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=768,
            context_length=512,
            supported_languages=["ar", "en", "fr", "es", "de", "zh", "100+"],
            notes="Balanced performance-to-latency multilingual embeddings.",
        ),
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": EmbeddingModelProfile(
            model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            provider="huggingface",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=768,
            context_length=512,
            supported_languages=["ar", "en", "fr", "es", "it", "de", "50+"],
            notes="Reliable 50+ language sentence transformer with native Arabic support.",
        ),
        "aubmindlab/bert-base-arabertv02": EmbeddingModelProfile(
            model_id="aubmindlab/bert-base-arabertv02",
            provider="huggingface",
            is_multilingual=False,
            supports_arabic=True,
            dimensions=768,
            context_length=512,
            supported_languages=["ar"],
            notes="Dedicated Arabic-only domain model with deep morphological understanding.",
        ),
        # Google Generative AI
        "models/embedding-001": EmbeddingModelProfile(
            model_id="models/embedding-001",
            provider="google",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=768,
            context_length=2048,
            supported_languages=["ar", "en", "100+"],
            notes="Google multilingual embedding model with high cross-lingual transfer.",
        ),
        "models/text-embedding-004": EmbeddingModelProfile(
            model_id="models/text-embedding-004",
            provider="google",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=768,
            context_length=2048,
            supported_languages=["ar", "en", "100+"],
            notes="Google latest generation multilingual embeddings.",
        ),
        # OpenAI Models
        "text-embedding-3-small": EmbeddingModelProfile(
            model_id="text-embedding-3-small",
            provider="openai",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=1536,
            context_length=8191,
            supported_languages=["ar", "en", "100+"],
            notes="Cost-efficient multilingual embedding model with excellent Arabic coverage.",
        ),
        "text-embedding-3-large": EmbeddingModelProfile(
            model_id="text-embedding-3-large",
            provider="openai",
            is_multilingual=True,
            supports_arabic=True,
            dimensions=3072,
            context_length=8191,
            supported_languages=["ar", "en", "100+"],
            notes="Top-tier multilingual embeddings with 3072 dimensions.",
        ),
        # English-Centric / Monolingual Models (Unsuitable for Arabic)
        "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelProfile(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            provider="huggingface",
            is_multilingual=False,
            supports_arabic=False,
            dimensions=384,
            context_length=256,
            supported_languages=["en"],
            notes="English-only lightweight model. Fails on Arabic with high out-of-vocabulary [UNK] rate.",
        ),
        "nomic-ai/nomic-embed-text-v1": EmbeddingModelProfile(
            model_id="nomic-ai/nomic-embed-text-v1",
            provider="huggingface",
            is_multilingual=False,
            supports_arabic=False,
            dimensions=768,
            context_length=8192,
            supported_languages=["en"],
            notes="English-optimized long-context model. Not recommended for Arabic.",
        ),
        "nomic-embed-text": EmbeddingModelProfile(
            model_id="nomic-embed-text",
            provider="ollama",
            is_multilingual=False,
            supports_arabic=False,
            dimensions=768,
            context_length=8192,
            supported_languages=["en"],
            notes="Ollama English-centric embedding model.",
        ),
    }

    @classmethod
    def get_profile(cls, model_name: str) -> EmbeddingModelProfile | None:
        """Lookup capability profile for a given model identifier."""
        clean_name = model_name.strip()
        # Direct match
        if clean_name in cls._PROFILES:
            return cls._PROFILES[clean_name]

        # Case-insensitive or basename match
        for key, profile in cls._PROFILES.items():
            if (
                key.lower() == clean_name.lower()
                or key.split("/")[-1].lower() == clean_name.lower()
            ):
                return profile

        return None

    @classmethod
    def is_multilingual_compatible(cls, model_name: str, language: str = "ar") -> bool:
        """Verify whether the model supports the requested language."""
        profile = cls.get_profile(model_name)
        if profile is None:
            # Unknown models are presumed unverified
            return True

        if language.lower() == "ar":
            return profile.supports_arabic
        return profile.is_multilingual or (language.lower() in profile.supported_languages)

    @classmethod
    def validate_language_compatibility(
        cls, model_name: str, target_language: str = "ar"
    ) -> tuple[bool, str]:
        """Audit embedding model compatibility for a specific target language."""
        profile = cls.get_profile(model_name)
        if profile is None:
            return True, f"Model '{model_name}' has uncataloged language profile."

        if target_language.lower() == "ar" and not profile.supports_arabic:
            rec = cls.recommend_model_for_language(language="ar", provider=profile.provider)
            msg = (
                f"Model '{model_name}' is monolingual or English-centric and does NOT support Arabic. "
                f"Non-Latin tokens will suffer severe [UNK] vocabulary collapse. "
                f"Recommended upgrade: '{rec}'."
            )
            logger.warning(msg)
            return False, msg

        return True, f"Model '{model_name}' supports {target_language}."

    @classmethod
    def recommend_model_for_language(
        cls, language: str = "ar", provider: str = "huggingface"
    ) -> str:
        """Recommend the optimal verified model for a given language and provider."""
        if language.lower() == "ar":
            if provider.lower() in ("huggingface", "local"):
                return "BAAI/bge-m3"
            if provider.lower() == "google":
                return "models/text-embedding-004"
            if provider.lower() == "openai":
                return "text-embedding-3-small"
            if provider.lower() == "ollama":
                return "bge-m3"

        return "BAAI/bge-m3"
