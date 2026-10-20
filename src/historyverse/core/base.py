"""Abstract base classes defining the core contracts of HistoryVerse."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "BaseEmbeddingProvider",
    "BaseEvaluationProvider",
    "BaseLLMProvider",
    "BaseSTTProvider",
    "BaseTTSProvider",
]


class BaseLLMProvider(ABC):
    """Abstract interface that all model backends must implement.

    Ensures zero coupling to any single vendor (Gemini, OpenAI, LLaMA, Ollama).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the fully qualified model identifier."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for the provided prompt."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check whether the provider is configured and available."""

    async def generate_with_confidence(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, float]:
        """Generate response alongside an estimated confidence score.

        Returns:
            Tuple of (response_text, confidence_score between 0.0 and 1.0).
        """
        response = await self.generate(
            prompt,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        confidence = self.estimate_confidence(prompt, response)
        logger.debug(
            "Generated response with model=%s confidence=%.2f",
            self.model_name,
            confidence,
        )
        return response, confidence

    def estimate_confidence(self, prompt: str, response: str) -> float:
        """Calculate a heuristic confidence score based on lexical and response properties."""
        score = 1.0
        low_confidence_phrases = [
            "i don't know",
            "i cannot answer",
            "not available in the context",
            "no information",
            "i am unable to",
            "i cannot verify",
        ]
        resp_lower = response.lower()
        for phrase in low_confidence_phrases:
            if phrase in resp_lower:
                score -= 0.35

        tokens = response.split()
        if len(tokens) < 8:
            score -= 0.25

        prompt_terms = {t.lower() for t in prompt.split() if len(t) > 3}
        resp_terms = {t.lower() for t in tokens if len(t) > 3}
        if prompt_terms:
            overlap = len(prompt_terms & resp_terms) / len(prompt_terms)
            if overlap < 0.15:
                score -= 0.20

        return max(0.0, min(1.0, round(score, 3)))


class BaseEmbeddingProvider(ABC):
    """Abstract interface for document and query embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""


class BaseSTTProvider(ABC):
    """Abstract interface for Speech-to-Text audio transcription."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""


class BaseTTSProvider(ABC):
    """Abstract interface for Text-to-Speech audio synthesis."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str | None = None) -> str:
        """Synthesize text into speech audio and return the file path."""


class BaseEvaluationProvider(ABC):
    """Abstract interface for RAG accuracy and hallucination evaluation."""

    @abstractmethod
    async def evaluate_qa(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        """Compute evaluation metrics for a Q&A pair."""
