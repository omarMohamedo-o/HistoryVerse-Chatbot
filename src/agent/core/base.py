"""Abstract base classes defining the core contracts of HistoryVerse."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "BaseEmbeddingProvider",
    "BaseEvaluationProvider",
    "BaseLLMProvider",
    "BaseSTTProvider",
    "BaseStreamingSTTProvider",
    "BaseStreamingTTSProvider",
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

    async def stream(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronously stream response tokens. Default fallback yields full generation."""
        output = await self.generate(
            prompt,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        yield output

    @abstractmethod
    async def is_available(self) -> bool:
        """Check whether the provider is configured and available."""

    @property
    def supports_image_generation(self) -> bool:
        """Indicate whether the provider natively supports image generation."""
        return False

    async def generate_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs: Any,
    ) -> str | None:
        """Natively synthesize an image if supported by the provider."""
        return None

    @property
    def supports_video_generation(self) -> bool:
        """Indicate whether the provider natively supports video generation."""
        return False

    async def generate_video(
        self,
        prompt: str,
        *,
        image_url: str | None = None,
        duration_seconds: int = 4,
        fps: int = 24,
        resolution: str = "1024x1024",
        **kwargs: Any,
    ) -> str | None:
        """Natively synthesize a video if supported by the provider."""
        return None

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

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text string using tokenizer or fast heuristic."""
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def get_context_window(self) -> int:
        """Return maximum context token capacity for the active model."""
        name = self.model_name.lower()
        if "gemini-1.5" in name or "gemini-2.0" in name:
            return 1000000
        elif "gpt-4" in name:
            return 128000
        elif "mistral" in name:
            return 32768
        return 8192 if ("llama-3" in name or "llama3" in name) else 4096

    def fit_to_context_budget(self, text: str, max_tokens: int | None = None) -> str:
        """Trim or truncate text if it exceeds the specified token context budget."""
        limit = max_tokens if max_tokens is not None else (self.get_context_window() - 512)
        token_count = self.count_tokens(text)
        if token_count <= limit:
            return text

        char_limit = limit * 4
        return f"{text[:char_limit]}\n...[Context truncated to fit token budget]"

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = 0.0,
        **kwargs: Any,
    ) -> T:
        """Generate structured output validated against a Pydantic schema."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        structured_prompt = (
            f"{prompt}\n\n"
            f"[MANDATORY STRUCTURED OUTPUT INSTRUCTION]:\n"
            f"You must respond ONLY with a valid JSON object matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any conversational preamble, commentary, or text outside the JSON object."
        )
        raw_output = await self.generate(
            structured_prompt, images=images, temperature=temperature, **kwargs
        )
        return self._parse_structured_json(raw_output, response_model)

    def _parse_structured_json(self, raw_output: str, model_cls: type[T]) -> T:
        """Extract and validate JSON object into target Pydantic schema."""
        clean_text = raw_output.strip()
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        candidate = json_match.group(1) if json_match else clean_text

        if not candidate.startswith("{"):
            brace_start = candidate.find("{")
            brace_end = candidate.rfind("}")
            candidate = (
                candidate[brace_start : brace_end + 1]
                if brace_start != -1 and brace_end != -1
                else candidate
            )

        data = json.loads(candidate)
        return model_cls.model_validate(data)


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


class BaseStreamingSTTProvider(BaseSTTProvider):
    """Abstract interface for real-time incremental Speech-to-Text streaming."""

    @abstractmethod
    async def feed_audio_chunk(self, chunk: bytes) -> tuple[str, bool]:
        """Process incoming audio bytes. Returns (current_transcript, is_final)."""

    @abstractmethod
    async def finish(self) -> str:
        """Finalize the current stream and return final transcript."""

    @abstractmethod
    async def reset(self) -> None:
        """Reset internal buffers for a new conversational turn."""


class BaseTTSProvider(ABC):
    """Abstract interface for Text-to-Speech audio synthesis."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str | None = None) -> str:
        """Synthesize text into speech audio and return the file path."""


class BaseStreamingTTSProvider(BaseTTSProvider):
    """Abstract interface for real-time streaming Text-to-Speech."""

    @abstractmethod
    async def synthesize_chunk(self, text_chunk: str) -> bytes | None:
        """Synthesize a text chunk (e.g. sentence/clause) into audio bytes."""

    @abstractmethod
    async def stream_speech(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Stream audio bytes as text tokens or sentences arrive."""


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
