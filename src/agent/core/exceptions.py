"""Custom exception hierarchy for the Agent application domain."""

from __future__ import annotations

__all__ = [
    "AgentError",
    "ConfigurationError",
    "DriftDetectedError",
    "EvaluationError",
    "HallucinationDetectedError",
    "HistoryVerseError",
    "IngestError",
    "LLMProviderError",
    "ModelNotFoundError",
    "RAGError",
    "RetrieverError",
    "VisionError",
    "VoiceError",
]


class AgentError(Exception):
    """Base exception for all domain errors within Agent."""

    def __init__(self, message: str = "An error occurred in Agent.") -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


HistoryVerseError = AgentError


class ConfigurationError(AgentError):
    """Raised when application configuration is invalid or missing required values."""


class LLMProviderError(AgentError):
    """Base error for LLM generation failures."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class ModelNotFoundError(LLMProviderError):
    """Raised when a specified local or remote model cannot be resolved or downloaded."""

    def __init__(self, model_identifier: str, provider: str = "local") -> None:
        super().__init__(
            f"Model '{model_identifier}' was not found and could not be loaded.", provider
        )
        self.model_identifier = model_identifier


class DriftDetectedError(AgentError):
    """Raised when generated output confidence or quality falls below acceptable thresholds."""

    def __init__(self, confidence_score: float, threshold: float, attempt: int = 1) -> None:
        message = (
            f"Model drift detected on attempt {attempt}: confidence {confidence_score:.2f} "
            f"below threshold {threshold:.2f}."
        )
        super().__init__(message)
        self.confidence_score = confidence_score
        self.threshold = threshold
        self.attempt = attempt


class HallucinationDetectedError(AgentError):
    """Raised when an answer contains factual contradictions or ungrounded claims."""

    def __init__(self, score: float, threshold: float) -> None:
        super().__init__(f"Hallucination score {score:.2f} exceeds tolerance {threshold:.2f}.")
        self.score = score
        self.threshold = threshold


class RAGError(AgentError):
    """Base error for retrieval-augmented generation failures."""


class IngestError(RAGError):
    """Raised when document parsing, batch loading, or chunking fails."""


class RetrieverError(RAGError):
    """Raised when ChromaDB index querying or vector similarity search fails."""


class VoiceError(AgentError):
    """Raised when audio transcription (STT) or speech synthesis (TTS) encounters an error."""


class VisionError(AgentError):
    """Raised when image loading, format validation, or visual inference fails."""


class EvaluationError(AgentError):
    """Raised when Ragas metric computation or evaluation pipeline fails."""
