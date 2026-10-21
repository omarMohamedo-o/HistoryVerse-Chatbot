"""Enumerations for the Agent application domain."""

from enum import Enum

__all__ = [
    "ConversationType",
    "DeviceType",
    "DriftAction",
    "EmbeddingProvider",
    "LLMProvider",
    "UnlearningStatus",
]


class LLMProvider(str, Enum):
    """Supported large language model providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    LOCAL = "local"
    OLLAMA = "ollama"


class ConversationType(str, Enum):
    """Interaction modality supported by the chatbot."""

    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"


class DeviceType(str, Enum):
    """Compute device targets for local models, embeddings, and STT."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class DriftAction(str, Enum):
    """Action to execute when model accuracy drift or hallucination is detected."""

    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    ALERT = "alert"
    DOCKER_REBUILD = "docker_rebuild"


class EmbeddingProvider(str, Enum):
    """Embedding model providers for RAG vectorization."""

    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    OPENAI = "openai"


class UnlearningStatus(str, Enum):
    """Status outcomes for selective knowledge unlearning verification."""

    FORGOTTEN_VERIFIED = "forgotten_verified"
    LEAKAGE_DETECTED = "leakage_detected"
    PENDING_VERIFICATION = "pending_verification"
    PARTIAL_FORGET = "partial_forget"
