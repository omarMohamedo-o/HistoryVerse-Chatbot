"""Unit tests for domain and infrastructure exceptions."""

from agent.core.exceptions import (
    ConfigurationError,
    DriftDetectedError,
    EvaluationError,
    HallucinationDetectedError,
    IngestError,
    LLMProviderError,
    ModelNotFoundError,
    RetrieverError,
    VisionError,
    VoiceError,
)


def test_custom_exceptions_instantiation():
    assert "config" in str(ConfigurationError("config missing")).lower()
    assert (
        "drift" in str(DriftDetectedError(confidence_score=0.4, threshold=0.8, attempt=1)).lower()
    )
    assert "eval" in str(EvaluationError("eval failed")).lower()
    assert "hallucination" in str(HallucinationDetectedError(score=0.9, threshold=0.5)).lower()
    assert "ingest" in str(IngestError("ingest failed")).lower()
    assert "openai" in str(LLMProviderError("call failed", provider="openai")).lower()
    assert "model" in str(ModelNotFoundError("llama3")).lower()
    assert "retriever" in str(RetrieverError("retriever chroma failure")).lower()
    assert "vision" in str(VisionError("vision image corrupt")).lower()
    assert "voice" in str(VoiceError("voice tts audio failed")).lower()
