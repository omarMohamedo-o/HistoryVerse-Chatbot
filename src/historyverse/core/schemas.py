"""Pydantic v2 domain schemas for request and response validation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from historyverse.core.enums import ConversationType, DeviceType, DriftAction, LLMProvider

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "DriftReport",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "RagasEvaluationRequest",
    "RagasEvaluationResponse",
]


class ChatRequest(BaseModel):
    """Payload for conversational interaction with the chatbot."""

    query: str = Field(..., min_length=1, description="User question or query text")
    conversation_id: str | None = Field(
        default=None, description="Optional ID for tracking multi-turn conversation history"
    )
    conversation_type: ConversationType = Field(
        default=ConversationType.TEXT, description="Modality of the conversation"
    )
    speak_response: bool = Field(
        default=False, description="Whether to generate an audio TTS response"
    )
    image_source: str | None = Field(
        default=None, description="URL or local path to an image for visual analysis"
    )
    video_source: str | None = Field(
        default=None, description="URL or local path to a video file for historical analysis"
    )
    file_source: str | None = Field(
        default=None, description="URL or local path to a document (e.g. PDF) for ad-hoc QA"
    )
    audio_file_path: str | None = Field(
        default=None, description="Path to audio file for voice queries"
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Optional per-request temperature override"
    )
    prompt_version: str | None = Field(
        default=None, description="Optional specific prompt version override"
    )


class ChatResponse(BaseModel):
    """Response returned by the chatbot."""

    question: str = Field(..., description="The original or transcribed user query")
    answer: str = Field(..., description="Synthesized answer from the AI system")
    sources: list[str] = Field(
        default_factory=list, description="List of source document citations or context chunks"
    )
    confidence_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Heuristic or model confidence score"
    )
    drift_detected: bool = Field(
        default=False, description="Flag indicating if low quality or drift triggered a retry"
    )
    retry_count: int = Field(default=0, ge=0, description="Number of retries executed")
    audio_url: str | None = Field(
        default=None, description="URL or path to generated TTS audio if requested"
    )
    model_used: str = Field(default="", description="Identifier of the model that generated the answer")
    trace_id: str | None = Field(
        default=None, description="Langfuse or telemetry trace identifier"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Evaluation metrics (faithfulness, relevancy, etc.)"
    )


class IngestRequest(BaseModel):
    """Parameters for ingesting raw documents into the vector store."""

    source_directory: str = Field(
        default="./dataset", description="Path containing documents to ingest"
    )
    device_type: DeviceType = Field(
        default=DeviceType.CPU, description="Target device for embedding computation"
    )
    chunk_size: int = Field(default=1000, gt=0, description="Character chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between consecutive chunks")


class IngestResponse(BaseModel):
    """Result of a document ingestion run."""

    documents_loaded: int = Field(default=0, ge=0, description="Number of raw files ingested")
    chunks_created: int = Field(default=0, ge=0, description="Number of text chunks embedded")
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Elapsed time for ingestion in seconds"
    )


class HealthResponse(BaseModel):
    """Health check status representation."""

    status: str = Field(default="ok", description="Service health indicator")
    version: str = Field(..., description="Application version")
    model_provider: LLMProvider = Field(..., description="Active LLM provider")
    active_model: str = Field(..., description="Current active model identifier")
    uptime_seconds: float = Field(default=0.0, ge=0.0, description="Uptime duration")


class DriftReport(BaseModel):
    """Diagnostics generated when drift is evaluated."""

    confidence_score: float = Field(..., description="Estimated confidence score")
    threshold: float = Field(..., description="Target threshold")
    action_taken: DriftAction = Field(..., description="Action selected by the drift policy")
    original_answer: str = Field(..., description="First generation attempt")
    final_answer: str = Field(..., description="Final selected answer after mitigations")
    retries: int = Field(default=0, description="Number of retry attempts executed")
    fallback_used: bool = Field(default=False, description="Whether fallback model was activated")
    docker_rebuild_triggered: bool = Field(
        default=False, description="Whether container rebuild signal was dispatched"
    )


class RagasEvaluationRequest(BaseModel):
    """Input payload for Ragas metric evaluation."""

    question: str = Field(..., description="User query evaluated")
    answer: str = Field(..., description="Generated answer to evaluate")
    contexts: list[str] = Field(default_factory=list, description="Retrieved context chunks")
    ground_truth: str | None = Field(default=None, description="Optional ground truth reference")


class RagasEvaluationResponse(BaseModel):
    """Output metrics from Ragas evaluation."""

    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    context_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    context_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    hallucination_detected: bool = Field(default=False)
    passed_safeguards: bool = Field(default=True)
