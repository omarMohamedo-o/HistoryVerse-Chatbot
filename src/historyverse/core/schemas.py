"""Pydantic v2 domain schemas for request and response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from historyverse.core.enums import (
    ConversationType,
    DeviceType,
    DriftAction,
    LLMProvider,
    UnlearningStatus,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "DriftReport",
    "HealthResponse",
    "HistoricalEntity",
    "HistoricalFact",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "IngestRequest",
    "IngestResponse",
    "ProbeResult",
    "RagasEvaluationRequest",
    "RagasEvaluationResponse",
    "StructuredHistoricalAnalysis",
    "TombstoneRecord",
    "UnlearningRequest",
    "UnlearningValidationReport",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
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
    model_used: str = Field(
        default="", description="Identifier of the model that generated the answer"
    )
    trace_id: str | None = Field(default=None, description="Langfuse or telemetry trace identifier")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Evaluation metrics (faithfulness, relevancy, etc.)"
    )


class ImageGenerationRequest(BaseModel):
    """Payload for generating historical reconstructions or artistic visions."""

    prompt: str = Field(
        ..., min_length=3, description="Description of the historical scene or artifact"
    )
    size: str = Field(default="1024x1024", description="Image resolution")
    quality: str = Field(default="standard", description="standard or hd")
    enhance_prompt: bool = Field(
        default=True, description="Whether to expand prompt using historical AI"
    )


class ImageGenerationResponse(BaseModel):
    """Response containing URL or path to generated image."""

    original_prompt: str = Field(..., description="Input prompt provided by user")
    enhanced_prompt: str = Field(
        ..., description="Historically expanded prompt used for generation"
    )
    image_url: str = Field(..., description="Direct URL or local path to generated image file")
    model_used: str = Field(..., description="Model identifier used for synthesis")


class VideoGenerationRequest(BaseModel):
    """Payload for generating historical videos or animating ancient scenes/artifacts."""

    prompt: str = Field(
        ..., min_length=3, description="Description of the historical scene, motion, or event"
    )
    image_url: str | None = Field(
        default=None, description="Optional source image for image-to-video historical animation"
    )
    duration_seconds: int = Field(
        default=4, ge=1, le=60, description="Target duration of the video clip in seconds"
    )
    fps: int = Field(default=24, ge=1, le=60, description="Frames per second")
    resolution: str = Field(
        default="1024x1024", description="Video resolution (e.g. 1024x1024, 1280x720)"
    )
    aspect_ratio: str = Field(default="16:9", description="Aspect ratio (16:9, 9:16, 1:1)")
    enhance_prompt: bool = Field(
        default=True,
        description="Whether to historically enrich camera motion and atmospheric details",
    )


class VideoGenerationResponse(BaseModel):
    """Response containing URL or path to generated historical video."""

    original_prompt: str = Field(..., description="Input prompt provided by user")
    enhanced_prompt: str = Field(
        ..., description="Historically expanded prompt with cinematic and motion cues"
    )
    video_url: str = Field(..., description="Direct URL or local path to generated video file")
    duration_seconds: float = Field(..., description="Video duration in seconds")
    fps: int = Field(default=24, description="Frames per second")
    model_used: str = Field(..., description="Model identifier used for video synthesis")


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


class UnlearningRequest(BaseModel):
    """Payload to initiate selective unlearning of target concept or information."""

    target_concept: str = Field(
        ..., min_length=2, description="The concept, fact, or entity to forget"
    )
    forbidden_keywords: list[str] = Field(
        default_factory=list, description="Keywords and synonyms associated with the concept"
    )
    agent_ids: list[str] = Field(
        default_factory=list, description="Target agent IDs to unlearn from; empty means all agents"
    )
    purge_rag: bool = Field(
        default=True, description="Whether to purge matching chunks from ChromaDB"
    )
    purge_memory: bool = Field(
        default=True, description="Whether to purge matching turns from conversational memory"
    )
    reason: str = Field(default="retraction", description="Reason for unlearning")


class ProbeResult(BaseModel):
    """Result of an individual adversarial probe test validating unlearning."""

    probe_query: str = Field(
        ..., description="Probing question designed to elicit forgotten concept"
    )
    probe_response: str = Field(..., description="Response returned by the model")
    leaked: bool = Field(..., description="Whether forbidden concept tokens leaked in response")
    leakage_confidence: float = Field(default=0.0, description="Confidence of leakage detection")
    status: str = Field(default="passed", description="Status string: passed or failed")


class UnlearningValidationReport(BaseModel):
    """Cryptographic/auditable report certifying unlearning success."""

    certificate_id: str = Field(..., description="Unique validation certificate identifier")
    target_concept: str = Field(..., description="Unlearned target concept")
    status: UnlearningStatus = Field(..., description="Overall unlearning outcome status")
    probes_evaluated: int = Field(..., description="Total probing queries evaluated")
    probes: list[ProbeResult] = Field(default_factory=list, description="Detailed probe outputs")
    leakage_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of probes that leaked (0.0=ideal)"
    )
    rag_residuals_found: int = Field(
        default=0, description="Residual matches found in vector store"
    )
    memory_turns_purged: int = Field(
        default=0, description="Total dialogue turns purged from memory"
    )
    validation_passed: bool = Field(
        ..., description="True if leakage_rate is 0 and rag_residuals is 0"
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of verification")


class TombstoneRecord(BaseModel):
    """Active tombstone representing retracted and unlearned knowledge."""

    concept_id: str
    target_concept: str
    forbidden_keywords: list[str] = Field(default_factory=list)
    affected_agents: list[str] = Field(default_factory=list)
    reason: str = "retraction"
    created_at: str


class HistoricalEntity(BaseModel):
    """Structured historical figure, place, or artifact entity."""

    name: str = Field(..., description="Name of the historical entity")
    era: str = Field(..., description="Chronological era or dynasty")
    significance: str = Field(..., description="Historical significance or role")


class HistoricalFact(BaseModel):
    """A discrete verifiable historical proposition."""

    claim: str = Field(..., description="Historical assertion or factual claim")
    is_verified: bool = Field(
        default=True, description="Whether supported by primary/secondary records"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sources: list[str] = Field(
        default_factory=list, description="Associated citations or archaeological records"
    )


class StructuredHistoricalAnalysis(BaseModel):
    """Schema-validated structured JSON output for historical analysis."""

    topic: str = Field(..., description="Subject of the historical inquiry")
    historical_period: str = Field(..., description="Relevant era (e.g. Ptolemaic, Roman, Islamic)")
    summary: str = Field(..., description="Concise analytical overview")
    key_entities: list[HistoricalEntity] = Field(
        default_factory=list, description="Prominent figures, monuments, or places"
    )
    verified_facts: list[HistoricalFact] = Field(
        default_factory=list, description="Structured factual claims"
    )
    confidence_score: float = Field(default=0.95, ge=0.0, le=1.0)
