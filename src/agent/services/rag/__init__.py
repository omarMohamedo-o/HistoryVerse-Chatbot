"""RAG service components: Ingestion, Retrieval, and Query Pipeline."""

from __future__ import annotations

from agent.services.rag.citation_guard import (
    CitationGuardrail,
    CitationVerificationResult,
    GroundingGuardrail,
    GroundingVerificationResult,
)
from agent.services.rag.refusal import DomainRefusalGate, LegalRefusalGate, RefusalDecision

__all__ = [
    "ChromaRetriever",
    "CitationGuardrail",
    "CitationVerificationResult",
    "DocumentIngestor",
    "DomainRefusalGate",
    "FileChatService",
    "GroundingGuardrail",
    "GroundingVerificationResult",
    "LegalRefusalGate",
    "RAGPipeline",
    "RefusalDecision",
]


def __getattr__(name: str):
    if name == "FileChatService":
        from agent.services.rag.file_chat import FileChatService

        return FileChatService
    if name == "DocumentIngestor":
        from agent.services.rag.ingest import DocumentIngestor

        return DocumentIngestor
    if name == "RAGPipeline":
        from agent.services.rag.pipeline import RAGPipeline

        return RAGPipeline
    if name == "ChromaRetriever":
        from agent.services.rag.retriever import ChromaRetriever

        return ChromaRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
