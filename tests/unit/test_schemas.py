"""Unit tests for Pydantic domain schemas."""

import pytest
from pydantic import ValidationError

from agentverse.core.enums import ConversationType
from agentverse.core.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
)


def test_chat_request_valid():
    req = ChatRequest(query="Tell me about Serapis", conversation_type=ConversationType.TEXT)
    assert req.query == "Tell me about Serapis"
    assert req.conversation_type == ConversationType.TEXT
    assert req.speak_response is False


def test_chat_request_empty_query_fails():
    with pytest.raises(ValidationError):
        ChatRequest(query="")


def test_chat_response_serialization():
    resp = ChatResponse(
        question="Who was Ptolemy I?",
        answer="Ptolemy I Soter was a Macedonian general of Alexander the Great.",
        sources=["History of Alexandria.pdf"],
        confidence_score=0.92,
        model_used="gemini/gemini-1.5-flash",
    )
    data = resp.model_dump()
    assert data["confidence_score"] == 0.92
    assert len(data["sources"]) == 1


def test_ingest_request_defaults():
    req = IngestRequest()
    assert req.source_directory == "./dataset"
    assert req.chunk_size == 1000
    assert req.chunk_overlap == 200
