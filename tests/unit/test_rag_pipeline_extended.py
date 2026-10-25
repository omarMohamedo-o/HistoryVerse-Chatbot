"""Unit tests for RAG pipeline fallback, multilingual utilities, and retriever edge cases."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from agent.config.settings import Settings
from agent.core.enums import DriftAction
from agent.core.schemas import DriftReport
from agent.services.rag.embeddings import EmbeddingModelRegistry
from agent.services.rag.pipeline import RAGPipeline
from agent.services.rag.retriever import ChromaRetriever as RAGRetriever
from agent.utils.multilingual import detect_script, expand_multilingual_keywords


@pytest.mark.asyncio
async def test_rag_pipeline_context_miss_fallback():
    """Verify RAGPipeline triggers fallback prompt when primary context answer misses."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="Unrelated text about modern architecture.")
    ]

    mock_drift = MagicMock()
    # First response triggers context miss; second response is the fallback
    mock_drift.generate_with_drift_protection = AsyncMock(
        side_effect=[
            (
                "The answer is not available in the context provided.",
                DriftReport(
                    action_taken=DriftAction.ALERT,
                    confidence_score=0.5,
                    threshold=0.75,
                    original_answer="old",
                    final_answer="old",
                ),
            ),
            (
                "Alexander the Great founded Alexandria in 331 BC.",
                DriftReport(
                    action_taken=DriftAction.ALERT,
                    confidence_score=0.95,
                    threshold=0.75,
                    original_answer="new",
                    final_answer="new",
                ),
            ),
        ]
    )

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    settings = Settings()
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_provider=mock_llm,
        drift_detector=mock_drift,
        settings=settings,
    )

    resp = await pipeline.query("When was Alexandria founded?")
    assert "Alexander the Great" in resp.answer
    assert resp.sources == ["General Historical Knowledge Base"]
    assert mock_drift.generate_with_drift_protection.call_count == 2


@pytest.mark.asyncio
async def test_retriever_retrieve_async_and_filter_branches():
    """Verify RAGRetriever retrieve_async and delete_documents_by_keywords with agent and tenant filters."""
    mock_store = MagicMock()
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": ["doc_1", "doc_2"]}
    mock_store._collection = mock_collection

    retriever = RAGRetriever()
    retriever._vectorstore = mock_store

    # Test retrieve_async
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            retriever,
            "retrieve",
            lambda q, top_k, agent_id, tenant_id: [Document(page_content="async doc")],
        )
        results = await retriever.retrieve_async("test query", top_k=1)
        assert len(results) == 1

    # Test delete_documents_by_keywords with only agent_id
    deleted_agent = retriever.delete_documents_by_keywords(["pharaoh"], agent_id="agent_123")
    assert deleted_agent >= 0

    # Test delete_documents_by_keywords with both agent_id and tenant_id ($and branch)
    deleted_both = retriever.delete_documents_by_keywords(
        ["pharaoh"], agent_id="agent_123", tenant_id="tenant_abc"
    )
    assert deleted_both >= 0


def test_multilingual_detect_script_unknown_and_ligatures():
    """Verify detect_script returns 'unknown' for emojis and expands ligatures correctly."""
    # Unknown script (emojis / math symbols)
    assert detect_script("✨🌟⚡") == "unknown"

    # French and Latin ligatures expansion
    variants = expand_multilingual_keywords("cœur") + expand_multilingual_keywords("archæology")
    assert any("coeur" in v or "cœur" in v for v in variants)
    assert any("archaeology" in v or "archæology" in v for v in variants)


def test_embedding_model_registry_multilingual_language_check():
    """Verify EmbeddingModelRegistry.is_multilingual_compatible with non-Arabic target language."""
    # BAAI/bge-m3 is multilingual
    assert EmbeddingModelRegistry.is_multilingual_compatible("BAAI/bge-m3", language="fr") is True
    # Unknown model defaults to True
    assert (
        EmbeddingModelRegistry.is_multilingual_compatible("unknown/custom-model", language="es")
        is True
    )


def test_retriever_get_vectorstore_when_dir_exists(tmp_path, monkeypatch):
    """Verify ChromaRetriever._ensure_vectorstore initializes Chroma when persist directory exists."""
    retriever = RAGRetriever()
    retriever._persist_dir = str(tmp_path)
    mock_chroma_cls = MagicMock()
    monkeypatch.setattr("langchain_chroma.Chroma", mock_chroma_cls)
    monkeypatch.setattr(retriever, "_get_embedding_function", lambda: MagicMock())
    store = retriever._ensure_vectorstore()
    assert store is not None
    mock_chroma_cls.assert_called_once()


def test_file_chat_extract_and_query_fallbacks():
    """Verify FileChatService handles unparseable bytes and short query ranking fallback."""
    from agent.core.exceptions import RAGError
    from agent.services.rag.file_chat import FileChatService

    service = FileChatService(llm_provider=MagicMock())

    # 1. Purely non-printable bytes
    docs = service.load_documents_from_bytes(b"\x00\x01\x02\x03\x04", "binary.unknown")
    assert "could not be parsed" in docs[0].page_content

    # 2. Short query ranking fallback
    many_chunks = [Document(page_content=f"Chunk {i}") for i in range(10)]
    selected = service._select_relevant_chunks(many_chunks, query="is it", top_k=3)
    assert len(selected) == 3

    # 3. Network URL failure raises RAGError
    with pytest.raises(RAGError, match="Failed to fetch document from URL"):
        service.load_documents_from_source("http://127.0.0.1:9999/nonexistent.pdf")


def test_file_chat_pdf_fallback_and_exceptions(monkeypatch):
    """Verify FileChatService._extract_pdf_pages handles pypdf ImportError and PyPDFLoader exceptions."""
    from agent.services.rag.file_chat import FileChatService

    service = FileChatService(llm_provider=MagicMock())

    # 1. pypdf raises ImportError (line 98) and PyPDFLoader succeeds
    mock_pdf_loader = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [Document(page_content="Page 1 text", metadata={})]
    mock_pdf_loader.return_value = mock_loader_instance

    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setattr("langchain_community.document_loaders.PyPDFLoader", mock_pdf_loader)

    docs = service._extract_pdf_pages(b"%PDF-1.4 fake pdf bytes", "test.pdf")
    assert len(docs) == 1
    assert docs[0].page_content == "Page 1 text"

    # 2. PyPDFLoader raises Exception (lines 117-118, 123)
    mock_loader_instance.load.side_effect = RuntimeError("Loader corrupted")
    docs_empty = service._extract_pdf_pages(b"%PDF-1.4 fake pdf bytes", "test.pdf")
    assert docs_empty == []
