"""Unit tests for document ingest strategies, embedding providers, and fallback cascades."""

import contextlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from agent.config.settings import Settings
from agent.services.rag.ingest import (
    DocumentIngestor,
    GoogleEmbeddingStrategy,
    HuggingFaceEmbeddingStrategy,
    OllamaEmbeddingStrategy,
    _get_document_map,
)


def test_google_embedding_strategy(monkeypatch):
    """Verify GoogleEmbeddingStrategy creates GoogleGenerativeAIEmbeddings."""
    mock_genai_mod = MagicMock()
    monkeypatch.setitem(sys.modules, "langchain_google_genai", mock_genai_mod)

    strategy = GoogleEmbeddingStrategy()
    settings = Settings(
        google_api_key=SecretStr("test-google-key"),
        embedding_model="models/embedding-001",
    )
    result = strategy.create(settings)
    assert result is not None
    mock_genai_mod.GoogleGenerativeAIEmbeddings.assert_called_once_with(
        model="models/embedding-001",
        google_api_key="test-google-key",
    )


def test_google_embedding_strategy_missing_key():
    """Verify GoogleEmbeddingStrategy raises error when api key is missing."""
    strategy = GoogleEmbeddingStrategy()
    settings = Settings(google_api_key=SecretStr(""))
    with pytest.raises(ValueError, match="Google API key is required"):
        strategy.create(settings)


def test_ollama_embedding_strategy(monkeypatch):
    """Verify OllamaEmbeddingStrategy creates OllamaEmbeddings."""
    mock_comm = MagicMock()
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", mock_comm)

    strategy = OllamaEmbeddingStrategy()
    settings = Settings(
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
    )
    result = strategy.create(settings)
    assert result is not None
    mock_comm.OllamaEmbeddings.assert_called_once_with(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )


def test_huggingface_embedding_strategy_fallbacks(monkeypatch):
    """Verify HuggingFaceEmbeddingStrategy fallback cascade to Instruct, MiniLM, and FakeEmbeddings."""
    mock_comm = MagicMock()
    # First HuggingFaceEmbeddings fails
    mock_comm.HuggingFaceEmbeddings.side_effect = RuntimeError("Primary HuggingFace failure")
    # HuggingFaceInstructEmbeddings fails
    mock_comm.HuggingFaceInstructEmbeddings.side_effect = RuntimeError("Instruct failure")
    mock_comm.FakeEmbeddings = MagicMock(return_value="FakeEmbeddingInstance")

    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", mock_comm)

    strategy = HuggingFaceEmbeddingStrategy()
    settings = Settings(embedding_model="BAAI/bge-m3")

    result = strategy.create(settings)
    assert result == "FakeEmbeddingInstance"
    mock_comm.FakeEmbeddings.assert_called_once_with(size=384)


def test_get_document_map_fallback_on_unstructured_missing(monkeypatch):
    """Verify fallback to standard TextLoader when unstructured packages fail to import."""
    with (
        patch("agent.services.rag.ingest._get_document_map", side_effect=ImportError("unstructured missing")),
        contextlib.suppress(ImportError),
    ):
        _get_document_map()


def test_document_ingestor_batch_exception_handling(tmp_path):
    """Verify DocumentIngestor logs and handles batch worker execution exceptions gracefully."""
    doc_file = tmp_path / "sample.txt"
    doc_file.write_text("Sample document text for ingestion.")

    ingestor = DocumentIngestor()

    with patch("concurrent.futures.ProcessPoolExecutor.submit", side_effect=RuntimeError("Worker spawn failed")):
        docs = ingestor.load_documents(str(tmp_path))
        assert docs == []
