"""Unit tests for ChromaRetriever, embedding strategies, and document loaders."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agent.config.settings import Settings
from agent.core.enums import EmbeddingProvider
from agent.core.exceptions import RetrieverError
from agent.services.rag.ingest import (
    DocumentIngestor,
    EmbeddingStrategyFactory,
    GoogleEmbeddingStrategy,
    HuggingFaceEmbeddingStrategy,
    OllamaEmbeddingStrategy,
    load_document_batch,
    load_single_document,
)
from agent.services.rag.retriever import ChromaRetriever


def test_embedding_strategy_factory_selection():
    strat_google = EmbeddingStrategyFactory.get_strategy(EmbeddingProvider.GOOGLE.value)
    assert isinstance(strat_google, GoogleEmbeddingStrategy)

    strat_ollama = EmbeddingStrategyFactory.get_strategy(EmbeddingProvider.OLLAMA.value)
    assert isinstance(strat_ollama, OllamaEmbeddingStrategy)

    strat_hf = EmbeddingStrategyFactory.get_strategy(EmbeddingProvider.HUGGINGFACE.value)
    assert isinstance(strat_hf, HuggingFaceEmbeddingStrategy)

    # Unknown defaults to HuggingFace
    strat_fallback = EmbeddingStrategyFactory.get_strategy("unknown_provider")
    assert isinstance(strat_fallback, HuggingFaceEmbeddingStrategy)


def test_load_single_document_text(tmp_path):
    text_file = tmp_path / "papyrus.txt"
    text_file.write_text("Hieroglyphs carved onto limestone stele.")

    doc = load_single_document(str(text_file))
    assert doc is not None
    assert "Hieroglyphs" in doc.page_content


def test_load_single_document_unsupported():
    doc = load_single_document("unknown_file.unsupported_ext")
    assert doc is None


def test_load_document_batch(tmp_path):
    f1 = tmp_path / "doc1.txt"
    f1.write_text("Document 1 content")
    f2 = tmp_path / "doc2.txt"
    f2.write_text("Document 2 content")

    docs, paths = load_document_batch([str(f1), str(f2)])
    assert len(docs) == 2
    assert len(paths) == 2


def test_document_ingestor_load_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_sources"
    empty_dir.mkdir()

    ingestor = DocumentIngestor()
    docs = ingestor.load_documents(str(empty_dir))
    assert docs == []


def test_retriever_uninitialized_directory():
    settings = Settings(chroma_persist_dir="/nonexistent/chroma_db_dir")
    retriever = ChromaRetriever(settings)

    assert retriever._ensure_vectorstore() is None
    results = retriever.retrieve("Any historical query")
    assert results == []


def test_retriever_retrieve_with_tombstone_filtering():
    retriever = ChromaRetriever(Settings())
    mock_store = MagicMock()

    doc_valid = Document(page_content="Ptolemy II built the Serapeum library branch.")
    doc_forbidden = Document(page_content="Debunked secret tunnel rumor.")
    mock_store.similarity_search.return_value = [doc_valid, doc_forbidden]

    retriever._vectorstore = mock_store

    with patch("agent.services.unlearning.registry.get_tombstone_registry") as mock_tb_reg:
        mock_registry = MagicMock()
        mock_registry.is_forbidden.side_effect = lambda content: (
            (True, "Secret Tunnel") if "tunnel" in content else (False, None)
        )
        mock_tb_reg.return_value = mock_registry

        results = retriever.retrieve("Library branch", top_k=2)
        assert len(results) == 1
        assert "Ptolemy II" in results[0].page_content


def test_retriever_search_exception_raises_retriever_error():
    retriever = ChromaRetriever(Settings())
    mock_store = MagicMock()
    mock_store.similarity_search.side_effect = Exception("Chroma index corrupted")
    retriever._vectorstore = mock_store

    with pytest.raises(RetrieverError, match="Chroma index corrupted"):
        retriever.retrieve("Any query")


def test_retriever_delete_documents_native_where_document():
    retriever = ChromaRetriever(Settings())
    mock_store = MagicMock()
    mock_collection = MagicMock()

    mock_collection.count.side_effect = [10, 8]  # 2 deleted
    mock_store._collection = mock_collection
    retriever._vectorstore = mock_store

    purged = retriever.delete_documents_by_keywords(["alien pyramid"])
    assert purged == 2
    mock_collection.delete.assert_called_once_with(where_document={"$contains": "alien pyramid"})


def test_retriever_delete_documents_fallback():
    retriever = ChromaRetriever(Settings())
    mock_store = MagicMock()
    mock_collection = MagicMock()

    mock_collection.delete.side_effect = [Exception("where_document not supported"), None]
    mock_collection.get.return_value = {"ids": ["doc-101", "doc-102"]}
    mock_store._collection = mock_collection
    retriever._vectorstore = mock_store

    purged = retriever.delete_documents_by_keywords(["retracted fact"])
    assert purged == 2


def test_document_ingestor_split_documents():
    ingestor = DocumentIngestor()
    docs = [
        Document(page_content="Text content", metadata={"source": "history.txt"}),
        Document(page_content="def script(): pass", metadata={"source": "analysis.py"}),
    ]
    text_docs, py_docs = ingestor.split_documents(docs)
    assert len(text_docs) == 1
    assert len(py_docs) == 1
    assert text_docs[0].metadata["source"] == "history.txt"
    assert py_docs[0].metadata["source"] == "analysis.py"


def test_huggingface_embedding_strategy_fallbacks():
    strategy = HuggingFaceEmbeddingStrategy()
    settings = Settings()

    # Fallback to SentenceTransformers or FakeEmbeddings
    with patch.dict("sys.modules", {"langchain_community.embeddings": MagicMock()}):
        res = strategy.create(settings)
        assert res is not None


def test_document_ingestor_ingest_pipeline(tmp_path):
    source_dir = tmp_path / "source_docs"
    source_dir.mkdir()
    (source_dir / "article.txt").write_text("The Rosetta Stone deciphered Egyptian hieroglyphs.")
    (source_dir / "code.py").write_text("# Python utility\ndef parse(): pass\n")

    persist_dir = tmp_path / "chroma_db"
    settings = Settings(
        source_directory=str(source_dir),
        chroma_persist_dir=str(persist_dir),
        ingest_threads=1,
    )
    ingestor = DocumentIngestor(settings=settings)

    with (
        patch.object(ingestor, "get_embeddings", return_value=MagicMock()),
        patch("langchain_chroma.Chroma.from_documents") as mock_chroma,
    ):
        resp = ingestor.ingest(str(source_dir))
        assert resp.documents_loaded == 2
        assert resp.chunks_created >= 2
        assert mock_chroma.called


def test_embedding_model_registry_and_recommendations():
    from agent.services.rag.embeddings import EmbeddingModelRegistry
    from agent.services.rag.ingest import OpenAIEmbeddingStrategy

    p1 = EmbeddingModelRegistry.get_profile("BAAI/bge-m3")
    assert p1 is not None
    p2 = EmbeddingModelRegistry.get_profile("bge-m3")
    assert p2 is not None
    assert EmbeddingModelRegistry.get_profile("nonexistent/model") is None

    assert EmbeddingModelRegistry.is_multilingual_compatible("BAAI/bge-m3", "ar") is True
    assert EmbeddingModelRegistry.is_multilingual_compatible("unknown-model", "ar") is True

    ok1, _ = EmbeddingModelRegistry.validate_language_compatibility("unknown-model")
    assert ok1 is True

    assert (
        EmbeddingModelRegistry.recommend_model_for_language("ar", "google")
        == "models/text-embedding-004"
    )
    assert (
        EmbeddingModelRegistry.recommend_model_for_language("ar", "openai")
        == "text-embedding-3-small"
    )
    assert EmbeddingModelRegistry.recommend_model_for_language("ar", "ollama") == "bge-m3"

    openai_strat = OpenAIEmbeddingStrategy()
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        openai_strat.create(Settings(openai_api_key=""))

    with (
        patch.dict("sys.modules", {"langchain_openai": None}),
        patch("langchain_community.embeddings.OpenAIEmbeddings") as mock_emb,
    ):
        mock_emb.return_value = MagicMock()
        res = openai_strat.create(Settings(openai_api_key="sk-testkey1234567890"))
        assert res is not None


@pytest.mark.asyncio
async def test_retriever_and_ingest_edge_cases(tmp_path):
    # Empty dir ingest
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    ingestor = DocumentIngestor(Settings())
    res_empty = ingestor.ingest(source_dir=str(empty_dir))
    assert res_empty.documents_loaded == 0

    # get_embeddings fallback
    with (
        patch(
            "agent.services.rag.ingest.GoogleEmbeddingStrategy.create",
            side_effect=Exception("Strategy error"),
        ),
        patch(
            "agent.services.rag.ingest.HuggingFaceEmbeddingStrategy.create",
            return_value=MagicMock(),
        ),
    ):
        emb = ingestor.get_embeddings()
        assert emb is not None

    # load_single_document exception
    with patch(
        "agent.services.rag.ingest.DOCUMENT_MAP",
        {".txt": MagicMock(side_effect=Exception("IO err"))},
    ):
        assert load_single_document("invalid.txt") is None

    # Retriever ChromaDB connect error
    retriever = ChromaRetriever(Settings(chroma_persist_dir=str(tmp_path)))
    with patch("langchain_chroma.Chroma", side_effect=Exception("Chroma init fail")):
        assert retriever._ensure_vectorstore() is None

    # Retriever purge error handling and async purge
    with patch.object(retriever, "_ensure_vectorstore", return_value=None):
        assert retriever.delete_documents_by_keywords(["pyramid"]) == 0

    mock_broken_store = MagicMock()
    mock_broken_store._collection.delete.side_effect = Exception("Chroma delete boom")
    with patch.object(retriever, "_ensure_vectorstore", return_value=mock_broken_store):
        assert retriever.delete_documents_by_keywords(["pyramid"]) == 0

    with patch.object(retriever, "delete_documents_by_keywords", return_value=5):
        assert await retriever.delete_documents_by_keywords_async(["pharaoh"]) == 5
