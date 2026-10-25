"""Unit tests for multi-agent and multi-tenant RAG isolation in ChromaRetriever."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from agent.config.settings import Settings
from agent.services.rag.retriever import ChromaRetriever


def test_rag_agent_and_tenant_metadata_filtering():
    settings = Settings()
    retriever = ChromaRetriever(settings=settings)

    # Mock vector store
    mock_store = MagicMock()
    retriever._vectorstore = mock_store

    doc_history = Document(
        page_content="The Lighthouse of Alexandria was built by the Ptolemaic Kingdom.",
        metadata={"agent_id": "history-agent", "tenant_id": "tenant-history"},
    )
    doc_medical = Document(
        page_content="Penicillin is an antibiotic derived from Penicillium fungi.",
        metadata={"agent_id": "medical-agent", "tenant_id": "tenant-medical"},
    )

    def mock_similarity_search(query, k=5, **kwargs):
        docs = [doc_history, doc_medical]
        filter_expr = kwargs.get("filter")
        if not filter_expr:
            return docs
        # Check single clause filter
        if "agent_id" in filter_expr:
            return [d for d in docs if d.metadata.get("agent_id") == filter_expr["agent_id"]]
        if "tenant_id" in filter_expr:
            return [d for d in docs if d.metadata.get("tenant_id") == filter_expr["tenant_id"]]
        # Check $and clause filter
        if "$and" in filter_expr:
            clauses = filter_expr["$and"]
            filtered = docs
            for clause in clauses:
                for k_attr, v_val in clause.items():
                    filtered = [d for d in filtered if d.metadata.get(k_attr) == v_val]
            return filtered
        return docs

    mock_store.similarity_search.side_effect = mock_similarity_search

    # 1. Query for history agent strictly isolates to history docs
    history_results = retriever.retrieve("tell me something", agent_id="history-agent")
    assert len(history_results) == 1
    assert "Lighthouse" in history_results[0].page_content
    mock_store.similarity_search.assert_called_with(
        "tell me something", k=settings.retrieval_top_k, filter={"agent_id": "history-agent"}
    )

    # 2. Query for medical agent strictly isolates to medical docs
    medical_results = retriever.retrieve("tell me something", agent_id="medical-agent")
    assert len(medical_results) == 1
    assert "Penicillin" in medical_results[0].page_content
    assert "Lighthouse" not in medical_results[0].page_content

    # 3. Dual filter ($and)
    dual_results = retriever.retrieve(
        "tell me something", agent_id="history-agent", tenant_id="tenant-history"
    )
    assert len(dual_results) == 1
    assert "Lighthouse" in dual_results[0].page_content

    # 4. Cross-tenant mismatch yields zero leakage
    mismatch_results = retriever.retrieve(
        "tell me something", agent_id="history-agent", tenant_id="tenant-medical"
    )
    assert len(mismatch_results) == 0


def test_document_ingestor_attaches_agent_and_tenant_metadata(tmp_path):
    from agent.services.rag.ingest import DocumentIngestor

    # Create a test document
    doc_file = tmp_path / "sample.txt"
    doc_file.write_text("Tutankhamun was an ancient Egyptian pharaoh.", encoding="utf-8")

    settings = Settings(chroma_persist_dir=str(tmp_path / "chroma"))
    ingestor = DocumentIngestor(settings=settings)

    # Mock embeddings and Chroma
    mock_embeddings = MagicMock()
    ingestor.get_embeddings = MagicMock(return_value=mock_embeddings)

    from unittest.mock import patch

    with patch("langchain_chroma.Chroma.from_documents") as mock_from_docs:
        resp = ingestor.ingest(
            source_dir=str(tmp_path),
            agent_id="ar-agent-mora",
            tenant_id="museum-egypt",
        )
        assert resp.documents_loaded == 1
        assert resp.chunks_created >= 1

        mock_from_docs.assert_called_once()
        passed_docs = mock_from_docs.call_args[1]["documents"]
        for d in passed_docs:
            assert d.metadata.get("agent_id") == "ar-agent-mora"
            assert d.metadata.get("tenant_id") == "museum-egypt"
