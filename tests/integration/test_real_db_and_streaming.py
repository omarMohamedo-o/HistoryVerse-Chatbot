"""Real database isolation and live streaming integration tests without external network calls."""

from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

from agent.config.settings import Settings
from agent.services.rag.retriever import ChromaRetriever
from agent.services.voice.streaming_tts import StreamingTTS


class DeterministicEmbeddings:
    """Fast, local, in-memory embedding function for real vector store tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float((hash(t) + i) % 100) / 100.0 for i in range(16)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float((hash(text) + i) % 100) / 100.0 for i in range(16)]


def test_real_chroma_db_tenant_isolation_and_purges(tmp_path):
    """Instantiate a real ChromaDB instance on disk and verify multi-tenant isolation and document purges."""
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma

    persist_dir = str(tmp_path / "real_chroma_db")
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = DeterministicEmbeddings()
    chroma_settings = ChromaSettings(anonymized_telemetry=False, is_persistent=True)

    # 1. Create real Chroma vector store on disk
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        client_settings=chroma_settings,
        collection_name="tenant_isolation_test",
    )

    doc_egypt = Document(
        page_content="tutankhamun ruled Egypt during the 18th Dynasty.",
        metadata={"agent_id": "agent_egypt", "tenant_id": "tenant_alpha"},
    )
    doc_rome = Document(
        page_content="Julius Caesar led the Roman Republic in the 1st century BC.",
        metadata={"agent_id": "agent_rome", "tenant_id": "tenant_alpha"},
    )
    doc_other_tenant = Document(
        page_content="Alexander the Great expanded Macedonia across Persia.",
        metadata={"agent_id": "agent_egypt", "tenant_id": "tenant_beta"},
    )

    vectorstore.add_documents([doc_egypt, doc_rome, doc_other_tenant])

    settings = Settings(chroma_persist_dir=persist_dir, retrieval_top_k=5)
    retriever = ChromaRetriever(settings=settings)
    retriever._vectorstore = vectorstore

    # 2. Query filtered by agent_id="agent_egypt" and tenant_id="tenant_alpha"
    results_egypt = retriever.retrieve(
        "Pharaoh",
        agent_id="agent_egypt",
        tenant_id="tenant_alpha",
    )
    assert len(results_egypt) == 1
    assert "tutankhamun" in results_egypt[0].page_content.lower()

    # 3. Query filtered by agent_id="agent_rome" and tenant_id="tenant_alpha"
    results_rome = retriever.retrieve(
        "Caesar",
        agent_id="agent_rome",
        tenant_id="tenant_alpha",
    )
    assert len(results_rome) == 1
    assert "Julius Caesar" in results_rome[0].page_content

    # 4. Cross-tenant inquiry (tenant_beta searching for Rome) MUST return 0 results
    results_cross = retriever.retrieve(
        "Caesar",
        agent_id="agent_rome",
        tenant_id="tenant_beta",
    )
    assert len(results_cross) == 0

    # 5. Purge by keyword
    purged_count = retriever.delete_documents_by_keywords(["tutankhamun"])
    assert purged_count >= 1

    # Verify document is gone from the real collection
    remaining = retriever.retrieve("Pharaoh", agent_id="agent_egypt", tenant_id="tenant_alpha")
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_real_streaming_generator_and_sentence_chunking():
    """Verify live async generator token consumption and sentence-chunked streaming TTS."""

    async def sample_token_stream():
        tokens = [
            "Ancient ",
            "Alexandria ",
            "had ",
            "a ",
            "great ",
            "library. ",
            "It ",
            "preserved ",
            "classical ",
            "knowledge.",
        ]
        for t in tokens:
            yield t

    collected_tokens: list[str] = []
    async for token in sample_token_stream():
        collected_tokens.append(token)

    assembled = "".join(collected_tokens)
    assert assembled == "Ancient Alexandria had a great library. It preserved classical knowledge."

    # Test SentenceChunker and StreamingTTS chunk synthesis
    from agent.services.voice.streaming_tts import SentenceChunker

    chunker = SentenceChunker(min_chunk_chars=10)
    chunks = []
    async for sentence in chunker.chunk_stream(sample_token_stream()):
        chunks.append(sentence)

    assert len(chunks) == 2
    assert "Ancient Alexandria had a great library." in chunks[0]
    assert "It preserved classical knowledge." in chunks[1]

    streaming_tts = StreamingTTS()
    audio_bytes = await streaming_tts.synthesize_chunk(chunks[0])
    assert audio_bytes is not None
    assert len(audio_bytes) > 0
