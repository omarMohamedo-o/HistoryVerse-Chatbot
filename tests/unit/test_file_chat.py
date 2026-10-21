"""Unit tests for FileChatService and document file QA endpoints."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from agent.api.app import create_app
from agent.api.dependencies import get_llm_provider
from agent.config.settings import Settings
from agent.services.rag.file_chat import FileChatService
from tests.conftest import MockLLMProvider


@pytest.fixture
def mock_llm():
    return MockLLMProvider("mock-gemini")


@pytest.fixture
def file_chat_service(mock_llm):
    settings = Settings()
    return FileChatService(llm_provider=mock_llm, settings=settings)


def test_load_documents_from_bytes_text(file_chat_service):
    content = (
        "# Serapeum of Alexandria\nThe Serapeum was an ancient Greek temple built by Ptolemy III."
    )
    docs = file_chat_service.load_documents_from_bytes(content.encode("utf-8"), "serapeum.md")
    assert len(docs) == 1
    assert "Serapeum" in docs[0].page_content
    assert docs[0].metadata["source"] == "serapeum.md"


def test_load_documents_from_bytes_empty(file_chat_service):
    docs = file_chat_service.load_documents_from_bytes(b"", "empty.txt")
    assert len(docs) == 1
    assert "Empty document" in docs[0].page_content


@pytest.mark.asyncio
async def test_chat_with_documents(file_chat_service):
    docs = [
        Document(
            page_content="The Pharos of Alexandria was a lighthouse built by the Ptolemaic Kingdom.",
            metadata={"source": "lighthouse.txt", "page": 1},
        ),
        Document(
            page_content="It was estimated to be at least 100 metres (330 ft) in overall height.",
            metadata={"source": "lighthouse.txt", "page": 2},
        ),
    ]

    response = await file_chat_service.chat_with_documents(
        documents=docs,
        question="How tall was the Pharos of Alexandria?",
        conversation_id="test-file-conv",
    )

    assert response.question == "How tall was the Pharos of Alexandria?"
    assert response.answer is not None
    assert "lighthouse.txt" in response.sources


@pytest.mark.asyncio
async def test_chat_with_file_source_local(tmp_path: Path, file_chat_service):
    sample_file = tmp_path / "hypatia.txt"
    sample_file.write_text(
        "Hypatia was a Neoplatonist philosopher, astronomer, and mathematician in Alexandria, Egypt.",
        encoding="utf-8",
    )

    response = await file_chat_service.chat_with_file_source(
        file_source=str(sample_file),
        question="Who was Hypatia?",
    )
    assert response.question == "Who was Hypatia?"
    assert response.answer is not None
    assert "hypatia.txt" in response.sources


def test_chat_file_multipart_api_endpoint(mock_llm):
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    file_bytes = b"Cleopatra VII Philopator was the queen of the Ptolemaic Kingdom of Egypt from 51 to 30 BC."
    file_obj = io.BytesIO(file_bytes)

    response = client.post(
        "/v1/chat/file",
        files={"file": ("cleopatra.txt", file_obj, "text/plain")},
        data={"query": "When did Cleopatra rule?", "speak_response": "false"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "When did Cleopatra rule?"
    assert data["answer"] is not None
    assert "cleopatra.txt" in data["sources"]


def test_chat_endpoint_with_file_source_json(tmp_path: Path, mock_llm):
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    doc_path = tmp_path / "alexander.txt"
    doc_path.write_text("Alexander the Great founded Alexandria in 331 BC.", encoding="utf-8")

    payload = {
        "query": "Who founded Alexandria?",
        "file_source": str(doc_path),
        "conversation_type": "pdf",
    }

    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "Who founded Alexandria?"
    assert data["answer"] is not None
    assert "alexander.txt" in data["sources"]


def test_extract_pdf_pages_with_mocked_pypdf(file_chat_service, monkeypatch):
    from unittest.mock import MagicMock

    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "Page 1: Library of Alexandria founded."
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "Page 2: Destroyed in antiquity."

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page_1, mock_page_2]

    mock_pypdf = MagicMock()
    mock_pypdf.PdfReader.return_value = mock_reader
    monkeypatch.setitem(__import__("sys").modules, "pypdf", mock_pypdf)

    docs = file_chat_service.load_documents_from_bytes(b"%PDF-1.4 dummy", "sample.pdf")
    assert len(docs) == 2
    assert "Page 1" in docs[0].page_content
    assert docs[0].metadata["page"] == 1
    assert "Page 2" in docs[1].page_content


def test_extract_docx_with_mock(file_chat_service, monkeypatch):
    from unittest.mock import MagicMock

    mock_docx2txt = MagicMock()
    mock_docx2txt.process.return_value = "Word document containing historical data."
    monkeypatch.setitem(__import__("sys").modules, "docx2txt", mock_docx2txt)

    docs = file_chat_service.load_documents_from_bytes(b"PK\x03\x04 dummy docx", "sample.docx")
    assert len(docs) == 1
    assert "Word document" in docs[0].page_content


def test_fallback_printable_text_extraction(file_chat_service):
    raw_binary = b"\x00\x01\x02Ancient Rosetta Stone\x00\x03Inscribed in three scripts."
    docs = file_chat_service.load_documents_from_bytes(raw_binary, "artifact.bin")
    assert len(docs) == 1
    assert "Ancient Rosetta Stone" in docs[0].page_content


def test_load_documents_from_source_url(file_chat_service, monkeypatch):
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.read.return_value = b"Archimedes principle discovered in Alexandria."
    mock_response.__enter__.return_value = mock_response

    mock_urlopen = MagicMock(return_value=mock_response)
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    docs = file_chat_service.load_documents_from_source("https://example.com/archimedes.txt")
    assert len(docs) == 1
    assert "Archimedes" in docs[0].page_content
    mock_urlopen.assert_called_once()


def test_load_documents_nonexistent_file_raises(file_chat_service):
    from agent.core.exceptions import RAGError

    with pytest.raises(RAGError, match="does not exist"):
        file_chat_service.load_documents_from_source("nonexistent_path_xyz.txt")


@pytest.mark.asyncio
async def test_chat_with_documents_empty_raises(file_chat_service):
    from agent.core.exceptions import RAGError

    with pytest.raises(RAGError, match="No documents provided"):
        await file_chat_service.chat_with_documents([], "Question?")


def test_select_relevant_chunks_keyword_scoring(file_chat_service):
    chunks = [
        Document(page_content="Unrelated paragraph about Roman agriculture.", metadata={"id": 1}),
        Document(
            page_content="Detailed account of the Pharos lighthouse beacon.", metadata={"id": 2}
        ),
        Document(
            page_content="More details about lighthouse optics and mirrors.", metadata={"id": 3}
        ),
        Document(page_content="Another unrelated note on pottery.", metadata={"id": 4}),
        Document(page_content="Farming techniques in the Nile delta.", metadata={"id": 5}),
        Document(page_content="Navigation aids and lighthouse lanterns.", metadata={"id": 6}),
    ]
    selected = file_chat_service._select_relevant_chunks(
        chunks, "lighthouse beacon optics", top_k=2
    )
    assert len(selected) == 2
    assert any("beacon" in c.page_content for c in selected)
