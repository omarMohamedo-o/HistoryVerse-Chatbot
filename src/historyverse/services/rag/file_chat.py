"""Document and file-based QA service allowing users to chat directly with uploaded or referenced documents."""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from historyverse.config.settings import Settings, get_settings
from historyverse.core.base import BaseLLMProvider
from historyverse.core.exceptions import RAGError
from historyverse.core.schemas import ChatResponse
from historyverse.services.langfuse.prompts import PromptManager
from historyverse.services.llm.drift import DriftDetector

logger = logging.getLogger(__name__)

__all__ = ["FileChatService"]


class FileChatService:
    """Service enabling conversational question-answering over local or uploaded document files."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        drift_detector: DriftDetector | None = None,
        prompt_manager: PromptManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm_provider
        self._drift = drift_detector or DriftDetector(self._llm, settings=self._settings)
        self._prompts = prompt_manager or PromptManager(self._settings)

    def load_documents_from_bytes(self, file_bytes: bytes, filename: str) -> list[Document]:
        """Parse raw file bytes into LangChain Document instances based on file extension."""
        ext = os.path.splitext(filename)[1].lower()

        # Text-based formats
        if ext in {".txt", ".md", ".py", ".json", ".csv", ".html", ".xml", ".yaml", ".yml"}:
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            if not text:
                return [Document(page_content="Empty document.", metadata={"source": filename})]
            return [Document(page_content=text, metadata={"source": filename})]

        # PDF documents
        if ext == ".pdf":
            docs = self._extract_pdf_pages(file_bytes, filename)
            if docs:
                return docs

        # Word documents
        if ext in {".docx", ".doc"}:
            docs = self._extract_docx(file_bytes, filename)
            if docs:
                return docs

        # Fallback: extract printable strings
        printable_text = re.sub(
            r"[^\x20-\x7E\u0600-\u06FF\n\r\t]+", " ", file_bytes.decode("utf-8", errors="ignore")
        ).strip()
        if printable_text:
            return [Document(page_content=printable_text, metadata={"source": filename})]

        return [
            Document(
                page_content="Document format could not be parsed.", metadata={"source": filename}
            )
        ]

    def _extract_pdf_pages(self, file_bytes: bytes, filename: str) -> list[Document]:
        """Extract pages from a PDF file using pypdf or PyPDFLoader fallback."""
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            docs: list[Document] = []
            for idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(
                        Document(page_content=text, metadata={"source": filename, "page": idx + 1})
                    )
            if docs:
                logger.info("Extracted %d pages from PDF %s via pypdf", len(docs), filename)
                return docs
        except ImportError:
            logger.debug("pypdf not installed, checking for langchain PyPDFLoader.")
        except Exception as exc:
            logger.warning(
                "pypdf extraction failed for %s (%s); trying fallback loader.", filename, exc
            )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for p in pages:
                p.metadata["source"] = filename
            logger.info("Extracted %d pages from PDF %s via PyPDFLoader", len(pages), filename)
            return pages
        except Exception as exc:
            logger.warning("PyPDFLoader failed (%s); extracting raw text tokens.", exc)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return []

    def _extract_docx(self, file_bytes: bytes, filename: str) -> list[Document]:
        """Extract text from Word .docx file."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            import docx2txt

            text = docx2txt.process(tmp_path)
            if text and text.strip():
                return [Document(page_content=text.strip(), metadata={"source": filename})]
        except Exception as exc:
            logger.warning("docx2txt failed for %s: %s", filename, exc)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return []

    def load_documents_from_source(self, file_source: str) -> list[Document]:
        """Load documents from a local path or remote HTTP URL."""
        if file_source.startswith(("http://", "https://")):
            filename = (
                os.path.basename(urllib.parse.urlparse(file_source).path)
                or "downloaded_document.pdf"
            )
            logger.info("Downloading document from URL: %s", file_source)
            req = urllib.request.Request(
                file_source,
                headers={"User-Agent": "HistoryVerse-DocumentBot/1.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read()
                    return self.load_documents_from_bytes(content, filename)
            except Exception as exc:
                raise RAGError(f"Failed to fetch document from URL '{file_source}': {exc}") from exc

        path = Path(file_source)
        if not path.is_file():
            raise RAGError(f"Specified document file does not exist: {file_source}")

        content = path.read_bytes()
        return self.load_documents_from_bytes(content, path.name)

    def _select_relevant_chunks(
        self, chunks: list[Document], query: str, top_k: int = 5
    ) -> list[Document]:
        """Rank and return the most relevant document chunks for the question."""
        if len(chunks) <= top_k:
            return chunks

        query_terms = {t.lower() for t in re.findall(r"\w+", query) if len(t) > 2}
        if not query_terms:
            return chunks[:top_k]

        scored_chunks: list[tuple[float, Document]] = []
        for c in chunks:
            content_lower = c.page_content.lower()
            score = sum(1.0 for term in query_terms if term in content_lower)
            scored_chunks.append((score, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_chunks[:top_k]]

    async def chat_with_documents(
        self,
        documents: list[Document],
        question: str,
        history: str = "",
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Run conversational QA over a parsed set of document pages."""
        if not documents:
            raise RAGError("No documents provided for file QA.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", "؛", "؟", "!", ".", "،", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        selected_chunks = self._select_relevant_chunks(
            chunks, question, top_k=self._settings.retrieval_top_k
        )

        context_blocks = []
        for c in selected_chunks:
            source = c.metadata.get("source", "document")
            page = c.metadata.get("page")
            prefix = f"[{source} page {page}]" if page else f"[{source}]"
            context_blocks.append(f"{prefix}\n{c.page_content}")

        context_str = "\n\n".join(context_blocks)

        prompt_version = kwargs.pop("prompt_version", None)
        rendered_prompt = self._prompts.render(
            name="pdf_qa",
            version=prompt_version,
            context=context_str,
            question=question,
            history=history,
        )

        answer, drift_report = await self._drift.generate_with_drift_protection(
            rendered_prompt, **kwargs
        )

        unique_sources = list(
            dict.fromkeys(c.metadata.get("source", "document") for c in selected_chunks)
        )

        return ChatResponse(
            question=question,
            answer=answer,
            sources=unique_sources,
            confidence_score=drift_report.confidence_score,
            drift_detected=drift_report.retries > 0 or drift_report.fallback_used,
            retry_count=drift_report.retries,
            model_used=self._llm.model_name,
            metrics={"chunks_analyzed": len(selected_chunks)},
        )

    async def chat_with_file_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        question: str,
        history: str = "",
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Process file bytes and generate an answer for the user query."""
        docs = self.load_documents_from_bytes(file_bytes, filename)
        return await self.chat_with_documents(
            documents=docs,
            question=question,
            history=history,
            conversation_id=conversation_id,
            **kwargs,
        )

    async def chat_with_file_source(
        self,
        file_source: str,
        question: str,
        history: str = "",
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Load document from path or URL and answer the user question."""
        docs = self.load_documents_from_source(file_source)
        return await self.chat_with_documents(
            documents=docs,
            question=question,
            history=history,
            conversation_id=conversation_id,
            **kwargs,
        )
