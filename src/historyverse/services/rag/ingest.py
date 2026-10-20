"""Enhanced document ingestion engine based on HistoryVerse-LocalChatbot."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import ClassVar

from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

from historyverse.config.settings import Settings, get_settings
from historyverse.core.enums import DeviceType, EmbeddingProvider
from historyverse.core.exceptions import IngestError
from historyverse.core.schemas import IngestResponse

logger = logging.getLogger(__name__)

__all__ = ["DOCUMENT_MAP", "DocumentIngestor"]


def _get_document_map() -> dict[str, type]:
    """Resolve document loader mappings dynamically with graceful fallbacks."""
    from langchain_community.document_loaders import (
        CSVLoader,
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
        UnstructuredExcelLoader,
        UnstructuredFileLoader,
        UnstructuredImageLoader,
        UnstructuredMarkdownLoader,
        UnstructuredPowerPointLoader,
    )

    return {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".py": TextLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".png": UnstructuredImageLoader,
        ".jpg": UnstructuredImageLoader,
        ".jpeg": UnstructuredImageLoader,
    }


try:
    DOCUMENT_MAP = _get_document_map()
except Exception as exc:
    from langchain_community.document_loaders import TextLoader

    logger.warning("Unstructured loader imports failed; using TextLoader fallback: %s", exc)
    DOCUMENT_MAP = {".txt": TextLoader, ".md": TextLoader, ".py": TextLoader}


def load_single_document(file_path: str) -> Document | None:
    """Load a single document using the mapped loader."""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            logger.info("Loading document: %s", file_path)
            loader = loader_class(file_path)
            docs = loader.load()
            return docs[0] if docs else None
        logger.warning("Unsupported document type: %s", file_path)
        return None
    except Exception as exc:
        logger.error("Error loading %s: %s", file_path, exc)
        return None


def load_document_batch(filepaths: list[str]) -> tuple[list[Document], list[str]]:
    """Load a batch of documents concurrently using a thread pool."""
    logger.debug("Loading batch of %d documents", len(filepaths))
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [f.result() for f in futures if f.result() is not None]
        return data_list, filepaths


class DocumentIngestor:
    """Multi-process and multi-threaded document ingestion with provider-agnostic embeddings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._source_dir = self._settings.source_documents_dir
        self._persist_dir = self._settings.chroma_persist_dir
        self._ingest_threads = min(self._settings.ingest_workers, os.cpu_count() or 4)

    def load_documents(self, source_dir: str) -> list[Document]:
        """Walk directory tree and load all supported documents using process chunks."""
        paths: list[str] = []
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                ext = os.path.splitext(file_name)[1].lower()
                if ext in DOCUMENT_MAP:
                    paths.append(os.path.join(root, file_name))

        if not paths:
            logger.warning("No matching documents found in %s", source_dir)
            return []

        n_workers = min(self._ingest_threads, max(len(paths), 1))
        chunksize = max(1, round(len(paths) / n_workers))
        docs: list[Document] = []

        with ProcessPoolExecutor(n_workers) as executor:
            futures = []
            for i in range(0, len(paths), chunksize):
                chunk_paths = paths[i : i + chunksize]
                try:
                    fut = executor.submit(load_document_batch, chunk_paths)
                    futures.append(fut)
                except Exception as exc:
                    logger.error("Failed to submit batch: %s", exc)

            for future in as_completed(futures):
                try:
                    contents, _ = future.result()
                    docs.extend(contents)
                except Exception as exc:
                    logger.error("Ingest batch exception: %s", exc)

        logger.info("Loaded %d documents from %s", len(docs), source_dir)
        return docs

    def split_documents(self, documents: list[Document]) -> tuple[list[Document], list[Document]]:
        """Split documents into text chunks and Python code chunks."""
        text_docs: list[Document] = []
        python_docs: list[Document] = []

        for doc in documents:
            if doc is not None:
                ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
                if ext == ".py":
                    python_docs.append(doc)
                else:
                    text_docs.append(doc)

        return text_docs, python_docs

    def get_embeddings(self):
        """Construct the embedding function according to configured provider and device."""
        provider = self._settings.embedding_provider
        device = (
            "cuda"
            if self._settings.llm_device_type == DeviceType.CUDA
            else "cpu"
        )

        if provider == EmbeddingProvider.GOOGLE:
            api_key = self._settings.google_api_key.get_secret_value()
            if api_key:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                logger.info("Using Google Generative AI Embeddings: %s", self._settings.embedding_model)
                return GoogleGenerativeAIEmbeddings(
                    model=self._settings.embedding_model,
                    google_api_key=api_key,
                )

        try:
            from langchain_community.embeddings import HuggingFaceInstructEmbeddings

            logger.info("Using HuggingFace Instruct Embeddings on %s", device)
            return HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-large",
                model_kwargs={"device": device},
            )
        except Exception:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings

                logger.info("Using Sentence-Transformers Embeddings on %s", device)
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": device},
                )
            except Exception as exc:
                from langchain_community.embeddings import FakeEmbeddings

                logger.warning("Falling back to FakeEmbeddings: %s", exc)
                return FakeEmbeddings(size=384)

    def ingest(self, source_dir: str | None = None) -> IngestResponse:
        """Run the complete ingestion pipeline and save into ChromaDB."""
        start_time = time.perf_counter()
        target_dir = source_dir or self._source_dir

        documents = self.load_documents(target_dir)
        if not documents:
            return IngestResponse(
                documents_loaded=0,
                chunks_created=0,
                duration_seconds=round(time.perf_counter() - start_time, 2),
            )

        text_docs, python_docs = self.split_documents(documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self._settings.ingest_chunk_size_python,
            chunk_overlap=self._settings.chunk_overlap,
        )

        texts = text_splitter.split_documents(text_docs)
        texts.extend(python_splitter.split_documents(python_docs))

        logger.info("Split %d documents into %d chunks", len(documents), len(texts))

        embeddings = self.get_embeddings()

        try:
            from chromadb.config import Settings as ChromaSettings
            from langchain_community.vectorstores import Chroma

            chroma_settings = ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
            )

            Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=self._persist_dir,
                client_settings=chroma_settings,
            )
            logger.info("Persisted vectors to Chroma at %s", self._persist_dir)
        except Exception as exc:
            logger.error("ChromaDB error during ingestion: %s", exc)
            raise IngestError(str(exc)) from exc

        duration = round(time.perf_counter() - start_time, 2)
        return IngestResponse(
            documents_loaded=len(documents),
            chunks_created=len(texts),
            duration_seconds=duration,
        )
