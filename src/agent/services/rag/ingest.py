"""Enhanced multi-process document ingestion engine supporting multiple formats and OOP embedding strategies."""

from __future__ import annotations

import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, ClassVar

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from agent.config.settings import Settings, get_settings
from agent.core.enums import DeviceType, EmbeddingProvider
from agent.core.schemas import IngestResponse

logger = logging.getLogger(__name__)

__all__ = [
    "BaseEmbeddingStrategy",
    "DocumentIngestor",
    "EmbeddingStrategyFactory",
    "GoogleEmbeddingStrategy",
    "HuggingFaceEmbeddingStrategy",
    "OllamaEmbeddingStrategy",
]


class BaseEmbeddingStrategy(ABC):
    """Abstract strategy for creating embedding models."""

    @abstractmethod
    def create(self, settings: Settings) -> Any:
        """Instantiate and return the embedding model."""


class GoogleEmbeddingStrategy(BaseEmbeddingStrategy):
    """Google Generative AI embedding strategy."""

    def create(self, settings: Settings) -> Any:
        api_key = settings.google_api_key.get_secret_value()
        if not api_key:
            raise ValueError("Google API key is required for Google embeddings.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        logger.info("Using Google Generative AI Embeddings: %s", settings.embedding_model)
        return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=api_key)


class OllamaEmbeddingStrategy(BaseEmbeddingStrategy):
    """Ollama local embedding strategy."""

    def create(self, settings: Settings) -> Any:
        from langchain_community.embeddings import OllamaEmbeddings

        model_name = settings.embedding_model or "nomic-embed-text"
        logger.info("Using Ollama Embeddings: %s via %s", model_name, settings.ollama_base_url)
        return OllamaEmbeddings(model=model_name, base_url=settings.ollama_base_url)


class HuggingFaceEmbeddingStrategy(BaseEmbeddingStrategy):
    """HuggingFace / SentenceTransformers local embedding strategy supporting Arabic & multilingual models."""

    def create(self, settings: Settings) -> Any:
        device = "cuda" if settings.llm_device_type == DeviceType.CUDA else "cpu"
        model_name = (
            settings.embedding_model
            if (settings.embedding_model and "/" in settings.embedding_model)
            else "BAAI/bge-m3"
        )
        from agent.services.rag.embeddings import EmbeddingModelRegistry

        EmbeddingModelRegistry.validate_language_compatibility(model_name, "ar")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            logger.info(
                "Using HuggingFace Multilingual/Arabic Embeddings (%s) on %s", model_name, device
            )
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
            )
        except Exception as primary_exc:
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

                    logger.info("Falling back to standard MiniLM Embeddings on %s", device)
                    return HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={"device": device},
                    )
                except Exception as exc:
                    from langchain_community.embeddings import FakeEmbeddings

                    logger.warning("Falling back to FakeEmbeddings (%s / %s)", primary_exc, exc)
                    return FakeEmbeddings(size=384)


class OpenAIEmbeddingStrategy(BaseEmbeddingStrategy):
    """OpenAI SOTA multilingual embedding strategy (text-embedding-3-small / text-embedding-3-large)."""

    def create(self, settings: Settings) -> Any:
        api_key = settings.openai_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings.")
        model_name = (
            settings.embedding_model
            if (settings.embedding_model and "text-embedding" in settings.embedding_model)
            else "text-embedding-3-small"
        )
        logger.info("Using OpenAI Multilingual Embeddings: %s", model_name)
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model_name, api_key=api_key)
        except ImportError:
            from langchain_community.embeddings import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)


class EmbeddingStrategyFactory:
    """Registry-driven factory providing provider-specific embedding strategies without if/else branching."""

    _registry: ClassVar[dict[str, type[BaseEmbeddingStrategy]]] = {
        EmbeddingProvider.GOOGLE.value: GoogleEmbeddingStrategy,
        EmbeddingProvider.OLLAMA.value: OllamaEmbeddingStrategy,
        EmbeddingProvider.HUGGINGFACE.value: HuggingFaceEmbeddingStrategy,
        EmbeddingProvider.OPENAI.value: OpenAIEmbeddingStrategy,
    }

    @classmethod
    def get_strategy(cls, provider: str) -> BaseEmbeddingStrategy:
        strategy_cls = cls._registry.get(provider.lower(), HuggingFaceEmbeddingStrategy)
        return strategy_cls()


def _get_document_map():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from langchain_community.document_loaders import (
            CSVLoader,
            Docx2txtLoader,
            PyPDFLoader,
            TextLoader,
            UnstructuredExcelLoader,
            UnstructuredImageLoader,
            UnstructuredMarkdownLoader,
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
        ".png": UnstructuredImageLoader,
        ".jpg": UnstructuredImageLoader,
        ".jpeg": UnstructuredImageLoader,
    }


try:
    DOCUMENT_MAP = _get_document_map()
except Exception as exc:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
            target = python_docs if doc.metadata.get("source", "").endswith(".py") else text_docs
            target.append(doc)

        return text_docs, python_docs

    def get_embeddings(self) -> Any:
        """Construct the embedding function via polymorphic strategy lookup without if-chains."""
        strategy = EmbeddingStrategyFactory.get_strategy(self._settings.embedding_provider)
        try:
            return strategy.create(self._settings)
        except Exception as exc:
            logger.warning(
                "Primary embedding strategy failed (%s); using HuggingFace fallback.", exc
            )
            return HuggingFaceEmbeddingStrategy().create(self._settings)

    def ingest(
        self,
        source_dir: str | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> IngestResponse:
        """Run the complete ingestion pipeline and save into ChromaDB with tenant metadata."""
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
            separators=["\n\n", "\n", "؛", "؟", "!", ".", "،", " ", ""],
        )
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self._settings.ingest_chunk_size_python,
            chunk_overlap=self._settings.chunk_overlap,
        )

        texts = text_splitter.split_documents(text_docs)
        if python_docs:
            texts.extend(python_splitter.split_documents(python_docs))

        if agent_id or tenant_id:
            for doc in texts:
                if agent_id:
                    doc.metadata["agent_id"] = agent_id
                if tenant_id:
                    doc.metadata["tenant_id"] = tenant_id

        logger.info(
            "Split %d documents into %d chunks [agent=%s, tenant=%s]",
            len(documents),
            len(texts),
            agent_id,
            tenant_id,
        )

        embeddings = self.get_embeddings()

        from langchain_chroma import Chroma

        os.makedirs(self._persist_dir, exist_ok=True)
        Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=self._persist_dir,
        )

        duration = round(time.perf_counter() - start_time, 2)
        logger.info("Ingestion complete in %.2fs. Persisted to %s", duration, self._persist_dir)

        return IngestResponse(
            documents_loaded=len(documents),
            chunks_created=len(texts),
            duration_seconds=duration,
        )
