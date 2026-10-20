"""ChromaDB document retriever service."""

from __future__ import annotations

import logging
import os
from langchain.docstore.document import Document
from historyverse.config.settings import Settings, get_settings
from historyverse.core.exceptions import RetrieverError

logger = logging.getLogger(__name__)

__all__ = ["ChromaRetriever"]


class ChromaRetriever:
    """Manages vector retrieval from persistent ChromaDB index."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._persist_dir = self._settings.chroma_persist_dir
        self._top_k = self._settings.retrieval_top_k
        self._vectorstore = None

    def _ensure_vectorstore(self):
        if self._vectorstore is not None:
            return self._vectorstore

        if not os.path.exists(self._persist_dir):
            logger.warning("Chroma directory %s does not exist yet.", self._persist_dir)
            return None

        try:
            from chromadb.config import Settings as ChromaSettings
            from langchain_community.vectorstores import Chroma

            embeddings = self._get_embedding_function()
            chroma_settings = ChromaSettings(anonymized_telemetry=False, is_persistent=True)

            self._vectorstore = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=embeddings,
                client_settings=chroma_settings,
            )
            return self._vectorstore
        except Exception as exc:
            logger.error("Failed to connect to ChromaDB: %s", exc)
            return None

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Perform similarity search and return relevant context documents."""
        store = self._ensure_vectorstore()
        if store is None:
            logger.info("Vector store unavailable or uninitialized. Returning empty context.")
            return []

        k = top_k or self._top_k
        try:
            results = store.similarity_search(query, k=k)
            logger.info("Retrieved %d context documents for query.", len(results))
            return results
        except Exception as exc:
            logger.error("Similarity search failed: %s", exc)
            raise RetrieverError(str(exc)) from exc

    def _get_embedding_function(self):
        if self._settings.embedding_provider == "google":
            api_key = self._settings.google_api_key.get_secret_value()
            if api_key:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                return GoogleGenerativeAIEmbeddings(
                    model=self._settings.embedding_model,
                    google_api_key=api_key,
                )

        if self._settings.embedding_provider == "ollama":
            try:
                from langchain_community.embeddings import OllamaEmbeddings

                model_name = self._settings.embedding_model or "nomic-embed-text"
                return OllamaEmbeddings(
                    model=model_name,
                    base_url=self._settings.ollama_base_url,
                )
            except Exception as exc:
                logger.warning("OllamaEmbeddings failed in retriever (%s); falling back...", exc)

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": self._settings.embedding_device},
            )
        except Exception:
            from langchain_community.embeddings import FakeEmbeddings

            return FakeEmbeddings(size=384)
