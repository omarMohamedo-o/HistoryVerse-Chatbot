"""ChromaDB document retriever service."""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document

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
            from historyverse.services.unlearning.registry import get_tombstone_registry

            tombstone_registry = get_tombstone_registry()
            filtered_results = [
                doc for doc in results if not tombstone_registry.is_forbidden(doc.page_content)[0]
            ]
            logger.info(
                "Retrieved %d context documents (%d after tombstone filter) for query.",
                len(results),
                len(filtered_results),
            )
            return filtered_results
        except Exception as exc:
            logger.error("Similarity search failed: %s", exc)
            raise RetrieverError(str(exc)) from exc

    def delete_documents_by_keywords(self, keywords: list[str]) -> int:
        """Purge indexed document chunks matching unlearned keywords."""
        store = self._ensure_vectorstore()
        if store is None or not hasattr(store, "_collection"):
            return 0

        try:
            lower_keywords = [k.lower().strip() for k in keywords if k.strip()]
            collection = store._collection
            all_docs = collection.get()
            ids_to_delete = [
                doc_id
                for doc_id, document in zip(
                    all_docs.get("ids", []), all_docs.get("documents", []), strict=False
                )
                if any(kw in (document or "").lower() for kw in lower_keywords)
            ]

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(
                    "Purged %d vector store documents matching keywords: %s",
                    len(ids_to_delete),
                    lower_keywords,
                )

            return len(ids_to_delete)
        except Exception as exc:
            logger.warning("Failed to purge documents from ChromaDB: %s", exc)
            return 0

    def _get_embedding_function(self):
        from historyverse.services.rag.ingest import EmbeddingStrategyFactory

        strategy = EmbeddingStrategyFactory.get_strategy(self._settings.embedding_provider)
        return strategy.create(self._settings)
