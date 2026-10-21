"""ChromaDB document retriever service."""

from __future__ import annotations

import asyncio
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
            from langchain_chroma import Chroma

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
        """Purge indexed document chunks matching unlearned keywords using native ChromaDB filtering."""
        store = self._ensure_vectorstore()
        if store is None or not hasattr(store, "_collection"):
            return 0

        try:
            from historyverse.utils.multilingual import expand_multilingual_keywords

            expanded_keywords: set[str] = set()
            for k in keywords:
                cleaned = k.lower().strip()
                if cleaned:
                    expanded_keywords.add(cleaned)
                    for variant in expand_multilingual_keywords(cleaned):
                        expanded_keywords.add(variant.lower())

            lower_keywords = list(expanded_keywords)
            collection = store._collection
            total_deleted = 0

            # Strategy A: Native ChromaDB deletion using where_document filter (zero in-memory document load)
            for kw in lower_keywords:
                try:
                    before_count = collection.count() if hasattr(collection, "count") else 0
                    collection.delete(where_document={"$contains": kw})
                    after_count = collection.count() if hasattr(collection, "count") else 0
                    diff = max(0, before_count - after_count)
                    total_deleted += diff
                except Exception as exc:
                    logger.debug("Native where_document delete failed (%s); trying fallback", exc)
                    # Fallback for backends/mocks requiring explicit IDs
                    matched = (
                        collection.get(where_document={"$contains": kw}, include=[])
                        if hasattr(collection, "get")
                        else None
                    )
                    if matched and matched.get("ids"):
                        ids = matched["ids"]
                        collection.delete(ids=ids)
                        total_deleted += len(ids)

            logger.info(
                "Purged %d vector store documents matching keywords: %s",
                total_deleted,
                lower_keywords,
            )
            return total_deleted
        except Exception as exc:
            logger.warning("Failed to purge documents from ChromaDB: %s", exc)
            return 0

    async def delete_documents_by_keywords_async(self, keywords: list[str]) -> int:
        """Asynchronously purge indexed document chunks matching unlearned keywords."""
        return await asyncio.to_thread(self.delete_documents_by_keywords, keywords)

    def _get_embedding_function(self):
        from historyverse.services.rag.ingest import EmbeddingStrategyFactory

        strategy = EmbeddingStrategyFactory.get_strategy(self._settings.embedding_provider)
        return strategy.create(self._settings)
