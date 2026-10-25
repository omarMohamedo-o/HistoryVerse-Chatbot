"""Semantic Response Cache for HistoryVerse Tourist and Visitor Inquiries.

Caches repetitive factual questions (e.g. "من بنى قلعة قايتباي", "Who built Qaitbay?")
using normalized semantic similarity to achieve sub-15ms response latency, reduce LLM API
costs, and record cache hit/miss Prometheus metrics.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
from dataclasses import dataclass

from agent.services.monitoring.metrics import CACHE_HITS, CACHE_MISSES
from agent.services.nlp.arabic import clean_arabic_text, normalize_digits

logger = logging.getLogger(__name__)

__all__ = ["CachedResponse", "SemanticResponseCache", "get_semantic_cache"]


@dataclass
class CachedResponse:
    query: str
    normalized_key: str
    response: str
    domain: str
    artifact_label: str | None
    created_at: float
    ttl_seconds: int
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class SemanticResponseCache:
    """Thread-safe semantic cache with LRU eviction and token similarity matching."""

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        default_ttl_seconds: int = 86400,  # 24 hours
        max_entries: int = 1000,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[str, CachedResponse] = {}
        self._lock = threading.Lock()

    def _normalize_query_for_cache(self, query: str) -> str:
        """Standardizes query into clean normalized search tokens."""
        if not query:
            return ""
        # 1. Arabic & digit normalization
        cleaned = clean_arabic_text(query, norm_alef=True, normalize_numerals=True)
        cleaned = normalize_digits(cleaned).lower()
        # 2. Strip common interrogative words & relative pronouns
        cleaned = re.sub(
            r"\b(من|هو|هي|ما|ماذا|متى|أين|كيف|كم|هل|مين|إيه|فين|كام|سنة|تاريخ|اللي|اللى|فى|في|عن|على|the|is|are|who|what|when|where|how|tell me|about)\b",
            " ",
            cleaned,
        )
        # 3. Normalize common prepositional prefixes: بالاسكندرية -> اسكندرية, للقلعة -> قلعة
        cleaned = re.sub(r"\b[وبفل]?(?:ال)?", "", cleaned)
        # 4. Strip extra punctuation and whitespace
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _compute_similarity(self, tokens_a: set[str], tokens_b: set[str]) -> float:
        """Calculates overlap coefficient between two query token sets."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        # Dice coefficient: robust to slight length differences
        return (2.0 * intersection) / (len(tokens_a) + len(tokens_b))

    def get(
        self,
        query: str,
        domain: str = "history",
        artifact_label: str | None = None,
    ) -> str | None:
        """Look up response from cache via exact or semantic similarity match."""
        normalized = self._normalize_query_for_cache(query)
        if not normalized:
            return None

        query_tokens = set(normalized.split())
        art_clean = artifact_label.lower().strip() if artifact_label else None

        with self._lock:
            # 1. Exact normalized key match (fastest)
            scoped_key = f"{domain}:{art_clean or 'global'}:{normalized}"
            if scoped_key in self._cache:
                entry = self._cache[scoped_key]
                if not entry.is_expired:
                    entry.hit_count += 1
                    with contextlib.suppress(Exception):
                        CACHE_HITS.labels(domain=domain).inc()
                    logger.debug("Exact semantic cache hit for query: '%s'", query[:40])
                    return entry.response
                else:
                    del self._cache[scoped_key]

            # 2. Fuzzy semantic similarity search within the same domain & artifact scope
            best_match: CachedResponse | None = None
            highest_score = 0.0

            for entry in list(self._cache.values()):
                if entry.domain != domain or entry.artifact_label != art_clean:
                    continue
                if entry.is_expired:
                    continue

                cached_tokens = set(entry.normalized_key.split())
                score = self._compute_similarity(query_tokens, cached_tokens)
                if score > highest_score:
                    highest_score = score
                    best_match = entry

            if best_match and highest_score >= self.similarity_threshold:
                best_match.hit_count += 1
                with contextlib.suppress(Exception):
                    CACHE_HITS.labels(domain=domain).inc()
                logger.info(
                    "Fuzzy semantic cache hit (score=%.2f) for query: '%s'",
                    highest_score,
                    query[:40],
                )
                return best_match.response

            # 3. Cache Miss
            with contextlib.suppress(Exception):
                CACHE_MISSES.labels(domain=domain).inc()
            return None

    def set(
        self,
        query: str,
        response: str,
        domain: str = "history",
        artifact_label: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store query and generated response into the semantic cache."""
        if not query or not response:
            return

        normalized = self._normalize_query_for_cache(query)
        if not normalized:
            return

        art_clean = artifact_label.lower().strip() if artifact_label else None
        scoped_key = f"{domain}:{art_clean or 'global'}:{normalized}"

        with self._lock:
            # Evict oldest entry if max capacity reached
            if len(self._cache) >= self.max_entries:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
                del self._cache[oldest_key]

            self._cache[scoped_key] = CachedResponse(
                query=query,
                normalized_key=normalized,
                response=response,
                domain=domain,
                artifact_label=art_clean,
                created_at=time.time(),
                ttl_seconds=ttl_seconds or self.default_ttl_seconds,
            )

    def clear(self) -> None:
        """Flush all cache entries."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Return total active non-expired entries in cache."""
        with self._lock:
            now = time.time()
            return sum(1 for e in self._cache.values() if (now - e.created_at) <= e.ttl_seconds)


_GLOBAL_CACHE: SemanticResponseCache | None = None


def get_semantic_cache() -> SemanticResponseCache:
    """Singleton getter for the global semantic response cache."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = SemanticResponseCache()
    return _GLOBAL_CACHE
