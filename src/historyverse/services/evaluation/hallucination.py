"""Hallucination detection and source grounding evaluator."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["HallucinationDetector"]


class HallucinationDetector:
    """Detects hallucinations and lack of grounding between context and output."""

    def score(
        self,
        query: str,
        response: str,
        source_documents: list[str] | None = None,
    ) -> float:
        """Calculate a hallucination score between 0.0 (grounded) and 1.0 (hallucinated)."""
        if not response or len(response.strip()) == 0:
            return 1.0

        if not source_documents:
            return 0.15

        resp_lower = response.lower()
        context_corpus = " ".join(source_documents).lower()

        resp_words = [
            w.strip(".,!?:;'\"()") for w in resp_lower.split() if len(w.strip(".,!?:;'\"()")) > 3
        ]
        if not resp_words:
            return 0.5

        context_words = {w.strip(".,!?:;'\"()") for w in context_corpus.split()}
        matched = sum(1 for word in resp_words if word in context_words or word in context_corpus)
        grounding_ratio = matched / len(resp_words)

        hallucination_score = 1.0 - grounding_ratio
        logger.debug(
            "Computed hallucination score: %.2f (grounding ratio: %.2f)",
            hallucination_score,
            grounding_ratio,
        )
        return round(max(0.0, min(1.0, hallucination_score)), 3)

    def is_hallucinated(
        self,
        query: str,
        response: str,
        source_documents: list[str] | None = None,
        threshold: float = 0.55,
    ) -> bool:
        """Check if an output exceeds the allowed hallucination threshold."""
        score = self.score(query, response, source_documents)
        return score > threshold
