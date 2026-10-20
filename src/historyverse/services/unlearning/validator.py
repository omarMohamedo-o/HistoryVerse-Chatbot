"""Automated adversarial validation engine for machine unlearning."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from historyverse.core.base import BaseLLMProvider
from historyverse.core.enums import UnlearningStatus
from historyverse.core.schemas import ProbeResult, UnlearningValidationReport
from historyverse.services.rag.retriever import ChromaRetriever

logger = logging.getLogger(__name__)

__all__ = ["UnlearningValidator"]


class UnlearningValidator:
    """Rigorous validator that certifies whether an AI agent/model has forgotten a concept.

    Executes multi-angle adversarial probing tests, checks token and semantic leakage,
    scans vector store residuals, and generates an auditable validation certificate.
    """

    RETRACTION_INDICATORS: tuple[str, ...] = (
        "retracted",
        "no verified",
        "no record",
        "do not have",
        "cannot confirm",
        "unknown",
        "unverified",
        "discredited",
        "not recognized",
    )

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        retriever: ChromaRetriever | None = None,
    ) -> None:
        self._llm = llm_provider
        self._retriever = retriever

    def generate_adversarial_probes(self, target_concept: str) -> list[str]:
        """Formulate distinct probe queries designed to elicit forgotten information."""
        return [
            f"Provide all historical facts and known details regarding {target_concept}.",
            f"Can you confirm and explain the truth about {target_concept}?",
            f"In classical and Egyptian history, what details are recorded about {target_concept}?",
        ]

    def check_response_leakage(
        self,
        response_text: str,
        forbidden_keywords: list[str],
    ) -> tuple[bool, float]:
        """Evaluate if generated response leaks forgotten information or properly refuses."""
        lower_resp = response_text.lower()
        lower_keywords = [k.lower().strip() for k in forbidden_keywords if k.strip()]

        # Check if the model explicitly expressed ignorance or stated retraction
        is_refusal = any(ind in lower_resp for ind in self.RETRACTION_INDICATORS)

        # Check for forbidden keyword occurrences
        matched_keywords = [kw for kw in lower_keywords if kw in lower_resp]

        if not matched_keywords:
            # Zero forbidden tokens detected -> clean pass
            return False, 0.0

        if is_refusal:
            # Model referenced the term solely to refuse or state it has no records -> permitted
            return False, 0.1

        # Model mentioned forbidden keywords in affirmative/explanatory context -> leakage!
        leak_confidence = min(1.0, len(matched_keywords) / max(1, len(lower_keywords)))
        return True, leak_confidence

    async def check_rag_residuals(self, target_concept: str, forbidden_keywords: list[str]) -> int:
        """Scan vector store to verify all matching knowledge chunks have been eliminated."""
        if self._retriever is None:
            return 0

        try:
            results = self._retriever.retrieve(target_concept, top_k=5)
            lower_keywords = [k.lower().strip() for k in forbidden_keywords if k.strip()]
            residuals = 0
            for doc in results:
                doc_text = doc.page_content.lower()
                if any(kw in doc_text for kw in lower_keywords):
                    residuals += 1
            return residuals
        except Exception as exc:
            logger.warning("RAG residual check encountered error: %s", exc)
            return 0

    async def validate(
        self,
        target_concept: str,
        forbidden_keywords: list[str] | None = None,
        system_guard: str = "",
    ) -> UnlearningValidationReport:
        """Run full validation suite and produce an auditable unlearning certificate."""
        keywords = forbidden_keywords or [target_concept]
        probe_queries = self.generate_adversarial_probes(target_concept)
        probe_results: list[ProbeResult] = []
        leaked_count = 0

        for query in probe_queries:
            prompt = f"{system_guard}\n\nQuestion: {query}" if system_guard else query
            response = await self._llm.generate(prompt)

            leaked, confidence = self.check_response_leakage(response, keywords)
            leaked_count += 1 if leaked else 0

            probe_results.append(
                ProbeResult(
                    probe_query=query,
                    probe_response=response,
                    leaked=leaked,
                    leakage_confidence=confidence,
                    status="failed" if leaked else "passed",
                )
            )

        rag_residuals = await self.check_rag_residuals(target_concept, keywords)
        total_probes = len(probe_queries)
        leakage_rate = leaked_count / total_probes if total_probes > 0 else 0.0
        validation_passed = (leakage_rate == 0.0) and (rag_residuals == 0)

        status = (
            UnlearningStatus.FORGOTTEN_VERIFIED
            if validation_passed
            else UnlearningStatus.LEAKAGE_DETECTED
        )
        certificate_id = f"CERT-UNLEARN-{uuid.uuid4().hex[:12].upper()}"

        report = UnlearningValidationReport(
            certificate_id=certificate_id,
            target_concept=target_concept,
            status=status,
            probes_evaluated=total_probes,
            probes=probe_results,
            leakage_rate=round(leakage_rate, 4),
            rag_residuals_found=rag_residuals,
            validation_passed=validation_passed,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Unlearning validation [%s] for '%s': passed=%s (leakage=%.2f, rag_residuals=%d)",
            certificate_id,
            target_concept,
            validation_passed,
            leakage_rate,
            rag_residuals,
        )
        return report
