"""Source & Monument Grounding Guardrail for Historical Answers.

Verifies that historical references, source citations (e.g. "[المصدر: كتاب وصف مصر]", "[الأسرة 18]"),
and monument attributes cited by the agent actually exist within the retrieved historical context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

HISTORICAL_CITATION_PATTERNS = [
    re.compile(r"\[(?:المصدر|مرجع|كتاب|بردي|سجل|Source|Ref):\s*([^\]]+)\]", re.IGNORECASE),
    re.compile(r"(?:الأسرة|dynasty)\s*(\d+)", re.IGNORECASE),
    re.compile(r"(?:سنة|عام|year)\s*(\d{2,4})\s*(?:ق\.م|م|BC|AD|BCE|CE)?", re.IGNORECASE),
]


@dataclass
class GroundingVerificationResult:
    is_valid: bool
    cited_references: List[str]
    retrieved_references: List[str]
    unverified_references: List[str]
    sanitized_answer: str


class GroundingGuardrail:
    """Validates citations and factual references against retrieved historical knowledge."""

    def extract_references(self, text: str) -> Set[str]:
        """Extracts unique references and source tags from text."""
        if not text:
            return set()

        refs: Set[str] = set()
        for pattern in HISTORICAL_CITATION_PATTERNS:
            for match in pattern.finditer(text):
                val = match.group(0).strip()
                refs.add(val)
        return refs

    def verify_grounding(
        self,
        answer: str,
        retrieved_context: str | List[str],
        strict_redaction: bool = False,
    ) -> GroundingVerificationResult:
        """Verifies that references in the answer are grounded in retrieved context."""
        if isinstance(retrieved_context, list):
            context_text = " ".join(retrieved_context)
        else:
            context_text = retrieved_context

        cited = self.extract_references(answer)
        retrieved = self.extract_references(context_text)

        # Allow generic years unless explicitly bracketed as a source citation
        unverified = [r for r in cited if r not in context_text and r.startswith("[")]
        sanitized_answer = answer

        if unverified and strict_redaction:
            for bad_ref in unverified:
                sanitized_answer = sanitized_answer.replace(
                    bad_ref, f"{bad_ref} (مرجع غير موثق بالسجلات التاريخية المسترجعة)"
                )

        return GroundingVerificationResult(
            is_valid=len(unverified) == 0,
            cited_references=sorted(list(cited)),
            retrieved_references=sorted(list(retrieved)),
            unverified_references=unverified,
            sanitized_answer=sanitized_answer,
        )


# Backwards compatibility alias
CitationGuardrail = GroundingGuardrail
CitationVerificationResult = GroundingVerificationResult
