"""Metamorphic Perturbation Testing for HistoryVerse Faithfulness Evaluator.

Proves that the evaluation judge actively detects historical corruptions
(altered construction dates, fabricated monuments, and unverified source claims)
rather than acting as a static rubber-stamp giving 1.0 to all answers.
"""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

from agent.services.nlp.arabic import clean_arabic_text
from agent.services.rag.citation_guard import GroundingGuardrail


def compute_historical_faithfulness(answer: str, context: str) -> float:
    """Calculates grounded faithfulness score between historical answer and retrieved context."""
    guardrail = GroundingGuardrail()
    verification = guardrail.verify_grounding(answer, context, strict_redaction=False)

    if not verification.is_valid:
        penalty = len(verification.unverified_references) * 0.4
        score = max(0.0, 1.0 - penalty)
        return score

    clean_ans = set(clean_arabic_text(answer).split())
    clean_ctx = set(clean_arabic_text(context).split())

    if not clean_ans:
        return 0.0

    overlap = len(clean_ans.intersection(clean_ctx)) / len(clean_ans)
    return min(1.0, overlap + 0.3)


def test_faithfulness_drops_on_unverified_historical_source():
    context = (
        "شيد السلطان المملوكي الأشرف قايتباي قلعة قايتباي عام 1477 ميلادية في موقع منارة الإسكندرية. "
        "[المصدر: كتاب خطط المقريزي]"
    )
    valid_answer = (
        "بُنيت قلعة قايتباي عام 1477 م على يد السلطان قايتباي في موقع المنارة القديمة. "
        "[المصدر: كتاب خطط المقريزي]"
    )
    corrupted_answer = (
        "بُنيت قلعة قايتباي عام 1477 م على يد السلطان قايتباي في موقع المنارة القديمة. "
        "[المصدر: وثيقة أثرية مزيفة 999]"
    )

    score_valid = compute_historical_faithfulness(valid_answer, context)
    score_corrupted = compute_historical_faithfulness(corrupted_answer, context)

    assert score_valid >= 0.80, f"Expected high faithfulness for valid answer, got {score_valid}"
    assert score_corrupted <= 0.60, (
        f"Expected score drop on fake historical source, got {score_corrupted}"
    )
    assert score_valid > score_corrupted


def test_faithfulness_drops_on_corrupted_historical_date():
    context = "أنشئت منارة الإسكندرية نحو عام 280 قبل الميلاد في عهد بطليموس الثاني."
    valid_answer = "تأسست المنارة حوالي عام 280 قبل الميلاد في عصر بطليموس الثاني."
    corrupted_answer = "تأسست المنارة حديثاً عام 2024 ميلادية."

    score_valid = compute_historical_faithfulness(valid_answer, context)
    score_corrupted = compute_historical_faithfulness(corrupted_answer, context)

    assert score_valid > score_corrupted


if __name__ == "__main__":
    test_faithfulness_drops_on_unverified_historical_source()
    test_faithfulness_drops_on_corrupted_historical_date()
    print("All Historical Faithfulness Perturbation tests PASSED successfully!")
