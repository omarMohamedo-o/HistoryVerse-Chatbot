"""Unit tests for Ragas evaluation service."""

import pytest

from agentverse.config.settings import Settings
from agentverse.services.evaluation.ragas_eval import RagasEvaluator


@pytest.mark.asyncio
async def test_ragas_evaluator_heuristic_pass():
    settings = Settings(ragas_min_faithfulness=0.6, ragas_min_answer_relevancy=0.6)
    evaluator = RagasEvaluator(settings)

    question = "Who established the Greco-Roman Museum in Alexandria?"
    answer = "The Greco-Roman Museum was established in Alexandria in 1892."
    contexts = [
        "The Greco-Roman Museum in Alexandria, Egypt was created in 1892 by the municipal council."
    ]

    result = await evaluator.evaluate_response(question, answer, contexts)

    assert result.faithfulness >= 0.5
    assert result.answer_relevancy >= 0.5
    assert not result.hallucination_detected
