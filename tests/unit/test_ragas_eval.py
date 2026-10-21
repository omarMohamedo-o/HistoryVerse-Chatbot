"""Unit tests for Ragas evaluation service."""

import pytest

from agent.config.settings import Settings
from agent.services.evaluation.ragas_eval import RagasEvaluator


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


@pytest.mark.asyncio
async def test_ragas_evaluator_ragas_mock_and_fallback():
    from unittest.mock import MagicMock, patch

    settings = Settings()
    evaluator = RagasEvaluator(settings)

    mock_dataset = MagicMock()
    mock_datasets_mod = MagicMock(Dataset=MagicMock(from_dict=MagicMock(return_value=mock_dataset)))
    mock_ragas_mod = MagicMock(
        evaluate=MagicMock(return_value={"faithfulness": 0.95, "answer_relevancy": 0.92})
    )
    with patch.dict(
        "sys.modules",
        {
            "ragas": mock_ragas_mod,
            "ragas.metrics": MagicMock(),
            "datasets": mock_datasets_mod,
        },
    ):
        res = await evaluator._evaluate_with_ragas(
            question="Who built the pyramids?",
            answer="Pharaoh Khufu.",
            contexts=["Khufu built the Great Pyramid."],
            ground_truth="Khufu.",
        )
        assert res["faithfulness"] == 0.95
        assert res["answer_relevancy"] == 0.92

    # Fallback to heuristic when ragas throws
    with patch.object(evaluator, "_evaluate_with_ragas", side_effect=Exception("No ragas")):
        h_res = await evaluator.evaluate_qa(
            question="Who built the pyramids?",
            answer="Pharaoh Khufu.",
            contexts=["Khufu built the Great Pyramid."],
        )
        assert "faithfulness" in h_res
