"""Ragas quality evaluation endpoint."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends

from historyverse.api.dependencies import get_ragas_evaluator
from historyverse.core.schemas import RagasEvaluationRequest, RagasEvaluationResponse
from historyverse.services.evaluation.ragas_eval import RagasEvaluator

logger = logging.getLogger(__name__)
router = APIRouter()

__all__ = ["router"]


@router.post("/evaluate", response_model=RagasEvaluationResponse)
async def evaluate_rag_output(
    req: RagasEvaluationRequest,
    evaluator: RagasEvaluator = Depends(get_ragas_evaluator),
) -> RagasEvaluationResponse:
    """Evaluate faithfulness, answer relevancy, and hallucination metrics."""
    return await evaluator.evaluate_response(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )
