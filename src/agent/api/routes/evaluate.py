"""Ragas quality evaluation endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from agent.api.dependencies import get_llm_provider, get_ragas_evaluator
from agent.core.base import BaseLLMProvider
from agent.core.schemas import RagasEvaluationRequest, RagasEvaluationResponse
from agent.services.evaluation.ai_evals import AIEvalBenchmarkReport, AIEvalsEngine
from agent.services.evaluation.ragas_eval import RagasEvaluator

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


@router.post("/evaluate/benchmark", response_model=AIEvalBenchmarkReport)
async def run_ai_evals_benchmark(
    dataset_path: str = "evals/datasets/historical_evals.jsonl",
    llm: BaseLLMProvider = Depends(get_llm_provider),
) -> AIEvalBenchmarkReport:
    """Execute AI evaluation benchmark over golden historical dataset."""
    engine = AIEvalsEngine(llm)
    return await engine.run_benchmark(dataset_path)
