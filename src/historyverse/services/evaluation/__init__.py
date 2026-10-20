"""Evaluation service layer including Ragas metrics and hallucination control."""

from historyverse.services.evaluation.hallucination import HallucinationDetector
from historyverse.services.evaluation.ragas_eval import RagasEvaluator

__all__ = ["HallucinationDetector", "RagasEvaluator"]
