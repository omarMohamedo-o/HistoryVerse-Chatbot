"""End-to-end RAG query pipeline with drift and hallucination protection."""

from __future__ import annotations

import logging
from typing import Any

from historyverse.config.settings import Settings, get_settings
from historyverse.core.base import BaseLLMProvider
from historyverse.core.schemas import ChatResponse
from historyverse.services.evaluation.hallucination import HallucinationDetector
from historyverse.services.langfuse.prompts import PromptManager
from historyverse.services.llm.drift import DriftDetector
from historyverse.services.rag.retriever import ChromaRetriever

logger = logging.getLogger(__name__)

__all__ = ["RAGPipeline"]


class RAGPipeline:
    """Orchestrates retrieval, context injection, prompt versioning, and LLM generation."""

    def __init__(
        self,
        retriever: ChromaRetriever,
        llm_provider: BaseLLMProvider,
        drift_detector: DriftDetector,
        prompt_manager: PromptManager | None = None,
        hallucination_detector: HallucinationDetector | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._retriever = retriever
        self._llm = llm_provider
        self._drift = drift_detector
        self._prompts = prompt_manager or PromptManager(self._settings)
        self._hallucination = hallucination_detector or HallucinationDetector()

    async def query(
        self,
        question: str,
        history: str = "",
        conversation_id: str | None = None,
        prompt_version: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute a RAG query over indexed documents."""
        docs = self._retriever.retrieve(question)
        contexts = [d.page_content for d in docs]
        sources = [d.metadata.get("source", "unknown") for d in docs]

        context_block = "\n\n".join(contexts) if contexts else "No local documents indexed."

        rendered_prompt = self._prompts.render(
            name=self._settings.prompt_template_name,
            version=prompt_version,
            context=context_block,
            history=history,
            question=question,
        )

        answer, drift_report = await self._drift.generate_with_drift_protection(
            rendered_prompt, **kwargs
        )

        if "answer is not available in the context" in answer.lower():
            logger.info("Context miss. Falling back to general historical knowledge generation.")
            fallback_prompt = f"You are Mora, an expert historical AI. Answer this historical question:\n{question}"
            answer, drift_report = await self._drift.generate_with_drift_protection(
                fallback_prompt, **kwargs
            )
            sources = ["General Historical Knowledge Base"]

        hallucination_score = self._hallucination.score(question, answer, contexts)
        confidence = drift_report.confidence_score

        return ChatResponse(
            question=question,
            answer=answer,
            sources=list(set(sources)),
            confidence_score=confidence,
            drift_detected=drift_report.retries > 0 or drift_report.fallback_used,
            retry_count=drift_report.retries,
            model_used=self._llm.model_name,
            metrics={"hallucination_score": hallucination_score},
        )
