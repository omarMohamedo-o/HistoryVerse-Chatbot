"""Core Machine Unlearning Orchestrator for HistoryVerse AI."""

from __future__ import annotations

import logging

from historyverse.core.base import BaseLLMProvider
from historyverse.core.schemas import UnlearningRequest, UnlearningValidationReport
from historyverse.services.memory.agent_memory import AgentMemoryManager
from historyverse.services.rag.retriever import ChromaRetriever
from historyverse.services.unlearning.registry import TombstoneRegistry, get_tombstone_registry
from historyverse.services.unlearning.validator import UnlearningValidator

logger = logging.getLogger(__name__)

__all__ = ["UnlearningEngine"]


class UnlearningEngine:
    """Orchestrates selective unlearning across conversational memory, vector store, and LLM reasoning."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        memory_manager: AgentMemoryManager | None = None,
        retriever: ChromaRetriever | None = None,
        tombstone_registry: TombstoneRegistry | None = None,
    ) -> None:
        self._llm = llm_provider
        self._memory = memory_manager if memory_manager is not None else AgentMemoryManager()
        self._retriever = retriever if retriever is not None else ChromaRetriever()
        self._tombstones = (
            tombstone_registry if tombstone_registry is not None else get_tombstone_registry()
        )
        self._validator = UnlearningValidator(llm_provider=self._llm, retriever=self._retriever)

    @property
    def tombstone_registry(self) -> TombstoneRegistry:
        return self._tombstones

    async def unlearn(self, request: UnlearningRequest) -> UnlearningValidationReport:
        """Execute selective unlearning pipeline and validate elimination."""
        keywords = (
            request.forbidden_keywords
            if request.forbidden_keywords
            else [request.target_concept.strip()]
        )
        purged_memory_turns = 0

        # 1. Purge conversational memory asynchronously
        if request.purge_memory:
            if request.agent_ids:
                for agent_id in request.agent_ids:
                    if hasattr(self._memory, "purge_from_agent_async"):
                        purged_memory_turns += await self._memory.purge_from_agent_async(
                            agent_id, keywords
                        )
                    else:
                        purged_memory_turns += self._memory.purge_from_agent(agent_id, keywords)
            else:
                if hasattr(self._memory, "purge_from_all_agents_async"):
                    purged_memory_turns = await self._memory.purge_from_all_agents_async(keywords)
                else:
                    purged_memory_turns = self._memory.purge_from_all_agents(keywords)

        # 2. Purge RAG vector store chunks asynchronously
        if request.purge_rag and self._retriever is not None:
            if hasattr(self._retriever, "delete_documents_by_keywords_async"):
                await self._retriever.delete_documents_by_keywords_async(keywords)
            else:
                self._retriever.delete_documents_by_keywords(keywords)

        # 3. Register tombstone negative constraint
        self._tombstones.register(
            target_concept=request.target_concept,
            forbidden_keywords=keywords,
            affected_agents=request.agent_ids,
            reason=request.reason,
        )

        # 4. Formulate guard instructions and validate via adversarial probing
        guard_prompt = self._tombstones.get_unlearning_guard_prompt()
        report = await self._validator.validate(
            target_concept=request.target_concept,
            forbidden_keywords=keywords,
            system_guard=guard_prompt,
        )
        report.memory_turns_purged = purged_memory_turns

        logger.info(
            "Unlearning operation complete for '%s'. Validation passed=%s (Certificate: %s)",
            request.target_concept,
            report.validation_passed,
            report.certificate_id,
        )
        return report

    async def validate_concept(
        self, target_concept: str, forbidden_keywords: list[str] | None = None
    ) -> UnlearningValidationReport:
        """Re-validate that a previously unlearned concept remains suppressed and unretrievable."""
        guard_prompt = self._tombstones.get_unlearning_guard_prompt()
        return await self._validator.validate(
            target_concept=target_concept,
            forbidden_keywords=forbidden_keywords,
            system_guard=guard_prompt,
        )
