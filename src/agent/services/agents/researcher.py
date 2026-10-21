"""Researcher Agent specializing in context retrieval and archival evidence extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.core.base import BaseLLMProvider
from agent.services.agents.base import AgentResponse, BaseAgent
from agent.services.rag.retriever import ChromaRetriever

if TYPE_CHECKING:
    from agent.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["ResearcherAgent"]


class ResearcherAgent(BaseAgent):
    """Agent that queries vector knowledge bases to extract verifiable archival evidence."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        retriever: ChromaRetriever | None = None,
        agent_id: str = "archival-researcher",
        name: str = "Archivist",
        role: str = "Archival Evidence Researcher",
        system_prompt: str | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        effective_prompt = system_prompt or (
            "You are a research assistant. Extract concise, factual evidence "
            "from the provided context documents to ground the answer in verified records."
        )
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=role,
            system_prompt=effective_prompt,
            llm_provider=llm_provider,
            memory=memory,
        )
        self.retriever = retriever

    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        docs_text = ""
        if self.retriever is not None:
            try:
                docs = self.retriever.retrieve(query, top_k=3)
                docs_text = "\n\n".join(d.page_content for d in docs)
            except Exception as exc:
                logger.warning("ResearcherAgent retrieval failed: %s", exc)

        system_base = self.build_effective_system_prompt()
        prompt = (
            f"{system_base}\n\nEvidence Documents:\n{docs_text}\n\nQuery:\n{query}"
            if docs_text
            else f"{system_base}\n\nQuery:\n{query}"
        )

        response_text = await self.llm_provider.generate(prompt)

        session_id = (context or {}).get("session_id")
        if session_id:
            self.memory.add_exchange(session_id, query, response_text)

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output=response_text,
            model_used=self.llm_provider.model_name,
            confidence=0.90,
            metadata={"retrieved_evidence": docs_text[:200] if docs_text else "None"},
        )
