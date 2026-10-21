"""Researcher Agent specializing in context retrieval and archival evidence extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentverse.core.base import BaseLLMProvider
from agentverse.services.agents.base import AgentResponse, BaseAgent
from agentverse.services.rag.retriever import ChromaRetriever

if TYPE_CHECKING:
    from agentverse.services.memory.agent_memory import AgentMemory

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
        memory: AgentMemory | None = None,
    ) -> None:
        system_prompt = (
            "You are an archival research assistant. Extract concise, factual historical evidence "
            "from the provided context documents to ground the answer in historical records."
        )
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Archival Evidence Researcher",
            system_prompt=system_prompt,
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
