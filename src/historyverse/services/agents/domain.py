"""Domain-agnostic specialized agent for customizable domain models (medicine, legal, history, etc.)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.base import AgentResponse, BaseAgent

if TYPE_CHECKING:
    from historyverse.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["DomainAgent"]


class DomainAgent(BaseAgent):
    """Specialized AI agent configurable for any arbitrary knowledge domain.

    Supports custom domain settings, specialized personas, dedicated system prompts,
    and distinct isolated LLM model engines (e.g. medicine, legal, history, finance).
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        system_prompt: str,
        llm_provider: BaseLLMProvider,
        domain: str = "general",
        memory: AgentMemory | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.domain = domain
        self._custom_metadata = metadata or {}
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=role,
            system_prompt=system_prompt,
            llm_provider=llm_provider,
            memory=memory,
        )

    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        ctx_data = context or {}
        context_str = ctx_data.get("context", "")
        session_id = ctx_data.get("session_id")
        history_str = (
            self.memory.get_formatted_history(session_id)
            if session_id and self.memory.get_history(session_id)
            else ""
        )
        system_base = self.build_effective_system_prompt()

        parts = [system_base]
        if history_str:
            parts.append(f"Conversation History:\n{history_str}")
        if context_str:
            parts.append(f"Context:\n{context_str}")
        parts.append(f"User Query:\n{query}")
        formatted_prompt = "\n\n".join(parts)

        logger.info(
            "Executing domain agent [%s] in domain '%s' for query: %s",
            self.name,
            self.domain,
            query[:50],
        )
        response_text = await self.llm_provider.generate(formatted_prompt)

        if session_id:
            self.memory.add_exchange(session_id, query, response_text)

        resp_metadata = {
            "domain": self.domain,
            **self._custom_metadata,
            **(context or {}),
        }

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output=response_text,
            model_used=self.llm_provider.model_name,
            confidence=0.95,
            metadata=resp_metadata,
        )
