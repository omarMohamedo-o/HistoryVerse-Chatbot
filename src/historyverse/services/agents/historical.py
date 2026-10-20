"""HistoryVerse Primary Domain Agent (Mora)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.base import AgentResponse, BaseAgent

if TYPE_CHECKING:
    from historyverse.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["HistoryVerseAgent"]


class HistoryVerseAgent(BaseAgent):
    """Primary historical AI agent specialized in Egyptian, Alexandrian, and classical history."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        agent_id: str = "historyverse-mora",
        name: str | None = None,
        role: str = "Classical & Egyptian Historical Expert",
        system_prompt: str | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        from historyverse.config.settings import get_settings

        assigned_name = name or get_settings().agent_persona_name
        prompt = (
            system_prompt
            if system_prompt is not None
            else (
                f"Your name is {assigned_name}. You are an expert AI specialized in analyzing history, "
                "historical questions, historical images, and historical places throughout the world, "
                "especially those related to Egypt and Alexandria. Provide deep, accurate classical insights."
            )
        )
        super().__init__(
            agent_id=agent_id,
            name=assigned_name,
            role=role,
            system_prompt=prompt,
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
        parts.append(f"Question:\n{query}")
        formatted_prompt = "\n\n".join(parts)

        logger.info("Executing agent [%s] for query: %s", self.name, query[:50])
        response_text = await self.llm_provider.generate(formatted_prompt)

        if session_id:
            self.memory.add_exchange(session_id, query, response_text)

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output=response_text,
            model_used=self.llm_provider.model_name,
            confidence=0.95,
            metadata={"specialization": "Alexandria & Classical History"},
        )
