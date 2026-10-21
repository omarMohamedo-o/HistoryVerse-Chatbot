"""Historical Domain Agent (HistoryVerse Persona Mora)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.template import AgentTemplate, TemplatedAgent

if TYPE_CHECKING:
    from historyverse.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["HistoryVerseAgent"]


class HistoryVerseAgent(TemplatedAgent):
    """Historical domain agent generated from the base AgentTemplate."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        agent_id: str | None = None,
        name: str | None = None,
        role: str = "Classical & Egyptian Historical Expert",
        system_prompt: str | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        from historyverse.config.settings import get_settings

        settings = get_settings()
        assigned_name = name or settings.agent_persona_name
        assigned_id = agent_id or settings.agent_persona_id
        prompt_text = (
            system_prompt
            if system_prompt is not None
            else (
                "Your name is {name}. You are an expert AI specialized in analyzing history, "
                "historical questions, historical images, and historical places throughout the world, "
                "especially those related to Egypt and Alexandria. Provide deep, accurate classical insights."
            )
        )

        template = AgentTemplate(
            agent_id=assigned_id,
            name=assigned_name,
            role=role,
            domain="history",
            system_prompt_template=prompt_text,
        )

        super().__init__(
            template=template,
            llm_provider=llm_provider,
            memory=memory,
            metadata={"specialization": "Alexandria & Classical History"},
        )
