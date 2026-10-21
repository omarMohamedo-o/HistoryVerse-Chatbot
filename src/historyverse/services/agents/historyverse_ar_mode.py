"""HistoryVerse AR Mode (Augmented Reality Artifact & Statue Guide) Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.template import AgentTemplate, TemplatedAgent
from historyverse.utils.multilingual import is_arabic

if TYPE_CHECKING:
    from historyverse.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["HistoryVerseARAgent", "HistoryVerseARModeAgent"]


class HistoryVerseARModeAgent(TemplatedAgent):
    """Augmented reality artifact tour guide agent generated from the base AgentTemplate."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        artifact_label: str = "Ancient Artifact",
        agent_id: str = "historyverse-ar-mode",
        name: str | None = None,
        role: str = "Augmented Reality Artifact Guide",
        system_prompt: str | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        from historyverse.config.settings import get_settings

        settings = get_settings()
        assigned_name = name or settings.agent_persona_name
        self.artifact_label = artifact_label

        prompt_text = (
            system_prompt
            if system_prompt is not None
            else (
                "Your name is {name}. You are an expert historical guide specialized in "
                "the monument or historical artifact: '{artifact_label}'. "
                "You have extensive historical, cultural, and archaeological knowledge regarding '{artifact_label}'. "
                "Answer questions about its origin, significance, architecture, and historical context. "
                "If the user asks about topics completely unrelated to '{artifact_label}', politely state that "
                "you can only analyze and answer questions related to '{artifact_label}'. "
                "Make sure your answers are in the same language as the user's question (supporting Arabic and English seamlessly)."
            )
        )

        template = AgentTemplate(
            agent_id=agent_id,
            name=assigned_name,
            role=role,
            domain="augmented-reality-history",
            system_prompt_template=prompt_text,
            default_variables={"artifact_label": self.artifact_label},
        )

        super().__init__(
            template=template,
            llm_provider=llm_provider,
            memory=memory,
            metadata={"artifact_label": self.artifact_label, "agent_type": "HistoryVerseARMode"},
        )

    def set_artifact_label(self, label: str) -> None:
        """Update active artifact label and re-render template prompt."""
        self.artifact_label = label
        self.update_variables(artifact_label=label)
        self._custom_metadata["artifact_label"] = label

    async def start_conversation(
        self,
        artifact_name: str | None = None,
        session_id: str | None = None,
        language: str = "en",
    ) -> str:
        """Generate introductory overview of the detected artifact (under 10 lines)."""
        label = artifact_name or self.artifact_label
        self.set_artifact_label(label)

        if language == "ar" or is_arabic(label):
            intro_query = f"قدم نبذة تاريخية شاملة وموجزة (في حدود 10 أسطر أو أقل) عن {label} وأهميتها التاريخية والأثرية."
        else:
            intro_query = f"Tell me everything you can about {label}, but try to keep it brief (10 lines or less)."

        context: dict[str, Any] = {"session_id": session_id} if session_id else {}
        response = await self.run(intro_query, context=context)
        return response.output


HistoryVerseARAgent = HistoryVerseARModeAgent
