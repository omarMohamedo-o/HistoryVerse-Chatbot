"""HistoryVerse AR Mode (Augmented Reality Artifact & Statue Guide) Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.base import AgentResponse, BaseAgent
from historyverse.utils.multilingual import is_arabic

if TYPE_CHECKING:
    from historyverse.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["HistoryVerseARAgent", "HistoryVerseARModeAgent"]


class HistoryVerseARModeAgent(BaseAgent):
    """Specialized historical AI agent anchored to a specific statue, monument, or artifact.

    Designed for interactive museum AR, camera scans, and guided artifact exploration.
    Strictly focuses dialogue on the detected artifact and responds in the visitor's language.
    """

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

        prompt = (
            system_prompt
            if system_prompt is not None
            else (
                f"Your name is {assigned_name}. You are an expert historical guide specialized in "
                f"the monument or historical artifact: '{self.artifact_label}'. "
                f"You have extensive historical, cultural, and archaeological knowledge regarding '{self.artifact_label}'. "
                f"Answer questions about its origin, significance, architecture, and historical context. "
                f"If the user asks about topics completely unrelated to '{self.artifact_label}', politely state that "
                f"you can only analyze and answer questions related to '{self.artifact_label}'. "
                "Make sure your answers are in the same language as the user's question (supporting Arabic and English seamlessly)."
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

    def set_artifact_label(self, label: str) -> None:
        """Update active artifact label and re-anchor system prompt."""
        self.artifact_label = label
        self.system_prompt = (
            f"Your name is {self.name}. You are an expert historical guide specialized in "
            f"the monument or historical artifact: '{self.artifact_label}'. "
            f"You have extensive historical, cultural, and archaeological knowledge regarding '{self.artifact_label}'. "
            f"Answer questions about its origin, significance, architecture, and historical context. "
            f"If the user asks about topics completely unrelated to '{self.artifact_label}', politely state that "
            f"you can only analyze and answer questions related to '{self.artifact_label}'. "
            "Make sure your answers are in the same language as the user's question (supporting Arabic and English seamlessly)."
        )

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
        parts.append(f"Question regarding {self.artifact_label}:\n{query}")
        formatted_prompt = "\n\n".join(parts)

        logger.info(
            "Executing AR agent [%s] for artifact '%s', query: %s",
            self.name,
            self.artifact_label,
            query[:50],
        )
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
            metadata={
                "artifact_label": self.artifact_label,
                "agent_type": "HistoryVerseARMode",
            },
        )


HistoryVerseARAgent = HistoryVerseARModeAgent
