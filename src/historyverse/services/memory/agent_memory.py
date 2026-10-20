"""Isolated, agent-scoped conversation memory manager.

Guarantees complete memory isolation between distinct agent instances.
"""

from __future__ import annotations

import logging

from historyverse.config.settings import Settings, get_settings
from historyverse.services.memory.manager import ConversationMemory

logger = logging.getLogger(__name__)

__all__ = ["AgentMemory", "AgentMemoryManager"]


class AgentMemory:
    """Isolated working and conversational memory scoped strictly to a single agent.

    Ensures that conversation turns and scratchpad contexts belonging to one agent
    never bleed into, contaminate, or overwrite another agent's memory state.
    """

    def __init__(
        self,
        agent_id: str,
        backend: ConversationMemory | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._settings = settings or get_settings()
        self._backend = backend if backend is not None else ConversationMemory(self._settings)

    def scope_session(self, session_id: str) -> str:
        """Derive an isolated session key strictly namespaced to this agent ID."""
        return f"{self.agent_id}::{session_id}"

    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        """Append a question-answer interaction to this agent's scoped memory."""
        scoped_id = self.scope_session(session_id)
        self._backend.add_exchange(scoped_id, question, answer)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve raw interaction turns belonging exclusively to this agent."""
        scoped_id = self.scope_session(session_id)
        return self._backend.get_history(scoped_id)

    def get_formatted_history(self, session_id: str, assistant_name: str | None = None) -> str:
        """Retrieve prompt-injectable conversational history for this agent."""
        scoped_id = self.scope_session(session_id)
        name = assistant_name or self.agent_id
        return self._backend.get_formatted_history(scoped_id, assistant_name=name)

    def clear(self, session_id: str) -> None:
        """Purge conversational history for a specific session of this agent."""
        scoped_id = self.scope_session(session_id)
        self._backend.clear(scoped_id)

    def unlearn_concept(self, keywords: list[str], session_id: str | None = None) -> int:
        """Purge turns containing forbidden keywords from this agent's scoped memory.

        If session_id is provided, purges turns only within that session.
        If omitted, purges matching turns across all sessions for this agent.
        """
        prefix = self.scope_session(session_id) if session_id else f"{self.agent_id}::"
        purged = self._backend.purge_concept(keywords, session_prefix=prefix)
        logger.info(
            "Agent [%s] unlearned %d memory turns matching keywords: %s",
            self.agent_id,
            purged,
            keywords,
        )
        return purged


class AgentMemoryManager:
    """Registry managing isolated AgentMemory instances for all active agents."""

    def __init__(
        self,
        shared_backend: ConversationMemory | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._backend = (
            shared_backend if shared_backend is not None else ConversationMemory(self._settings)
        )
        self._agent_memories: dict[str, AgentMemory] = {}

    def get_agent_memory(self, agent_id: str) -> AgentMemory:
        """Retrieve or lazily instantiate an isolated memory space for an agent."""
        if agent_id not in self._agent_memories:
            self._agent_memories[agent_id] = AgentMemory(
                agent_id=agent_id,
                backend=self._backend,
                settings=self._settings,
            )
        return self._agent_memories[agent_id]

    def purge_from_all_agents(self, keywords: list[str]) -> int:
        """Execute cross-agent conversational memory unlearning."""
        return self._backend.purge_concept(keywords)

    def purge_from_agent(self, agent_id: str, keywords: list[str]) -> int:
        """Execute conversational memory unlearning scoped to a single agent."""
        agent_mem = self.get_agent_memory(agent_id)
        return agent_mem.unlearn_concept(keywords)
