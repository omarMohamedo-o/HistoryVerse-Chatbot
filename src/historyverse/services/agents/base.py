"""Abstract Base Class and Domain Schemas for the Multi-Agent System."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from historyverse.core.base import BaseLLMProvider
from historyverse.services.memory.agent_memory import AgentMemory


class AgentMessage(BaseModel):
    """Message exchanged between agents or from user."""

    sender: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Execution output from a single agent."""

    agent_id: str
    agent_name: str
    role: str
    output: str
    model_used: str
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TeamExecutionResponse(BaseModel):
    """Aggregated output from a collaborative multi-agent team pipeline."""

    query: str
    final_answer: str
    agent_contributions: list[AgentResponse]
    drift_detected: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all autonomous or specialized agents."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        system_prompt: str,
        llm_provider: BaseLLMProvider,
        memory: AgentMemory | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm_provider = llm_provider
        self.memory = memory if memory is not None else AgentMemory(agent_id=agent_id)

    def build_effective_system_prompt(self) -> str:
        """Compose system prompt including dynamic negative unlearning guard constraints."""
        from historyverse.services.unlearning.registry import get_tombstone_registry

        guard = get_tombstone_registry().get_unlearning_guard_prompt(agent_id=self.agent_id)
        return f"{self.system_prompt}\n\n{guard}" if guard else self.system_prompt

    @abstractmethod
    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        """Execute agent reasoning and return response."""
