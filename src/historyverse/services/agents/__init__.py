"""Multi-agent models generation, specialized domain agents, and team orchestrator."""

from historyverse.services.agents.base import (
    AgentMessage,
    AgentResponse,
    BaseAgent,
    TeamExecutionResponse,
)
from historyverse.services.agents.critic import CriticAgent
from historyverse.services.agents.domain import DomainAgent
from historyverse.services.agents.historical import HistoryVerseAgent
from historyverse.services.agents.historyverse_ar_mode import HistoryVerseARModeAgent
from historyverse.services.agents.orchestrator import AgentOrchestrator, CustomAgent
from historyverse.services.agents.researcher import ResearcherAgent
from historyverse.services.agents.template import AgentTemplate, TemplatedAgent

__all__ = [
    "AgentMessage",
    "AgentOrchestrator",
    "AgentResponse",
    "AgentTemplate",
    "BaseAgent",
    "CriticAgent",
    "CustomAgent",
    "DomainAgent",
    "HistoryVerseARModeAgent",
    "HistoryVerseAgent",
    "ResearcherAgent",
    "TeamExecutionResponse",
    "TemplatedAgent",
]
