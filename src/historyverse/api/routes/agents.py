"""API routes for multi-agent execution, dynamic agent generation, and team orchestration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from historyverse.api.dependencies import (
    get_hallucination_detector,
    get_llm_provider,
    get_retriever,
)
from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.critic import CriticAgent
from historyverse.services.agents.historical import HistoryVerseAgent
from historyverse.services.agents.orchestrator import AgentOrchestrator
from historyverse.services.agents.researcher import ResearcherAgent
from historyverse.services.evaluation.hallucination import HallucinationDetector
from historyverse.services.rag.retriever import ChromaRetriever

router = APIRouter(prefix="/v1/agents", tags=["Agents"])

# Global orchestrator instance initialized with default agents
_ORCHESTRATOR = AgentOrchestrator()
_INITIALIZED = False


def get_agent_orchestrator(
    llm: BaseLLMProvider = Depends(get_llm_provider),
    retriever: ChromaRetriever = Depends(get_retriever),
    hallucination: HallucinationDetector = Depends(get_hallucination_detector),
) -> AgentOrchestrator:
    """Dependency providing the agent orchestrator populated with default agents."""
    global _INITIALIZED
    if not _INITIALIZED:
        mora = HistoryVerseAgent(llm_provider=llm)
        researcher = ResearcherAgent(llm_provider=llm, retriever=retriever)
        critic = CriticAgent(llm_provider=llm, hallucination_detector=hallucination)

        _ORCHESTRATOR.register_agent(researcher)
        _ORCHESTRATOR.register_agent(mora)
        _ORCHESTRATOR.register_agent(critic)
        _INITIALIZED = True
    return _ORCHESTRATOR


class RunAgentRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User prompt or task for the agent")
    context: dict[str, Any] = Field(default_factory=dict, description="Optional extra context")


class RunTeamRequest(BaseModel):
    query: str = Field(
        ..., min_length=2, description="User prompt for collaborative team execution"
    )
    pipeline: list[str] = Field(
        default_factory=lambda: ["archival-researcher", "historyverse-mora", "historical-critic"],
        description="Ordered sequence of agent IDs to execute",
    )


class CreateAgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=2)
    name: str = Field(..., min_length=2)
    role: str = Field(..., min_length=2)
    system_prompt: str = Field(..., min_length=10)


@router.get("", summary="List all active agents")
async def list_agents(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> list[dict[str, str]]:
    """List all registered agents and their assigned model engines."""
    return orchestrator.list_agents()


@router.post("/custom", summary="Dynamically spawn a new specialized agent")
async def create_custom_agent(
    req: CreateAgentRequest,
    llm: BaseLLMProvider = Depends(get_llm_provider),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> dict[str, str]:
    """Dynamically register a new specialized agent model."""
    orchestrator.create_custom_agent(
        agent_id=req.agent_id,
        name=req.name,
        role=req.role,
        system_prompt=req.system_prompt,
        llm_provider=llm,
    )
    return {"status": "created", "agent_id": req.agent_id, "name": req.name}


@router.post("/{agent_id}/run", summary="Execute a single agent individually")
async def run_individual_agent(
    agent_id: str,
    req: RunAgentRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
):
    """Execute a single agent model independently."""
    try:
        return await orchestrator.run_individual(agent_id, req.query, req.context)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/team/run", summary="Execute a collaborative multi-agent team pipeline")
async def run_team_pipeline(
    req: RunTeamRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
):
    """Execute a collaborative multi-agent workflow where models cooperate."""
    return await orchestrator.run_team(query=req.query, agent_pipeline=req.pipeline)
