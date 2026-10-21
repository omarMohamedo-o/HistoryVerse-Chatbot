"""API routes for multi-agent execution, dynamic domain agent generation, and team orchestration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.api.dependencies import (
    get_hallucination_detector,
    get_llm_provider,
    get_retriever,
)
from agent.config.settings import Settings, get_settings
from agent.core.base import BaseLLMProvider
from agent.services.agents.critic import CriticAgent
from agent.services.agents.orchestrator import AgentOrchestrator
from agent.services.agents.researcher import ResearcherAgent
from agent.services.agents.template import AgentTemplate, TemplatedAgent
from agent.services.evaluation.hallucination import HallucinationDetector
from agent.services.rag.retriever import ChromaRetriever

router = APIRouter(prefix="/v1/agents", tags=["Agents"])

# Global orchestrator instance initialized with default agents
_ORCHESTRATOR = AgentOrchestrator()
_INITIALIZED = False


def get_agent_orchestrator(
    llm: BaseLLMProvider = Depends(get_llm_provider),
    retriever: ChromaRetriever = Depends(get_retriever),
    hallucination: HallucinationDetector = Depends(get_hallucination_detector),
    settings: Settings = Depends(get_settings),
) -> AgentOrchestrator:
    """Dependency providing the agent orchestrator populated with default agents."""
    global _INITIALIZED
    if not _INITIALIZED:
        template = AgentTemplate.resolve(settings.default_agent_template, settings.prompts_dir)
        default_agent = TemplatedAgent(template=template, llm_provider=llm)

        researcher = ResearcherAgent(llm_provider=llm, retriever=retriever)
        critic = CriticAgent(llm_provider=llm, hallucination_detector=hallucination)

        _ORCHESTRATOR.register_agent(researcher)
        _ORCHESTRATOR.register_agent(default_agent)
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
        default_factory=lambda: [
            "archival-researcher",
            get_settings().agent_persona_id,
            "historical-critic",
        ],
        description="Ordered sequence of agent IDs to execute",
    )


class CreateAgentRequest(BaseModel):
    agent_id: str = Field(
        ..., min_length=2, description="Unique agent identifier (e.g. 'medical-diagnostic')"
    )
    name: str = Field(..., min_length=2, description="Agent persona name (e.g. 'Dr. Sarah')")
    role: str = Field(
        ...,
        min_length=2,
        description="Agent professional role (e.g. 'Clinical Diagnostic Consultant')",
    )
    domain: str = Field(
        default="general",
        description="Knowledge domain (e.g. 'medicine', 'history', 'law', 'finance')",
    )
    system_prompt: str = Field(
        ..., min_length=10, description="System instructions and domain behavioral guardrails"
    )
    provider: str | None = Field(
        default=None,
        description="Optional LLM provider engine ('openai', 'gemini', 'ollama', 'local')",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional model identifier override (e.g. 'gpt-4o', 'gemini-1.5-flash', 'llama3')",
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Optional temperature parameter"
    )
    api_key: str | None = Field(default=None, description="Optional provider API key override")


@router.get("", summary="List all active agents")
async def list_agents(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> list[dict[str, Any]]:
    """List all registered agents and their assigned domain and model engines."""
    return orchestrator.list_agents()


@router.post("/custom", summary="Dynamically spawn a new specialized domain agent")
async def create_custom_agent(
    req: CreateAgentRequest,
    llm: BaseLLMProvider = Depends(get_llm_provider),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> dict[str, Any]:
    """Dynamically register a new specialized domain agent model with customizable settings and provider."""
    use_default_llm = (
        req.provider is None
        and req.model_name is None
        and req.temperature is None
        and req.api_key is None
    )
    agent = orchestrator.create_custom_agent(
        agent_id=req.agent_id,
        name=req.name,
        role=req.role,
        domain=req.domain,
        system_prompt=req.system_prompt,
        llm_provider=llm if use_default_llm else None,
        provider_name=req.provider,
        model_name=req.model_name,
        temperature=req.temperature,
        api_key=req.api_key,
    )
    return {
        "status": "created",
        "agent_id": agent.agent_id,
        "name": agent.name,
        "role": agent.role,
        "domain": agent.domain,
        "model_used": agent.llm_provider.model_name,
    }


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
