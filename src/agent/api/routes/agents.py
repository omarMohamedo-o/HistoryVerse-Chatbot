"""API routes for multi-agent execution, dynamic domain agent generation, and team orchestration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from agent.api.dependencies import (
    get_hallucination_detector,
    get_llm_provider,
    get_retriever,
    verify_api_key,
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


class UpdateAgentRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, description="Updated persona name")
    role: str | None = Field(default=None, min_length=2, description="Updated professional role")
    domain: str | None = Field(default=None, description="Updated domain")
    system_prompt: str | None = Field(
        default=None, min_length=10, description="Updated system prompt"
    )
    provider: str | None = Field(default=None, description="Updated provider")
    model_name: str | None = Field(default=None, description="Updated model identifier")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


# Agent Listing and Provider Discovery
@router.get("", summary="List all active agents")
async def list_agents(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> list[dict[str, Any]]:
    """List all registered agents and their assigned domain and model engines."""
    return orchestrator.list_agents()


@router.get("/providers", summary="Discover supported and available model providers")
async def discover_providers(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> list[dict[str, Any]]:
    """Discover genuinely supported LLM engines, models, and real-time capability status."""
    return orchestrator.discover_providers()


@router.post("/custom", summary="Dynamically spawn a new specialized domain agent")
async def create_custom_agent(
    req: CreateAgentRequest,
    llm: BaseLLMProvider = Depends(get_llm_provider),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    """Dynamically register a new specialized domain agent model with customizable settings and provider."""
    use_default_llm = (
        req.provider is None
        and req.model_name is None
        and req.temperature is None
        and req.api_key is None
    )
    metadata = {"tenant_id": x_tenant_id} if x_tenant_id else None
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
        metadata=metadata,
    )
    return {
        "status": "created",
        "agent_id": agent.agent_id,
        "name": agent.name,
        "role": agent.role,
        "domain": agent.domain,
        "model_used": agent.llm_provider.model_name,
    }


@router.post("/team/run", summary="Execute a collaborative multi-agent team pipeline")
async def run_team_pipeline(
    req: RunTeamRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
):
    """Execute a collaborative multi-agent workflow where models cooperate."""
    return await orchestrator.run_team(query=req.query, agent_pipeline=req.pipeline)


# Multi-Agent Team Management
class CreateTeamRequest(BaseModel):
    team_id: str = Field(..., min_length=2, description="Unique team identifier")
    name: str = Field(..., min_length=2, description="Team display name")
    strategy: str = Field(
        default="sequential",
        description="Execution strategy: sequential, parallel, supervisor, conditional",
    )
    members: list[str] = Field(..., min_length=1, description="List of agent IDs in the team")
    description: str = Field(default="", description="Team mandate description")
    config: dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")


class RunWorkflowRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Task prompt for the team")
    require_human_approval: bool = Field(
        default=False, description="Pause for human approval before execution"
    )


class ApproveWorkflowRequest(BaseModel):
    approved: bool = Field(default=True, description="Whether to approve or reject the run")
    comment: str | None = Field(default=None, description="Optional supervisor notes")


@router.get("/teams", summary="List all multi-agent teams")
async def list_teams(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> list[dict[str, Any]]:
    """Retrieve all configured multi-agent teams and their execution strategies."""
    return orchestrator.team_store.list_teams(tenant_id=x_tenant_id)


@router.post("/teams", summary="Create or register a multi-agent team")
async def create_team(
    req: CreateTeamRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
) -> dict[str, Any]:
    """Register a new multi-agent team with defined collaboration strategy."""
    return orchestrator.team_store.save_team(
        team_id=req.team_id,
        name=req.name,
        strategy=req.strategy,
        members=req.members,
        description=req.description,
        config=req.config,
    )


@router.get("/teams/{team_id}", summary="Get multi-agent team details")
async def get_team(
    team_id: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    """Retrieve details for a specific multi-agent team."""
    team = orchestrator.team_store.get_team(team_id, tenant_id=x_tenant_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
    return team


@router.delete("/teams/{team_id}", summary="Delete a multi-agent team")
async def delete_team(
    team_id: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    """Remove a multi-agent team definition."""
    deleted = orchestrator.team_store.delete_team(team_id, tenant_id=x_tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
    return {"status": "deleted", "team_id": team_id}


@router.post("/teams/{team_id}/run", summary="Execute a multi-agent team workflow")
async def execute_team_workflow(
    team_id: str,
    req: RunWorkflowRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
):
    """Execute team workflow with sequential, parallel, supervisor, or conditional strategy."""
    try:
        return await orchestrator.run_team_workflow(
            team_id=team_id,
            query=req.query,
            require_human_approval=req.require_human_approval,
            tenant_id=x_tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# Workflow Execution Auditing & Human Approval
@router.get("/executions", summary="List workflow executions")
async def list_executions(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> list[dict[str, Any]]:
    """List recent workflow execution runs with status and step audits."""
    return orchestrator.team_store.list_executions(tenant_id=x_tenant_id)


@router.get("/executions/{run_id}", summary="Get workflow execution run details")
async def get_execution(
    run_id: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    """Retrieve execution status, inter-agent step outputs, and metrics."""
    record = orchestrator.team_store.get_execution(run_id, tenant_id=x_tenant_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Execution run '{run_id}' not found.")
    return record


@router.post(
    "/executions/{run_id}/approve", summary="Approve or reject a paused workflow checkpoint"
)
async def approve_workflow_run(
    run_id: str,
    req: ApproveWorkflowRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    """Approve or reject a workflow paused at a human-in-the-loop checkpoint."""
    try:
        return await orchestrator.resume_workflow(
            run_id=run_id,
            approved=req.approved,
            user_comment=req.comment,
            tenant_id=x_tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# Parameterized Individual Agent Operations
@router.get("/{agent_id}", summary="Get agent details")
async def get_agent_details(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    """Retrieve details of a registered agent by identifier."""
    agent = orchestrator.get_agent(agent_id, tenant_id=x_tenant_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "role": agent.role,
        "domain": getattr(agent, "domain", "general"),
        "model_used": agent.llm_provider.model_name,
    }


@router.put("/{agent_id}", summary="Update an existing custom domain agent")
async def update_custom_agent(
    agent_id: str,
    req: UpdateAgentRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
) -> dict[str, Any]:
    """Update properties of a custom domain agent."""
    try:
        agent = orchestrator.update_custom_agent(
            agent_id=agent_id,
            name=req.name,
            role=req.role,
            domain=req.domain,
            system_prompt=req.system_prompt,
            provider_name=req.provider,
            model_name=req.model_name,
            temperature=req.temperature,
        )
        return {
            "status": "updated",
            "agent_id": agent.agent_id,
            "name": agent.name,
            "role": agent.role,
            "domain": getattr(agent, "domain", "general"),
            "model_used": agent.llm_provider.model_name,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{agent_id}", summary="Delete a custom domain agent")
async def delete_custom_agent(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    """Remove a custom domain agent from the orchestrator and persistent store."""
    deleted = orchestrator.delete_custom_agent(agent_id, tenant_id=x_tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return {"status": "deleted", "agent_id": agent_id}


@router.post("/{agent_id}/run", summary="Execute a single agent individually")
async def run_individual_agent(
    agent_id: str,
    req: RunAgentRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    _auth: bool = Depends(verify_api_key),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
):
    """Execute a single agent model independently."""
    try:
        return await orchestrator.run_individual(
            agent_id, req.query, req.context, tenant_id=x_tenant_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
