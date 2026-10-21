"""Multi-Agent Team Orchestrator for collaborative or individual model execution."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

from agent.core.base import BaseLLMProvider
from agent.services.agents.base import AgentResponse, BaseAgent, TeamExecutionResponse
from agent.services.agents.registry_store import AgentRegistryStore
from agent.services.agents.team_store import TeamRegistryStore
from agent.services.agents.template import AgentTemplate, TemplatedAgent

if TYPE_CHECKING:
    from agent.config.settings import Settings
    from agent.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["AgentOrchestrator", "CustomAgent"]

CustomAgent = TemplatedAgent


class AgentOrchestrator:
    """Orchestrates individual agent invocations or collaborative multi-agent team pipelines.

    Uses declarative AgentTemplates to spawn domain-agnostic agents with independent
    customized settings, isolated memories, and decoupled LLM providers.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        store: AgentRegistryStore | None = None,
        team_store: TeamRegistryStore | None = None,
    ) -> None:
        from agent.config.settings import get_settings

        self._settings = settings or get_settings()
        self._registry: dict[str, BaseAgent] = {}
        self._store = store or AgentRegistryStore(settings=self._settings)
        self._team_store = team_store or TeamRegistryStore(settings=self._settings)
        self.rehydrate()

    @property
    def team_store(self) -> TeamRegistryStore:
        """Return the multi-agent team and execution registry store."""
        return self._team_store

    @property
    def agent_store(self) -> AgentRegistryStore:
        """Return the custom agent registry store."""
        return self._store

    def rehydrate(self) -> int:
        """Load persisted custom agents from persistent store into runtime."""
        records = self._store.list_agents()
        loaded = 0
        for rec in records:
            aid = rec["agent_id"]
            if aid in self._registry:
                continue
            try:
                self.create_custom_agent(
                    agent_id=rec["agent_id"],
                    name=rec["name"],
                    role=rec["role"],
                    domain=rec.get("domain", "general"),
                    system_prompt=rec["system_prompt"],
                    provider_name=rec.get("provider_name"),
                    model_name=rec.get("model_name"),
                    temperature=rec.get("temperature"),
                    metadata=rec.get("metadata"),
                    persist=False,
                )
                loaded += 1
            except Exception as exc:
                logger.warning("Failed to rehydrate custom agent '%s': %s", aid, exc)
        return loaded

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent instance in the orchestrator registry."""
        self._registry[agent.agent_id] = agent
        logger.info(
            "Registered agent: %s (%s) [domain: %s]",
            agent.name,
            agent.agent_id,
            getattr(agent, "domain", "general"),
        )

    def get_agent(self, agent_id: str) -> BaseAgent | None:
        """Retrieve an agent by its unique identifier."""
        return self._registry.get(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """Return metadata for all registered agents across all domains."""
        results = []
        for a in self._registry.values():
            provider_type = "unknown"
            if hasattr(a.llm_provider, "settings") and hasattr(
                a.llm_provider.settings, "llm_provider"
            ):
                provider_type = str(a.llm_provider.settings.llm_provider)
            results.append(
                {
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "role": a.role,
                    "domain": getattr(a, "domain", "general"),
                    "model_used": a.llm_provider.model_name,
                    "provider": provider_type,
                }
            )
        return results

    def create_custom_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        system_prompt: str,
        domain: str = "general",
        llm_provider: BaseLLMProvider | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        memory: AgentMemory | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> TemplatedAgent:
        """Spawn a new custom domain agent from an AgentTemplate with isolated settings and LLM engine."""
        if llm_provider is not None:
            resolved_provider = llm_provider
        elif provider_name or model_name or temperature is not None or api_key:
            updates: dict[str, Any] = {}
            if provider_name:
                from agent.core.enums import LLMProvider

                updates["llm_provider"] = LLMProvider(provider_name.lower())
            if model_name:
                updates["llm_model_name"] = model_name
            if temperature is not None:
                updates["llm_temperature"] = temperature
            if api_key:
                from pydantic import SecretStr

                if provider_name and "openai" in provider_name.lower():
                    updates["openai_api_key"] = SecretStr(api_key)
                else:
                    updates["google_api_key"] = SecretStr(api_key)
            agent_settings = self._settings.model_copy(update=updates)
            from agent.services.llm.factory import LLMFactory

            resolved_provider = LLMFactory.create(agent_settings)
        else:
            from agent.services.llm.factory import LLMFactory

            resolved_provider = LLMFactory.create(self._settings)

        if memory is None:
            from agent.services.memory.agent_memory import AgentMemory
            from agent.services.memory.manager import ConversationMemory

            backend = ConversationMemory(self._settings)
            memory = AgentMemory(agent_id=agent_id, backend=backend)

        template = AgentTemplate(
            agent_id=agent_id,
            name=name,
            role=role,
            domain=domain,
            system_prompt_template=system_prompt,
            provider_type=provider_name,
            model_name=model_name,
            temperature=temperature,
        )

        agent = TemplatedAgent(
            template=template,
            llm_provider=resolved_provider,
            memory=memory,
            metadata=metadata,
        )
        self.register_agent(agent)

        if persist:
            self._store.save_agent(
                agent_id=agent_id,
                name=name,
                role=role,
                domain=domain,
                system_prompt=system_prompt,
                provider_name=provider_name,
                model_name=model_name,
                temperature=temperature,
                metadata=metadata,
            )

        return agent

    def delete_custom_agent(self, agent_id: str) -> bool:
        """Delete custom agent from memory registry and persistent store."""
        removed = False
        if agent_id in self._registry:
            del self._registry[agent_id]
            removed = True
        store_removed = self._store.delete_agent(agent_id)
        return removed or store_removed

    def update_custom_agent(
        self,
        agent_id: str,
        name: str | None = None,
        role: str | None = None,
        domain: str | None = None,
        system_prompt: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemplatedAgent:
        """Update properties of an existing custom agent and persist changes."""
        existing_rec = self._store.get_agent(agent_id)
        current_agent = self.get_agent(agent_id)
        if not existing_rec and not current_agent:
            raise ValueError(f"Agent '{agent_id}' does not exist.")

        updated_name = name or (current_agent.name if current_agent else existing_rec["name"])
        updated_role = role or (current_agent.role if current_agent else existing_rec["role"])
        updated_domain = domain or (
            getattr(current_agent, "domain", "general")
            if current_agent
            else existing_rec.get("domain", "general")
        )
        updated_prompt = system_prompt or (
            existing_rec["system_prompt"]
            if existing_rec
            else getattr(current_agent, "system_prompt", "")
        )
        updated_provider = provider_name or (
            existing_rec.get("provider_name") if existing_rec else None
        )
        updated_model = model_name or (existing_rec.get("model_name") if existing_rec else None)
        updated_temp = (
            temperature
            if temperature is not None
            else (existing_rec.get("temperature") if existing_rec else None)
        )
        updated_meta = (
            metadata
            if metadata is not None
            else (existing_rec.get("metadata", {}) if existing_rec else {})
        )

        return self.create_custom_agent(
            agent_id=agent_id,
            name=updated_name,
            role=updated_role,
            domain=updated_domain,
            system_prompt=updated_prompt,
            provider_name=updated_provider,
            model_name=updated_model,
            temperature=updated_temp,
            metadata=updated_meta,
            persist=True,
        )

    async def run_individual(
        self, agent_id: str, query: str, context: dict[str, Any] | None = None
    ) -> AgentResponse:
        """Execute a single agent independently."""
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' is not registered in the orchestrator.")
        return await agent.run(query, context)

    async def run_team(
        self,
        query: str,
        agent_pipeline: list[str] | None = None,
    ) -> TeamExecutionResponse:
        """Execute a collaborative multi-agent workflow where agents build upon each other's work."""
        primary_id = self._settings.agent_persona_id
        pipeline = agent_pipeline or [
            "archival-researcher",
            primary_id,
            "historical-critic",
        ]
        contributions: list[AgentResponse] = []
        shared_context: dict[str, Any] = {}

        for step_agent_id in pipeline:
            agent = self.get_agent(step_agent_id)
            if not agent:
                logger.warning("Agent '%s' in pipeline not found; skipping step.", step_agent_id)
                continue

            step_response = await agent.run(query, shared_context)
            contributions.append(step_response)

            # Pass forward data to subsequent agents in the pipeline
            if agent.agent_id == "archival-researcher":
                shared_context["context"] = step_response.output
                shared_context["evidence"] = [step_response.output]
            elif agent.agent_id == primary_id or agent.name == self._settings.agent_persona_name:
                shared_context["draft_answer"] = step_response.output

        # Final answer is taken from the primary domain agent or the last answering agent
        primary_contrib = next(
            (
                c
                for c in contributions
                if c.agent_id == primary_id or c.agent_name == self._settings.agent_persona_name
            ),
            None,
        )
        final_answer = (
            primary_contrib.output
            if primary_contrib
            else (contributions[-1].output if contributions else "No output generated.")
        )

        critic_contrib = next((c for c in contributions if c.agent_id == "historical-critic"), None)
        drift_detected = (
            critic_contrib.metadata.get("is_hallucinated", False) if critic_contrib else False
        )

        return TeamExecutionResponse(
            query=query,
            final_answer=final_answer,
            agent_contributions=contributions,
            drift_detected=drift_detected,
            metadata={"pipeline_steps": len(contributions)},
        )

    async def run_parallel(
        self,
        query: str,
        agent_ids: list[str],
        aggregator_agent_id: str | None = None,
    ) -> TeamExecutionResponse:
        """Execute multiple agents concurrently and aggregate their contributions."""
        agents = [self.get_agent(aid) for aid in agent_ids]
        valid_agents = [a for a in agents if a is not None]
        if not valid_agents:
            raise ValueError("No valid agents specified for parallel execution.")

        tasks = [agent.run(query) for agent in valid_agents]
        contributions: list[AgentResponse] = list(await asyncio.gather(*tasks))

        if aggregator_agent_id:
            aggregator = self.get_agent(aggregator_agent_id)
            if aggregator:
                summary_prompt = (
                    f"User Query: {query}\n\n"
                    f"Specialist Findings:\n"
                    + "\n\n".join(f"[{c.agent_name}]:\n{c.output}" for c in contributions)
                    + "\n\nPlease synthesize a comprehensive final response integrating the above findings."
                )
                agg_res = await aggregator.run(summary_prompt)
                contributions.append(agg_res)
                final_answer = agg_res.output
            else:
                final_answer = "\n\n---\n\n".join(
                    f"**{c.agent_name}**:\n{c.output}" for c in contributions
                )
        else:
            final_answer = "\n\n---\n\n".join(
                f"**{c.agent_name}**:\n{c.output}" for c in contributions
            )

        return TeamExecutionResponse(
            query=query,
            final_answer=final_answer,
            agent_contributions=contributions,
            drift_detected=False,
            metadata={"strategy": "parallel", "parallel_agents": len(valid_agents)},
        )

    async def run_supervisor(
        self,
        query: str,
        supervisor_id: str,
        worker_ids: list[str],
        max_steps: int = 4,
    ) -> TeamExecutionResponse:
        """Supervisor agent delegates subtasks to specialist workers and evaluates results."""
        supervisor = self.get_agent(supervisor_id)
        if not supervisor:
            raise ValueError(f"Supervisor agent '{supervisor_id}' not found.")

        workers = {
            wid: self.get_agent(wid) for wid in worker_ids if self.get_agent(wid) is not None
        }
        if not workers:
            raise ValueError("No valid worker agents provided for supervisor delegation.")

        contributions: list[AgentResponse] = []
        worker_descriptions = "\n".join(
            f"- ID: {wid} | Name: {w.name} | Role: {w.role}" for wid, w in workers.items()
        )

        delegation_prompt = (
            f"You are the team supervisor managing these specialists:\n{worker_descriptions}\n\n"
            f"User Goal: {query}\n\n"
            f"Break down this goal and assign specific research subtasks to the team."
        )
        sup_initial = await supervisor.run(delegation_prompt)
        contributions.append(sup_initial)

        subtask_results = []
        steps_executed = 0
        for _wid, worker in workers.items():
            if steps_executed >= max_steps:
                break
            worker_task = (
                f"Subtask delegated by supervisor:\n{sup_initial.output}\n\nContext: {query}"
            )
            w_res = await worker.run(worker_task)
            contributions.append(w_res)
            subtask_results.append(f"[{worker.name}]:\n{w_res.output}")
            steps_executed += 1

        eval_prompt = (
            f"Original User Goal: {query}\n\n"
            f"Specialist Findings:\n" + "\n\n".join(subtask_results) + "\n\n"
            "As supervisor, deliver the comprehensive final resolution."
        )
        final_sup_res = await supervisor.run(eval_prompt)
        contributions.append(final_sup_res)

        return TeamExecutionResponse(
            query=query,
            final_answer=final_sup_res.output,
            agent_contributions=contributions,
            drift_detected=False,
            metadata={
                "strategy": "supervisor",
                "supervisor": supervisor_id,
                "steps": steps_executed,
            },
        )

    async def run_conditional(
        self,
        query: str,
        router_agent_id: str,
        branches: dict[str, str],
        fallback_agent_id: str | None = None,
    ) -> TeamExecutionResponse:
        """Classify query intent with router agent and conditionally delegate to matching branch."""
        router = self.get_agent(router_agent_id)
        if not router:
            raise ValueError(f"Router agent '{router_agent_id}' not found.")

        branch_keys = list(branches.keys())
        router_prompt = (
            f"Analyze this query and classify which category it belongs to: {branch_keys}.\n"
            f"User Query: {query}\n"
            f"Respond with the matching category."
        )
        router_resp = await router.run(router_prompt)
        selected_key = router_resp.output.strip().lower()

        matched_target = None
        for k, target_agent_id in branches.items():
            if k.lower() in selected_key or selected_key in k.lower():
                matched_target = target_agent_id
                break

        if not matched_target:
            matched_target = fallback_agent_id or next(iter(branches.values()))

        target_agent = self.get_agent(matched_target) or router
        target_resp = await target_agent.run(query)
        contributions = [router_resp, target_resp]

        return TeamExecutionResponse(
            query=query,
            final_answer=target_resp.output,
            agent_contributions=contributions,
            drift_detected=False,
            metadata={
                "strategy": "conditional",
                "router": router_agent_id,
                "selected_branch": selected_key,
                "delegated_agent": matched_target,
            },
        )

    async def run_team_workflow(
        self,
        team_id: str,
        query: str,
        require_human_approval: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a team workflow with step-by-step auditing, persistent tracking, and optional human approval."""
        team = self._team_store.get_team(team_id)
        if not team:
            raise ValueError(f"Team '{team_id}' not found.")

        exec_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        strategy = team.get("strategy", "sequential").lower()
        members = team.get("members", [])
        config = team.get("config", {})

        # Record initial execution in store
        self._team_store.create_execution(
            run_id=exec_id,
            team_id=team_id,
            strategy=strategy,
            query=query,
            status="WAITING_APPROVAL" if require_human_approval else "RUNNING",
        )

        if require_human_approval:
            self._team_store.update_execution(
                run_id=exec_id,
                status="WAITING_APPROVAL",
                requires_approval=True,
                approval_status="PENDING",
            )
            return {
                "run_id": exec_id,
                "team_id": team_id,
                "status": "WAITING_APPROVAL",
                "requires_approval": True,
                "approval_status": "PENDING",
                "message": "Workflow paused at human approval checkpoint.",
            }

        t_start = time.perf_counter()
        try:
            if strategy == "parallel":
                resp = await self.run_parallel(
                    query=query,
                    agent_ids=members,
                    aggregator_agent_id=config.get("aggregator_id"),
                )
            elif strategy == "supervisor":
                supervisor_id = config.get("supervisor_id") or (members[0] if members else "")
                worker_ids = [m for m in members if m != supervisor_id]
                resp = await self.run_supervisor(
                    query=query,
                    supervisor_id=supervisor_id,
                    worker_ids=worker_ids,
                    max_steps=config.get("max_steps", 4),
                )
            elif strategy == "conditional":
                router_id = config.get("router_id") or (members[0] if members else "")
                branches = config.get("branches") or {m: m for m in members if m != router_id}
                resp = await self.run_conditional(
                    query=query,
                    router_agent_id=router_id,
                    branches=branches,
                    fallback_agent_id=config.get("fallback_id"),
                )
            else:
                resp = await self.run_team(query=query, agent_pipeline=members if members else None)

            duration = round(time.perf_counter() - t_start, 2)
            steps = [
                {
                    "step_id": f"step_{i + 1}",
                    "agent_id": c.agent_id,
                    "agent_name": c.agent_name,
                    "role": c.role,
                    "output": c.output,
                    "model_used": c.model_used,
                    "confidence": c.confidence,
                }
                for i, c in enumerate(resp.agent_contributions)
            ]

            self._team_store.update_execution(
                run_id=exec_id,
                status="COMPLETED",
                final_answer=resp.final_answer,
                steps=steps,
                requires_approval=False,
            )
            return {
                "run_id": exec_id,
                "team_id": team_id,
                "status": "COMPLETED",
                "strategy": strategy,
                "final_answer": resp.final_answer,
                "steps": steps,
                "duration_seconds": duration,
                "drift_detected": resp.drift_detected,
            }
        except Exception as exc:
            logger.error("Team workflow '%s' failed: %s", exec_id, exc)
            self._team_store.update_execution(
                run_id=exec_id,
                status="FAILED",
                final_answer=f"Execution error: {exc}",
            )
            raise

    async def resume_workflow(
        self,
        run_id: str,
        approved: bool,
        user_comment: str | None = None,
    ) -> dict[str, Any]:
        """Resume or reject a workflow paused at a human approval checkpoint."""
        record = self._team_store.get_execution(run_id)
        if not record:
            raise ValueError(f"Execution run '{run_id}' not found.")

        if not approved:
            self._team_store.update_execution(
                run_id=run_id,
                status="REJECTED",
                final_answer="Workflow rejected by user.",
                approval_status="REJECTED",
                approval_comment=user_comment or "Action was rejected by user.",
            )
            return self._team_store.get_execution(run_id) or {}

        self._team_store.update_execution(
            run_id=run_id,
            status="RUNNING",
            approval_status="APPROVED",
            approval_comment=user_comment or "Action approved by user.",
        )
        return await self.run_team_workflow(
            team_id=record["team_id"],
            query=record["query"],
            require_human_approval=False,
            run_id=run_id,
        )

    def discover_providers(self) -> list[dict[str, Any]]:
        """Report genuine availability and capabilities for supported providers."""
        gemini_has_key = bool(self._settings.google_api_key.get_secret_value())
        openai_has_key = bool(self._settings.openai_api_key.get_secret_value())
        local_weights_exist = os.path.exists(self._settings.llm_local_model_path)

        return [
            {
                "provider": "gemini",
                "name": "Google Gemini",
                "status": "active" if gemini_has_key else "unconfigured",
                "requires_key": True,
                "models": ["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro"],
                "capabilities": ["text", "streaming", "vision", "audio"],
            },
            {
                "provider": "openai",
                "name": "OpenAI",
                "status": "active" if openai_has_key else "unconfigured",
                "requires_key": True,
                "models": ["gpt-4o", "gpt-4o-mini"],
                "capabilities": ["text", "streaming", "vision"],
            },
            {
                "provider": "ollama",
                "name": "Ollama Local Engine",
                "status": "active",
                "requires_key": False,
                "base_url": self._settings.ollama_base_url,
                "models": ["llama3", "mistral", "nomic-embed-text"],
                "capabilities": ["text", "streaming", "local_offline"],
            },
            {
                "provider": "local",
                "name": "Local LLaMA (llama.cpp GGUF)",
                "status": "active" if local_weights_exist else "unconfigured",
                "requires_key": False,
                "models": [self._settings.llm_model_basename],
                "capabilities": ["text", "local_offline", "cpu_gpu"],
            },
        ]
