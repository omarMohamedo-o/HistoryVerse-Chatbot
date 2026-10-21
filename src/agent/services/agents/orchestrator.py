"""Multi-Agent Team Orchestrator for collaborative or individual model execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.core.base import BaseLLMProvider
from agent.services.agents.base import AgentResponse, BaseAgent, TeamExecutionResponse
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

    def __init__(self, settings: Settings | None = None) -> None:
        from agent.config.settings import get_settings

        self._settings = settings or get_settings()
        self._registry: dict[str, BaseAgent] = {}

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
        return agent

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
