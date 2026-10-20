"""Multi-Agent Team Orchestrator for collaborative or individual model execution."""

from __future__ import annotations

import logging
from typing import Any

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.base import AgentResponse, BaseAgent, TeamExecutionResponse

logger = logging.getLogger(__name__)

__all__ = ["AgentOrchestrator", "CustomAgent"]


class CustomAgent(BaseAgent):
    """Dynamically generated agent with user-defined prompt and assigned LLM model."""

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
        parts.append(f"Query:\n{query}")
        prompt = "\n\n".join(parts)

        output = await self.llm_provider.generate(prompt)

        if session_id:
            self.memory.add_exchange(session_id, query, output)
        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output=output,
            model_used=self.llm_provider.model_name,
            confidence=0.9,
            metadata=context or {},
        )


class AgentOrchestrator:
    """Orchestrates individual agent invocations or collaborative multi-agent team pipelines."""

    def __init__(self) -> None:
        self._registry: dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent instance in the orchestrator registry."""
        self._registry[agent.agent_id] = agent
        logger.info("Registered agent: %s (%s)", agent.name, agent.agent_id)

    def get_agent(self, agent_id: str) -> BaseAgent | None:
        """Retrieve an agent by its unique identifier."""
        return self._registry.get(agent_id)

    def list_agents(self) -> list[dict[str, str]]:
        """Return metadata for all registered agents."""
        return [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "role": a.role,
                "model_used": a.llm_provider.model_name,
            }
            for a in self._registry.values()
        ]

    def create_custom_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        system_prompt: str,
        llm_provider: BaseLLMProvider,
    ) -> CustomAgent:
        """Spawn a new custom agent model dynamically and register it."""
        agent = CustomAgent(
            agent_id=agent_id,
            name=name,
            role=role,
            system_prompt=system_prompt,
            llm_provider=llm_provider,
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
        pipeline = agent_pipeline or [
            "archival-researcher",
            "historyverse-mora",
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
            elif agent.agent_id == "historyverse-mora":
                shared_context["draft_answer"] = step_response.output

        # Final answer is taken from the primary domain agent (Mora) or the last answering agent
        mora_contrib = next((c for c in contributions if c.agent_id == "historyverse-mora"), None)
        final_answer = (
            mora_contrib.output
            if mora_contrib
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
