"""Critic and Drift Verification Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentverse.core.base import BaseLLMProvider
from agentverse.services.agents.base import AgentResponse, BaseAgent
from agentverse.services.evaluation.hallucination import HallucinationDetector

if TYPE_CHECKING:
    from agentverse.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["CriticAgent"]


class CriticAgent(BaseAgent):
    """Agent that critiques historical claims, detects hallucinations, and assesses drift."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        hallucination_detector: HallucinationDetector | None = None,
        agent_id: str = "historical-critic",
        name: str = "Critic",
        role: str = "Peer Reviewer & Fact Verifier",
        system_prompt: str | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        effective_prompt = system_prompt or (
            "You are a rigorous peer reviewer and critic. Analyze the drafted answer for "
            "factual hallucinations, inconsistencies, and unverifiable claims. Verify whether the claims "
            "are grounded in provided evidence and facts."
        )
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=role,
            system_prompt=effective_prompt,
            llm_provider=llm_provider,
            memory=memory,
        )
        self.detector = hallucination_detector or HallucinationDetector()

    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        draft_answer = (context or {}).get("draft_answer", "")
        evidence = (context or {}).get("evidence", [])

        h_score = self.detector.score(query, draft_answer, evidence) if draft_answer else 0.0

        system_base = self.build_effective_system_prompt()
        prompt = (
            f"{system_base}\n\nUser Question:\n{query}\n\nDrafted Historical Answer:\n{draft_answer}\n\n"
            f"Hallucination Risk Score: {h_score:.2f}\n\nProvide your brief verification and critique:"
        )

        critique = await self.llm_provider.generate(prompt)

        session_id = (context or {}).get("session_id")
        if session_id:
            self.memory.add_exchange(session_id, query, critique)

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output=critique,
            model_used=self.llm_provider.model_name,
            confidence=round(1.0 - h_score, 2),
            metadata={"hallucination_score": h_score, "is_hallucinated": h_score > 0.5},
        )
