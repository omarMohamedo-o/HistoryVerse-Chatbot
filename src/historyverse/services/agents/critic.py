"""Critic and Drift Verification Agent."""

from __future__ import annotations

import logging
from typing import Any

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.base import AgentResponse, BaseAgent
from historyverse.services.evaluation.hallucination import HallucinationDetector

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
    ) -> None:
        system_prompt = (
            "You are a rigorous historical peer reviewer and critic. Analyze the drafted answer for "
            "anachronisms, factual hallucinations, and unverifiable claims. Verify whether the claims "
            "are grounded in known historical facts."
        )
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Peer Reviewer & Fact Verifier",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
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
