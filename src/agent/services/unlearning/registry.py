"""Tombstone Registry for selective machine unlearning and negative constraints."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

from agent.core.schemas import TombstoneRecord

logger = logging.getLogger(__name__)

__all__ = ["TombstoneRegistry", "get_tombstone_registry"]


class TombstoneRegistry:
    """Thread-safe registry of active tombstones representing unlearned knowledge.

    Maintains negative constraints that prevent models from retrieving or
    generating retracted, hallucinated, or deliberately forgotten historical facts.
    """

    def __init__(self) -> None:
        self._tombstones: dict[str, TombstoneRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        target_concept: str,
        forbidden_keywords: list[str] | None = None,
        affected_agents: list[str] | None = None,
        reason: str = "retraction",
    ) -> TombstoneRecord:
        """Register a new unlearned concept tombstone."""
        concept_id = f"tombstone-{uuid.uuid4().hex[:8]}"
        keywords = forbidden_keywords or [target_concept]
        record = TombstoneRecord(
            concept_id=concept_id,
            target_concept=target_concept,
            forbidden_keywords=[k.lower().strip() for k in keywords if k.strip()],
            affected_agents=affected_agents or [],
            reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._tombstones[concept_id] = record

        logger.info(
            "Registered unlearning tombstone [%s] for concept: %s (keywords: %s)",
            concept_id,
            target_concept,
            record.forbidden_keywords,
        )
        return record

    def get(self, concept_id: str) -> TombstoneRecord | None:
        """Retrieve tombstone record by ID."""
        with self._lock:
            return self._tombstones.get(concept_id)

    def list_tombstones(self) -> list[TombstoneRecord]:
        """Return list of all active unlearned tombstones."""
        with self._lock:
            return list(self._tombstones.values())

    def delete(self, concept_id: str) -> bool:
        """Remove a tombstone (reinstating the concept if appropriate)."""
        with self._lock:
            removed = self._tombstones.pop(concept_id, None)
        return removed is not None

    def is_forbidden(self, text: str, agent_id: str | None = None) -> tuple[bool, str | None]:
        """Check if a text snippet contains tokens from any active unlearned tombstone."""
        lower_text = text.lower()
        with self._lock:
            tombstones = list(self._tombstones.values())

        for tb in tombstones:
            agent_applies = not tb.affected_agents or (
                agent_id is not None and agent_id in tb.affected_agents
            )
            if not agent_applies:
                continue

            for kw in tb.forbidden_keywords:
                if kw in lower_text:
                    return True, tb.target_concept

        return False, None

    def get_unlearning_guard_prompt(self, agent_id: str | None = None) -> str:
        """Generate negative constraint prompt instructions for active tombstones."""
        with self._lock:
            applicable = [
                tb
                for tb in self._tombstones.values()
                if not tb.affected_agents
                or (agent_id is not None and agent_id in tb.affected_agents)
            ]

        if not applicable:
            return ""

        concept_list = ", ".join(f"'{tb.target_concept}'" for tb in applicable)
        return (
            f"[MANDATORY UNLEARNING DIRECTIVE]: The following topics/claims have been retracted, "
            f"unlearned, and declared obsolete: {concept_list}. You must NEVER affirm, detail, "
            f"or speculate on these topics. If directly asked about any of them, state clearly: "
            f"'I do not have verified historical records regarding this topic, as the claim has been retracted.'"
        )


_GLOBAL_TOMBSTONE_REGISTRY = TombstoneRegistry()


def get_tombstone_registry() -> TombstoneRegistry:
    """Return the global singleton TombstoneRegistry."""
    return _GLOBAL_TOMBSTONE_REGISTRY
