"""API routes for selective machine unlearning, memory purging, and unlearning validation."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from historyverse.api.dependencies import get_unlearning_engine
from historyverse.core.schemas import (
    TombstoneRecord,
    UnlearningRequest,
    UnlearningValidationReport,
)
from historyverse.services.unlearning.engine import UnlearningEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/unlearning", tags=["Machine Unlearning"])

__all__ = ["router"]


class ValidationRequest(BaseModel):
    """Payload to trigger re-validation of forgotten information."""

    target_concept: str = Field(..., min_length=2)
    forbidden_keywords: list[str] = Field(default_factory=list)


@router.post("/forget", response_model=UnlearningValidationReport)
async def forget_concept(
    request: UnlearningRequest,
    engine: UnlearningEngine = Depends(get_unlearning_engine),
) -> UnlearningValidationReport:
    """Execute unlearning across memory and RAG, apply negative constraints, and validate."""
    try:
        report = await engine.unlearn(request)
        return report
    except Exception as exc:
        logger.error("Unlearning failed for concept '%s': %s", request.target_concept, exc)
        raise HTTPException(status_code=500, detail=f"Unlearning failed: {exc}") from exc


@router.post("/validate", response_model=UnlearningValidationReport)
async def validate_forgetting(
    request: ValidationRequest,
    engine: UnlearningEngine = Depends(get_unlearning_engine),
) -> UnlearningValidationReport:
    """Run adversarial probes and residual checks to certify concept remains unlearned."""
    try:
        report = await engine.validate_concept(
            target_concept=request.target_concept,
            forbidden_keywords=request.forbidden_keywords,
        )
        return report
    except Exception as exc:
        logger.error("Validation failed for concept '%s': %s", request.target_concept, exc)
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}") from exc


@router.get("/tombstones", response_model=list[TombstoneRecord])
async def list_active_tombstones(
    engine: UnlearningEngine = Depends(get_unlearning_engine),
) -> list[TombstoneRecord]:
    """List all registered unlearning tombstones enforcing negative constraints."""
    return engine.tombstone_registry.list_tombstones()


@router.delete("/tombstones/{concept_id}")
async def remove_tombstone(
    concept_id: str,
    engine: UnlearningEngine = Depends(get_unlearning_engine),
) -> dict[str, Any]:
    """Remove an unlearned tombstone, lifting negative constraints."""
    success = engine.tombstone_registry.delete(concept_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Tombstone '{concept_id}' not found in registry."
        )
    return {"status": "removed", "concept_id": concept_id}
