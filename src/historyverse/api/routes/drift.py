"""Drift monitoring and diagnostic endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from historyverse.api.dependencies import get_app_settings, get_drift_detector
from historyverse.config.settings import Settings
from historyverse.core.schemas import DriftReport
from historyverse.services.llm.drift import DriftDetector

logger = logging.getLogger(__name__)
router = APIRouter()

__all__ = ["router"]


@router.get("/drift/status")
async def get_drift_status(
    settings: Settings = Depends(get_app_settings),
) -> dict[str, str | float | int]:
    """Return active drift thresholds, max retries, fallback model, and action."""
    return {
        "confidence_threshold": settings.drift_confidence_threshold,
        "max_retries": settings.drift_max_retries,
        "fallback_model": settings.drift_fallback_model,
        "drift_action": settings.drift_action.value,
        "docker_rebuild_command": settings.docker_rebuild_command,
    }


@router.post("/drift/test", response_model=DriftReport)
async def test_drift_trigger(
    prompt: str,
    detector: DriftDetector = Depends(get_drift_detector),
) -> DriftReport:
    """Simulate query generation to test drift detection and retry behaviors."""
    _, report = await detector.generate_with_drift_protection(prompt)
    return report


@router.post("/drift/rebuild")
async def trigger_rebuild(
    detector: DriftDetector = Depends(get_drift_detector),
) -> dict[str, str | bool]:
    """Manually trigger the Docker container rebuild hook."""
    success = detector.trigger_docker_rebuild()
    return {"status": "triggered" if success else "failed", "success": success}


@router.post("/drift/reingest")
async def trigger_reingest(
    detector: DriftDetector = Depends(get_drift_detector),
) -> dict[str, str | bool]:
    """Trigger automated document re-ingestion to eliminate context drift."""
    success = detector.trigger_reingest()
    return {"status": "reingested" if success else "failed", "success": success}
