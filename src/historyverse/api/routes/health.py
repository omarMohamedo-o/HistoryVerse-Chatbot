"""Health check endpoint."""

from __future__ import annotations

import time
from fastapi import APIRouter, Depends
from historyverse import __version__
from historyverse.api.dependencies import get_app_settings
from historyverse.config.settings import Settings
from historyverse.core.schemas import HealthResponse

router = APIRouter()
_start_time = time.time()

__all__ = ["router"]


@router.get("/health", response_model=HealthResponse)
async def check_health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Return health status, active provider, model name, and uptime."""
    return HealthResponse(
        status="ok",
        version=__version__,
        model_provider=settings.llm_provider,
        active_model=settings.llm_model_name,
        uptime_seconds=round(time.time() - _start_time, 2),
    )
