"""Health and readiness checks with Prometheus metrics exporter."""

from __future__ import annotations

import os
import shutil
import time

from fastapi import APIRouter, Depends, Response

from historyverse import __version__
from historyverse.api.dependencies import get_app_settings, get_llm_provider
from historyverse.config.settings import Settings
from historyverse.core.base import BaseLLMProvider
from historyverse.core.schemas import HealthResponse

router = APIRouter()
_start_time = time.time()

__all__ = ["router"]


@router.get("/health", response_model=HealthResponse)
@router.get("/health/live", response_model=HealthResponse)
async def check_liveness(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Liveness probe indicating that the HTTP service is responsive."""
    return HealthResponse(
        status="ok",
        version=__version__,
        model_provider=settings.llm_provider,
        active_model=settings.llm_model_name,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get("/health/ready")
async def check_readiness(
    settings: Settings = Depends(get_app_settings),
    provider: BaseLLMProvider = Depends(get_llm_provider),
) -> dict[str, str | bool | dict]:
    """Readiness probe checking LLM provider, vector store, and filesystem health."""
    provider_available = await provider.is_available()
    chroma_accessible = os.path.exists(settings.chroma_persist_dir)

    _, _, free = shutil.disk_usage("/")
    free_gb = round(free / (1024**3), 2)
    disk_healthy = free_gb > settings.min_disk_free_gb

    all_healthy = provider_available and disk_healthy

    return {
        "ready": all_healthy,
        "checks": {
            "llm_provider": provider_available,
            "chroma_dir_exists": chroma_accessible,
            "disk_healthy": disk_healthy,
            "free_disk_gb": free_gb,
        },
    }


@router.get("/metrics")
async def export_prometheus_metrics() -> Response:
    """Expose Prometheus formatted metrics for scrapers."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")
