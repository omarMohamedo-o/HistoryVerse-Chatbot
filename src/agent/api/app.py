"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent import __version__
from agent.api.middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from agent.api.routes import (
    agents_router,
    ar_router,
    chat_router,
    drift_router,
    evaluate_router,
    health_router,
    ingest_router,
    legacy_ar_router,
    mcp_router,
    unlearning_router,
    vision_router,
    voice_router,
)
from agent.config.settings import get_settings

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle events: startup and shutdown."""
    settings = get_settings()
    logger.info(
        "Agent starting up: version=%s provider=%s model=%s",
        __version__,
        settings.llm_provider,
        settings.llm_model_name,
    )
    yield
    logger.info("Agent shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Agent API",
        description="Domain-agnostic, multi-provider multimodal AI agent microservices",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health_router, tags=["health"])
    app.include_router(chat_router, prefix="/v1", tags=["chat"])
    app.include_router(ar_router)
    app.include_router(legacy_ar_router)
    app.include_router(ingest_router, prefix="/v1", tags=["ingest"])
    app.include_router(voice_router, prefix="/v1", tags=["voice", "sts"])
    app.include_router(vision_router, prefix="/v1", tags=["vision"])
    app.include_router(evaluate_router, prefix="/v1", tags=["evaluation"])
    app.include_router(drift_router, prefix="/v1", tags=["drift"])
    app.include_router(agents_router)
    app.include_router(unlearning_router)
    app.include_router(mcp_router)

    return app
