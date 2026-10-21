"""API route endpoints."""

from agentverse.api.routes.agents import router as agents_router
from agentverse.api.routes.ar import (
    ar_router,
    historyverse_ar_mode_router,
    legacy_ar_router,
)
from agentverse.api.routes.chat import router as chat_router
from agentverse.api.routes.drift import router as drift_router
from agentverse.api.routes.evaluate import router as evaluate_router
from agentverse.api.routes.health import router as health_router
from agentverse.api.routes.ingest import router as ingest_router
from agentverse.api.routes.mcp import router as mcp_router
from agentverse.api.routes.unlearning import router as unlearning_router
from agentverse.api.routes.vision import router as vision_router
from agentverse.api.routes.voice import router as voice_router

__all__ = [
    "agents_router",
    "ar_router",
    "chat_router",
    "drift_router",
    "evaluate_router",
    "health_router",
    "historyverse_ar_mode_router",
    "ingest_router",
    "legacy_ar_router",
    "mcp_router",
    "unlearning_router",
    "vision_router",
    "voice_router",
]
