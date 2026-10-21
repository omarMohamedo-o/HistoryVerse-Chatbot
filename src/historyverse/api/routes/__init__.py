"""API route endpoints."""

from historyverse.api.routes.agents import router as agents_router
from historyverse.api.routes.chat import router as chat_router
from historyverse.api.routes.drift import router as drift_router
from historyverse.api.routes.evaluate import router as evaluate_router
from historyverse.api.routes.health import router as health_router
from historyverse.api.routes.ingest import router as ingest_router
from historyverse.api.routes.mcp import router as mcp_router
from historyverse.api.routes.unlearning import router as unlearning_router
from historyverse.api.routes.vision import router as vision_router
from historyverse.api.routes.voice import router as voice_router

__all__ = [
    "agents_router",
    "chat_router",
    "drift_router",
    "evaluate_router",
    "health_router",
    "ingest_router",
    "mcp_router",
    "unlearning_router",
    "vision_router",
    "voice_router",
]
