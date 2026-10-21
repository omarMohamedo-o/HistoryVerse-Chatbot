"""API routes for Model Context Protocol (MCP) and Agent Harness tool execution."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from historyverse.api.dependencies import get_llm_provider, get_retriever
from historyverse.config.settings import get_settings
from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.historical import HistoryVerseAgent
from historyverse.services.mcp.harness import AgentHarness
from historyverse.services.mcp.schemas import (
    MCPRequest,
    MCPResponse,
    MCPToolDefinition,
    ToolResult,
)
from historyverse.services.mcp.server import MCPServer
from historyverse.services.mcp.tools import (
    ArtifactInspectionTool,
    HistoricalSearchTool,
    ToolRegistry,
    UnlearningStatusTool,
)
from historyverse.services.rag.retriever import ChromaRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/mcp", tags=["Model Context Protocol & Tools"])

__all__ = ["router"]

# Singleton harness and server instance for API lifecycle
_TOOL_REGISTRY: ToolRegistry | None = None
_HARNESS: AgentHarness | None = None
_MCP_SERVER: MCPServer | None = None


def get_mcp_server(retriever: ChromaRetriever = Depends(get_retriever)) -> MCPServer:
    global _TOOL_REGISTRY, _HARNESS, _MCP_SERVER
    if _MCP_SERVER is None:
        _TOOL_REGISTRY = ToolRegistry()
        _TOOL_REGISTRY.register(HistoricalSearchTool(retriever=retriever))
        _TOOL_REGISTRY.register(ArtifactInspectionTool())
        _TOOL_REGISTRY.register(UnlearningStatusTool())
        _HARNESS = AgentHarness(registry=_TOOL_REGISTRY)
        _MCP_SERVER = MCPServer(harness=_HARNESS)
    return _MCP_SERVER


class ToolExecutionRequest(BaseModel):
    """Payload to invoke a tool directly through the harness."""

    arguments: dict[str, Any] = Field(default_factory=dict)


class HarnessAgentRunRequest(BaseModel):
    """Payload to execute an agent equipped with tool-calling harness capabilities."""

    query: str = Field(..., min_length=1)
    agent_id: str = Field(default_factory=lambda: get_settings().agent_persona_id)
    max_turns: int = Field(default=3, ge=1, le=5)


@router.post("", response_model=MCPResponse)
async def handle_mcp_jsonrpc(
    request: MCPRequest,
    server: MCPServer = Depends(get_mcp_server),
) -> MCPResponse:
    """Standard Model Context Protocol JSON-RPC 2.0 endpoint."""
    return await server.handle_request(request)


@router.get("/tools", response_model=list[MCPToolDefinition])
async def list_tools(
    server: MCPServer = Depends(get_mcp_server),
) -> list[MCPToolDefinition]:
    """List all available tools in the MCP specification format."""
    return server.harness.registry.get_mcp_definitions()


@router.post("/tools/{tool_name}/execute", response_model=ToolResult)
async def execute_tool(
    tool_name: str,
    payload: ToolExecutionRequest,
    server: MCPServer = Depends(get_mcp_server),
) -> ToolResult:
    """Execute a single tool through the sandboxed Agent Harness."""
    result = await server.harness.call_tool(tool_name, payload.arguments)
    if not result.is_success:
        raise HTTPException(status_code=400, detail=result.error_message)
    return result


@router.post("/agent/run")
async def run_harness_agent(
    payload: HarnessAgentRunRequest,
    llm: BaseLLMProvider = Depends(get_llm_provider),
    server: MCPServer = Depends(get_mcp_server),
) -> dict[str, Any]:
    """Execute an agent equipped with tool-calling capabilities via the harness."""
    agent = HistoryVerseAgent(llm_provider=llm, agent_id=payload.agent_id)
    response = await server.harness.execute_agent_with_harness(
        agent=agent,
        query=payload.query,
        max_turns=payload.max_turns,
    )
    return response.model_dump()
