"""Model Context Protocol (MCP) and Agent Harness Domain Schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MCPError",
    "MCPRequest",
    "MCPResourceDefinition",
    "MCPResponse",
    "MCPToolDefinition",
    "ToolCall",
    "ToolResult",
]


class MCPToolDefinition(BaseModel):
    """MCP specification of a callable tool."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")


class MCPResourceDefinition(BaseModel):
    """MCP specification of a readable data resource."""

    model_config = ConfigDict(populate_by_name=True)

    uri: str
    name: str
    description: str = ""
    mime_type: str = Field(default="text/plain", alias="mimeType")


class ToolCall(BaseModel):
    """An instruction emitted by an agent or user to execute a tool."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""


class ToolResult(BaseModel):
    """Standardized output produced by the Agent Harness after executing a tool."""

    tool_name: str
    status: str = "success"  # "success" or "error"
    data: Any = None
    error_message: str | None = None
    duration_ms: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.status == "success"


class MCPError(BaseModel):
    """JSON-RPC 2.0 error body."""

    code: int
    message: str
    data: Any = None


class MCPRequest(BaseModel):
    """Incoming JSON-RPC 2.0 request compliant with Model Context Protocol."""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPResponse(BaseModel):
    """Outgoing JSON-RPC 2.0 response compliant with Model Context Protocol."""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    result: Any = None
    error: MCPError | None = None
