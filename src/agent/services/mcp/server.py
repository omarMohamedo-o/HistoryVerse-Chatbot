"""Model Context Protocol (MCP) Server implementing JSON-RPC 2.0 interface."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.services.mcp.harness import AgentHarness
from agent.services.mcp.schemas import (
    MCPError,
    MCPRequest,
    MCPResourceDefinition,
    MCPResponse,
)

logger = logging.getLogger(__name__)

__all__ = ["MCPServer"]


class MCPServer:
    """Standard Model Context Protocol (MCP) server handling JSON-RPC 2.0 tool requests."""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, harness: AgentHarness | None = None) -> None:
        self.harness = harness if harness is not None else AgentHarness()
        self._resources: list[MCPResourceDefinition] = [
            MCPResourceDefinition(
                uri="historyverse://archives/alexandria",
                name="Alexandria Classical Archives",
                description="Greco-Roman and Ptolemaic historical texts and archaeological surveys",
                mime_type="application/json",
            ),
            MCPResourceDefinition(
                uri="historyverse://registry/tombstones",
                name="Machine Unlearning Tombstones",
                description="Catalogue of retracted and unlearned historical assertions",
                mime_type="application/json",
            ),
        ]

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Dispatch incoming MCP JSON-RPC 2.0 requests to dedicated method handlers."""
        method = request.method
        params = request.params

        logger.info("Handling MCP method call: %s (id=%s)", method, request.id)

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            elif method == "resources/list":
                result = await self._handle_resources_list()
            elif method == "resources/read":
                result = await self._handle_resources_read(params)
            elif method == "ping":
                result = {}
            else:
                return MCPResponse(
                    id=request.id,
                    error=MCPError(
                        code=-32601, message=f"Method '{method}' not found in MCP server."
                    ),
                )

            return MCPResponse(id=request.id, result=result)
        except Exception as exc:
            logger.error("Unhandled error processing MCP method %s: %s", method, exc)
            return MCPResponse(
                id=request.id,
                error=MCPError(code=-32603, message=f"Internal server error: {exc}"),
            )

    async def _handle_initialize(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": False, "listChanged": True},
            },
            "serverInfo": {
                "name": "HistoryVerse-MCP-Server",
                "version": "2.0.0",
            },
        }

    async def _handle_tools_list(self) -> dict[str, Any]:
        tools = self.harness.registry.get_mcp_definitions()
        return {"tools": [t.model_dump() for t in tools]}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        tool_name = params.get("name") or params.get("tool_name", "")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Parameter 'name' is required for tools/call.")

        tool_result = await self.harness.call_tool(tool_name, arguments)
        content_text = (
            json.dumps(tool_result.data) if tool_result.is_success else tool_result.error_message
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": str(content_text),
                }
            ],
            "isError": not tool_result.is_success,
            "duration_ms": tool_result.duration_ms,
        }

    async def _handle_resources_list(self) -> dict[str, Any]:
        return {"resources": [r.model_dump() for r in self._resources]}

    async def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri", "")
        if "alexandria" in uri:
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(
                            {
                                "archive": "Alexandria Classical Collection",
                                "coverage": "Ptolemaic, Roman, and Byzantine Egypt",
                                "monuments": ["Serapeum", "Pharos", "Library of Alexandria"],
                            }
                        ),
                    }
                ]
            }
        elif "tombstones" in uri:
            from agent.services.unlearning.registry import get_tombstone_registry

            tombstones = get_tombstone_registry().list_tombstones()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps([t.model_dump() for t in tombstones]),
                    }
                ]
            }

        raise ValueError(f"Resource with URI '{uri}' not found.")
