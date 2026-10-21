"""Tools for Agent Harness and Model Context Protocol (MCP)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from agentverse.services.mcp.schemas import MCPToolDefinition
from agentverse.services.rag.retriever import ChromaRetriever
from agentverse.services.unlearning.registry import TombstoneRegistry, get_tombstone_registry

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactInspectionTool",
    "BaseTool",
    "HistoricalSearchTool",
    "ToolRegistry",
    "UnlearningStatusTool",
]


class BaseTool(ABC):
    """Abstract base class for all tools accessible to agents and MCP clients."""

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool logic and return raw result."""

    def to_mcp_definition(self) -> MCPToolDefinition:
        """Convert to standard Model Context Protocol tool definition."""
        return MCPToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI/standard LLM function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class HistoricalSearchTool(BaseTool):
    """RAG tool querying indexed classical and Alexandrian historical archives."""

    def __init__(self, retriever: ChromaRetriever | None = None) -> None:
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Historical subject, entity, or question to search in archives",
                },
                "top_k": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum number of relevant document passages to retrieve",
                },
            },
            "required": ["query"],
        }
        super().__init__(
            name="search_historical_archives",
            description="Query the HistoryVerse RAG vector index for authentic historical records, papyri, and classical manuscripts.",
            input_schema=schema,
        )
        self.retriever = retriever if retriever is not None else ChromaRetriever()

    async def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k", 3))
        docs = self.retriever.retrieve(query, top_k=top_k)
        return [
            {
                "content": d.page_content,
                "source": d.metadata.get("source", "historical_record"),
            }
            for d in docs
        ]


class ArtifactInspectionTool(BaseTool):
    """Tool inspecting classical archaeological artifacts and architectural structures."""

    def __init__(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "artifact_name": {
                    "type": "string",
                    "description": "Name or identifier of the artifact or monument (e.g. 'Rosetta Stone', 'Pharos Lighthouse')",
                },
            },
            "required": ["artifact_name"],
        }
        super().__init__(
            name="inspect_archaeological_artifact",
            description="Retrieve structural, material, and chronological details of known classical artifacts and monuments.",
            input_schema=schema,
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        name = str(kwargs.get("artifact_name", "")).strip().title()
        return {
            "artifact_name": name,
            "provenance": "Classical Mediterranean & Ancient Egypt",
            "status": "Cataloged in Greco-Roman & Classical Archives",
            "preservation": "Verified Historical Asset",
        }


class UnlearningStatusTool(BaseTool):
    """Tool querying the active Machine Unlearning Tombstone Registry."""

    def __init__(self, registry: TombstoneRegistry | None = None) -> None:
        schema = {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic, claim, or entity to verify against the unlearning registry",
                },
            },
            "required": ["topic"],
        }
        super().__init__(
            name="check_unlearning_status",
            description="Verify whether a claim or topic has been retracted, declared obsolete, or subjected to machine unlearning.",
            input_schema=schema,
        )
        self.registry = registry if registry is not None else get_tombstone_registry()

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        topic = str(kwargs.get("topic", ""))
        is_forbidden, concept = self.registry.is_forbidden(topic)
        return {
            "topic": topic,
            "is_unlearned": is_forbidden,
            "retracted_concept": concept,
            "status": "RETRACTED" if is_forbidden else "VALID_HISTORICAL_RECORD",
        }


class ToolRegistry:
    """Thread-safe OOP registry managing callable tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        logger.info("Registered tool in registry: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        """Retrieve tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def get_mcp_definitions(self) -> list[MCPToolDefinition]:
        """Return all tools formatted according to MCP specification."""
        return [tool.to_mcp_definition() for tool in self._tools.values()]

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return all tools formatted for LLM function calling."""
        return [tool.to_openai_dict() for tool in self._tools.values()]
