"""Agent Harness implementing defensive tool execution and ReAct observation loops."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from agentverse.services.agents.base import AgentResponse, BaseAgent
from agentverse.services.mcp.schemas import ToolCall, ToolResult
from agentverse.services.mcp.tools import (
    ArtifactInspectionTool,
    HistoricalSearchTool,
    ToolRegistry,
    UnlearningStatusTool,
)

logger = logging.getLogger(__name__)

__all__ = ["AgentHarness"]


class AgentHarness:
    """Execution harness providing sandboxed tool execution, error handling, and ReAct loop coordination."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        if registry is not None:
            self.registry = registry
        else:
            reg = ToolRegistry()
            reg.register(HistoricalSearchTool())
            reg.register(ArtifactInspectionTool())
            reg.register(UnlearningStatusTool())
            self.registry = reg

        self._audit_log: list[ToolResult] = []

    def get_audit_log(self) -> list[ToolResult]:
        """Return chronological record of all harness tool executions."""
        return list(self._audit_log)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool with defensive sandboxing, validation, and latency tracking."""
        start_time = time.perf_counter()
        tool = self.registry.get(tool_name)

        if tool is None:
            result = ToolResult(
                tool_name=tool_name,
                status="error",
                error_message=f"Tool '{tool_name}' is not recognized in the harness registry.",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
            self._audit_log.append(result)
            return result

        # Validate required arguments against tool schema
        required_props = tool.input_schema.get("required", [])
        missing = [param for param in required_props if param not in arguments]
        if missing:
            result = ToolResult(
                tool_name=tool_name,
                status="error",
                error_message=f"Missing required parameters for tool '{tool_name}': {missing}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
            self._audit_log.append(result)
            return result

        try:
            raw_output = await tool.execute(**arguments)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result = ToolResult(
                tool_name=tool_name,
                status="success",
                data=raw_output,
                duration_ms=elapsed_ms,
            )
            self._audit_log.append(result)
            logger.info("Harness executed [%s] in %.2fms", tool_name, elapsed_ms)
            return result
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result = ToolResult(
                tool_name=tool_name,
                status="error",
                error_message=f"Execution error in tool '{tool_name}': {exc}",
                duration_ms=elapsed_ms,
            )
            self._audit_log.append(result)
            logger.error("Harness error in [%s]: %s", tool_name, exc)
            return result

    def format_tools_prompt(self) -> str:
        """Render available tools into prompt guidance for LLM agents."""
        tool_descriptions: list[str] = []
        for t in self.registry.list_tools():
            schema_str = json.dumps(t.input_schema)
            tool_descriptions.append(
                f"- **{t.name}**: {t.description}\n  Parameters: `{schema_str}`"
            )
        return "\n".join(tool_descriptions)

    def extract_tool_call(self, text: str) -> ToolCall | None:
        """Parse structured tool call requests from agent output."""
        # Try JSON format: {"tool": "name", "arguments": {...}}
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = json_match.group(1) if json_match else text

        try:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1:
                obj = json.loads(candidate[start : end + 1])
                tool_name = obj.get("tool") or obj.get("tool_name") or obj.get("name")
                args = obj.get("arguments") or obj.get("params") or {}
                if tool_name and self.registry.get(str(tool_name)):
                    return ToolCall(tool_name=str(tool_name), arguments=args)
        except Exception:
            pass

        # Try Action syntax: Action: tool_name(param="value")
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\((.*?)\)", text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            raw_args = action_match.group(2)
            if self.registry.get(tool_name):
                # Simple keyword extraction
                args_dict = {}
                for part in raw_args.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        args_dict[k.strip()] = v.strip().strip("'\"")
                return ToolCall(tool_name=tool_name, arguments=args_dict)

        return None

    async def execute_agent_with_harness(
        self,
        agent: BaseAgent,
        query: str,
        context: dict[str, Any] | None = None,
        max_turns: int = 3,
    ) -> AgentResponse:
        """Run an autonomous agent inside the execution harness with tool-calling capabilities."""
        tools_guide = self.format_tools_prompt()
        harness_system_prompt = (
            f"{agent.build_effective_system_prompt()}\n\n"
            f"[AVAILABLE TOOLS]:\n{tools_guide}\n\n"
            f"[TOOL USAGE INSTRUCTION]:\n"
            f"If you need verified evidence to answer, emit a tool call in this JSON block:\n"
            f'```json\n{{"tool": "<tool_name>", "arguments": {{...}}}}\n```\n'
            f"When you have sufficient information to conclude, output your final synthesis directly."
        )

        scratchpad = f"Question: {query}"
        tools_invoked: list[str] = []

        for turn in range(max_turns):
            prompt = f"{harness_system_prompt}\n\nScratchpad:\n{scratchpad}\n\nYour reasoning or tool call:"
            output = await agent.llm_provider.generate(prompt)

            tool_call = self.extract_tool_call(output)
            if tool_call is None:
                # Agent concluded and provided final answer
                return AgentResponse(
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    role=agent.role,
                    output=output,
                    model_used=agent.llm_provider.model_name,
                    confidence=0.95,
                    metadata={
                        "tools_invoked": tools_invoked,
                        "turns_taken": turn + 1,
                        "harness_assisted": len(tools_invoked) > 0,
                    },
                )

            # Execute tool through sandboxed harness
            tools_invoked.append(tool_call.tool_name)
            tool_result = await self.call_tool(tool_call.tool_name, tool_call.arguments)
            observation_data = (
                tool_result.data
                if tool_result.is_success
                else f"Error: {tool_result.error_message}"
            )

            scratchpad += (
                f"\n\nThought: Calling tool {tool_call.tool_name} with {tool_call.arguments}\n"
                f"Observation: {json.dumps(observation_data)}"
            )

        # Fallback if max turns reached without direct conclusion
        final_prompt = (
            f"{agent.build_effective_system_prompt()}\n\n"
            f"Observations Gathered:\n{scratchpad}\n\n"
            f"Based on these observations, synthesize your final historical answer to: {query}"
        )
        final_answer = await agent.llm_provider.generate(final_prompt)

        return AgentResponse(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            role=agent.role,
            output=final_answer,
            model_used=agent.llm_provider.model_name,
            confidence=0.90,
            metadata={
                "tools_invoked": tools_invoked,
                "turns_taken": max_turns,
                "harness_assisted": True,
            },
        )
