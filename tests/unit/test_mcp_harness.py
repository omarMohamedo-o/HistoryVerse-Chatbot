"""Unit tests for Model Context Protocol (MCP), Tools, and the Agent Harness."""

import pytest

from agent.services.agents.template import TemplatedAgent
from agent.services.mcp.harness import AgentHarness
from agent.services.mcp.schemas import MCPRequest
from agent.services.mcp.server import MCPServer
from agent.services.mcp.tools import (
    ArtifactInspectionTool,
    HistoricalSearchTool,
    ToolRegistry,
    UnlearningStatusTool,
)
from tests.conftest import MockLLMProvider


@pytest.mark.asyncio
async def test_tools_execution():
    # 1. Historical Search Tool
    search_tool = HistoricalSearchTool()
    res = await search_tool.execute(query="Pharos Lighthouse", top_k=2)
    assert isinstance(res, list)

    # 2. Artifact Inspection Tool
    artifact_tool = ArtifactInspectionTool()
    art_res = await artifact_tool.execute(artifact_name="Rosetta Stone")
    assert art_res["artifact_name"] == "Rosetta Stone"
    assert "Verified" in art_res["preservation"]

    # 3. Unlearning Status Tool
    unlearn_tool = UnlearningStatusTool()
    status_res = await unlearn_tool.execute(topic="Pharos")
    assert status_res["is_unlearned"] is False


def test_tool_registry_and_schema_conversion():
    reg = ToolRegistry()
    tool = ArtifactInspectionTool()
    reg.register(tool)

    assert reg.get("inspect_archaeological_artifact") is tool
    assert len(reg.list_tools()) == 1

    # Check MCP definition
    mcp_defs = reg.get_mcp_definitions()
    assert len(mcp_defs) == 1
    assert mcp_defs[0].name == "inspect_archaeological_artifact"
    assert "required" in mcp_defs[0].input_schema

    # Check OpenAI function tool schema
    openai_tools = reg.get_openai_tools()
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "inspect_archaeological_artifact"


@pytest.mark.asyncio
async def test_agent_harness_defensive_engineering():
    harness = AgentHarness()

    # 1. Successful execution
    result = await harness.call_tool(
        "inspect_archaeological_artifact", {"artifact_name": "Alexander Mosaic"}
    )
    assert result.is_success is True
    assert result.duration_ms >= 0.0
    assert result.data["artifact_name"] == "Alexander Mosaic"

    # 2. Missing required parameters
    bad_result = await harness.call_tool("inspect_archaeological_artifact", {})
    assert bad_result.is_success is False
    assert "Missing required parameters" in bad_result.error_message

    # 3. Unrecognized tool
    unknown_result = await harness.call_tool("nonexistent_teleport_tool", {})
    assert unknown_result.is_success is False
    assert "not recognized" in unknown_result.error_message

    # 4. Check audit log
    audit = harness.get_audit_log()
    assert len(audit) == 3


@pytest.mark.asyncio
async def test_agent_harness_react_loop():
    harness = AgentHarness()
    mock_llm = MockLLMProvider()
    agent = TemplatedAgent.create("historical_guide", llm_provider=mock_llm)

    response = await harness.execute_agent_with_harness(
        agent=agent,
        query="Tell me about the Serapeum of Alexandria",
        max_turns=2,
    )

    assert response.agent_name == "Mora"
    assert "Serapeum" in response.output
    assert "turns_taken" in response.metadata


@pytest.mark.asyncio
async def test_mcp_server_jsonrpc_protocol():
    server = MCPServer()

    # 1. initialize
    init_req = MCPRequest(id=1, method="initialize", params={})
    init_res = await server.handle_request(init_req)
    assert init_res.error is None
    assert init_res.result["protocolVersion"] == "2024-11-05"

    # 2. tools/list
    list_req = MCPRequest(id=2, method="tools/list", params={})
    list_res = await server.handle_request(list_req)
    assert list_res.error is None
    assert len(list_res.result["tools"]) >= 3

    # 3. tools/call
    call_req = MCPRequest(
        id=3,
        method="tools/call",
        params={
            "name": "inspect_archaeological_artifact",
            "arguments": {"artifact_name": "Sphinx"},
        },
    )
    call_res = await server.handle_request(call_req)
    assert call_res.error is None
    assert call_res.result["isError"] is False
    assert "Sphinx" in call_res.result["content"][0]["text"]

    # 4. resources/list and resources/read
    res_list_req = MCPRequest(id=4, method="resources/list", params={})
    res_list_res = await server.handle_request(res_list_req)
    assert len(res_list_res.result["resources"]) >= 2

    res_read_req = MCPRequest(
        id=5, method="resources/read", params={"uri": "historyverse://archives/alexandria"}
    )
    res_read_res = await server.handle_request(res_read_req)
    assert "Alexandria Classical Collection" in res_read_res.result["contents"][0]["text"]

    # 5. Invalid method error code -32601
    bad_req = MCPRequest(id=6, method="unknown_method_xyz", params={})
    bad_res = await server.handle_request(bad_req)
    assert bad_res.error is not None
    assert bad_res.error.code == -32601


@pytest.mark.asyncio
async def test_mcp_server_additional_methods_and_errors():
    server = MCPServer()

    # ping method
    ping_req = MCPRequest(id=7, method="ping")
    res_ping = await server.handle_request(ping_req)
    assert res_ping.error is None
    assert res_ping.result == {}

    # tools/call with missing tool name
    empty_tool_req = MCPRequest(id=8, method="tools/call", params={})
    res_empty = await server.handle_request(empty_tool_req)
    assert res_empty.error is not None
    assert "required" in res_empty.error.message

    # resources/read with tombstones
    tomb_req = MCPRequest(
        id=9, method="resources/read", params={"uri": "historyverse://tombstones"}
    )
    res_tomb = await server.handle_request(tomb_req)
    assert res_tomb.error is None
    assert len(res_tomb.result["contents"]) == 1

    # resources/read with unknown uri
    bad_res_req = MCPRequest(id=10, method="resources/read", params={"uri": "unknown://path"})
    res_bad = await server.handle_request(bad_res_req)
    assert res_bad.error is not None
    assert res_bad.error.code == -32603


@pytest.mark.asyncio
async def test_mcp_harness_action_syntax_and_error_handling():
    from agent.services.mcp.tools import BaseTool

    reg = ToolRegistry()
    reg.register(HistoricalSearchTool())
    harness = AgentHarness(reg)

    # 1. Action: syntax parsing
    call = harness.extract_tool_call('Action: search_historical_archives(query="Tutankhamun")')
    assert call is not None
    assert call.tool_name == "search_historical_archives"
    assert call.arguments["query"] == "Tutankhamun"

    # 2. Markdown JSON syntax parsing
    json_text = (
        '```json\n{"tool": "search_historical_archives", "arguments": {"query": "Luxor"}}\n```'
    )
    call_json = harness.extract_tool_call(json_text)
    assert call_json is not None
    assert call_json.tool_name == "search_historical_archives"

    # 3. Broken tool execution error logging
    class BrokenTool(BaseTool):
        def __init__(self):
            super().__init__(name="broken_tool", description="Broken", input_schema={})

        async def execute(self, **kwargs):
            raise RuntimeError("Database failure")

    reg.register(BrokenTool())
    res = await harness.call_tool("broken_tool", {})
    assert res.status == "error"
    assert "Database failure" in res.error_message
