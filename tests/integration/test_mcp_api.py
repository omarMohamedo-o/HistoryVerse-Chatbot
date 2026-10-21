"""Integration tests for Model Context Protocol (MCP) REST and JSON-RPC API endpoints."""

from fastapi.testclient import TestClient


def test_mcp_tools_list_endpoint(client: TestClient):
    response = client.get("/v1/mcp/tools")
    assert response.status_code == 200
    tools = response.json()
    assert isinstance(tools, list)
    assert len(tools) >= 3
    tool_names = [t["name"] for t in tools]
    assert "search_historical_archives" in tool_names
    assert "inspect_archaeological_artifact" in tool_names
    assert "check_unlearning_status" in tool_names


def test_mcp_execute_tool_endpoint(client: TestClient):
    payload = {"arguments": {"artifact_name": "Rosetta Stone"}}
    response = client.post("/v1/mcp/tools/inspect_archaeological_artifact/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["tool_name"] == "inspect_archaeological_artifact"
    assert data["data"]["artifact_name"] == "Rosetta Stone"


def test_mcp_jsonrpc_endpoint(client: TestClient):
    payload = {
        "jsonrpc": "2.0",
        "id": "req-101",
        "method": "tools/list",
        "params": {},
    }
    response = client.post("/v1/mcp", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "req-101"
    assert "tools" in body["result"]


def test_mcp_harness_agent_run_endpoint(client: TestClient):
    payload = {
        "query": "Tell me about the Serapeum of Alexandria",
        "agent_id": "historyverse-mora",
        "max_turns": 2,
    }
    response = client.post("/v1/mcp/agent/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == "historyverse-mora"
    assert "Serapeum" in data["output"]
