"""Unit tests for in-process MCP client/server adapter."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.core.skill_contracts import SkillEnvelope
from agents.mcp.client import MCPClient, qualify_tool_name
from agents.mcp.server import InProcessMCPServer


class _SimpleInput(BaseModel):
    topic: str = Field(min_length=1)


def _simple_handler(payload: _SimpleInput) -> SkillEnvelope:
    return SkillEnvelope(
        tool="custom_tool",
        status="ok",
        request=payload.model_dump(mode="python"),
        data={"topic": payload.topic},
        evidence=[],
    )


def test_mcp_client_routes_namespaced_tool() -> None:
    server = InProcessMCPServer("testdb")
    server.register_tool("custom_tool", _SimpleInput, _simple_handler, "custom description")

    client = MCPClient()
    client.register_server(server)

    qualified = qualify_tool_name("testdb", "custom_tool")
    envelope = client.call_tool(qualified, {"topic": "AI"})
    assert envelope.status == "ok"
    assert envelope.tool == "custom_tool"
    assert envelope.data["topic"] == "AI"
    assert envelope.diagnostics["mcp_server"] == "testdb"


def test_mcp_client_unknown_tool_returns_error() -> None:
    client = MCPClient()
    envelope = client.call_tool("mcp__missing__tool", {"topic": "AI"})
    assert envelope.status == "error"
    assert envelope.error == "mcp_unknown_namespaced_tool"
