"""Unit tests for in-process MCP client/server adapter."""

from __future__ import annotations

import sys
import textwrap
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from agents.core.skill_contracts import SkillEnvelope
from agents.mcp.client import MCPClient, StdioMCPServer, qualify_tool_name
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


def test_mcp_client_with_stdio_backend() -> None:
    workspace_tmp = Path("agents/tests/unit/.tmp_mcp_stdio")
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    test_tmp = workspace_tmp / f"case_{uuid.uuid4().hex}"
    test_tmp.mkdir(parents=True, exist_ok=True)

    script = test_tmp / "fake_stdio_mcp.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            def result(req_id, payload):
                return {"jsonrpc": "2.0", "id": req_id, "result": payload}

            def error(req_id, code, message):
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                req = json.loads(line)
                method = req.get("method")
                req_id = req.get("id")
                params = req.get("params") or {}

                if method == "initialize":
                    payload = {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "fake", "version": "0.1.0"},
                        "capabilities": {"tools": {}},
                    }
                    if req_id is not None:
                        sys.stdout.write(json.dumps(result(req_id, payload)) + "\\n")
                        sys.stdout.flush()
                    continue

                if method == "notifications/initialized":
                    continue

                if method == "tools/list":
                    payload = {
                        "tools": [
                            {
                                "name": "custom_tool",
                                "description": "fake custom tool",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"topic": {"type": "string"}},
                                    "required": ["topic"],
                                },
                            }
                        ]
                    }
                    sys.stdout.write(json.dumps(result(req_id, payload)) + "\\n")
                    sys.stdout.flush()
                    continue

                if method == "tools/call":
                    name = str(params.get("name") or "")
                    arguments = params.get("arguments") or {}
                    envelope = {
                        "tool": name,
                        "status": "ok",
                        "request": arguments,
                        "data": {"topic": arguments.get("topic")},
                        "evidence": [],
                        "diagnostics": {},
                    }
                    payload = {
                        "structuredContent": envelope,
                        "content": [{"type": "text", "text": json.dumps(envelope)}],
                        "isError": False,
                    }
                    sys.stdout.write(json.dumps(result(req_id, payload)) + "\\n")
                    sys.stdout.flush()
                    continue

                if req_id is not None:
                    sys.stdout.write(json.dumps(error(req_id, -32601, "method not found")) + "\\n")
                    sys.stdout.flush()
            """
        ),
        encoding="utf-8",
    )

    server = StdioMCPServer(
        server_name="testdb",
        command=[sys.executable, str(script.resolve())],
        cwd=str(test_tmp),
    )
    try:
        client = MCPClient()
        client.register_server(server)
        envelope = client.call_tool(qualify_tool_name("testdb", "custom_tool"), {"topic": "AI"})
    finally:
        server.close()
        try:
            script.unlink(missing_ok=True)
            test_tmp.rmdir()
        except Exception:
            pass

    assert envelope.status == "ok"
    assert envelope.tool == "custom_tool"
    assert envelope.data["topic"] == "AI"
    assert envelope.diagnostics["mcp_transport"] == "stdio"
