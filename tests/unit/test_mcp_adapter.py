"""Unit tests for in-process MCP client/server adapter."""

from __future__ import annotations

import sys
import textwrap
import uuid
from shutil import rmtree
from pathlib import Path

from pydantic import BaseModel, Field

from agent.core.skill_catalog import iter_skill_definitions
from agent.core.skill_contracts import SkillEnvelope
from agent.mcp.client import MCPClient, StdioMCPServer, qualify_tool_name
from agent.mcp.server import InProcessMCPServer, build_newsdb_server


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


def test_newsdb_server_registration_matches_skill_catalog() -> None:
    server = build_newsdb_server("newsdb")
    listed = {row["name"] for row in server.list_tools()}
    expected = {
        definition.mcp_name or definition.name
        for definition in iter_skill_definitions()
        if definition.expose_in_mcp
    }
    assert listed == expected


def test_mcp_client_with_stdio_backend() -> None:
    tmp_root = Path("tests/unit/.tmp_mcp_stdio")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"case_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        script = tmp_dir / "fake_stdio_mcp.py"
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
            cwd=str(tmp_dir),
        )
        try:
            client = MCPClient()
            client.register_server(server)
            envelope = client.call_tool(qualify_tool_name("testdb", "custom_tool"), {"topic": "AI"})
        finally:
            server.close()
    finally:
        rmtree(tmp_dir, ignore_errors=True)

    assert envelope.status == "ok"
    assert envelope.tool == "custom_tool"
    assert envelope.data["topic"] == "AI"
    assert envelope.diagnostics["mcp_transport"] == "stdio"


def test_stdio_mcp_ignores_interleaved_notifications() -> None:
    tmp_root = Path("tests/unit/.tmp_mcp_stdio")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"case_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        script = tmp_dir / "fake_stdio_notify.py"
        script.write_text(
            textwrap.dedent(
                """
                import json
                import sys

                def result(req_id, payload):
                    return {"jsonrpc": "2.0", "id": req_id, "result": payload}

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

                    # Interleaved notification without id.
                    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/log", "params": {"msg": "hello"}}) + "\\n")
                    sys.stdout.flush()

                    if method == "tools/list":
                        payload = {"tools": [{"name": "custom_tool", "description": "x", "inputSchema": {}}]}
                        sys.stdout.write(json.dumps(result(req_id, payload)) + "\\n")
                        sys.stdout.flush()
                        continue

                    if method == "tools/call":
                        name = str(params.get("name") or "")
                        arguments = params.get("arguments") or {}
                        envelope = {"tool": name, "status": "ok", "request": arguments, "data": {"ok": True}, "evidence": [], "diagnostics": {}}
                        payload = {"structuredContent": envelope, "content": [{"type": "text", "text": json.dumps(envelope)}], "isError": False}
                        sys.stdout.write(json.dumps(result(req_id, payload)) + "\\n")
                        sys.stdout.flush()
                        continue
                """
            ),
            encoding="utf-8",
        )

        server = StdioMCPServer(
            server_name="testdb",
            command=[sys.executable, str(script.resolve())],
            cwd=str(tmp_dir),
        )
        try:
            client = MCPClient()
            client.register_server(server)
            envelope = client.call_tool(qualify_tool_name("testdb", "custom_tool"), {"topic": "AI"})
        finally:
            server.close()
    finally:
        rmtree(tmp_dir, ignore_errors=True)

    assert envelope.status == "ok"
    assert envelope.tool == "custom_tool"


def test_stdio_mcp_timeout_returns_transport_error(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MCP_RPC_TIMEOUT_SEC", "1")
    tmp_root = Path("tests/unit/.tmp_mcp_stdio")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"case_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        script = tmp_dir / "fake_stdio_timeout.py"
        script.write_text(
            textwrap.dedent(
                """
                import json
                import sys

                def result(req_id, payload):
                    return {"jsonrpc": "2.0", "id": req_id, "result": payload}

                for raw in sys.stdin:
                    line = raw.strip()
                    if not line:
                        continue
                    req = json.loads(line)
                    method = req.get("method")
                    req_id = req.get("id")
                    if method == "initialize":
                        payload = {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "fake", "version": "0.1.0"},
                            "capabilities": {"tools": {}},
                        }
                        if req_id is not None:
                            sys.stdout.write(json.dumps(result(req_id, payload)) + "\\n")
                            sys.stdout.flush()
                    elif method == "notifications/initialized":
                        continue
                    elif method == "tools/call":
                        # Intentionally no response to trigger timeout.
                        continue
                """
            ),
            encoding="utf-8",
        )

        server = StdioMCPServer(
            server_name="testdb",
            command=[sys.executable, str(script.resolve())],
            cwd=str(tmp_dir),
        )
        try:
            envelope = server.call_tool("custom_tool", {"topic": "AI"})
        finally:
            server.close()
    finally:
        rmtree(tmp_dir, ignore_errors=True)

    assert envelope.status == "error"
    assert envelope.error == "mcp_client_transport_error"

