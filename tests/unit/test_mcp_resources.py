"""MCP resources primitive: registration, listing, reading, stdio dispatch."""

from __future__ import annotations

import pytest

from agent.mcp.server import InProcessMCPServer
from agent.mcp.stdio_server import _handle_request


def _server_with_resource() -> InProcessMCPServer:
    server = InProcessMCPServer("test")
    server.register_resource(
        uri="news://x",
        name="X",
        description="demo resource",
        mime_type="text/plain",
        reader=lambda: "hello",
    )
    return server


def test_register_list_read_resource():
    server = _server_with_resource()
    assert server.list_resources() == [
        {"uri": "news://x", "name": "X", "description": "demo resource", "mimeType": "text/plain"}
    ]
    assert server.read_resource("news://x") == {
        "uri": "news://x",
        "mimeType": "text/plain",
        "text": "hello",
    }


def test_read_unknown_resource_raises():
    with pytest.raises(KeyError):
        _server_with_resource().read_resource("news://missing")


def test_stdio_resources_list_and_read():
    server = _server_with_resource()
    listed = _handle_request(server, {"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
    assert listed["result"]["resources"][0]["uri"] == "news://x"
    read = _handle_request(
        server,
        {"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": "news://x"}},
    )
    assert read["result"]["contents"][0]["text"] == "hello"


def test_initialize_advertises_resources():
    resp = _handle_request(_server_with_resource(), {"jsonrpc": "2.0", "id": 3, "method": "initialize"})
    assert "resources" in resp["result"]["capabilities"]
