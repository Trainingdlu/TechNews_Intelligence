"""Standalone stdio MCP server entrypoint for NewsDB tools."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    from .server import InProcessMCPServer, build_newsdb_server
except ImportError:  # pragma: no cover - direct script fallback
    from server import InProcessMCPServer, build_newsdb_server  # type: ignore[no-redef]


def _jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _handle_request(server: InProcessMCPServer, request: dict[str, Any]) -> dict[str, Any] | None:
    method = str(request.get("method") or "").strip()
    request_id = request.get("id")
    params = request.get("params")
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        raise ValueError("params must be an object")

    if method == "initialize":
        # Minimal initialize response compatible with tools-based clients.
        result = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": f"{server.server_name}-stdio", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        }
        return _jsonrpc_result(request_id, result)

    if method == "notifications/initialized":
        # Notification: no response required.
        return None

    if method == "ping":
        return _jsonrpc_result(request_id, {"ok": True})

    if method == "tools/list":
        tools: list[dict[str, Any]] = []
        for row in server.list_tools():
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            tools.append(
                {
                    "name": name,
                    "description": str(row.get("description") or ""),
                    "inputSchema": row.get("input_schema") if isinstance(row.get("input_schema"), dict) else {},
                }
            )
        return _jsonrpc_result(request_id, {"tools": tools})

    if method == "tools/call":
        tool_name = str(params.get("name") or "").strip()
        if not tool_name:
            raise ValueError("tools/call requires params.name")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}

        envelope = server.call_tool(tool_name, arguments)
        return _jsonrpc_result(
            request_id,
            {
                "structuredContent": envelope.to_dict(),
                "content": [{"type": "text", "text": envelope.to_json()}],
                "isError": envelope.status == "error",
            },
        )

    raise ValueError(f"Unknown method: {method}")


def _serve_loop(server: InProcessMCPServer) -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception:
            response = _jsonrpc_error(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        if not isinstance(request, dict):
            response = _jsonrpc_error(None, -32600, "Invalid Request")
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        try:
            response = _handle_request(server, request)
        except ValueError as exc:
            response = _jsonrpc_error(request.get("id"), -32602, str(exc))
        except Exception as exc:  # noqa: BLE001
            response = _jsonrpc_error(request.get("id"), -32603, f"Internal error: {type(exc).__name__}")

        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NewsDB MCP stdio server")
    parser.add_argument("--server-name", default="newsdb", help="Logical server name for tool namespace")
    args = parser.parse_args(argv)

    server = build_newsdb_server(server_name=str(args.server_name).strip() or "newsdb")
    return _serve_loop(server)


if __name__ == "__main__":
    raise SystemExit(main())

