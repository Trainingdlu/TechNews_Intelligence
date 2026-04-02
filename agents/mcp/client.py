"""MCP client with namespaced tool routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from core.skill_contracts import SkillEnvelope, build_error_envelope
    from mcp.server import InProcessMCPServer, build_newsdb_server
except ImportError:  # package-style import fallback
    from ..core.skill_contracts import SkillEnvelope, build_error_envelope
    from .server import InProcessMCPServer, build_newsdb_server


@dataclass(frozen=True)
class MCPToolRoute:
    qualified_name: str
    server_name: str
    tool_name: str
    description: str = ""


def qualify_tool_name(server_name: str, tool_name: str) -> str:
    return f"mcp__{server_name}__{tool_name}"


class MCPClient:
    """In-process MCP client for discovery and namespaced call routing."""

    def __init__(self) -> None:
        self._servers: dict[str, InProcessMCPServer] = {}
        self._routes: dict[str, MCPToolRoute] = {}

    def register_server(self, server: InProcessMCPServer) -> None:
        self._servers[server.server_name] = server
        self.refresh_server(server.server_name)

    def refresh_server(self, server_name: str) -> None:
        server = self._servers.get(server_name)
        if server is None:
            return

        stale_keys = [k for k, v in self._routes.items() if v.server_name == server_name]
        for key in stale_keys:
            self._routes.pop(key, None)

        for row in server.list_tools():
            qualified = str(row.get("qualified_name") or "")
            tool_name = str(row.get("name") or "")
            if not qualified or not tool_name:
                continue
            self._routes[qualified] = MCPToolRoute(
                qualified_name=qualified,
                server_name=server_name,
                tool_name=tool_name,
                description=str(row.get("description") or ""),
            )

    def list_tools(self) -> list[MCPToolRoute]:
        return [self._routes[k] for k in sorted(self._routes.keys())]

    def call_tool(self, qualified_name: str, payload: dict[str, Any] | None = None) -> SkillEnvelope:
        request_payload = payload or {}
        route = self._routes.get(qualified_name)
        if route is None:
            return build_error_envelope(
                tool=qualified_name,
                request=request_payload,
                error="mcp_unknown_namespaced_tool",
                diagnostics={"available_tools": sorted(self._routes.keys())},
            )

        server = self._servers.get(route.server_name)
        if server is None:
            return build_error_envelope(
                tool=qualified_name,
                request=request_payload,
                error="mcp_server_unavailable",
                diagnostics={"server_name": route.server_name},
            )

        envelope = server.call_tool(route.tool_name, request_payload)
        envelope.diagnostics.update(
            {
                "mcp_qualified_name": qualified_name,
                "mcp_server": route.server_name,
                "mcp_tool": route.tool_name,
            }
        )
        return envelope


_DEFAULT_CLIENT: MCPClient | None = None


def build_default_mcp_client() -> MCPClient:
    """Build local default MCP client with NewsDB server mounted."""

    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        return _DEFAULT_CLIENT

    client = MCPClient()
    client.register_server(build_newsdb_server(server_name="newsdb"))
    _DEFAULT_CLIENT = client
    return client
