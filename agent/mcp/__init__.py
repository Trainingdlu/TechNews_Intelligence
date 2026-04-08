"""MCP adapter package (in-process baseline implementation)."""

from .client import MCPClient, build_default_mcp_client
from .server import InProcessMCPServer, build_newsdb_server

__all__ = [
    "MCPClient",
    "build_default_mcp_client",
    "InProcessMCPServer",
    "build_newsdb_server",
]
