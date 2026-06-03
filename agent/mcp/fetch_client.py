"""MCP client for an external fetch server (read URLs outside the news corpus).

Uses the official MCP Python SDK to spawn a standard stdio MCP server (the
reference ``mcp-server-fetch`` by default, via ``uvx``) and call its ``fetch``
tool. Kept separate from ``client.py`` — that one speaks this project's own
ToolEnvelope format, whereas external servers return standard MCP text content.

``mcp`` is imported lazily so importing the tool catalog never hard-depends on
the SDK being present.
"""

from __future__ import annotations

import asyncio
import os
import shlex

_DEFAULT_COMMAND = "uvx mcp-server-fetch"


def _command_parts() -> list[str]:
    raw = str(os.getenv("AGENT_FETCH_MCP_COMMAND", _DEFAULT_COMMAND)).strip() or _DEFAULT_COMMAND
    return shlex.split(raw, posix=os.name != "nt")


async def _afetch(url: str, max_length: int) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parts = _command_parts()
    params = StdioServerParameters(command=parts[0], args=parts[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("fetch", {"url": url, "max_length": max_length})
            if getattr(result, "isError", False):
                raise RuntimeError(f"fetch server returned an error for {url}")
            chunks = [
                str(getattr(item, "text", "")).strip()
                for item in (result.content or [])
                if str(getattr(item, "text", "")).strip()
            ]
            return "\n\n".join(chunks).strip()


def fetch_external_url(url: str, *, max_length: int = 5000, timeout_sec: float | None = None) -> str:
    """Fetch an external URL's readable content as text (sync; spawns per call)."""
    if timeout_sec is None:
        try:
            timeout_sec = float(str(os.getenv("AGENT_FETCH_MCP_TIMEOUT_SEC", "30")).strip())
        except (TypeError, ValueError):
            timeout_sec = 30.0
    return asyncio.run(asyncio.wait_for(_afetch(str(url), int(max_length)), timeout_sec))
