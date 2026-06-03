"""Tool: read an external (non-corpus) URL via the fetch MCP server (MCP client)."""

from __future__ import annotations

from ..core.tool_contracts import ToolEnvelope, build_tool_error_envelope
from .schemas import FetchExternalUrlInput


def fetch_external_url_tool(input: FetchExternalUrlInput) -> ToolEnvelope:
    """Read a user-provided external URL not in the news database."""
    # Lazy import: avoids a tool_catalog -> agent.mcp -> ... -> tool_catalog cycle.
    from ..mcp.fetch_client import fetch_external_url

    req = input.model_dump()
    url = str(input.url)
    try:
        text = fetch_external_url(url, max_length=int(input.max_length))
        return ToolEnvelope(
            tool="fetch_external_url",
            status="ok",
            request=req,
            data={"raw_output": text, "url": url},
            evidence=[{"url": url, "title": None, "source": None, "created_at": None, "score": None, "rank": 1}],
            diagnostics={"evidence_count": 1, "mcp_transport": "stdio", "mcp_server": "fetch"},
        )
    except Exception as e:
        return build_tool_error_envelope(
            "fetch_external_url",
            req,
            "fetch_external_url_failed",
            diagnostics={"exception_type": type(e).__name__, "exception_message": str(e)},
        )
