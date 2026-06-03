"""fetch_external_url tool: envelope shaping + catalog registration (client mocked)."""

from __future__ import annotations

from unittest.mock import patch

from agent.tools.fetch_external import fetch_external_url_tool
from agent.tools.schemas import FetchExternalUrlInput


def test_fetch_tool_success():
    with patch("agent.mcp.fetch_client.fetch_external_url", return_value="hello world"):
        env = fetch_external_url_tool(FetchExternalUrlInput(url="https://example.com/x"))
    assert env.status == "ok"
    assert env.data["raw_output"] == "hello world"
    assert env.evidence and env.evidence[0].url == "https://example.com/x"


def test_fetch_tool_error_degrades():
    with patch("agent.mcp.fetch_client.fetch_external_url", side_effect=RuntimeError("boom")):
        env = fetch_external_url_tool(FetchExternalUrlInput(url="https://example.com/x"))
    assert env.status == "error"


def test_fetch_tool_registered_in_catalog():
    from agent.core.tool_catalog import tool_definition_by_name

    definition = tool_definition_by_name("fetch_external_url")
    assert definition.expose_in_mcp is False
    assert definition.requires_evidence is True
