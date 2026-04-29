"""Basic lookup and state primitives exposed as tools."""

from __future__ import annotations

from ..core.tool_contracts import ToolEnvelope, build_tool_error_envelope
from .news_ops import get_db_stats, list_topics, read_news_content
from .schemas import GetDbStatsInput, ListTopicsInput, ReadNewsContentInput


def get_db_stats_tool(input: GetDbStatsInput) -> ToolEnvelope:
    """Wrapped get_db_stats."""
    req = input.model_dump()
    try:
        text = get_db_stats()
        return ToolEnvelope(
            tool="get_db_stats",
            status="ok",
            request=req,
            data={"raw_output": text},
            diagnostics={"evidence_count": 0},
        )
    except Exception as e:
        return build_tool_error_envelope(
            "get_db_stats",
            req,
            "get_db_stats_failed",
            diagnostics={"exception_type": type(e).__name__, "exception_message": str(e)},
        )


def list_topics_tool(input: ListTopicsInput) -> ToolEnvelope:
    """Wrapped list_topics."""
    req = input.model_dump()
    try:
        text = list_topics()
        return ToolEnvelope(
            tool="list_topics",
            status="ok",
            request=req,
            data={"raw_output": text},
            diagnostics={"evidence_count": 0},
        )
    except Exception as e:
        return build_tool_error_envelope(
            "list_topics",
            req,
            "list_topics_failed",
            diagnostics={"exception_type": type(e).__name__, "exception_message": str(e)},
        )


def read_news_content_tool(input: ReadNewsContentInput) -> ToolEnvelope:
    """Wrapped read_news_content."""
    req = input.model_dump()
    try:
        text = read_news_content(str(input.url))
        return ToolEnvelope(
            tool="read_news_content",
            status="ok",
            request=req,
            data={"raw_output": text, "url": input.url},
            evidence=[{"url": str(input.url), "title": None, "source": None, "created_at": None, "score": None, "rank": 1}],
            diagnostics={"evidence_count": 1},
        )
    except Exception as e:
        return build_tool_error_envelope(
            "read_news_content",
            req,
            "read_news_content_failed",
            diagnostics={"exception_type": type(e).__name__, "exception_message": str(e)},
        )

