"""Single source of truth for tool registration metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from .tool_contracts import ToolEnvelope
from ..tools.analyze_landscape import analyze_landscape_tool
from ..tools.build_timeline import build_timeline_tool
from ..tools.compare_sources import compare_sources_tool
from ..tools.compare_topics import compare_topics_tool
from ..tools.fulltext_batch import fulltext_batch_tool
from ..tools.query_news import query_news_tool
from ..tools.schemas import (
    AnalyzeLandscapeToolInput,
    BuildTimelineToolInput,
    CompareSourcesToolInput,
    CompareTopicsToolInput,
    FulltextBatchToolInput,
    QueryNewsToolInput,
    SearchNewsToolInput,
    TrendAnalysisToolInput,
    GetDbStatsInput,
    ListTopicsInput,
    ReadNewsContentInput,
)
from ..tools.search_news import search_news_tool
from ..tools.trend_analysis import trend_analysis_tool
from ..tools.basic_tools import (
    get_db_stats_tool,
    list_topics_tool,
    read_news_content_tool,
)

ToolHandler = Callable[[BaseModel], ToolEnvelope]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    input_model: type[BaseModel]
    handler: ToolHandler
    description: str
    capability: str
    tool_group: str
    requires_evidence: bool
    allowed_for_intents: tuple[str, ...]
    mcp_name: str | None = None
    expose_in_mcp: bool = True


TOOL_CATALOG: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="query_news",
        input_model=QueryNewsToolInput,
        handler=query_news_tool,
        description=(
            "Retrieve recent news records with optional query, source, category, sentiment, "
            "sort, time-window, and limit filters. Evidence items are the returned article URLs."
        ),
        capability="news_retrieval",
        tool_group="retrieval",
        requires_evidence=True,
        allowed_for_intents=("roundup_listing", "analysis", "conflict_resolution", "generic"),
        mcp_name="query_news_vector",
    ),
    ToolDefinition(
        name="trend_analysis",
        input_model=TrendAnalysisToolInput,
        handler=trend_analysis_tool,
        description=(
            "Measure recent-vs-previous topic momentum from article counts and engagement. "
            "Evidence items are representative articles supporting the trend."
        ),
        capability="trend_analysis",
        tool_group="analysis",
        requires_evidence=True,
        allowed_for_intents=("analysis", "conflict_resolution"),
    ),
    ToolDefinition(
        name="search_news",
        input_model=SearchNewsToolInput,
        handler=search_news_tool,
        description=(
            "Run hybrid semantic, lexical, and exact news search for a specific query. "
            "Evidence items are ranked matching article URLs with retrieval scores when available."
        ),
        capability="hybrid_search",
        tool_group="retrieval",
        requires_evidence=True,
        allowed_for_intents=("roundup_listing", "analysis", "conflict_resolution", "generic"),
    ),
    ToolDefinition(
        name="compare_sources",
        input_model=CompareSourcesToolInput,
        handler=compare_sources_tool,
        description=(
            "Compare source coverage, sentiment, and top articles for one topic across sources. "
            "Evidence items are source-attributed article URLs."
        ),
        capability="source_comparison",
        tool_group="analysis",
        requires_evidence=True,
        allowed_for_intents=("analysis", "conflict_resolution"),
    ),
    ToolDefinition(
        name="compare_topics",
        input_model=CompareTopicsToolInput,
        handler=compare_topics_tool,
        description=(
            "Compare two distinct topics or entities using non-overlapping evidence pools "
            "and match-score arbitration. Evidence items are assigned to the best matching side."
        ),
        capability="topic_comparison",
        tool_group="analysis",
        requires_evidence=True,
        allowed_for_intents=("analysis", "conflict_resolution"),
    ),
    ToolDefinition(
        name="build_timeline",
        input_model=BuildTimelineToolInput,
        handler=build_timeline_tool,
        description=(
            "Build a chronological timeline for one topic, company, or product. "
            "Evidence items are the article URLs backing individual timeline events."
        ),
        capability="timeline",
        tool_group="analysis",
        requires_evidence=True,
        allowed_for_intents=("analysis", "roundup_listing", "conflict_resolution"),
    ),
    ToolDefinition(
        name="analyze_landscape",
        input_model=AnalyzeLandscapeToolInput,
        handler=analyze_landscape_tool,
        description=(
            "Analyze a competitive landscape across entities with coverage, source mix, "
            "signals, confidence, and evidence URLs."
        ),
        capability="landscape_analysis",
        tool_group="analysis",
        requires_evidence=True,
        allowed_for_intents=("analysis", "conflict_resolution"),
    ),
    ToolDefinition(
        name="fulltext_batch",
        input_model=FulltextBatchToolInput,
        handler=fulltext_batch_tool,
        description=(
            "Read full text for explicit URLs, or auto-select relevant articles from a query. "
            "Evidence items are the URLs whose full text was read."
        ),
        capability="fulltext_retrieval",
        tool_group="content",
        requires_evidence=True,
        allowed_for_intents=("analysis", "conflict_resolution", "generic"),
    ),
    ToolDefinition(
        name="get_db_stats",
        input_model=GetDbStatsInput,
        handler=get_db_stats_tool,
        description="Get database freshness stats and total article count; no article evidence is expected.",
        capability="database_metadata",
        tool_group="metadata",
        requires_evidence=False,
        allowed_for_intents=("smalltalk_or_capability", "generic"),
        expose_in_mcp=False,
    ),
    ToolDefinition(
        name="list_topics",
        input_model=ListTopicsInput,
        handler=list_topics_tool,
        description="Get daily article volume distribution for recent 21 days; no article evidence is expected.",
        capability="topic_distribution",
        tool_group="metadata",
        requires_evidence=False,
        allowed_for_intents=("roundup_listing", "analysis", "generic"),
        expose_in_mcp=False,
    ),
    ToolDefinition(
        name="read_news_content",
        input_model=ReadNewsContentInput,
        handler=read_news_content_tool,
        description=(
            "Read full-text content for a URL already provided by the user or returned in prior tool evidence. "
            "Evidence is the same URL being read."
        ),
        capability="article_reader",
        tool_group="content",
        requires_evidence=True,
        allowed_for_intents=("analysis", "conflict_resolution", "generic"),
        expose_in_mcp=False,
    ),
)


def iter_tool_definitions() -> tuple[ToolDefinition, ...]:
    return TOOL_CATALOG


def tool_definition_by_name(name: str) -> ToolDefinition:
    normalized = str(name).strip()
    for definition in TOOL_CATALOG:
        if definition.name == normalized:
            return definition
    raise KeyError(f"Unknown tool: {name}")

