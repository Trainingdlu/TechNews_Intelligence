"""Single source of truth for skill registration metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from .skill_contracts import SkillEnvelope
from ..skills.analyze_landscape import analyze_landscape_skill
from ..skills.build_timeline import build_timeline_skill
from ..skills.compare_sources import compare_sources_skill
from ..skills.compare_topics import compare_topics_skill
from ..skills.fulltext_batch import fulltext_batch_skill
from ..skills.query_news import query_news_skill
from ..skills.schemas import (
    AnalyzeLandscapeSkillInput,
    BuildTimelineSkillInput,
    CompareSourcesSkillInput,
    CompareTopicsSkillInput,
    FulltextBatchSkillInput,
    QueryNewsSkillInput,
    SearchNewsSkillInput,
    TrendAnalysisSkillInput,
    GetDbStatsInput,
    ListTopicsInput,
    ReadNewsContentInput,
)
from ..skills.search_news import search_news_skill
from ..skills.trend_analysis import trend_analysis_skill
from ..skills.basic_tools import (
    get_db_stats_skill,
    list_topics_skill,
    read_news_content_skill,
)

SkillHandler = Callable[[BaseModel], SkillEnvelope]


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    input_model: type[BaseModel]
    handler: SkillHandler
    description: str
    mcp_name: str | None = None
    expose_in_mcp: bool = True


SKILL_CATALOG: tuple[SkillDefinition, ...] = (
    SkillDefinition(
        name="query_news",
        input_model=QueryNewsSkillInput,
        handler=query_news_skill,
        description="Filter recent news by optional query/source/category/sentiment; returns records with evidence URLs.",
        mcp_name="query_news_vector",
    ),
    SkillDefinition(
        name="trend_analysis",
        input_model=TrendAnalysisSkillInput,
        handler=trend_analysis_skill,
        description="Compare recent-vs-previous topic momentum and attach representative evidence URLs.",
    ),
    SkillDefinition(
        name="search_news",
        input_model=SearchNewsSkillInput,
        handler=search_news_skill,
        description="Hybrid semantic, lexical, and exact news search for a specific query; returns ranked evidence.",
    ),
    SkillDefinition(
        name="compare_sources",
        input_model=CompareSourcesSkillInput,
        handler=compare_sources_skill,
        description="Compare source coverage, sentiment, and top evidence for one topic across HackerNews and TechCrunch.",
    ),
    SkillDefinition(
        name="compare_topics",
        input_model=CompareTopicsSkillInput,
        handler=compare_topics_skill,
        description="Compare two distinct topics or entities using non-overlapping evidence pools and match-score arbitration.",
    ),
    SkillDefinition(
        name="build_timeline",
        input_model=BuildTimelineSkillInput,
        handler=build_timeline_skill,
        description="Build a chronological evidence-backed timeline for one topic, company, or product.",
    ),
    SkillDefinition(
        name="analyze_landscape",
        input_model=AnalyzeLandscapeSkillInput,
        handler=analyze_landscape_skill,
        description="Analyze a competitive landscape across entities with coverage, source mix, signals, and evidence URLs.",
    ),
    SkillDefinition(
        name="fulltext_batch",
        input_model=FulltextBatchSkillInput,
        handler=fulltext_batch_skill,
        description="Read full text for explicit URLs, or auto-select relevant articles from a query and return URL evidence.",
    ),
    SkillDefinition(
        name="get_db_stats",
        input_model=GetDbStatsInput,
        handler=get_db_stats_skill,
        description="Get database freshness stats and total article count; no article evidence is expected.",
        expose_in_mcp=False,
    ),
    SkillDefinition(
        name="list_topics",
        input_model=ListTopicsInput,
        handler=list_topics_skill,
        description="Get daily article volume distribution for recent 21 days; no article evidence is expected.",
        expose_in_mcp=False,
    ),
    SkillDefinition(
        name="read_news_content",
        input_model=ReadNewsContentInput,
        handler=read_news_content_skill,
        description="Read full-text content for a URL already provided by the user or prior tool evidence.",
        expose_in_mcp=False,
    ),
)


def iter_skill_definitions() -> tuple[SkillDefinition, ...]:
    return SKILL_CATALOG


def skill_definition_by_name(name: str) -> SkillDefinition:
    normalized = str(name).strip()
    for definition in SKILL_CATALOG:
        if definition.name == normalized:
            return definition
    raise KeyError(f"Unknown skill: {name}")
