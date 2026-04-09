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
        description="Structured news retrieval",
        mcp_name="query_news_vector",
    ),
    SkillDefinition(
        name="trend_analysis",
        input_model=TrendAnalysisSkillInput,
        handler=trend_analysis_skill,
        description="Structured trend momentum analysis",
    ),
    SkillDefinition(
        name="search_news",
        input_model=SearchNewsSkillInput,
        handler=search_news_skill,
        description="Hybrid semantic+keyword news search",
    ),
    SkillDefinition(
        name="compare_sources",
        input_model=CompareSourcesSkillInput,
        handler=compare_sources_skill,
        description="HackerNews vs TechCrunch source comparison",
    ),
    SkillDefinition(
        name="compare_topics",
        input_model=CompareTopicsSkillInput,
        handler=compare_topics_skill,
        description="A-vs-B entity comparison with evidence",
    ),
    SkillDefinition(
        name="build_timeline",
        input_model=BuildTimelineSkillInput,
        handler=build_timeline_skill,
        description="Chronological event timeline construction",
    ),
    SkillDefinition(
        name="analyze_landscape",
        input_model=AnalyzeLandscapeSkillInput,
        handler=analyze_landscape_skill,
        description="Competitive landscape analysis with entity stats",
    ),
    SkillDefinition(
        name="fulltext_batch",
        input_model=FulltextBatchSkillInput,
        handler=fulltext_batch_skill,
        description="Batch full-text article reading",
    ),
    SkillDefinition(
        name="get_db_stats",
        input_model=GetDbStatsInput,
        handler=get_db_stats_skill,
        description="Get database freshness stats and total article count",
        expose_in_mcp=False,
    ),
    SkillDefinition(
        name="list_topics",
        input_model=ListTopicsInput,
        handler=list_topics_skill,
        description="Get daily article volume distribution for recent 21 days",
        expose_in_mcp=False,
    ),
    SkillDefinition(
        name="read_news_content",
        input_model=ReadNewsContentInput,
        handler=read_news_content_skill,
        description="Read full-text content of a single article by URL",
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
