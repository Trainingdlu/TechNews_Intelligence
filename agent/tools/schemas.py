"""Pydantic input schemas for tool handlers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryNewsToolInput(BaseModel):
    """Typed input contract for structured query_news tool."""

    query: str = ""
    source: str = "all"
    days: int = Field(default=21, ge=1, le=365)
    category: str = ""
    sentiment: str = ""
    sort: str = "time_desc"
    limit: int = Field(default=8, ge=1, le=30)


class TrendAnalysisToolInput(BaseModel):
    """Typed input contract for structured trend_analysis tool."""

    topic: str = Field(min_length=1)
    window: int = Field(default=7, ge=3, le=60)


class SearchNewsToolInput(BaseModel):
    """Typed input contract for structured search_news tool."""

    query: str = Field(min_length=1)
    days: int = Field(default=21, ge=1, le=365)


class CompareSourcesToolInput(BaseModel):
    """Typed input contract for structured compare_sources tool."""

    topic: str = Field(min_length=1)
    days: int = Field(default=14, ge=1, le=90)


class CompareTopicsToolInput(BaseModel):
    """Typed input contract for structured compare_topics tool."""

    topic_a: str = Field(min_length=1)
    topic_b: str = Field(min_length=1)
    days: int = Field(default=14, ge=1, le=90)


class BuildTimelineToolInput(BaseModel):
    """Typed input contract for structured build_timeline tool."""

    topic: str = Field(min_length=1)
    days: int = Field(default=30, ge=1, le=180)
    limit: int = Field(default=12, ge=3, le=40)


class AnalyzeLandscapeToolInput(BaseModel):
    """Typed input contract for structured analyze_landscape tool."""

    topic: str = ""
    days: int = Field(default=30, ge=7, le=180)
    entities: str = ""
    limit_per_entity: int = Field(default=3, ge=1, le=5)


class FulltextBatchToolInput(BaseModel):
    """Typed input contract for structured fulltext_batch tool."""

    urls: str = Field(min_length=1)
    max_chars_per_article: int = Field(default=4000, ge=800, le=12000)


class GetDbStatsInput(BaseModel):
    """Typed input contract for structured get_db_stats tool."""
    pass


class ListTopicsInput(BaseModel):
    """Typed input contract for structured list_topics tool."""
    pass


class ReadNewsContentInput(BaseModel):
    """Typed input contract for structured read_news_content tool."""
    url: str = Field(min_length=1)


