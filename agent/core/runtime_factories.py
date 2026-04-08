"""Shared runtime factories for skill registry and tool hooks.

This module is the canonical home of the runtime component factories used by
the unified ReAct agent path.
"""

from __future__ import annotations

from .skill_registry import SkillRegistry
from .tool_hooks import ToolHookRunner
from ..tools import (
    AnalyzeLandscapeSkillInput,
    BuildTimelineSkillInput,
    CompareSourcesSkillInput,
    CompareTopicsSkillInput,
    FulltextBatchSkillInput,
    QueryNewsSkillInput,
    SearchNewsSkillInput,
    TrendAnalysisSkillInput,
    analyze_landscape_skill,
    build_timeline_skill,
    compare_sources_skill,
    compare_topics_skill,
    fulltext_batch_skill,
    query_news_skill,
    search_news_skill,
    trend_analysis_skill,
)


_DEFAULT_REGISTRY: SkillRegistry | None = None
_DEFAULT_HOOK_RUNNER: ToolHookRunner | None = None


def build_default_registry() -> SkillRegistry:
    """Build the default in-process skill registry."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    registry = SkillRegistry()
    registry.register(
        name="query_news",
        input_model=QueryNewsSkillInput,
        handler=lambda payload: query_news_skill(payload),
        description="Structured news retrieval",
    )
    registry.register(
        name="trend_analysis",
        input_model=TrendAnalysisSkillInput,
        handler=lambda payload: trend_analysis_skill(payload),
        description="Structured trend momentum analysis",
    )
    registry.register(
        name="search_news",
        input_model=SearchNewsSkillInput,
        handler=lambda payload: search_news_skill(payload),
        description="Hybrid semantic+keyword news search",
    )
    registry.register(
        name="compare_sources",
        input_model=CompareSourcesSkillInput,
        handler=lambda payload: compare_sources_skill(payload),
        description="HackerNews vs TechCrunch source comparison",
    )
    registry.register(
        name="compare_topics",
        input_model=CompareTopicsSkillInput,
        handler=lambda payload: compare_topics_skill(payload),
        description="A-vs-B entity comparison with evidence",
    )
    registry.register(
        name="build_timeline",
        input_model=BuildTimelineSkillInput,
        handler=lambda payload: build_timeline_skill(payload),
        description="Chronological event timeline construction",
    )
    registry.register(
        name="analyze_landscape",
        input_model=AnalyzeLandscapeSkillInput,
        handler=lambda payload: analyze_landscape_skill(payload),
        description="Competitive landscape analysis with entity stats",
    )
    registry.register(
        name="fulltext_batch",
        input_model=FulltextBatchSkillInput,
        handler=lambda payload: fulltext_batch_skill(payload),
        description="Batch full-text article reading",
    )
    _DEFAULT_REGISTRY = registry
    return registry


def build_default_hook_runner() -> ToolHookRunner:
    """Build the default shared hook runner."""

    global _DEFAULT_HOOK_RUNNER
    if _DEFAULT_HOOK_RUNNER is not None:
        return _DEFAULT_HOOK_RUNNER
    _DEFAULT_HOOK_RUNNER = ToolHookRunner()
    return _DEFAULT_HOOK_RUNNER
