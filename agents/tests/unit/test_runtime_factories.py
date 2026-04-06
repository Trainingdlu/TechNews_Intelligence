"""Tests for shared runtime factory builders.

The old Router->Miner->Analyst->Formatter workflow graph was retired.
This module now validates the shared SkillRegistry / ToolHookRunner builders
used by the unified ReAct runtime.
"""

from __future__ import annotations

from agents.core.skill_contracts import SkillEnvelope
from agents.core.runtime_factories import build_default_hook_runner, build_default_registry


EXPECTED_SKILLS = {
    "query_news",
    "trend_analysis",
    "search_news",
    "compare_sources",
    "compare_topics",
    "build_timeline",
    "analyze_landscape",
    "fulltext_batch",
}


def test_build_default_registry_is_singleton() -> None:
    reg_a = build_default_registry()
    reg_b = build_default_registry()
    assert reg_a is reg_b


def test_build_default_registry_contains_expected_skills() -> None:
    registry = build_default_registry()
    assert set(registry.list_skills()) == EXPECTED_SKILLS


def test_build_default_registry_exposes_input_schema() -> None:
    registry = build_default_registry()
    schema = registry.input_schema("query_news")
    assert schema.get("type") == "object"
    assert "properties" in schema
    assert "query" in schema["properties"]
    assert "days" in schema["properties"]


def test_build_default_registry_has_descriptions_for_all_skills() -> None:
    registry = build_default_registry()
    for skill_name in EXPECTED_SKILLS:
        spec = registry.get(skill_name)
        assert spec.description.strip()


def test_build_default_hook_runner_is_singleton() -> None:
    hooks_a = build_default_hook_runner()
    hooks_b = build_default_hook_runner()
    assert hooks_a is hooks_b


def test_default_hook_runner_denies_invalid_time_window() -> None:
    hooks = build_default_hook_runner()
    decision = hooks.pre_tool_use("query_news", {"query": "OpenAI", "days": 999})
    assert decision.action == "deny"
    assert "between 1 and 365" in str(decision.reason)


def test_default_hook_runner_warns_when_evidence_missing() -> None:
    hooks = build_default_hook_runner()
    envelope = SkillEnvelope(
        tool="query_news",
        status="ok",
        request={"query": "OpenAI", "days": 7},
        data={"count": 0, "records": []},
        evidence=[],
    )
    decision = hooks.post_tool_use("query_news", {"query": "OpenAI", "days": 7}, envelope)
    assert decision.action == "warn"
    assert "no_evidence_urls_in_skill_output" in str(decision.reason)


def test_default_hook_runner_allows_non_evidence_tool_output() -> None:
    hooks = build_default_hook_runner()
    envelope = SkillEnvelope(
        tool="list_topics",
        status="ok",
        request={},
        data={"rows": []},
        evidence=[],
    )
    decision = hooks.post_tool_use("list_topics", {}, envelope)
    assert decision.action == "allow"
