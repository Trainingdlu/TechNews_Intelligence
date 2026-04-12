"""Agent-level clarification trigger tests for ambiguous scope and source conflict."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.clarification import (
    CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
    CLARIFICATION_REASON_SOURCE_CONFLICT,
    ClarificationRequiredError,
)

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def test_generate_response_core_wide_query_triggers_ambiguous_scope() -> None:
    from agent.agent import _generate_response_core

    response_text = (
        "2025-01-01 到 2025-04-20 的跨来源信息覆盖 OpenAI、NVIDIA、Google、Microsoft、Meta、Anthropic。"
    )
    urls = [
        "https://news.ycombinator.com/item?id=1",
        "https://news.ycombinator.com/item?id=2",
        "https://techcrunch.com/2025/01/10/openai-update/",
        "https://techcrunch.com/2025/03/11/google-ai/",
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
        "https://example.com/d",
    ]
    with patch("agent.agent._generate_react", return_value=(response_text, urls)):
        with patch("agent.agent._get_accumulated_tool_calls", return_value={"query_news", "compare_sources"}):
            with pytest.raises(ClarificationRequiredError) as ei:
                _generate_response_core([], "帮我做 AI 行业全景总结")

    payload = ei.value.clarification.to_dict()
    assert payload["reason"] == CLARIFICATION_REASON_AMBIGUOUS_SCOPE
    assert "最近 7 天还是 30 天" in payload["question"]


def test_generate_response_core_conflicting_sources_triggers_source_conflict() -> None:
    from agent.agent import _generate_response_core

    response_text = (
        "TechCrunch 对 OpenAI 商业化前景较乐观，增长强劲。\n"
        "HackerNews 对 OpenAI 商业化更谨慎，强调风险与成本压力。"
    )
    urls = [
        "https://techcrunch.com/2025/04/01/openai-growth/",
        "https://news.ycombinator.com/item?id=100",
    ]
    with patch("agent.agent._generate_react", return_value=(response_text, urls)):
        with patch("agent.agent._get_accumulated_tool_calls", return_value={"compare_sources"}):
            with pytest.raises(ClarificationRequiredError) as ei:
                _generate_response_core([], "OpenAI 现在前景怎么样？")

    payload = ei.value.clarification.to_dict()
    assert payload["reason"] == CLARIFICATION_REASON_SOURCE_CONFLICT
    assert "冲突" in payload["question"] or "分歧" in payload["question"]


def test_generate_response_core_specific_query_with_evidence_remains_normal() -> None:
    from agent.agent import _generate_response_core

    response_text = "OpenAI 在最近 30 天于 TechCrunch 的报道热度上升。"
    urls = [
        "https://techcrunch.com/2025/04/01/openai-growth/",
        "https://techcrunch.com/2025/04/10/openai-product/",
    ]
    with patch("agent.agent._generate_react", return_value=(response_text, urls)):
        with patch("agent.agent._get_accumulated_tool_calls", return_value={"query_news", "trend_analysis"}):
            text, valid_urls = _generate_response_core([], "最近30天只看TechCrunch，分析OpenAI趋势")

    assert text == response_text
    assert valid_urls == urls

