"""Agent-level tests for soft HITL clarification on risk guards."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def test_generate_response_core_wide_query_appends_soft_hitl_followup() -> None:
    from agent.agent import _generate_response_core

    response_text = (
        "Cross-source overview covers OpenAI, NVIDIA, Google and Microsoft "
        "across a broad period with mixed signals."
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
            with patch("agent.agent._build_hitl_soft_followup", return_value="你更希望先限定时间范围还是来源范围？"):
                text, valid_urls = _generate_response_core([], "帮我做 AI 行业全景总结")

    assert text.startswith(response_text)
    assert text.endswith("你更希望先限定时间范围还是来源范围？")
    assert valid_urls == urls


def test_generate_response_core_conflict_risk_appends_soft_hitl_followup() -> None:
    from agent.agent import _generate_response_core

    response_text = (
        "TechCrunch is optimistic on OpenAI commercialization growth, while "
        "HackerNews highlights cost and execution risks."
    )
    urls = [
        "https://techcrunch.com/2025/04/01/openai-growth/",
        "https://news.ycombinator.com/item?id=100",
    ]

    with patch("agent.agent._generate_react", return_value=(response_text, urls)):
        with patch("agent.agent._get_accumulated_tool_calls", return_value={"compare_sources"}):
            with patch("agent.agent._build_hitl_soft_followup", return_value="是否只看单一来源后再给结论？"):
                text, valid_urls = _generate_response_core([], "OpenAI 现在前景怎么样？")

    assert text.startswith(response_text)
    assert text.endswith("是否只看单一来源后再给结论？")
    assert valid_urls == urls


def test_generate_response_core_specific_query_keeps_original_answer() -> None:
    from agent.agent import _generate_response_core

    response_text = "OpenAI trend is up on TechCrunch within the last 30 days."
    urls = [
        "https://techcrunch.com/2025/04/01/openai-growth/",
        "https://techcrunch.com/2025/04/10/openai-product/",
    ]

    with patch("agent.agent._generate_react", return_value=(response_text, urls)):
        with patch("agent.agent._get_accumulated_tool_calls", return_value={"query_news", "trend_analysis"}):
            with patch("agent.agent._build_hitl_soft_followup") as followup_mock:
                text, valid_urls = _generate_response_core([], "最近30天只看TechCrunch，分析OpenAI趋势")

    assert text == response_text
    assert valid_urls == urls
    assert not followup_mock.called

