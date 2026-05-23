"""Unit tests for the custom LangGraph runtime."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agent.core.run_context import agent_run_context
from agent.core.runtime_factories import build_default_tool_runtime
from agent.core.tool_contracts import ToolEnvelope, ToolEvidence
from agent.graph.builder import invoke_custom_graph
from agent.graph.nodes import (
    GraphDependencies,
    GraphNodeRunner,
    _heuristic_intent,
    _heuristic_tool_calls,
    _recent_context_snippet,
)
from agent.graph.state import GraphModelHandle, GraphModels, GraphRuntimeConfig

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def _null_models() -> GraphModels:
    return GraphModels(
        context_curator=GraphModelHandle("context", "test", "none", None),
        intent_router=GraphModelHandle("intent", "test", "none", None),
        tool_worker=GraphModelHandle("tool", "test", "none", None),
        final_synthesizer=GraphModelHandle("final", "test", "none", None),
    )


def _runner() -> GraphNodeRunner:
    return GraphNodeRunner(
        GraphDependencies(
            models=_null_models(),
            tool_runtime=build_default_tool_runtime(),
            config=GraphRuntimeConfig(max_tool_rounds=2),
        )
    )


def test_custom_graph_smalltalk_uses_direct_answer_without_tools() -> None:
    with patch.dict("os.environ", {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}):
        result = invoke_custom_graph([], "hello", models=_null_models())

    assert result.clarification is None
    assert result.urls == []
    assert "帮助" in result.text or "help" in result.text.lower()


def test_tool_selection_trend_exposes_only_relevant_tools() -> None:
    update = _runner().tool_selection(
        {"intent": {"route": "needs_tools", "intent_type": "trend"}}
    )

    assert update["selected_tools"] == ["trend_analysis", "search_news", "fulltext_batch"]


@pytest.mark.parametrize(
    ("route", "title", "detail"),
    [
        ("direct_answer", "正在组织回答", "无需检索直接回答"),
        ("needs_clarification", "正在确认分析范围", "需要补充信息"),
        ("needs_tools", "正在规划检索", "需要调用工具获取证据"),
    ],
)
def test_intent_router_progress_copy_matches_route(route: str, title: str, detail: str) -> None:
    events: list[dict] = []
    with patch(
        "agent.graph.nodes._invoke_json_model",
        return_value={
            "route": route,
            "intent_type": "news_analysis",
            "reason": "test",
            "confidence": 1.0,
            "requires_tools": route == "needs_tools",
        },
    ):
        with agent_run_context(progress_callback=events.append):
            update = _runner().intent_router({"user_message": "测试路由文案"})

    assert update["intent"]["route"] == route
    progress = [event for event in events if event.get("event") == "progress"]
    assert progress[-1]["title"] == title
    assert progress[-1]["detail"] == detail


def test_tool_policy_blocks_unselected_tool_name() -> None:
    update = _runner().tool_policy(
        {
            "user_message": "分析 OpenAI 趋势",
            "selected_tools": ["search_news"],
            "pending_tool_calls": [{"name": "compare_topics", "args": {"topic_a": "A", "topic_b": "B"}}],
            "llm_input_messages": [],
            "evidence_urls": [],
            "tool_results": [],
        }
    )

    assert update["clarification"]["kind"] == "clarification_required"


def test_evidence_normalizer_uses_tool_envelope_evidence() -> None:
    envelope = ToolEnvelope(
        tool="search_news",
        status="ok",
        request={"query": "OpenAI"},
        evidence=[
            ToolEvidence(url="https://example.com/a", title="A", source="TechCrunch", rank=1),
            ToolEvidence(url="https://example.com/a", title="A duplicate", source="TechCrunch", rank=2),
        ],
    )

    update = _runner().evidence_normalizer({"tool_results": [envelope]})

    assert update["evidence_urls"] == ["https://example.com/a"]
    assert "TechCrunch" in update["evidence_brief"]


def test_output_guard_keeps_only_evidence_urls_and_adds_source_url() -> None:
    update = _runner().output_guard(
        {
            "final_text": "结论参考 https://unknown.example.com/x",
            "evidence_urls": ["https://example.com/source"],
        }
    )

    assert "https://unknown.example.com/x" not in update["final_text"]
    assert "https://example.com/source" in update["final_text"]
    assert update["valid_urls"] == ["https://example.com/source"]


def test_followup_evidence_url_drives_single_article_read() -> None:
    wrapped = (
        "User question: 详细说说OpenAI联手Yubico推出了ChatGPT高级账户安全计划\n"
        "Related previous context: 对比openai和谷歌最近30天的事件\n"
        "Previous evidence URLs:\n"
        "- [7] [AI] OpenAI联手Yubico推出ChatGPT高级账户安全计划 | https://example.com/openai-yubico\n"
        "Instruction: if this question depends on previous context, resolve it before retrieval."
    )

    calls = _heuristic_tool_calls(
        user_message=wrapped,
        intent={"intent_type": "article_read"},
        selected_tools=["read_news_content", "fulltext_batch", "search_news"],
        tool_results=[],
    )

    assert calls == [{"name": "read_news_content", "args": {"url": "https://example.com/openai-yubico"}}]


def test_recent_context_snippet_keeps_enough_previous_answer_for_followup() -> None:
    answer = (
        "OpenAI released GPT-5.5 Instant with better context handling. "
        "It also launched GPT-Realtime-2 and GPT-Realtime-Translate. "
        "The company introduced GPT-5.5 Cyber with access controls for security users. "
        "Greg Brockman testified in a court case about OpenAI restructuring. "
    ) * 4

    snippet = _recent_context_snippet(
        [
            {"role": "user", "parts": [{"text": "what happened with openai"}]},
            {
                "role": "model",
                "parts": [{"text": answer}],
                "citation_urls": ["https://example.com/openai"],
            },
        ]
    )

    assert "GPT-5.5 Cyber" in snippet
    assert "Greg Brockman testified" in snippet
    assert "https://example.com/openai" in snippet
    assert len(snippet) > 240


def test_chinese_compare_topics_extracts_openai_and_google() -> None:
    calls = _heuristic_tool_calls(
        user_message="对比openai和谷歌最近的战略差异",
        intent={"intent_type": "topic_comparison"},
        selected_tools=["compare_topics", "search_news", "fulltext_batch"],
        tool_results=[],
    )

    assert calls == [{"name": "compare_topics", "args": {"topic_a": "OpenAI", "topic_b": "Google", "days": 14}}]


def test_followup_compare_context_keeps_entities_for_narrowing() -> None:
    wrapped = (
        "User question: 企业市场商业化布局\n"
        "Related previous context: 对比openai和谷歌最近的战略差异"
    )
    calls = _heuristic_tool_calls(
        user_message=wrapped,
        intent={"intent_type": "topic_comparison"},
        selected_tools=["compare_topics", "search_news", "fulltext_batch"],
        tool_results=[],
    )

    assert calls == [{"name": "compare_topics", "args": {"topic_a": "OpenAI", "topic_b": "Google", "days": 14}}]


def test_compare_split_handles_unknown_lowercase_entities() -> None:
    message = "对比perplexity和mistral最近的商业化策略"

    assert _heuristic_intent(message)["intent_type"] == "topic_comparison"
    calls = _heuristic_tool_calls(
        user_message=message,
        intent={"intent_type": "topic_comparison"},
        selected_tools=["compare_topics", "search_news", "query_news", "fulltext_batch"],
        tool_results=[],
    )

    assert calls == [{"name": "compare_topics", "args": {"topic_a": "perplexity", "topic_b": "mistral", "days": 14}}]


def test_compare_split_handles_chinese_entities_with_strategy_dimension() -> None:
    message = "月之暗面和智谱AI最近的商业化布局差异"

    assert _heuristic_intent(message)["intent_type"] == "topic_comparison"
    calls = _heuristic_tool_calls(
        user_message=message,
        intent={"intent_type": "topic_comparison"},
        selected_tools=["compare_topics", "search_news", "query_news", "fulltext_batch"],
        tool_results=[],
    )

    assert calls == [{"name": "compare_topics", "args": {"topic_a": "月之暗面", "topic_b": "智谱AI", "days": 14}}]


def test_empty_analytical_result_falls_back_to_search_before_insufficient() -> None:
    update = _runner().tool_loop_decider(
        {
            "user_message": "对比openai和谷歌最近的战略差异",
            "intent": {"route": "needs_tools", "intent_type": "topic_comparison"},
            "selected_tools": ["compare_topics", "search_news", "query_news", "fulltext_batch"],
            "tool_results": [
                ToolEnvelope(
                    tool="compare_topics",
                    status="empty",
                    request={"topic_a": "OpenAI", "topic_b": "Google", "days": 14},
                    evidence=[],
                )
            ],
            "evidence_urls": [],
            "tool_round": 1,
            "max_tool_rounds": 2,
        }
    )

    assert update["next_step"] == "more_tools"
    assert [call["name"] for call in update["pending_tool_calls"]] == ["search_news", "query_news"]
    assert "OpenAI" in update["pending_tool_calls"][0]["args"]["query"]
    assert "Google" in update["pending_tool_calls"][0]["args"]["query"]
