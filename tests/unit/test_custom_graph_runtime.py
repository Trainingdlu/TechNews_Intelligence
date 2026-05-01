"""Unit tests for the custom LangGraph runtime."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agent.core.runtime_factories import build_default_tool_runtime
from agent.core.tool_contracts import ToolEnvelope, ToolEvidence
from agent.graph.builder import invoke_custom_graph
from agent.graph.nodes import GraphDependencies, GraphNodeRunner
from agent.graph.state import GraphModelHandle, GraphModels, GraphRuntimeConfig

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def _null_models() -> GraphModels:
    return GraphModels(
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
