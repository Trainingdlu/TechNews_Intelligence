"""Unit tests for agent memory trimming and tool-call policy hooks."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.clarification import ClarificationRequiredError
from agent.core.metrics import get_route_metrics_snapshot, reset_route_metrics
from agent.core.trace import finalize_request_trace, request_trace_context
from agent.graph_factory import make_post_model_hook, pre_model_hook
from agent.tool_policy import evaluate_tool_calls


def test_pre_model_hook_uses_llm_input_messages_without_overwriting_state() -> None:
    messages = []
    for idx in range(10):
        messages.append(HumanMessage(content=f"user-{idx}"))
        messages.append(AIMessage(content=f"assistant-{idx}"))
    messages.append(HumanMessage(content="latest question"))
    state = {"messages": list(messages)}

    with patch.dict(
        "os.environ",
        {
            "AGENT_CONTEXT_TRIM_ENABLED": "true",
            "AGENT_CONTEXT_KEEP_LAST_MESSAGES": "5",
            "AGENT_CONTEXT_MAX_TOKENS": "100000",
        },
    ):
        update = pre_model_hook(state)

    assert "llm_input_messages" in update
    assert len(update["llm_input_messages"]) <= 5
    assert update["llm_input_messages"][-1].content == "latest question"
    assert len(state["messages"]) == len(messages)


def test_tool_policy_blocks_out_of_range_numeric_args() -> None:
    schemas = {
        "query_news": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 365},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
        }
    }
    messages = [
        HumanMessage(content="latest AI"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "query_news",
                    "args": {"query": "AI", "days": 999, "limit": 5},
                }
            ],
        ),
    ]

    decision = evaluate_tool_calls(
        messages,
        allowed_tool_names={"query_news"},
        input_schemas=schemas,
    )

    assert not decision.allowed
    assert decision.reason == "tool_arg_out_of_range"
    assert decision.details["arg"] == "days"
    assert decision.details["source"] == "pydantic_schema"


def test_tool_policy_allows_valid_tool_call() -> None:
    schemas = {
        "query_news": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 365},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
        }
    }
    messages = [
        HumanMessage(content="latest OpenAI updates"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "query_news",
                    "args": {"query": "OpenAI", "days": 7, "limit": 5},
                }
            ],
        ),
    ]

    decision = evaluate_tool_calls(
        messages,
        allowed_tool_names={"query_news"},
        input_schemas=schemas,
    )

    assert decision.allowed


def test_post_model_hook_raises_clarification_for_invalid_tool_call() -> None:
    reset_route_metrics()
    hook = make_post_model_hook(
        allowed_tool_names={"query_news"},
        input_schemas={
            "query_news": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "minimum": 1, "maximum": 365},
                },
            }
        },
    )
    state = {
        "messages": [
            HumanMessage(content="latest AI"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "query_news",
                        "args": {"query": "AI", "days": 999},
                    }
                ],
            ),
        ]
    }

    with request_trace_context(user_message="latest AI", request_id="req-tool-policy"):
        with pytest.raises(ClarificationRequiredError) as exc_info:
            hook(state)
        summary = finalize_request_trace(
            final_status="clarification_required",
            error_code="clarification_ambiguous_scope",
        )

    assert exc_info.value.clarification.reason == "ambiguous_scope"
    snapshot = get_route_metrics_snapshot()
    assert snapshot["tool_policy_blocked_total"] == 1
    assert snapshot["tool_policy_blocked_tool_arg_out_of_range"] == 1
    assert summary["runtime"]["tool_policy_blocks"][0]["reason"] == "tool_arg_out_of_range"


def test_tool_policy_uses_schema_specific_bounds() -> None:
    schemas = {
        "build_timeline": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 180},
                "limit": {"type": "integer", "minimum": 3, "maximum": 40},
            },
        }
    }
    messages = [
        HumanMessage(content="build timeline"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "build_timeline",
                    "args": {"topic": "OpenAI", "days": 365, "limit": 12},
                }
            ],
        ),
    ]

    decision = evaluate_tool_calls(
        messages,
        allowed_tool_names={"build_timeline"},
        input_schemas=schemas,
    )

    assert not decision.allowed
    assert decision.details["maximum"] == 180


def test_tool_policy_uses_schema_required_and_min_length() -> None:
    schemas = {
        "search_news": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "days": {"type": "integer", "minimum": 1, "maximum": 365},
            },
        }
    }
    missing_messages = [
        HumanMessage(content="search"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "search_news",
                    "args": {"days": 7},
                }
            ],
        ),
    ]
    short_messages = [
        HumanMessage(content="search"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-2",
                    "name": "search_news",
                    "args": {"query": "", "days": 7},
                }
            ],
        ),
    ]

    missing = evaluate_tool_calls(
        missing_messages,
        allowed_tool_names={"search_news"},
        input_schemas=schemas,
    )
    short = evaluate_tool_calls(
        short_messages,
        allowed_tool_names={"search_news"},
        input_schemas=schemas,
    )

    assert missing.reason == "missing_required_tool_arg"
    assert missing.details["source"] == "pydantic_schema"
    assert short.reason == "tool_arg_too_short"


def test_tool_policy_blocks_read_news_content_without_context_urls() -> None:
    messages = [
        HumanMessage(content="please read this article"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "read_news_content",
                    "args": {"url": "https://example.com/hallucinated"},
                }
            ],
        ),
    ]

    decision = evaluate_tool_calls(
        messages,
        allowed_tool_names={"read_news_content"},
        evidence_urls=[],
    )

    assert not decision.allowed
    assert decision.reason == "read_news_content_no_context_urls"


def test_tool_policy_allows_read_news_content_when_user_message_contains_same_url() -> None:
    url = "https://example.com/a"
    messages = [
        HumanMessage(content=f"please read this article: {url}"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "read_news_content",
                    "args": {"url": url},
                }
            ],
        ),
    ]

    decision = evaluate_tool_calls(
        messages,
        allowed_tool_names={"read_news_content"},
        evidence_urls=[],
    )

    assert decision.allowed


def test_tool_policy_allows_read_news_content_when_evidence_contains_same_url() -> None:
    url = "https://example.com/a"
    messages = [
        HumanMessage(content="continue the analysis"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "read_news_content",
                    "args": {"url": url},
                }
            ],
        ),
    ]

    decision = evaluate_tool_calls(
        messages,
        allowed_tool_names={"read_news_content"},
        evidence_urls=[url],
    )

    assert decision.allowed


def test_tool_policy_blocks_read_news_content_when_url_not_in_context() -> None:
    messages = [
        HumanMessage(content="read this one https://example.com/a"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "read_news_content",
                    "args": {"url": "https://example.com/b"},
                }
            ],
        ),
    ]

    decision = evaluate_tool_calls(
        messages,
        allowed_tool_names={"read_news_content"},
        evidence_urls=[],
    )

    assert not decision.allowed
    assert decision.reason == "read_news_content_url_not_in_context"
