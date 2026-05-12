"""Unit tests for agent memory trimming and graph-native tool-call policy."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.core.trace import finalize_request_trace, request_trace_context, trace_span
from agent.memory_policy import build_llm_input_messages
from agent.tool_policy import evaluate_pending_tool_calls


def _pending_call(name: str, args: dict) -> list[dict]:
    return [{"id": "call-1", "name": name, "args": args}]


def test_memory_policy_uses_llm_input_messages_without_overwriting_state() -> None:
    messages = []
    for idx in range(10):
        messages.append(HumanMessage(content=f"user-{idx}"))
        messages.append(AIMessage(content=f"assistant-{idx}"))
    messages.append(HumanMessage(content="latest question"))
    original = list(messages)

    import os
    from unittest.mock import patch

    with patch.dict(
        "os.environ",
        {
            "AGENT_CONTEXT_TRIM_ENABLED": "true",
            "AGENT_CONTEXT_KEEP_LAST_MESSAGES": "5",
            "AGENT_CONTEXT_MAX_TOKENS": "100000",
        },
    ):
        trimmed = build_llm_input_messages(messages)

    assert len(trimmed) <= 5
    assert trimmed[-1].content == "latest question"
    assert messages == original


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
    decision = evaluate_pending_tool_calls(
        _pending_call("query_news", {"query": "AI", "days": 999, "limit": 5}),
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
    decision = evaluate_pending_tool_calls(
        _pending_call("query_news", {"query": "OpenAI", "days": 7, "limit": 5}),
        allowed_tool_names={"query_news"},
        input_schemas=schemas,
    )

    assert decision.allowed


def test_pending_tool_policy_records_trace_block_for_invalid_tool_call() -> None:
    schemas = {
        "query_news": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 365},
            },
        },
    }
    pending = [{"name": "query_news", "args": {"query": "AI", "days": 999}}]

    with request_trace_context(user_message="latest AI", request_id="req-tool-policy"):
        decision = evaluate_pending_tool_calls(
            pending,
            allowed_tool_names={"query_news"},
            input_schemas=schemas,
        )
        with trace_span(
            "guard",
            "tool_policy",
            input_summary={"pending_tool_count": len(pending), "selected_tools": ["query_news"]},
            metadata={"node": "tool_policy"},
        ) as span:
            span.set_output(
                {
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "details": decision.details,
                }
            )
            if not decision.allowed:
                span.status = "blocked"
        summary = finalize_request_trace(
            final_status="clarification_required",
            error_code=f"tool_policy_{decision.reason}",
        )

    assert not decision.allowed
    guard_spans = [
        item
        for item in summary["spans"]
        if item["span_type"] == "guard" and item["name"] == "tool_policy"
    ]
    assert guard_spans[0]["status"] == "blocked"
    assert guard_spans[0]["output_summary"]["reason"] == "tool_arg_out_of_range"


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
    decision = evaluate_pending_tool_calls(
        _pending_call("build_timeline", {"topic": "OpenAI", "days": 365, "limit": 12}),
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
    missing = evaluate_pending_tool_calls(
        _pending_call("search_news", {"days": 7}),
        allowed_tool_names={"search_news"},
        input_schemas=schemas,
    )
    short = evaluate_pending_tool_calls(
        _pending_call("search_news", {"query": "", "days": 7}),
        allowed_tool_names={"search_news"},
        input_schemas=schemas,
    )

    assert missing.reason == "missing_required_tool_arg"
    assert missing.details["source"] == "pydantic_schema"
    assert short.reason == "tool_arg_too_short"


def test_tool_policy_blocks_read_news_content_without_context_urls() -> None:
    decision = evaluate_pending_tool_calls(
        _pending_call("read_news_content", {"url": "https://example.com/hallucinated"}),
        allowed_tool_names={"read_news_content"},
        evidence_urls=[],
        user_messages=["please read this article"],
    )

    assert not decision.allowed
    assert decision.reason == "read_news_content_no_context_urls"


def test_tool_policy_allows_read_news_content_when_user_message_contains_same_url() -> None:
    url = "https://example.com/a"
    decision = evaluate_pending_tool_calls(
        _pending_call("read_news_content", {"url": url}),
        allowed_tool_names={"read_news_content"},
        evidence_urls=[],
        user_messages=[f"please read this article: {url}"],
    )

    assert decision.allowed


def test_tool_policy_allows_read_news_content_when_evidence_contains_same_url() -> None:
    url = "https://example.com/a"
    decision = evaluate_pending_tool_calls(
        _pending_call("read_news_content", {"url": url}),
        allowed_tool_names={"read_news_content"},
        evidence_urls=[url],
        user_messages=["continue the analysis"],
    )

    assert decision.allowed


def test_tool_policy_blocks_read_news_content_when_url_not_in_context() -> None:
    decision = evaluate_pending_tool_calls(
        _pending_call("read_news_content", {"url": "https://example.com/b"}),
        allowed_tool_names={"read_news_content"},
        evidence_urls=[],
        user_messages=["read this one https://example.com/a"],
    )

    assert not decision.allowed
    assert decision.reason == "read_news_content_url_not_in_context"
