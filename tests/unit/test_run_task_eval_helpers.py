from __future__ import annotations

from eval.run_task_eval import (
    _collect_retrieved_urls,
    _model_calls_from_trace,
    _tool_calls_detailed_from_trace,
)


def test_tool_calls_detailed_prefers_input_payload() -> None:
    summary = {
        "spans": [
            {
                "span_type": "tool_call",
                "name": "search_news",
                "started_at_ms": 10,
                "input_summary": {
                    "tool": "search_news",
                    "args": {"query": "OpenAI", "days": 14, "limit": 5},
                },
            }
        ]
    }
    rows = _tool_calls_detailed_from_trace(summary)
    assert rows == [
        {
            "tool": "search_news",
            "args": {"query": "OpenAI", "days": 14, "limit": 5},
        }
    ]


def test_tool_calls_detailed_falls_back_to_input_summary() -> None:
    summary = {
        "spans": [
            {
                "span_type": "tool_call",
                "name": "query_news",
                "started_at_ms": 10,
                "input_summary": {"query": "AI funding"},
            }
        ]
    }
    rows = _tool_calls_detailed_from_trace(summary)
    assert rows == [{"tool": "query_news", "args": {"query": "AI funding"}}]


def test_collect_retrieved_urls_only_uses_payload_and_trace() -> None:
    payload = {"valid_urls": ["https://a.example.com", "https://a.example.com"]}
    trace_summary = {
        "spans": [
            {
                "span_type": "tool_call",
                "output_summary": {
                    "context_docs": [
                        {"url": "https://b.example.com"},
                        {"url": "https://a.example.com"},
                    ]
                }
            }
        ]
    }
    urls = _collect_retrieved_urls(payload, trace_summary)
    assert urls == ["https://a.example.com", "https://b.example.com"]


def test_tool_calls_detailed_prefers_span_trace() -> None:
    summary = {
        "spans": [
            {
                "span_type": "tool_call",
                "name": "search_news",
                "started_at_ms": 20,
                "input_summary": {"tool": "search_news", "args": {"query": "OpenAI", "limit": 5}},
            }
        ],
    }
    rows = _tool_calls_detailed_from_trace(summary)
    assert rows == [{"tool": "search_news", "args": {"query": "OpenAI", "limit": 5}}]


def test_collect_retrieved_urls_uses_tool_span_evidence_urls() -> None:
    payload = {"valid_urls": []}
    trace_summary = {
        "spans": [
            {
                "span_type": "tool_call",
                "output_summary": {"evidence_urls": ["https://span.example.com"]},
            }
        ],
    }
    urls = _collect_retrieved_urls(payload, trace_summary)
    assert urls == ["https://span.example.com"]


def test_model_calls_from_trace_links_model_io_to_span() -> None:
    summary = {
        "spans": [
            {
                "span_id": "span-model",
                "span_type": "model_call",
                "name": "tool_worker",
                "status": "success",
                "started_at_ms": 1,
                "latency_ms": 42,
                "metadata": {"provider": "gemini_api", "model": "fallback"},
            }
        ],
        "model_io": [
            {
                "span_id": "span-model",
                "node": "tool_worker",
                "provider": "vertex",
                "model": "gemini-test",
                "token_usage": {"total_tokens": 9},
            }
        ],
    }
    assert _model_calls_from_trace(summary) == [
        {
            "span_id": "span-model",
            "node": "tool_worker",
            "provider": "vertex",
            "model": "gemini-test",
            "status": "success",
            "latency_ms": 42,
            "token_usage": {"total_tokens": 9},
            "has_full_io": True,
        }
    ]
