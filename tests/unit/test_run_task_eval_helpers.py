from __future__ import annotations

from eval.run_task_eval import _collect_retrieved_urls, _tool_calls_detailed_from_trace


def test_tool_calls_detailed_prefers_input_payload() -> None:
    summary = {
        "tool_events": [
            {
                "tool_name": "search_news",
                "input_payload": {"query": "OpenAI", "days": 14, "limit": 5},
                "input_summary": {"query": "OpenAI", "days": "<int>", "limit": "<int>"},
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
        "tool_events": [
            {
                "tool_name": "query_news",
                "input_summary": {"query": "AI funding"},
            }
        ]
    }
    rows = _tool_calls_detailed_from_trace(summary)
    assert rows == [{"tool": "query_news", "args": {"query": "AI funding"}}]


def test_collect_retrieved_urls_only_uses_payload_and_trace() -> None:
    payload = {"valid_urls": ["https://a.example.com", "https://a.example.com"]}
    trace_summary = {
        "tool_events": [
            {
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
