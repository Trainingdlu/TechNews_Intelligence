"""Unit tests for trace persistence storage adapter."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from services import agent_trace_store as store_mod


class _FakeCursor:
    def __init__(self):
        self.execute_calls: list[tuple[str, tuple]] = []
        self.executemany_calls: list[tuple[str, list[tuple]]] = []

    def execute(self, sql: str, params: tuple = ()):
        self.execute_calls.append((sql, params))

    def executemany(self, sql: str, params_seq):
        self.executemany_calls.append((sql, list(params_seq)))


def _transaction_context(cursor: _FakeCursor):
    @contextmanager
    def _ctx():
        yield None, cursor

    return _ctx


def _build_summary() -> dict:
    return {
        "request_id": "req-1001",
        "thread_id": "thread-1",
        "user_message": "latest ai updates",
        "final_status": "success",
        "latency_ms": 321,
        "evidence_count": 2,
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "error_code": None,
        "error_message": None,
        "exception_chain": [],
        "tool_call_chain": ["query_news", "compare_topics"],
        "final_answer_metadata": {"response_kind": "text", "answer_chars": 1200, "source_count": 2},
        "runtime": {"route": "react", "provider": "gemini_api", "model": "gemini-2.5-pro"},
        "tool_events": [
            {
                "event_index": 1,
                "tool_name": "query_news",
                "status": "success",
                "latency_ms": 123,
                "input_summary": {"query": "openai"},
                "output_summary": {"evidence_count": 1},
                "error_code": None,
                "error_message": None,
                "exception_chain": [],
            },
            {
                "event_index": 2,
                "tool_name": "compare_topics",
                "status": "empty",
                "latency_ms": 45,
                "input_summary": {"topic_a": "openai", "topic_b": "anthropic"},
                "output_summary": {"evidence_count": 0},
                "error_code": None,
                "error_message": None,
                "exception_chain": [],
            },
        ],
    }


def test_persist_request_trace_writes_agent_runs_and_tool_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _FakeCursor()
    monkeypatch.setattr(store_mod, "Json", lambda value: value)
    monkeypatch.setattr(store_mod, "db_transaction", _transaction_context(cursor))

    ok = store_mod.persist_request_trace(_build_summary())

    assert ok is True
    assert len(cursor.execute_calls) == 1
    run_sql, run_params = cursor.execute_calls[0]
    assert "INSERT INTO public.agent_runs" in run_sql
    assert run_params[0] == "req-1001"
    assert run_params[1] == "thread-1"
    assert run_params[2] == "latest ai updates"
    assert run_params[3] == "success"
    assert run_params[4] == 321
    assert run_params[5] == 2
    assert run_params[6]["total_tokens"] == 30
    assert run_params[11]["summary"]["request_id"] == "req-1001"
    assert run_params[11]["final_answer_metadata"]["response_kind"] == "text"
    assert run_params[11]["runtime"]["route"] == "react"

    assert len(cursor.executemany_calls) == 1
    tool_sql, tool_rows = cursor.executemany_calls[0]
    assert "INSERT INTO public.agent_tool_events" in tool_sql
    assert len(tool_rows) == 2
    assert tool_rows[0][0] == "req-1001"
    assert tool_rows[0][1] == 1
    assert tool_rows[0][2] == "query_news"
    assert tool_rows[0][3] == "success"
    assert tool_rows[0][4] == 123
    assert tool_rows[1][2] == "compare_topics"
    assert tool_rows[1][3] == "empty"


def test_persist_request_trace_skips_tool_batch_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _build_summary()
    summary["tool_events"] = []

    cursor = _FakeCursor()
    monkeypatch.setattr(store_mod, "Json", lambda value: value)
    monkeypatch.setattr(store_mod, "db_transaction", _transaction_context(cursor))

    ok = store_mod.persist_request_trace(summary)

    assert ok is True
    assert len(cursor.execute_calls) == 1
    assert len(cursor.executemany_calls) == 0
