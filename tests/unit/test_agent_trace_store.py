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
        "runtime": {"route": "custom_graph", "provider": "gemini_api", "model": "gemini-2.5-pro"},
    }


def test_persist_request_trace_writes_agent_runs_only_without_spans(
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
    assert run_params[11]["runtime"]["route"] == "custom_graph"
    assert len(cursor.executemany_calls) == 0


def test_persist_request_trace_skips_span_batches_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _build_summary()

    cursor = _FakeCursor()
    monkeypatch.setattr(store_mod, "Json", lambda value: value)
    monkeypatch.setattr(store_mod, "db_transaction", _transaction_context(cursor))

    ok = store_mod.persist_request_trace(summary)

    assert ok is True
    assert len(cursor.execute_calls) == 1
    assert len(cursor.executemany_calls) == 0


def test_persist_request_trace_writes_spans_and_model_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _build_summary()
    summary["spans"] = [
        {
            "span_id": "span-1",
            "parent_span_id": None,
            "span_type": "graph_node",
            "name": "intent_router",
            "status": "success",
            "started_at_ms": 100,
            "finished_at_ms": 130,
            "latency_ms": 30,
            "input_summary": {"message_count": 1},
            "output_summary": {"route": "needs_tools"},
            "error_code": None,
            "error_message": None,
            "exception_chain": [],
            "metadata": {"node": "intent_router"},
        },
        {
            "span_id": "span-2",
            "parent_span_id": "span-1",
            "span_type": "model_call",
            "name": "intent_router",
            "status": "success",
            "started_at_ms": 101,
            "finished_at_ms": 120,
            "latency_ms": 19,
            "input_summary": {"message_count": 2},
            "output_summary": {"output_chars": 12},
            "error_code": None,
            "error_message": None,
            "exception_chain": [],
            "metadata": {"provider": "gemini_api", "model": "gemini-test"},
        },
    ]
    summary["model_io"] = [
        {
            "request_id": "req-1001",
            "span_id": "span-2",
            "node": "intent_router",
            "provider": "gemini_api",
            "model": "gemini-test",
            "input_messages": [{"type": "human", "content": "full prompt"}],
            "raw_output": {"type": "ai", "content": "full output"},
            "parsed_output": {"route": "needs_tools"},
            "token_usage": {"total_tokens": 3},
        }
    ]

    cursor = _FakeCursor()
    monkeypatch.setattr(store_mod, "Json", lambda value: value)
    monkeypatch.setattr(store_mod, "db_transaction", _transaction_context(cursor))

    ok = store_mod.persist_request_trace(summary)

    assert ok is True
    assert len(cursor.execute_calls) == 1
    assert len(cursor.executemany_calls) == 2
    span_sql, span_rows = cursor.executemany_calls[0]
    assert "INSERT INTO public.agent_trace_spans" in span_sql
    assert span_rows[0][0] == "req-1001"
    assert span_rows[0][1] == "span-1"
    assert span_rows[1][2] == "span-1"

    model_sql, model_rows = cursor.executemany_calls[1]
    assert "INSERT INTO public.agent_model_io" in model_sql
    assert model_rows[0][1] == "span-2"
    assert model_rows[0][5][0]["content"] == "full prompt"
    assert model_rows[0][6]["content"] == "full output"

    run_payload = cursor.execute_calls[0][1][11]
    assert "spans" not in run_payload["summary"]
    assert "model_io" not in run_payload["summary"]
