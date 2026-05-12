"""Unit tests for the standalone trace dashboard API."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def trace_api_mod(agent_dependency_stubs, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001
    monkeypatch.setenv("TRACE_DASHBOARD_TOKEN", "unit-secret")
    monkeypatch.setenv("TRACE_DASHBOARD_ADMIN_EMAIL", "trace-admin@example.com")
    monkeypatch.delitem(sys.modules, "app.trace_api", raising=False)
    return importlib.import_module("app.trace_api")


class _FakeCursor:
    def __init__(self, handler):  # noqa: ANN001
        self.handler = handler
        self.calls: list[tuple[str, tuple]] = []
        self.description = []
        self.rows = []

    def execute(self, sql, params=()):  # noqa: ANN001
        sql_text = str(sql)
        params_tuple = tuple(params or ())
        self.calls.append((sql_text, params_tuple))
        columns, rows = self.handler(sql_text, params_tuple)
        self.description = [(column,) for column in columns]
        self.rows = rows

    def fetchall(self):  # noqa: ANN001
        return self.rows


def _install_db(monkeypatch: pytest.MonkeyPatch, trace_api_mod, handler):  # noqa: ANN001
    cursor = _FakeCursor(handler)

    @contextmanager
    def _ctx():
        yield None, cursor

    monkeypatch.setattr(trace_api_mod, "db_cursor", _ctx)
    return cursor


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer unit-secret"}


def test_trace_api_rejects_missing_or_wrong_token(trace_api_mod) -> None:  # noqa: ANN001
    with TestClient(trace_api_mod.app) as client:
        missing = client.get("/trace-api/meta")
        wrong = client.get("/trace-api/meta", headers={"Authorization": "Bearer wrong"})

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_trace_api_meta_returns_admin_and_langsmith(trace_api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "unit-project")

    with TestClient(trace_api_mod.app) as client:
        response = client.get("/trace-api/meta", headers=_auth_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["admin_email"] == "trace-admin@example.com"
    assert payload["langsmith"]["enabled"] is True
    assert payload["langsmith"]["project"] == "unit-project"


def test_list_runs_filters_and_returns_compact_rows(trace_api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    created_at = datetime(2026, 5, 12, 8, 30, tzinfo=timezone.utc)

    def _handler(sql: str, params: tuple):
        if "COUNT(*)" in sql:
            return ["total"], [(1,)]
        assert "final_status = %s" in sql
        assert "error_code = %s" in sql
        assert "user_message ILIKE %s" in sql
        assert "COALESCE(error_code, '') ILIKE %s" in sql
        assert params[:7] == (
            "error",
            "tool_failed",
            "%OpenAI%",
            "%OpenAI%",
            "%OpenAI%",
            "%OpenAI%",
            "%OpenAI%",
        )
        return (
            [
                "request_id",
                "thread_id",
                "user_message",
                "final_status",
                "latency_ms",
                "evidence_count",
                "token_usage",
                "error_code",
                "error_message",
                "exception_chain",
                "tool_call_chain",
                "trace_payload",
                "created_at",
            ],
            [
                (
                    "req-1",
                    "thread-1",
                    "OpenAI news",
                    "error",
                    1200,
                    2,
                    {"total_tokens": 9},
                    "tool_failed",
                    "boom",
                    [],
                    ["search_news"],
                    {"runtime": {"langsmith": {"enabled": False, "project": "p"}}},
                    created_at,
                )
            ],
        )

    _install_db(monkeypatch, trace_api_mod, _handler)

    with TestClient(trace_api_mod.app) as client:
        response = client.get(
            "/trace-api/runs?status=error&error_code=tool_failed&q=OpenAI",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["request_id"] == "req-1"
    assert payload["items"][0]["langsmith"]["project"] == "p"
    assert "T08:30:00" in payload["items"][0]["created_at"]


def test_get_run_builds_span_tree_with_chinese_labels(trace_api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    def _handler(sql: str, _params: tuple):
        if "FROM public.agent_runs" in sql:
            return (
                [
                    "request_id",
                    "thread_id",
                    "user_message",
                    "final_status",
                    "latency_ms",
                    "evidence_count",
                    "token_usage",
                    "error_code",
                    "error_message",
                    "exception_chain",
                    "tool_call_chain",
                    "trace_payload",
                    "created_at",
                ],
                [
                    (
                        "req-2",
                        "thread-2",
                        "latest ai",
                        "success",
                        330,
                        1,
                        None,
                        None,
                        None,
                        [],
                        ["search_news"],
                        {"runtime": {}},
                        datetime(2026, 5, 12, tzinfo=timezone.utc),
                    )
                ],
            )
        return (
            [
                "request_id",
                "span_id",
                "parent_span_id",
                "span_type",
                "name",
                "status",
                "started_at_ms",
                "finished_at_ms",
                "latency_ms",
                "input_summary",
                "output_summary",
                "error_code",
                "error_message",
                "exception_chain",
                "metadata",
                "created_at",
            ],
            [
                (
                    "req-2",
                    "span-root",
                    None,
                    "graph_node",
                    "intent_router",
                    "success",
                    1,
                    2,
                    1,
                    {},
                    {},
                    None,
                    None,
                    [],
                    {},
                    datetime(2026, 5, 12, tzinfo=timezone.utc),
                ),
                (
                    "req-2",
                    "span-child",
                    "span-root",
                    "model_call",
                    "intent_router",
                    "success",
                    2,
                    5,
                    3,
                    {},
                    {},
                    None,
                    None,
                    [],
                    {},
                    datetime(2026, 5, 12, tzinfo=timezone.utc),
                ),
            ],
        )

    _install_db(monkeypatch, trace_api_mod, _handler)

    with TestClient(trace_api_mod.app) as client:
        response = client.get("/trace-api/runs/req-2", headers=_auth_headers())

    assert response.status_code == 200
    tree = response.json()["span_tree"]
    assert tree[0]["display_name"] == "判断问题类型"
    assert tree[0]["children"][0]["display_name"] == "模型调用：判断问题类型"
    assert tree[0]["children"][0]["span_type_label"] == "模型调用"


def test_model_io_endpoint_returns_full_payload(trace_api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    def _handler(sql: str, _params: tuple):
        if "FROM public.agent_trace_spans" in sql:
            return ["request_id", "span_id", "span_type", "name"], [("req-3", "span-model", "model_call", "final_synthesizer")]
        return (
            [
                "request_id",
                "span_id",
                "node",
                "provider",
                "model",
                "input_messages",
                "raw_output",
                "parsed_output",
                "token_usage",
                "created_at",
            ],
            [
                (
                    "req-3",
                    "span-model",
                    "final_synthesizer",
                    "vertex",
                    "gemini-test",
                    [{"role": "human", "content": "full prompt"}],
                    {"content": "full output"},
                    {"answer": "ok"},
                    {"total_tokens": 12},
                    datetime(2026, 5, 12, tzinfo=timezone.utc),
                )
            ],
        )

    _install_db(monkeypatch, trace_api_mod, _handler)

    with TestClient(trace_api_mod.app) as client:
        response = client.get(
            "/trace-api/spans/span-model/model-io?request_id=req-3",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()["model_io"]
    assert payload["input_messages"][0]["content"] == "full prompt"
    assert payload["raw_output"]["content"] == "full output"


def test_model_io_endpoint_rejects_non_model_span(trace_api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    _install_db(
        monkeypatch,
        trace_api_mod,
        lambda _sql, _params: (
            ["request_id", "span_id", "span_type", "name"],
            [("req-4", "span-tool", "tool_call", "search_news")],
        ),
    )

    with TestClient(trace_api_mod.app) as client:
        response = client.get(
            "/trace-api/spans/span-tool/model-io?request_id=req-4",
            headers=_auth_headers(),
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "model_io_only_available_for_model_call"
