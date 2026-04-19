"""Tests for request-level tracing in agent runtime."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.core.skill_contracts import SkillEnvelope, SkillEvidence
from agent.core.trace import (
    finalize_request_trace,
    request_trace_context,
    trace_tool_finish_error,
    trace_tool_finish_with_envelope,
    trace_tool_start,
)

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def test_trace_records_tool_success_empty_and_error() -> None:
    with request_trace_context(user_message="latest ai updates", request_id="req-trace-001"):
        evt_ok = trace_tool_start("query_news", {"query": "openai", "days": 7})
        trace_tool_finish_with_envelope(
            evt_ok,
            SkillEnvelope(
                tool="query_news",
                status="ok",
                request={"query": "openai", "days": 7},
                data={"records": [{"title": "a"}]},
                evidence=[SkillEvidence(url="https://a.example.com")],
            ),
        )

        evt_empty = trace_tool_start("query_news", {"query": "unknown", "days": 7})
        trace_tool_finish_with_envelope(
            evt_empty,
            SkillEnvelope(
                tool="query_news",
                status="empty",
                request={"query": "unknown", "days": 7},
                data={"records": []},
                evidence=[],
            ),
        )

        evt_error = trace_tool_start("read_news_content", {"url": "https://a.example.com"})
        err = RuntimeError("fetch failed")
        trace_tool_finish_error(
            evt_error,
            error_code="tool_runtimeerror",
            error_message=str(err),
            error=err,
        )

        summary = finalize_request_trace(
            final_status="error",
            error_code="react_unexpected_runtime_error",
            error_message="runtime failed",
            error=RuntimeError("runtime failed"),
        )

    assert summary is not None
    assert summary["request_id"] == "req-trace-001"
    assert summary["user_message"] == "latest ai updates"
    assert summary["final_status"] == "error"
    assert summary["error_code"] == "react_unexpected_runtime_error"
    assert summary["evidence_count"] == 1
    assert summary["tool_call_chain"] == ["query_news", "query_news", "read_news_content"]

    event_statuses = [event["status"] for event in summary["tool_events"]]
    assert event_statuses == ["success", "empty", "error"]
    assert summary["tool_events"][2]["error_code"] == "tool_runtimeerror"
    assert summary["tool_events"][0]["output_summary"]["context_count"] == 1


def test_generate_response_finalizes_success_trace() -> None:
    from agent.agent import generate_response, get_last_request_trace_summary

    expected = "Main analysis [1].\n\n## Sources\n- [1] https://a.example.com"
    with (
        patch("agent.agent._run_generation_core", return_value=("Main analysis cites https://a.example.com.", {"https://a.example.com"})),
        patch("agent.agent._decorate_response_with_sources", return_value=(expected, {})),
    ):
        out = generate_response([], "summarize recent ai updates")

    summary = get_last_request_trace_summary()
    assert out == expected
    assert summary is not None
    assert summary["final_status"] == "success"
    assert summary["evidence_count"] == 1
    assert summary["user_message"] == "summarize recent ai updates"
    assert isinstance(summary["request_id"], str) and summary["request_id"]


def test_generate_response_finalizes_blocked_trace() -> None:
    from agent.agent import AgentGenerationError, generate_response, get_last_request_trace_summary

    with (
        patch("agent.agent._run_generation_core", return_value=("Main analysis body.", {"https://a.example.com"})),
        patch(
            "agent.agent._decorate_response_with_sources",
            return_value=("Main analysis body.\n\n## Sources\n- [1] https://a.example.com", {}),
        ),
        patch.dict("os.environ", {"AGENT_STRICT_INLINE_CITATIONS": "true"}),
    ):
        with pytest.raises(AgentGenerationError) as ei:
            generate_response([], "analyze ai trend")

    summary = get_last_request_trace_summary()
    assert ei.value.code == "react_inline_citation_missing"
    assert summary is not None
    assert summary["final_status"] == "blocked"
    assert summary["error_code"] == "react_inline_citation_missing"


def test_finalize_request_trace_reentry_does_not_persist_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.core.trace as trace_mod

    persisted: list[str] = []

    def _fake_persist(summary: dict) -> bool:
        persisted.append(str(summary.get("request_id")))
        return True

    monkeypatch.setattr(trace_mod, "_persist_request_trace", _fake_persist)

    with request_trace_context(user_message="hello", request_id="req-reentry-1"):
        first = finalize_request_trace(final_status="success", evidence_count=0)
        second = finalize_request_trace(final_status="error", error_code="should_not_override")

    assert first is not None
    assert second is not None
    assert first["request_id"] == "req-reentry-1"
    assert second["request_id"] == "req-reentry-1"
    assert first["final_status"] == "success"
    assert second["final_status"] == "success"
    assert persisted == ["req-reentry-1"]
