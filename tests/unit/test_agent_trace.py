"""Tests for request-level tracing in agent runtime."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from agent.core.trace import (
    finalize_request_trace,
    request_trace_context,
    trace_span,
)

pytestmark = pytest.mark.usefixtures("agent_dependency_stubs")


def test_trace_records_tool_call_spans_and_chain() -> None:
    with request_trace_context(user_message="latest ai updates", request_id="req-trace-001"):
        with trace_span("tool_call", "query_news", input_summary={"args": {"query": "openai", "days": 7}}) as span:
            span.set_output({"status": "success", "evidence_count": 1, "evidence_urls": ["https://a.example.com"]})
        with trace_span("tool_call", "query_news", input_summary={"args": {"query": "unknown", "days": 7}}) as span:
            span.status = "empty"
            span.set_output({"status": "empty", "evidence_count": 0})
        with trace_span("tool_call", "read_news_content", input_summary={"args": {"url": "https://a.example.com"}}) as span:
            err = RuntimeError("fetch failed")
            span.set_error(error_code="tool_runtimeerror", error_message=str(err), error=err)

        summary = finalize_request_trace(
            final_status="error",
            error_code="graph_unexpected_runtime_error",
            error_message="runtime failed",
            error=RuntimeError("runtime failed"),
        )

    assert summary is not None
    assert summary["request_id"] == "req-trace-001"
    assert summary["user_message"] == "latest ai updates"
    assert summary["final_status"] == "error"
    assert summary["error_code"] == "graph_unexpected_runtime_error"
    assert summary["tool_call_chain"] == ["query_news", "query_news", "read_news_content"]

    tool_spans = [span for span in summary["spans"] if span["span_type"] == "tool_call"]
    assert [span["status"] for span in tool_spans] == ["success", "empty", "error"]
    assert tool_spans[2]["error_code"] == "tool_runtimeerror"
    assert tool_spans[0]["output_summary"]["evidence_urls"] == ["https://a.example.com"]


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
    assert ei.value.code == "graph_inline_citation_missing"
    assert summary is not None
    assert summary["final_status"] == "blocked"
    assert summary["error_code"] == "graph_inline_citation_missing"


def test_generate_response_eval_payload_preserves_tool_call_sequence() -> None:
    from agent.agent import generate_response_eval_payload

    expected = "Main analysis [1].\n\n## Sources\n- [1] https://a.example.com"
    tool_chain = ["search_news", "query_news", "search_news"]
    with (
        patch(
            "agent.agent._generate_response_core",
            return_value=("Main analysis cites https://a.example.com.", {"https://a.example.com"}),
        ),
        patch("agent.agent._decorate_response_with_sources", return_value=(expected, {})),
        patch(
            "agent.agent._finalize_request_trace",
            return_value={"tool_call_chain": tool_chain},
        ),
        patch("agent.agent._get_accumulated_tool_call_chain", return_value=tool_chain),
    ):
        payload = generate_response_eval_payload([], "summarize recent ai updates")

    assert payload["tool_calls"] == tool_chain


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


def test_trace_span_records_parent_child_and_full_model_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_TRACE_FULL_MODEL_IO", "true")
    monkeypatch.setenv("AGENT_TRACE_SECRET_REDACTION", "true")
    persisted: list[dict] = []
    monkeypatch.setattr("agent.core.trace._persist_request_trace", lambda summary: persisted.append(summary) or True)

    with request_trace_context(user_message="hello", request_id="req-span-1"):
        with trace_span("graph_node", "intent_router") as parent:
            parent.set_output({"route": "needs_tools"})
            with trace_span("model_call", "intent_router") as child:
                child.set_model_io(
                    node="intent_router",
                    provider="gemini_api",
                    model="gemini-test",
                    input_messages=[
                        HumanMessage(
                            content=(
                                "business prompt remains intact. "
                                "GEMINI_API_KEY=sk-12345678901234567890"
                            )
                        )
                    ],
                    raw_output=AIMessage(
                        content=(
                            "model output remains intact. "
                            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
                        )
                    ),
                    token_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                )
                child.set_output({"parsed": True})

        summary = finalize_request_trace(final_status="success", evidence_count=0)

    assert summary is not None
    spans = summary["spans"]
    assert [span["name"] for span in spans] == ["intent_router", "intent_router"]
    assert spans[1]["parent_span_id"] == spans[0]["span_id"]
    assert spans[1]["span_type"] == "model_call"

    model_io = persisted[0]["model_io"]
    assert len(model_io) == 1
    assert model_io[0]["span_id"] == spans[1]["span_id"]
    content = model_io[0]["input_messages"][0]["content"]
    assert "business prompt remains intact" in content
    assert "sk-12345678901234567890" not in content
    assert "GEMINI_API_KEY=[REDACTED]" in content
    raw_content = model_io[0]["raw_output"]["content"]
    assert "model output remains intact" in raw_content
    assert "abcdefghijklmnopqrstuvwxyz" not in raw_content
    assert summary["model_io"] == [
        {
            "span_id": spans[1]["span_id"],
            "node": "intent_router",
            "provider": "gemini_api",
            "model": "gemini-test",
            "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    ]


def test_trace_full_model_io_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_TRACE_FULL_MODEL_IO", "false")
    monkeypatch.setattr("agent.core.trace._persist_request_trace", lambda _summary: True)

    with request_trace_context(user_message="hello", request_id="req-no-model-io"):
        with trace_span("model_call", "intent_router") as span:
            span.set_model_io(
                node="intent_router",
                provider="gemini_api",
                model="gemini-test",
                input_messages=[HumanMessage(content="full prompt")],
                raw_output=AIMessage(content="full output"),
            )
        summary = finalize_request_trace(final_status="success", evidence_count=0)

    assert summary is not None
    assert summary["model_io"] == []


def test_missing_model_client_still_records_model_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.graph.nodes import _invoke_text_model
    from agent.graph.state import GraphModelHandle

    monkeypatch.setenv("AGENT_TRACE_FULL_MODEL_IO", "true")
    persisted: list[dict] = []
    monkeypatch.setattr("agent.core.trace._persist_request_trace", lambda summary: persisted.append(summary) or True)

    with request_trace_context(user_message="hello", request_id="req-missing-client"):
        text = _invoke_text_model(
            GraphModelHandle(
                role="intent_router",
                provider="gemini_api",
                model="gemini-test",
                client=None,
                fallback=True,
                error="missing config",
            ),
            node="intent_router",
            messages=[HumanMessage(content="full prompt")],
        )
        summary = finalize_request_trace(final_status="success", evidence_count=0)

    assert text == ""
    assert summary is not None
    assert summary["spans"][0]["span_type"] == "model_call"
    assert persisted[0]["model_io"][0]["raw_output"] == {
        "status": "skipped",
        "reason": "missing_client",
    }


def test_trace_span_marks_exceptions_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent.core.trace._persist_request_trace", lambda _summary: True)

    with request_trace_context(user_message="hello", request_id="req-span-error"):
        with pytest.raises(ValueError):
            with trace_span("guard", "output_guard"):
                raise ValueError("bad url")
        summary = finalize_request_trace(final_status="error", error_code="guard_failed")

    assert summary is not None
    span = summary["spans"][0]
    assert span["status"] == "error"
    assert span["error_code"] == "guard_valueerror"
    assert span["exception_chain"][0]["type"] == "ValueError"
