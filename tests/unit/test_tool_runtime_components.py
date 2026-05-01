"""Unit tests for v2 tool runtime building blocks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.core.role_policy import assert_tool_allowed
from agent.core.tool_contracts import (
    ToolEnvelope,
    build_tool_empty_envelope,
    build_tool_error_envelope,
)
from agent.core.tool_registry import ToolRegistry
from agent.core.tool_runtime import ToolRuntime, ToolRuntimeContext, format_tool_envelope_for_model
from agent.core.tool_runtime_hooks import ToolRuntimeHooks


class _DummyInput(BaseModel):
    value: int = Field(ge=1)


def _dummy_handler(payload: _DummyInput) -> dict:
    return {
        "tool": "dummy_tool",
        "status": "ok",
        "request": payload.model_dump(mode="python"),
        "data": {"double": payload.value * 2},
        "evidence": [],
    }


def _empty_handler(payload: _DummyInput) -> ToolEnvelope:
    return build_tool_empty_envelope(
        "dummy_tool",
        payload.model_dump(mode="python"),
        "no_dummy_records",
    )


def _error_handler(payload: _DummyInput) -> ToolEnvelope:
    return build_tool_error_envelope(
        "dummy_tool",
        payload.model_dump(mode="python"),
        "dummy_failed",
        error_code="dummy_failed",
    )


def test_tool_registry_executes_valid_payload() -> None:
    registry = ToolRegistry()
    registry.register("dummy_tool", _DummyInput, _dummy_handler, "test tool")

    envelope = registry.execute("dummy_tool", {"value": 3})
    assert envelope.status == "ok"
    assert envelope.tool == "dummy_tool"
    assert envelope.data["double"] == 6


def test_tool_registry_rejects_invalid_payload() -> None:
    registry = ToolRegistry()
    registry.register("dummy_tool", _DummyInput, _dummy_handler, "test tool")

    envelope = registry.execute("dummy_tool", {"value": 0})
    assert envelope.status == "error"
    assert envelope.error == "input_validation_failed"


def test_tool_registry_rejects_empty_name() -> None:
    registry = ToolRegistry()
    try:
        registry.register("   ", _DummyInput, _dummy_handler, "test tool")
        raise AssertionError("Expected ValueError for empty tool name")
    except ValueError as exc:
        assert "must not be empty" in str(exc)


def test_tool_registry_normalizes_lookup_names() -> None:
    registry = ToolRegistry()
    registry.register("dummy_tool", _DummyInput, _dummy_handler, "test tool")

    assert registry.has("  dummy_tool  ")
    assert registry.get("  dummy_tool  ").name == "dummy_tool"
    envelope = registry.execute("  dummy_tool  ", {"value": 2})
    assert envelope.status == "ok"
    assert envelope.tool == "dummy_tool"


def test_role_policy_denies_unknown_role() -> None:
    allowed, reason = assert_tool_allowed("unknown_role", "query_news")
    assert not allowed
    assert reason == "unknown_role:unknown_role"


def test_tool_hook_runner_denies_invalid_window() -> None:
    hooks = ToolRuntimeHooks()
    decision = hooks.pre_tool_use("trend_analysis", {"topic": "OpenAI", "window": 999})
    assert decision.action == "deny"
    assert "between 3 and 60" in str(decision.reason)


def test_tool_hook_runner_denies_invalid_integer_payload() -> None:
    hooks = ToolRuntimeHooks()
    decision_days = hooks.pre_tool_use("query_news", {"query": "OpenAI", "days": "NaN"})
    decision_window = hooks.pre_tool_use("trend_analysis", {"topic": "OpenAI", "window": "bad"})
    assert decision_days.action == "deny"
    assert "integer" in str(decision_days.reason)
    assert decision_window.action == "deny"
    assert "integer" in str(decision_window.reason)


def test_tool_runtime_execute_returns_ok_envelope() -> None:
    registry = ToolRegistry()
    registry.register("dummy_tool", _DummyInput, _dummy_handler, "test tool")
    runtime = ToolRuntime(registry=registry, hooks=ToolRuntimeHooks())

    envelope = runtime.execute(
        "dummy_tool",
        {"value": 4},
        ToolRuntimeContext(trace=False, emit_progress_events=False),
    )

    assert envelope.status == "ok"
    assert envelope.tool == "dummy_tool"
    assert envelope.data["double"] == 8


def test_tool_envelope_has_no_runtime_version_field() -> None:
    envelope = ToolEnvelope(
        tool="dummy_tool",
        status="ok",
        request={"value": 1},
        data={"value": 2},
    )

    assert "schema" + "_version" not in envelope.to_dict()


def test_tool_runtime_execute_returns_empty_envelope() -> None:
    registry = ToolRegistry()
    registry.register("dummy_tool", _DummyInput, _empty_handler, "test tool")
    runtime = ToolRuntime(registry=registry, hooks=ToolRuntimeHooks())

    envelope = runtime.execute(
        "dummy_tool",
        {"value": 1},
        ToolRuntimeContext(trace=False, emit_progress_events=False),
    )

    assert envelope.status == "empty"
    assert envelope.diagnostics["empty_reason"] == "no_dummy_records"


def test_tool_runtime_execute_returns_error_envelope() -> None:
    registry = ToolRegistry()
    registry.register("dummy_tool", _DummyInput, _error_handler, "test tool")
    runtime = ToolRuntime(registry=registry, hooks=ToolRuntimeHooks())

    envelope = runtime.execute(
        "dummy_tool",
        {"value": 1},
        ToolRuntimeContext(trace=False, emit_progress_events=False),
    )

    assert envelope.status == "error"
    assert envelope.error_code == "dummy_failed"
    assert format_tool_envelope_for_model(envelope).startswith("[Error]")


def test_tool_runtime_policy_block_does_not_execute_handler() -> None:
    calls = {"count": 0}

    def handler(payload: _DummyInput) -> dict:
        calls["count"] += 1
        return _dummy_handler(payload)

    registry = ToolRegistry()
    registry.register("query_news", _DummyInput, handler, "test tool")
    runtime = ToolRuntime(registry=registry, hooks=ToolRuntimeHooks())

    envelope = runtime.execute(
        "query_news",
        {"value": 1, "days": 999},
        ToolRuntimeContext(trace=False, emit_progress_events=False),
    )

    assert envelope.status == "error"
    assert envelope.error_code == "tool_pre_hook_denied"
    assert calls["count"] == 0


