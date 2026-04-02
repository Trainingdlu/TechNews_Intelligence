"""Unit tests for v2 skill runtime building blocks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.core.role_policy import assert_skill_allowed
from agents.core.skill_registry import SkillRegistry
from agents.core.tool_hooks import ToolHookRunner
from agents.prompts import get_role_system_instruction


class _DummyInput(BaseModel):
    value: int = Field(ge=1)


def _dummy_handler(payload: _DummyInput) -> dict:
    return {
        "tool": "dummy_skill",
        "status": "ok",
        "request": payload.model_dump(mode="python"),
        "data": {"double": payload.value * 2},
        "evidence": [],
    }


def test_skill_registry_executes_valid_payload() -> None:
    registry = SkillRegistry()
    registry.register("dummy_skill", _DummyInput, _dummy_handler, "test skill")

    envelope = registry.execute("dummy_skill", {"value": 3})
    assert envelope.status == "ok"
    assert envelope.tool == "dummy_skill"
    assert envelope.data["double"] == 6


def test_skill_registry_rejects_invalid_payload() -> None:
    registry = SkillRegistry()
    registry.register("dummy_skill", _DummyInput, _dummy_handler, "test skill")

    envelope = registry.execute("dummy_skill", {"value": 0})
    assert envelope.status == "error"
    assert envelope.error == "input_validation_failed"


def test_skill_registry_rejects_empty_name() -> None:
    registry = SkillRegistry()
    try:
        registry.register("   ", _DummyInput, _dummy_handler, "test skill")
        raise AssertionError("Expected ValueError for empty skill name")
    except ValueError as exc:
        assert "must not be empty" in str(exc)


def test_skill_registry_normalizes_lookup_names() -> None:
    registry = SkillRegistry()
    registry.register("dummy_skill", _DummyInput, _dummy_handler, "test skill")

    assert registry.has("  dummy_skill  ")
    assert registry.get("  dummy_skill  ").name == "dummy_skill"
    envelope = registry.execute("  dummy_skill  ", {"value": 2})
    assert envelope.status == "ok"
    assert envelope.tool == "dummy_skill"


def test_role_policy_denies_unknown_role() -> None:
    allowed, reason = assert_skill_allowed("unknown_role", "query_news")
    assert not allowed
    assert reason == "unknown_role:unknown_role"


def test_tool_hook_runner_denies_invalid_window() -> None:
    hooks = ToolHookRunner()
    decision = hooks.pre_tool_use("trend_analysis", {"topic": "OpenAI", "window": 999})
    assert decision.action == "deny"
    assert "between 3 and 60" in str(decision.reason)


def test_tool_hook_runner_denies_invalid_integer_payload() -> None:
    hooks = ToolHookRunner()
    decision_days = hooks.pre_tool_use("query_news", {"query": "OpenAI", "days": "NaN"})
    decision_window = hooks.pre_tool_use("trend_analysis", {"topic": "OpenAI", "window": "bad"})
    assert decision_days.action == "deny"
    assert "integer" in str(decision_days.reason)
    assert decision_window.action == "deny"
    assert "integer" in str(decision_window.reason)


def test_role_prompt_lookup() -> None:
    text = get_role_system_instruction("miner")
    assert "Miner subagent" in text
