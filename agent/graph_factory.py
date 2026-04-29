"""LangGraph ReAct graph construction and runtime hooks."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from .clarification import (
    CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
    ClarificationRequiredError,
    build_clarification_payload,
)
from .core.metrics import metrics_inc
from .core.run_context import get_evidence_urls as _get_accumulated_evidence
from .core.trace import record_tool_policy_block
from .memory_policy import build_llm_input_messages
from .tool_policy import evaluate_tool_calls


def pre_model_hook(state: dict[str, Any]) -> dict[str, Any]:
    """Build an LLM-only trimmed message view without mutating graph messages."""
    messages = _state_messages(state)
    return {"llm_input_messages": build_llm_input_messages(messages)}


def make_post_model_hook(
    *,
    allowed_tool_names: set[str],
    input_schemas: dict[str, dict[str, Any]] | None = None,
):
    """Create a post-model hook bound to the current tool inventory."""

    def post_model_hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = _state_messages(state)
        decision = evaluate_tool_calls(
            messages,
            allowed_tool_names=allowed_tool_names,
            input_schemas=input_schemas or {},
            evidence_urls=_get_accumulated_evidence(),
        )
        if decision.allowed:
            return {}

        metrics_inc("tool_policy_blocked_total")
        if decision.reason:
            metrics_inc(f"tool_policy_blocked_{decision.reason}")
        record_tool_policy_block(
            reason=decision.reason,
            details=decision.details or {},
        )

        user_message = _latest_user_message(messages)
        payload = build_clarification_payload(
            user_message=user_message,
            reason=CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
            context={
                "tool_policy_reason": decision.reason,
                "tool_policy_details": decision.details or {},
            },
        )
        raise ClarificationRequiredError(payload)

    return post_model_hook


def build_react_graph(
    *,
    create_agent: Any,
    model: Any,
    tools: list[Any],
    prompt_kwargs: dict[str, Any],
    input_schemas: dict[str, dict[str, Any]] | None = None,
):
    """Build the project ReAct graph with memory and tool-policy hooks."""
    allowed_tool_names = {
        str(getattr(tool, "name", "")).strip()
        for tool in tools
        if str(getattr(tool, "name", "")).strip()
    }
    return create_agent(
        model=model,
        tools=tools,
        pre_model_hook=pre_model_hook,
        post_model_hook=make_post_model_hook(
            allowed_tool_names=allowed_tool_names,
            input_schemas=input_schemas or {},
        ),
        version="v2",
        **prompt_kwargs,
    )


def _state_messages(state: dict[str, Any]) -> list[BaseMessage]:
    messages = state.get("messages", []) if isinstance(state, dict) else []
    return [message for message in list(messages or []) if isinstance(message, BaseMessage)]


def _latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        if message_type != "human":
            continue
        content = getattr(message, "content", "")
        return str(content or "").strip()
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            return str(getattr(message, "content", "") or "").strip()
    return ""
