"""Conditional routing helpers for the custom graph."""

from __future__ import annotations

from .state import AgentGraphState


def route_after_intent(state: AgentGraphState) -> str:
    intent = state.get("intent") or {}
    route = str(intent.get("route") or "").strip()
    if route in {"direct_answer", "needs_clarification", "needs_tools"}:
        return route
    if state.get("clarification"):
        return "needs_clarification"
    return "needs_tools"


def route_after_policy(state: AgentGraphState) -> str:
    return "blocked" if state.get("clarification") else "allowed"


def route_after_loop_decider(state: AgentGraphState) -> str:
    next_step = str(state.get("next_step") or "").strip()
    if next_step in {"more_tools", "enough_evidence", "insufficient_evidence"}:
        return next_step
    if state.get("clarification"):
        return "insufficient_evidence"
    return "enough_evidence"
