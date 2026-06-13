"""Custom LangGraph builder and invocation entrypoint."""

from __future__ import annotations

import os
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.core.runtime_factories import build_default_tool_runtime
from agent.core.trace import get_current_request_id, get_current_thread_id

from .checkpointer import get_checkpointer
from .models import build_graph_models
from .nodes import GraphDependencies, GraphNodeRunner
from .routing import (
    route_after_clarification,
    route_after_intent,
    route_after_loop_decider,
    route_after_policy,
)
from .state import AgentGraphState, AgentRunResult, GraphModels, GraphRuntimeConfig


_USE_DEFAULT_CHECKPOINTER = object()


def build_custom_graph(
    *,
    models: GraphModels | None = None,
    config: GraphRuntimeConfig | None = None,
    checkpointer: Any = _USE_DEFAULT_CHECKPOINTER,
) -> Any:
    runtime_config = config or GraphRuntimeConfig(max_tool_rounds=_max_tool_rounds())
    deps = GraphDependencies(
        models=models or build_graph_models(),
        tool_runtime=build_default_tool_runtime(),
        config=runtime_config,
    )
    runner = GraphNodeRunner(deps)

    graph = StateGraph(AgentGraphState)
    graph.add_node("prepare_context", runner.prepare_context)
    graph.add_node("intent_router", runner.intent_router)
    graph.add_node("tool_selection", runner.tool_selection)
    graph.add_node("tool_worker", runner.tool_worker)
    graph.add_node("tool_policy", runner.tool_policy)
    graph.add_node("tool_executor", runner.tool_executor)
    graph.add_node("evidence_normalizer", runner.evidence_normalizer)
    graph.add_node("tool_loop_decider", runner.tool_loop_decider)
    graph.add_node("final_synthesizer", runner.final_synthesizer)
    graph.add_node("clarification_response", runner.clarification_response)
    graph.add_node("insufficient_evidence_response", runner.insufficient_evidence_response)

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "direct_answer": "final_synthesizer",
            "needs_clarification": "clarification_response",
            "needs_tools": "tool_selection",
        },
    )
    graph.add_edge("tool_selection", "tool_worker")
    graph.add_edge("tool_worker", "tool_policy")
    graph.add_conditional_edges(
        "tool_policy",
        route_after_policy,
        {"blocked": "clarification_response", "allowed": "tool_executor"},
    )
    graph.add_edge("tool_executor", "evidence_normalizer")
    graph.add_edge("evidence_normalizer", "tool_loop_decider")
    graph.add_conditional_edges(
        "tool_loop_decider",
        route_after_loop_decider,
        {
            "more_tools": "tool_worker",
            "enough_evidence": "final_synthesizer",
            "insufficient_evidence": "insufficient_evidence_response",
        },
    )
    graph.add_edge("final_synthesizer", END)
    graph.add_conditional_edges(
        "clarification_response",
        route_after_clarification,
        {"answer": "tool_selection", "end": END},
    )
    graph.add_edge("insufficient_evidence_response", END)
    resolved_checkpointer = (
        get_checkpointer() if checkpointer is _USE_DEFAULT_CHECKPOINTER else checkpointer
    )
    return graph.compile(name="custom_agent_graph", checkpointer=resolved_checkpointer)


def invoke_custom_graph(
    history: list[dict],
    user_message: str,
    *,
    request_id: str | None = None,
    thread_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    models: GraphModels | None = None,
    config: GraphRuntimeConfig | None = None,
) -> AgentRunResult:
    del progress_callback  # progress is delivered through request run context.
    runtime_config = config or GraphRuntimeConfig(max_tool_rounds=_max_tool_rounds())
    graph = build_custom_graph(models=models, config=runtime_config)
    resolved_request_id = str(request_id or get_current_request_id() or "")
    resolved_thread_id = str(thread_id or get_current_thread_id() or "") or None
    # Full per-turn reset: with a checkpointer the thread state persists across
    # turns, so a fresh turn must overwrite every channel to avoid state bleed.
    initial: AgentGraphState = {
        "request_id": resolved_request_id,
        "thread_id": resolved_thread_id,
        "user_message": str(user_message or ""),
        "history": list(history or []),
        "llm_input_messages": [],
        "context_pack": {},
        "intent": {},
        "selected_tools": [],
        "pending_tool_calls": [],
        "tool_results": [],
        "tool_round": 0,
        "max_tool_rounds": runtime_config.max_tool_rounds,
        "evidence_urls": [],
        "evidence_brief": "",
        "final_text": "",
        "valid_urls": [],
        "clarification": None,
        "next_step": "",
        "clarified": False,
        "clarify_count": 0,
        "clar_route": "",
    }
    invoke_config = None
    if get_checkpointer() is not None:
        checkpoint_thread = resolved_thread_id or resolved_request_id or uuid4().hex
        invoke_config = {"configurable": {"thread_id": checkpoint_thread}}
    if invoke_config is not None:
        pending = bool(getattr(graph.get_state(invoke_config), "next", None))
        if pending:
            # A clarification interrupt is waiting on this thread: treat the new
            # message as the user's reply and resume from the checkpoint.
            out = graph.invoke(Command(resume=str(user_message or "")), config=invoke_config)
        else:
            out = graph.invoke(initial, config=invoke_config)
    else:
        out = graph.invoke(initial)

    interrupts = out.get("__interrupt__") if isinstance(out, dict) else None
    if interrupts:
        value = getattr(interrupts[0], "value", interrupts[0])
        payload = value if isinstance(value, dict) else {"question": str(value)}
        return AgentRunResult(text=str(payload.get("question") or "").strip(), urls=[], clarification=payload)

    final_state = out
    clarification = final_state.get("clarification")
    return AgentRunResult(
        text=str(final_state.get("final_text") or "").strip(),
        urls=[str(url).strip() for url in (final_state.get("valid_urls") or final_state.get("evidence_urls") or []) if str(url).strip()],
        citable_urls=[str(url).strip() for url in (final_state.get("citable_urls") or []) if str(url).strip()],
        clarification=clarification if isinstance(clarification, dict) else None,
    )


def _max_tool_rounds() -> int:
    raw = os.getenv("AGENT_GRAPH_MAX_TOOL_ROUNDS", "2")
    try:
        return max(1, min(int(str(raw).strip()), 6))
    except Exception:
        return 2
