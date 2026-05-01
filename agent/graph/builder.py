"""Custom LangGraph builder and invocation entrypoint."""

from __future__ import annotations

import os
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from agent.core.runtime_factories import build_default_tool_runtime
from agent.core.trace import get_current_request_id, get_current_thread_id

from .models import build_graph_models
from .nodes import GraphDependencies, GraphNodeRunner
from .routing import route_after_intent, route_after_loop_decider, route_after_policy
from .state import AgentGraphState, AgentRunResult, GraphModels, GraphRuntimeConfig


def build_custom_graph(
    *,
    models: GraphModels | None = None,
    config: GraphRuntimeConfig | None = None,
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
    graph.add_node("output_guard", runner.output_guard)
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
    graph.add_edge("final_synthesizer", "output_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("clarification_response", END)
    graph.add_edge("insufficient_evidence_response", "output_guard")
    return graph.compile(name="custom_agent_graph")


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
    initial: AgentGraphState = {
        "request_id": str(request_id or get_current_request_id() or ""),
        "thread_id": str(thread_id or get_current_thread_id() or "") or None,
        "user_message": str(user_message or ""),
        "history": list(history or []),
        "tool_results": [],
        "evidence_urls": [],
        "valid_urls": [],
        "tool_round": 0,
        "max_tool_rounds": runtime_config.max_tool_rounds,
        "node_audit": [],
        "model_usage": {},
    }
    final_state = graph.invoke(initial)
    clarification = final_state.get("clarification")
    return AgentRunResult(
        text=str(final_state.get("final_text") or "").strip(),
        urls=[str(url).strip() for url in (final_state.get("valid_urls") or final_state.get("evidence_urls") or []) if str(url).strip()],
        clarification=clarification if isinstance(clarification, dict) else None,
    )


def _max_tool_rounds() -> int:
    raw = os.getenv("AGENT_GRAPH_MAX_TOOL_ROUNDS", "2")
    try:
        return max(1, min(int(str(raw).strip()), 6))
    except Exception:
        return 2
