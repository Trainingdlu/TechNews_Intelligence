"""Node implementations for the custom LangGraph agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import wraps
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.clarification import (
    build_clarification_payload,
    infer_clarification_reason,
)
from agent.context_manager import (
    active_question,
    build_context_curator_messages,
    build_context_pack,
    build_history_manifest,
    normalize_context_curator_result,
    render_context_for_prompt,
    should_use_context_curator,
)
from agent.core.evidence import normalize_url_for_match
from agent.core.runtime_factories import build_default_registry, build_default_tool_runtime
from agent.core.tool_contracts import ToolEnvelope
from agent.core.tool_runtime import (
    ToolRuntime,
    ToolRuntimeContext,
    format_tool_results_for_final_synthesis,
)
from agent.core.trace import trace_span
from agent.memory_policy import build_llm_input_messages
from agent.tool_policy import evaluate_pending_tool_calls

from .evidence_flow import (
    _empty_evidence_fallback_calls,
    _fulltext_calls_from_evidence,
    _normalize_evidence,
    _requires_evidence,
    _should_read_after_search,
)
from .intent_heuristics import _heuristic_intent, _merge_intent
from .message_history import (
    _history_to_messages,
    _human_texts,
    _load_thread_memory_summary_safe,
    _recent_context_snippet,
)
from .model_io import _invoke_json_model, _invoke_text_model
from .output_guard import _fallback_final_text, _guard_output_urls
from .prompts import (
    _FINAL_SYSTEM_PROMPT,
    _INTENT_ROUTER_SYSTEM_PROMPT,
    _TOOL_WORKER_SYSTEM_PROMPT,
)
from .state import AgentGraphState, GraphModelHandle, GraphModels, GraphRuntimeConfig
from .stream import emit_graph_evidence, emit_graph_progress, evidence_status_items
from .tool_planning import (
    _emit_tool_running_status,
    _heuristic_tool_calls,
    _normalize_tool_calls,
    _select_tools,
    _tool_schema_brief,
)


@dataclass
class GraphDependencies:
    models: GraphModels
    tool_runtime: ToolRuntime
    config: GraphRuntimeConfig


def _graph_node_span(name: str):
    def _decorator(func):
        @wraps(func)
        def _wrapped(self: "GraphNodeRunner", state: AgentGraphState) -> dict[str, Any]:
            with trace_span(
                "graph_node",
                name,
                input_summary=_graph_state_summary(state),
                metadata={"node": name},
            ) as span:
                result = func(self, state)
                span.set_output(_graph_update_summary(result))
                return result

        return _wrapped

    return _decorator


def _graph_state_summary(state: AgentGraphState) -> dict[str, Any]:
    context_pack = state.get("context_pack") or {}
    return {
        "user_message_chars": len(str(state.get("user_message") or "")),
        "history_count": len(state.get("history") or []),
        "llm_input_count": len(state.get("llm_input_messages") or []),
        "context_strategy": ((context_pack.get("trim_report") or {}).get("strategy") if isinstance(context_pack, dict) else None),
        "selected_tools": list(state.get("selected_tools") or []),
        "pending_tool_count": len(state.get("pending_tool_calls") or []),
        "tool_result_count": len(state.get("tool_results") or []),
        "tool_round": int(state.get("tool_round") or 0),
        "evidence_count": len(state.get("evidence_urls") or state.get("valid_urls") or []),
        "next_step": state.get("next_step"),
    }


def _graph_update_summary(update: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(update, dict):
        return {"type": type(update).__name__}
    context_pack = update.get("context_pack") if isinstance(update.get("context_pack"), dict) else {}
    return {
        "keys": sorted(update.keys()),
        "intent_route": (update.get("intent") or {}).get("route") if isinstance(update.get("intent"), dict) else None,
        "context_strategy": ((context_pack.get("trim_report") or {}).get("strategy") if isinstance(context_pack, dict) else None),
        "pending_tool_count": len(update.get("pending_tool_calls") or []),
        "tool_result_count": len(update.get("tool_results") or []),
        "evidence_count": len(update.get("evidence_urls") or update.get("valid_urls") or []),
        "final_text_chars": len(str(update.get("final_text") or "")),
        "next_step": update.get("next_step"),
    }


def _route_progress_copy(route: str) -> tuple[str, str]:
    """Return user-facing progress copy for the resolved intent route."""
    if route == "direct_answer":
        return "正在组织回答", "无需检索直接回答"
    if route == "needs_clarification":
        return "正在确认分析范围", "需要补充信息"
    if route == "needs_tools":
        return "正在规划检索", "需要调用工具获取证据"
    return "正在理解问题", ""


def _synthesis_progress_copy(route: str) -> tuple[str, str]:
    """Return final-generation progress copy for the resolved intent route."""
    if route == "needs_tools":
        return "正在生成分析", "输出结论、依据"
    if route == "needs_clarification":
        return "正在确认分析范围", "需要补充信息"
    return "正在生成回答", "输出回答"


class GraphNodeRunner:
    def __init__(self, deps: GraphDependencies) -> None:
        self.deps = deps
        self.registry = build_default_registry()

    @_graph_node_span("prepare_context")
    def prepare_context(self, state: AgentGraphState) -> dict[str, Any]:
        emit_graph_progress("understanding", "正在理解问题", detail="")
        user_message = str(state.get("user_message") or "")
        history = list(state.get("history") or [])
        thread_id = str(state.get("thread_id") or "").strip()
        memory_summary = _load_thread_memory_summary_safe(thread_id)
        with trace_span(
            "context",
            "history_manifest_builder",
            input_summary={"history_count": len(history), "thread_id": thread_id},
            metadata={"node": "prepare_context"},
        ) as context_span:
            history_manifest = build_history_manifest(history)
            context_span.set_output(
                {
                    "turn_count": len(history_manifest),
                    "memory_summary_available": bool(memory_summary),
                }
            )
        curator_result: dict[str, Any] | None = None
        curator_error = ""
        curator_used = should_use_context_curator(
            user_message=user_message,
            history_manifest=history_manifest,
            memory_summary=memory_summary,
        )
        if curator_used:
            raw_curator_result = _invoke_json_model(
                self.deps.models.context_curator,
                node="context_curator",
                messages=build_context_curator_messages(
                    user_message=user_message,
                    history_manifest=history_manifest,
                    memory_summary=memory_summary,
                ),
            )
            curator_result = normalize_context_curator_result(
                raw_curator_result,
                history_manifest,
                memory_summary,
            )
            if not isinstance(curator_result, dict):
                curator_error = "empty_or_invalid_curator_output"
        context_pack = build_context_pack(
            user_message=user_message,
            history=history,
            history_manifest=history_manifest,
            memory_summary=memory_summary,
            curator_result=curator_result,
            curator_used=curator_used,
            curator_error=curator_error,
        )
        with trace_span(
            "context",
            "context_pack_builder",
            input_summary={"curator_used": curator_used, "manifest_turn_count": len(history_manifest)},
            metadata={"node": "prepare_context"},
        ) as context_span:
            context_span.set_output(
                {
                    "strategy": (context_pack.get("trim_report") or {}).get("strategy"),
                    "selected_turn_count": len(context_pack.get("selected_turns") or []),
                    "selected_evidence_count": len(context_pack.get("selected_evidence_urls") or []),
                    "depends_on_history": context_pack.get("depends_on_history"),
                }
            )
        context_prompt = render_context_for_prompt(context_pack)
        llm_input = build_llm_input_messages(
            [
                HumanMessage(
                    content=(
                        f"Current user message:\n{user_message}\n\n"
                        f"Context pack:\n{context_prompt}"
                    )
                )
            ]
        )
        return {
            "context_pack": context_pack,
            "llm_input_messages": llm_input,
            "tool_results": list(state.get("tool_results") or []),
            "tool_round": int(state.get("tool_round") or 0),
            "max_tool_rounds": int(state.get("max_tool_rounds") or self.deps.config.max_tool_rounds),
        }

    @_graph_node_span("intent_router")
    def intent_router(self, state: AgentGraphState) -> dict[str, Any]:
        user_message = active_question(state.get("context_pack"), str(state.get("user_message") or ""))
        context_snippet = render_context_for_prompt(state.get("context_pack"))
        intent = _heuristic_intent(user_message)
        model_intent = _invoke_json_model(
            self.deps.models.intent_router,
            node="intent_router",
            messages=[
                SystemMessage(content=_INTENT_ROUTER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Recent conversation context:\n{context_snippet}\n\n"
                        f"Current user message:\n{user_message}\n\n"
                        "Return JSON only."
                    )
                ),
            ],
        )
        if isinstance(model_intent, dict):
            intent = _merge_intent(intent, model_intent)
        title, detail = _route_progress_copy(str(intent.get("route") or ""))
        emit_graph_progress("understanding", title, detail=detail)
        return {
            "intent": intent,
        }

    @_graph_node_span("tool_selection")
    def tool_selection(self, state: AgentGraphState) -> dict[str, Any]:
        intent = state.get("intent") or {}
        selected = _select_tools(intent)
        emit_graph_progress("selecting_tools", "正在调用工具")
        return {
            "selected_tools": selected,
        }

    @_graph_node_span("tool_worker")
    def tool_worker(self, state: AgentGraphState) -> dict[str, Any]:
        selected_tools = list(state.get("selected_tools") or [])
        user_message = active_question(state.get("context_pack"), str(state.get("user_message") or ""))
        context_snippet = render_context_for_prompt(state.get("context_pack"))
        tool_results = list(state.get("tool_results") or [])
        existing_calls = _normalize_tool_calls({"tool_calls": state.get("pending_tool_calls") or []})
        if existing_calls:
            emit_graph_progress("selecting_tools", "正在调用工具")
            return {
                "pending_tool_calls": existing_calls,
            }
        schema_brief = _tool_schema_brief(selected_tools)
        model_plan = _invoke_json_model(
            self.deps.models.tool_worker,
            node="tool_worker",
            messages=[
                SystemMessage(content=_TOOL_WORKER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Recent conversation context:\n{context_snippet}\n\n"
                        f"Current user message:\n{user_message}\n\n"
                        f"Selected tools:\n{schema_brief}\n\n"
                        f"Already executed tools: {[item.tool for item in tool_results]}\n"
                        "Return JSON only."
                    )
                ),
            ],
        )
        calls = _normalize_tool_calls(model_plan)
        if not calls:
            calls = _heuristic_tool_calls(
                user_message=user_message,
                intent=state.get("intent") or {},
                selected_tools=selected_tools,
                tool_results=tool_results,
            )
        emit_graph_progress("selecting_tools", "正在调用工具")
        return {
            "pending_tool_calls": calls,
        }

    @_graph_node_span("tool_policy")
    def tool_policy(self, state: AgentGraphState) -> dict[str, Any]:
        selected_tools = set(state.get("selected_tools") or [])
        input_schemas = {name: self.registry.input_schema(name) for name in selected_tools}
        user_messages = _human_texts(state.get("llm_input_messages") or [])
        previous_tool_calls = [item.tool for item in state.get("tool_results") or []]
        decision = evaluate_pending_tool_calls(
            list(state.get("pending_tool_calls") or []),
            allowed_tool_names=selected_tools,
            input_schemas=input_schemas,
            evidence_urls=state.get("evidence_urls") or [],
            user_messages=user_messages,
            previous_tool_calls=previous_tool_calls,
        )
        with trace_span(
            "guard",
            "tool_policy",
            input_summary={
                "pending_tool_count": len(state.get("pending_tool_calls") or []),
                "selected_tools": sorted(selected_tools),
                "previous_tool_calls": previous_tool_calls,
            },
            metadata={"node": "tool_policy"},
        ) as guard_span:
            guard_span.set_output(
                {
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "details": decision.details or {},
                }
            )
            if not decision.allowed:
                guard_span.status = "blocked"
        if decision.allowed:
            return {}

        clarification = build_clarification_payload(
            str(state.get("user_message") or ""),
            reason=infer_clarification_reason(str(state.get("user_message") or "")),
            context={"policy_reason": decision.reason, "policy_details": decision.details or {}},
        ).to_dict()
        return {
            "clarification": clarification,
        }

    @_graph_node_span("tool_executor")
    def tool_executor(self, state: AgentGraphState) -> dict[str, Any]:
        results = list(state.get("tool_results") or [])
        evidence_urls = list(state.get("evidence_urls") or [])
        seen_urls = {normalize_url_for_match(url) for url in evidence_urls}
        for call in state.get("pending_tool_calls") or []:
            name = str(call.get("name") or "").strip()
            args = dict(call.get("args") or {})
            _emit_tool_running_status(name, args)
            envelope = self.deps.tool_runtime.execute(
                name,
                args,
                ToolRuntimeContext(trace=True, emit_progress_events=True),
            )
            results.append(envelope)
            for item in envelope.evidence or []:
                normalized = normalize_url_for_match(item.url)
                if normalized and normalized not in seen_urls:
                    evidence_urls.append(item.url)
                    seen_urls.add(normalized)
            emit_graph_evidence(envelope, limit=self.deps.config.max_evidence_events)

        return {
            "tool_results": results,
            "evidence_urls": evidence_urls,
            "tool_round": int(state.get("tool_round") or 0) + 1,
            "pending_tool_calls": [],
        }

    @_graph_node_span("evidence_normalizer")
    def evidence_normalizer(self, state: AgentGraphState) -> dict[str, Any]:
        results = list(state.get("tool_results") or [])
        with trace_span(
            "postprocess",
            "evidence_normalizer",
            input_summary={"tool_result_count": len(results)},
            metadata={"node": "evidence_normalizer"},
        ) as post_span:
            evidence_urls, brief = _normalize_evidence(results)
            post_span.set_output({"evidence_count": len(evidence_urls), "brief_chars": len(brief)})
        items = evidence_status_items(results, limit=self.deps.config.max_status_items)
        if items:
            emit_graph_progress("retrieving", f"已找到 {len(evidence_urls)} 篇相关报道", items=items)
        emit_graph_progress("analyzing", "正在整理信息")
        return {
            "evidence_urls": evidence_urls,
            "valid_urls": evidence_urls,
            "evidence_brief": brief,
        }

    @_graph_node_span("tool_loop_decider")
    def tool_loop_decider(self, state: AgentGraphState) -> dict[str, Any]:
        results = list(state.get("tool_results") or [])
        evidence_urls = list(state.get("evidence_urls") or [])
        max_rounds = int(state.get("max_tool_rounds") or self.deps.config.max_tool_rounds)
        tool_round = int(state.get("tool_round") or 0)
        if not evidence_urls and _requires_evidence(state):
            fallback_calls = _empty_evidence_fallback_calls(state)
            if tool_round < max_rounds and fallback_calls:
                next_step = "more_tools"
                return {
                    "next_step": next_step,
                    "pending_tool_calls": fallback_calls,
                }
            next_step = "insufficient_evidence"
        elif tool_round < max_rounds and _should_read_after_search(results, state):
            next_step = "more_tools"
            read_calls = _fulltext_calls_from_evidence(evidence_urls, state)
            return {
                "next_step": next_step,
                "pending_tool_calls": read_calls,
            }
        else:
            next_step = "enough_evidence"
        return {
            "next_step": next_step,
        }

    @_graph_node_span("final_synthesizer")
    def final_synthesizer(self, state: AgentGraphState) -> dict[str, Any]:
        intent = state.get("intent") or {}
        title, detail = _synthesis_progress_copy(str(intent.get("route") or ""))
        emit_graph_progress("synthesizing", title, detail=detail)
        results = list(state.get("tool_results") or [])
        user_message = active_question(state.get("context_pack"), str(state.get("user_message") or ""))
        context_pack_text = render_context_for_prompt(state.get("context_pack"))
        context = format_tool_results_for_final_synthesis(results)
        evidence_brief = str(state.get("evidence_brief") or "")
        text = _invoke_text_model(
            self.deps.models.final_synthesizer,
            node="final_synthesizer",
            messages=[
                SystemMessage(content=_FINAL_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"User question:\n{user_message}\n\n"
                        f"Conversation context pack:\n{context_pack_text}\n\n"
                        f"Intent:\n{json.dumps(state.get('intent') or {}, ensure_ascii=False)}\n\n"
                        f"Evidence brief:\n{evidence_brief}\n\n"
                        f"Tool results:\n{context}\n\n"
                        "Write the final answer now."
                    )
                ),
            ],
        )
        if not text:
            text = _fallback_final_text(user_message, state)
        return {
            "final_text": text,
        }

    @_graph_node_span("output_guard")
    def output_guard(self, state: AgentGraphState) -> dict[str, Any]:
        text = str(state.get("final_text") or "").strip()
        valid_urls = list(state.get("evidence_urls") or state.get("valid_urls") or [])
        with trace_span(
            "postprocess",
            "output_guard",
            input_summary={"text_chars": len(text), "valid_url_count": len(valid_urls)},
            metadata={"node": "output_guard"},
        ) as guard_span:
            guarded_text, guard_metadata = _guard_output_urls(text, valid_urls)
            guard_span.set_output({"guarded_text_chars": len(guarded_text), **guard_metadata})
        return {
            "final_text": guarded_text,
            "valid_urls": valid_urls,
        }

    @_graph_node_span("clarification_response")
    def clarification_response(self, state: AgentGraphState) -> dict[str, Any]:
        payload = state.get("clarification")
        if not isinstance(payload, dict):
            payload = build_clarification_payload(str(state.get("user_message") or "")).to_dict()
        with trace_span(
            "guard",
            "clarification_response",
            input_summary={"reason": payload.get("reason")},
            metadata={"node": "clarification_response"},
        ) as guard_span:
            guard_span.set_output({"question_chars": len(str(payload.get("question") or ""))})
        emit_graph_progress("clarification_required", "需要补充信息", detail=str(payload.get("question") or ""))
        return {
            "clarification": payload,
            "final_text": str(payload.get("question") or ""),
            "valid_urls": [],
        }

    @_graph_node_span("insufficient_evidence_response")
    def insufficient_evidence_response(self, state: AgentGraphState) -> dict[str, Any]:
        text = (
            "目前没有找到足够可靠的相关新闻证据支撑这个分析结论。"
            "请缩小时间范围、补充更具体的实体或提供相关 URL 后我再继续。"
        )
        emit_graph_progress("analyzing", "正在整理信息", detail="当前证据不足")
        return {
            "final_text": text,
            "valid_urls": [],
        }
