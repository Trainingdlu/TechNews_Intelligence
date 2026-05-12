"""Node implementations for the custom LangGraph agent."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import wraps
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.clarification import (
    build_clarification_payload,
    infer_clarification_reason,
)
from agent.core.evidence import extract_urls, normalize_url_for_match
from agent.core.intent import classify_user_intent, extract_user_intent_text as _extract_user_intent_text
from agent.core.runtime_factories import build_default_registry, build_default_tool_runtime
from agent.core.tool_catalog import ToolDefinition, iter_tool_definitions, tool_definition_by_name
from agent.core.tool_contracts import ToolEnvelope
from agent.core.tool_runtime import (
    ToolRuntime,
    ToolRuntimeContext,
    format_tool_results_for_final_synthesis,
)
from agent.core.trace import (
    extract_token_usage,
    set_request_token_usage,
    trace_span,
)
from agent.memory_policy import build_llm_input_messages
from agent.prompts import SYSTEM_INSTRUCTION
from agent.tool_policy import evaluate_pending_tool_calls

from .state import AgentGraphState, GraphModelHandle, GraphModels, GraphRuntimeConfig
from .stream import emit_graph_evidence, emit_graph_progress, evidence_status_items


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
    return {
        "user_message_chars": len(str(state.get("user_message") or "")),
        "history_count": len(state.get("history") or []),
        "llm_input_count": len(state.get("llm_input_messages") or []),
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
    return {
        "keys": sorted(update.keys()),
        "intent_route": (update.get("intent") or {}).get("route") if isinstance(update.get("intent"), dict) else None,
        "pending_tool_count": len(update.get("pending_tool_calls") or []),
        "tool_result_count": len(update.get("tool_results") or []),
        "evidence_count": len(update.get("evidence_urls") or update.get("valid_urls") or []),
        "final_text_chars": len(str(update.get("final_text") or "")),
        "next_step": update.get("next_step"),
    }


class GraphNodeRunner:
    def __init__(self, deps: GraphDependencies) -> None:
        self.deps = deps
        self.registry = build_default_registry()

    @_graph_node_span("prepare_context")
    def prepare_context(self, state: AgentGraphState) -> dict[str, Any]:
        emit_graph_progress("understanding", "正在理解问题", detail="")
        messages = _history_to_messages(state.get("history") or [])
        user_message = str(state.get("user_message") or "")
        messages.append(HumanMessage(content=user_message))
        llm_input = build_llm_input_messages(messages)
        return {
            "llm_input_messages": llm_input,
            "tool_results": list(state.get("tool_results") or []),
            "tool_round": int(state.get("tool_round") or 0),
            "max_tool_rounds": int(state.get("max_tool_rounds") or self.deps.config.max_tool_rounds),
        }

    @_graph_node_span("intent_router")
    def intent_router(self, state: AgentGraphState) -> dict[str, Any]:
        user_message = str(state.get("user_message") or "")
        context_snippet = _recent_context_snippet(state.get("history") or [])
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
        detail = "需要进一步分析" if intent.get("route") == "needs_tools" else ""
        if intent.get("route") == "needs_clarification":
            detail = "需要补充信息"
        elif intent.get("route") == "direct_answer":
            detail = "可以直接回答"
        emit_graph_progress("understanding", "正在理解问题", detail=detail)
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
        user_message = str(state.get("user_message") or "")
        context_snippet = _recent_context_snippet(state.get("history") or [])
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
        emit_graph_progress("synthesizing", "正在生成分析", detail="输出结论、依据")
        results = list(state.get("tool_results") or [])
        user_message = str(state.get("user_message") or "")
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


def _history_to_messages(history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        text = _message_text(item)
        if not text:
            continue
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def _message_text(item: dict[str, Any]) -> str:
    parts = item.get("parts")
    if isinstance(parts, list):
        return "\n".join(
            str(part.get("text", "")).strip()
            for part in parts
            if isinstance(part, dict) and str(part.get("text", "")).strip()
        ).strip()
    return str(item.get("text", "") or "").strip()


def _human_texts(messages: list[BaseMessage]) -> list[str]:
    return [
        str(getattr(message, "content", "") or "")
        for message in messages
        if isinstance(message, HumanMessage)
    ]


def _invoke_json_model(handle: GraphModelHandle, *, node: str, messages: list[BaseMessage]) -> dict[str, Any] | None:
    text = _invoke_text_model(handle, node=node, messages=messages)
    if not text:
        return None
    return _extract_json_object(text)


def _invoke_text_model(handle: GraphModelHandle, *, node: str, messages: list[BaseMessage]) -> str:
    with trace_span(
        "model_call",
        node,
        input_summary={"message_count": len(messages)},
        metadata={
            "node": node,
            "provider": handle.provider,
            "model": handle.model,
            "fallback": handle.fallback,
            "handle_error": handle.error,
        },
    ) as span:
        if handle.client is None:
            span.set_output({"status": "skipped", "reason": "missing_client"})
            span.set_model_io(
                node=node,
                provider=handle.provider,
                model=handle.model,
                input_messages=messages,
                raw_output={"status": "skipped", "reason": "missing_client"},
            )
            return ""
        try:
            result = handle.client.invoke(messages)
            usage = extract_token_usage([result])
            if usage:
                set_request_token_usage(usage)
            text = _coerce_to_text(getattr(result, "content", result)).strip()
            span.set_output({"status": "success", "output_chars": len(text), "token_usage": usage})
            span.set_model_io(
                node=node,
                provider=handle.provider,
                model=handle.model,
                input_messages=messages,
                raw_output=result,
                token_usage=usage,
            )
            return text
        except Exception as exc:  # noqa: BLE001
            span.set_error(
                error_code=f"model_{type(exc).__name__.lower()}",
                error_message=str(exc),
                error=exc,
            )
            span.set_output({"status": "error", "fallback": True, "error": str(exc)})
            span.set_model_io(
                node=node,
                provider=handle.provider,
                model=handle.model,
                input_messages=messages,
                raw_output={"error": str(exc), "exception_type": type(exc).__name__},
            )
            return ""


def _coerce_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and part.get("text"):
                chunks.append(str(part.get("text")))
        return "\n".join(chunks)
    return "" if content is None else str(content)


_KNOWN_ENTITY_ALIASES: tuple[tuple[str, str], ...] = (
    (r"(?<![a-z0-9])openai(?![a-z0-9])", "OpenAI"),
    (r"(?<![a-z0-9])google(?![a-z0-9])", "Google"),
    (r"谷歌", "Google"),
    (r"(?<![a-z0-9])gemini(?![a-z0-9])", "Gemini"),
    (r"(?<![a-z0-9])anthropic(?![a-z0-9])", "Anthropic"),
    (r"(?<![a-z0-9])claude(?![a-z0-9])", "Claude"),
    (r"(?<![a-z0-9])microsoft(?![a-z0-9])", "Microsoft"),
    (r"微软", "Microsoft"),
    (r"(?<![a-z0-9])meta(?![a-z0-9])", "Meta"),
    (r"(?<![a-z0-9])amazon(?![a-z0-9])", "Amazon"),
    (r"亚马逊", "Amazon"),
    (r"(?<![a-z0-9])apple(?![a-z0-9])", "Apple"),
    (r"苹果", "Apple"),
    (r"(?<![a-z0-9])nvidia(?![a-z0-9])", "NVIDIA"),
    (r"英伟达", "NVIDIA"),
    (r"(?<![a-z0-9])tesla(?![a-z0-9])", "Tesla"),
    (r"特斯拉", "Tesla"),
    (r"(?<![a-z0-9])xai(?![a-z0-9])", "xAI"),
    (r"(?<![a-z0-9])grok(?![a-z0-9])", "Grok"),
    (r"(?<![a-z0-9])deepseek(?![a-z0-9])", "DeepSeek"),
)
_COMPARE_DIMENSION_RE = re.compile(
    r"差异|区别|不同|战略|策略|商业化|企业市场|定价|开源|生态|布局|路线|侧重点|"
    r"strategy|pricing|enterprise|commerciali[sz]ation|ecosystem|difference|different",
    re.IGNORECASE,
)
_COMPARE_SIDE_TRAILING_RE = re.compile(
    r"(?:最近|近期|当前|目前|过去|近来|在|上|方面|的|战略|策略|商业化|企业市场|定价|开源|生态|布局|"
    r"差异|区别|不同|表现|动态|事件|新闻|产品|路线|方向|侧重点|"
    r"recent|latest|current|strategy|pricing|enterprise|commerciali[sz]ation|ecosystem|difference|different)",
    re.IGNORECASE,
)
_COMPARE_SIDE_PREFIX_RE = re.compile(
    r"^(?:帮我|请|看看|看一下|对比一下|比较一下|对比|比较|分析一下|分析|一下|"
    r"please|compare|analyze)\s*",
    re.IGNORECASE,
)
_GENERIC_COMPARE_SIDE_TERMS = {
    "ai",
    "人工智能",
    "企业市场",
    "商业化",
    "战略",
    "策略",
    "定价",
    "生态",
    "布局",
    "差异",
    "区别",
    "不同",
    "technology news",
}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _merge_intent(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    route = str(override.get("route") or "").strip()
    if route in {"direct_answer", "needs_clarification", "needs_tools"}:
        out["route"] = route
    for key in ("intent_type", "reason", "analysis_depth", "entities", "time_window", "risk_flags"):
        if key in override and override.get(key) not in (None, "", []):
            out[key] = override.get(key)
    try:
        out["confidence"] = float(override.get("confidence", out.get("confidence", 0.5)))
    except Exception:
        out["confidence"] = out.get("confidence", 0.5)
    out["requires_tools"] = out.get("route") == "needs_tools"
    return out


def _heuristic_intent(user_message: str) -> dict[str, Any]:
    text = _extract_user_intent_text(user_message).strip() or str(user_message or "").strip()
    lowered = text.lower()
    urls = extract_urls(text)
    rule_intent = classify_user_intent(text)
    if rule_intent == "smalltalk_or_capability":
        route = "direct_answer"
        intent_type = "smalltalk_or_capability"
    elif _looks_like_article_reference_without_url(text):
        route = "needs_clarification"
        intent_type = "article_read"
    else:
        route = "needs_tools"
        intent_type = "news_analysis"
    if urls:
        route = "needs_tools"
        intent_type = "article_read"
    if _compare_topics_hit(text):
        intent_type = "topic_comparison"
    elif "hackernews" in lowered or "techcrunch" in lowered or "source" in lowered or "来源" in text:
        if re.search(r"compare|comparison|vs|versus", lowered) or "对比" in text or "比较" in text:
            intent_type = "source_comparison"
    elif re.search(r"timeline|时间线|脉络|发生了什么", lowered):
        intent_type = "timeline"
    elif re.search(r"landscape|格局|全景|竞争|生态|排行", lowered):
        intent_type = "landscape"
    elif re.search(r"trend|趋势|动向|变化|增长|下降", lowered):
        intent_type = "trend"
    elif rule_intent == "roundup_listing":
        intent_type = "roundup_listing"

    return {
        "route": route,
        "intent_type": intent_type,
        "reason": "heuristic_fallback",
        "confidence": 0.55,
        "requires_tools": route == "needs_tools",
        "analysis_depth": "deep" if re.search(r"深度|深入|研判|判断|前景|risk|outlook", lowered) else "standard",
        "entities": _extract_entity_hints(text),
        "time_window": {"days": _extract_days(text)},
        "risk_flags": [],
    }


def _looks_like_article_reference_without_url(text: str) -> bool:
    lowered = str(text or "").lower()
    if extract_urls(text):
        return False
    return bool(
        re.search(r"这篇|这条|文章|链接|原文|报道", text)
        or re.search(r"\b(this|that)\s+(article|link|story|post)\b", lowered)
    )


def _compare_topics_hit(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    if re.search(r"\bvs\b|versus|compare|comparison", lowered) or "对比" in raw or "比较" in raw:
        return True
    if _COMPARE_DIMENSION_RE.search(raw) and len(_extract_entity_hints(raw)) >= 2:
        return True
    if _COMPARE_DIMENSION_RE.search(raw) and _has_two_specific_compare_sides(raw):
        return True
    return False


def _extract_days(text: str, default: int = 14) -> int:
    lowered = str(text or "").lower()
    match = re.search(r"(?:最近|过去|last|past)\s*(\d{1,3})\s*(?:天|day|days)", lowered)
    if match:
        try:
            return max(1, min(int(match.group(1)), 365))
        except Exception:
            return default
    if "今天" in text or "today" in lowered:
        return 1
    if "一周" in text or "week" in lowered:
        return 7
    if "一个月" in text or "month" in lowered:
        return 30
    return default


def _extract_entity_hints(text: str) -> list[str]:
    entities: list[str] = []
    raw = str(text or "")
    lowered = raw.lower()
    for pattern, label in _KNOWN_ENTITY_ALIASES:
        if re.search(pattern, lowered, flags=re.IGNORECASE) and label not in entities:
            entities.append(label)
    for token in re.findall(r"\b[A-Z][A-Za-z0-9+._-]{1,}\b", str(text or "")):
        if token.lower() in {"today", "latest", "recent", "news"}:
            continue
        if token not in entities:
            entities.append(token)
    return entities[:8]


def _select_tools(intent: dict[str, Any]) -> list[str]:
    intent_type = str(intent.get("intent_type") or "").strip()
    route = str(intent.get("route") or "").strip()
    if route != "needs_tools":
        return []
    mapping = {
        "trend": ["trend_analysis", "search_news", "fulltext_batch"],
        "topic_comparison": ["compare_topics", "search_news", "query_news", "fulltext_batch"],
        "source_comparison": ["compare_sources", "search_news", "query_news", "fulltext_batch"],
        "timeline": ["build_timeline", "search_news", "fulltext_batch"],
        "landscape": ["analyze_landscape", "search_news", "fulltext_batch"],
        "article_read": ["read_news_content", "fulltext_batch", "search_news"],
        "roundup_listing": ["query_news", "search_news", "fulltext_batch"],
    }
    selected = mapping.get(intent_type, ["search_news", "query_news", "fulltext_batch"])
    known = {definition.name for definition in iter_tool_definitions()}
    return [name for name in selected if name in known]


def _tool_schema_brief(names: list[str]) -> str:
    blocks: list[str] = []
    for name in names:
        try:
            definition = tool_definition_by_name(name)
        except KeyError:
            continue
        schema = definition.input_model.model_json_schema()
        blocks.append(
            json.dumps(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": schema,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(blocks)


def _normalize_tool_calls(model_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(model_plan, dict):
        return []
    raw_calls = model_plan.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tool") or "").strip()
        args = item.get("args") or item.get("arguments") or {}
        if name and isinstance(args, dict):
            calls.append({"name": name, "args": dict(args)})
    return calls


def _heuristic_tool_calls(
    *,
    user_message: str,
    intent: dict[str, Any],
    selected_tools: list[str],
    tool_results: list[ToolEnvelope],
) -> list[dict[str, Any]]:
    text = _extract_user_intent_text(user_message).strip() or str(user_message or "").strip()
    days = _extract_days(text)
    urls = extract_urls(text)
    executed = {item.tool for item in tool_results}
    if len(urls) > 1 and "fulltext_batch" in selected_tools and "fulltext_batch" not in executed:
        return [{"name": "fulltext_batch", "args": {"urls": "\n".join(urls[:6]), "max_chars_per_article": 4000}}]
    if urls and "read_news_content" in selected_tools and "read_news_content" not in executed:
        return [{"name": "read_news_content", "args": {"url": urls[0]}}]
    if "fulltext_batch" in selected_tools and executed and "fulltext_batch" not in executed:
        return [{"name": "fulltext_batch", "args": {"urls": text, "max_chars_per_article": 4000}}]
    intent_type = str(intent.get("intent_type") or "")
    if intent_type == "topic_comparison" and "compare_topics" in selected_tools:
        topic_a, topic_b = _split_compare_topics(text)
        return [{"name": "compare_topics", "args": {"topic_a": topic_a, "topic_b": topic_b, "days": min(days, 90)}}]
    if intent_type == "source_comparison" and "compare_sources" in selected_tools:
        return [{"name": "compare_sources", "args": {"topic": _topic_from_message(text), "days": min(days, 90)}}]
    if intent_type == "timeline" and "build_timeline" in selected_tools:
        return [{"name": "build_timeline", "args": {"topic": _topic_from_message(text), "days": min(days, 180), "limit": 12}}]
    if intent_type == "landscape" and "analyze_landscape" in selected_tools:
        return [{"name": "analyze_landscape", "args": {"topic": _topic_from_message(text), "days": max(7, min(days, 180))}}]
    if intent_type == "trend" and "trend_analysis" in selected_tools:
        return [{"name": "trend_analysis", "args": {"topic": _topic_from_message(text), "window": max(3, min(days, 60))}}]
    if "search_news" in selected_tools:
        return [{"name": "search_news", "args": {"query": text, "days": days}}]
    if "query_news" in selected_tools:
        return [{"name": "query_news", "args": {"query": text, "days": days, "limit": 8}}]
    return []


def _split_compare_topics(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    entities = _extract_entity_hints(raw)
    if len(entities) >= 2:
        return entities[0], entities[1]
    for pattern in (r"(.+?)\s+(?:vs|versus)\s+(.+)",):
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return _clean_compare_topic(match.group(1)), _clean_compare_topic(match.group(2))
    for pattern in (
        r"(?:对比|比较)\s*(.+?)\s*(?:和|与|跟|同|及|以及)\s*(.+?)(?:的|在|上|方面|差异|区别|不同|$)",
        r"(.+?)\s*(?:和|与|跟|同|及|以及|、)\s*(.+?)(?:的)?(?:差异|区别|不同)",
        r"(.+?)(?:对比|比较)(.+)",
    ):
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return _clean_compare_topic(match.group(1)), _clean_compare_topic(match.group(2))
    return _topic_from_message(raw), "competitors"


def _has_two_specific_compare_sides(text: str) -> bool:
    left, right = _split_compare_topics(text)
    if right == "competitors":
        return False
    return _is_specific_compare_side(left) and _is_specific_compare_side(right)


def _is_specific_compare_side(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if len(normalized) < 2:
        return False
    if normalized in _GENERIC_COMPARE_SIDE_TERMS:
        return False
    if re.fullmatch(r"(?:最近|近期|当前|目前|过去|近来|recent|latest|current)", normalized):
        return False
    return True


def _topic_from_message(text: str) -> str:
    entities = _extract_entity_hints(text)
    if entities:
        return " ".join(entities[:3])
    cleaned = _clean_topic(text)
    return cleaned or "technology news"


def _clean_topic(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", "", str(text or "")).strip()
    cleaned = re.sub(r"(最近|过去|last|past)\s*\d{1,3}\s*(天|days?)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(分析|对比|比较|趋势|时间线|格局|新闻|动态|帮我|please|analyze|compare|trend|timeline|landscape)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。?？:：")
    return cleaned[:120] or "technology news"


def _clean_compare_topic(text: str) -> str:
    cleaned = _clean_topic(text)
    cleaned = _COMPARE_SIDE_PREFIX_RE.sub("", cleaned).strip(" ，,。?？:：")
    parts = _COMPARE_SIDE_TRAILING_RE.split(cleaned, maxsplit=1)
    if parts and parts[0].strip():
        cleaned = parts[0].strip(" ，,。?？:：")
    return cleaned[:120] or "technology news"


def _emit_tool_running_status(name: str, args: dict[str, Any]) -> None:
    if name in {"search_news", "query_news"}:
        query = str(args.get("query") or "").strip()
        days = args.get("days")
        detail = f"{query} 最近 {days} 天" if query and days else query
        emit_graph_progress("retrieving", "正在检索相关新闻", detail=detail)
    elif name in {"read_news_content", "fulltext_batch"}:
        emit_graph_progress("retrieving", "正在准备读取文章")
    else:
        emit_graph_progress("analyzing", "正在整理信息")


def _normalize_evidence(results: list[ToolEnvelope]) -> tuple[list[str], str]:
    urls: list[str] = []
    seen: set[str] = set()
    lines: list[str] = []
    for envelope in results:
        for item in envelope.evidence or []:
            normalized = normalize_url_for_match(item.url)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            urls.append(item.url)
            source = str(item.source or "").strip()
            title = str(item.title or item.url).strip()
            created = str(item.created_at or "").strip()
            lines.append(f"- {source} · {title} · {created} · {item.url}".strip())
    return urls, "\n".join(lines[:12])


def _guard_output_urls(text: str, valid_urls: list[str]) -> tuple[str, dict[str, Any]]:
    raw = str(text or "").strip()
    allowed: dict[str, str] = {}
    for item in valid_urls or []:
        url = str(item or "").strip()
        normalized = normalize_url_for_match(url)
        if normalized and normalized not in allowed:
            allowed[normalized] = url

    output_urls = extract_urls(raw)
    removed_count = 0
    has_valid_body_url = False
    guarded = raw
    for url in output_urls:
        normalized = normalize_url_for_match(url)
        if normalized and normalized in allowed:
            has_valid_body_url = True
            continue
        guarded = guarded.replace(url, "").strip()
        removed_count += 1

    guarded = re.sub(r"[ \t]+([，,。.;；:：!?！？])", r"\1", guarded)
    guarded = re.sub(r"\n{3,}", "\n\n", guarded).strip()

    appended_evidence_url = False
    if allowed and not has_valid_body_url:
        first_url = next(iter(allowed.values()))
        guarded = f"{guarded.rstrip()}\n\n来源：{first_url}".strip()
        appended_evidence_url = True

    return guarded, {
        "removed_unknown_url_count": removed_count,
        "appended_evidence_url": appended_evidence_url,
    }


def _empty_evidence_fallback_calls(state: AgentGraphState) -> list[dict[str, Any]]:
    selected = list(state.get("selected_tools") or [])
    executed = {item.tool for item in (state.get("tool_results") or [])}
    text = _extract_user_intent_text(str(state.get("user_message") or "")) or str(state.get("user_message") or "")
    query = _fallback_retrieval_query(text)
    days = _extract_days(text)
    calls: list[dict[str, Any]] = []
    if "search_news" in selected and "search_news" not in executed:
        calls.append({"name": "search_news", "args": {"query": query, "days": days}})
    if "query_news" in selected and "query_news" not in executed:
        calls.append({"name": "query_news", "args": {"query": query, "days": days, "limit": 8}})
    if calls:
        return calls
    if "fulltext_batch" in selected and "fulltext_batch" not in executed:
        return [{"name": "fulltext_batch", "args": {"urls": query, "max_chars_per_article": 4000}}]
    return []


def _fallback_retrieval_query(text: str) -> str:
    raw = str(text or "").strip()
    entities = _extract_entity_hints(raw)
    focus_terms = re.findall(
        r"企业市场|商业化|战略|策略|定价|开源|生态|产品|大模型|多模态|布局|差异|enterprise|commerciali[sz]ation|pricing|strategy|ecosystem",
        raw,
        flags=re.IGNORECASE,
    )
    parts: list[str] = []
    seen: set[str] = set()
    for item in [*entities[:4], *focus_terms]:
        normalized = str(item or "").strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            parts.append(normalized)
    if parts:
        return " ".join(parts)
    return raw[:180] or "technology news"


def _requires_evidence(state: AgentGraphState) -> bool:
    intent = state.get("intent") or {}
    return str(intent.get("route") or "") == "needs_tools"


def _should_read_after_search(results: list[ToolEnvelope], state: AgentGraphState) -> bool:
    selected = set(state.get("selected_tools") or [])
    if "fulltext_batch" not in selected:
        return False
    if any(item.tool == "fulltext_batch" for item in results):
        return False
    if not any(item.tool in {"search_news", "query_news"} and item.evidence for item in results):
        return False
    intent_type = str((state.get("intent") or {}).get("intent_type") or "")
    return intent_type in {
        "news_analysis",
        "trend",
        "roundup_listing",
        "article_read",
        "topic_comparison",
        "source_comparison",
        "landscape",
    }


def _fulltext_calls_from_evidence(evidence_urls: list[str], state: AgentGraphState) -> list[dict[str, Any]]:
    if evidence_urls:
        return [
            {
                "name": "fulltext_batch",
                "args": {"urls": "\n".join(evidence_urls[:3]), "max_chars_per_article": 4000},
            }
        ]
    return [
        {
            "name": "fulltext_batch",
            "args": {
                "urls": _extract_user_intent_text(str(state.get("user_message") or "")) or str(state.get("user_message") or ""),
                "max_chars_per_article": 4000,
            },
        }
    ]


def _recent_context_snippet(history: list[dict], max_messages: int = 4, max_chars: int = 1200) -> str:
    if not history:
        return "(none)"
    chunks: list[str] = []
    for item in history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        text = _message_text(item)
        if not text:
            continue
        label = "User" if role == "user" else "Assistant"
        urls = _history_item_context_urls(item)
        if urls:
            url_context = " ".join(urls[:3])
            chunks.append(f"[{label}] {text[:240]}\nEvidence URLs: {url_context}")
        else:
            chunks.append(f"[{label}] {text[:320]}")
    merged = "\n".join(chunks).strip()
    if not merged:
        return "(none)"
    if len(merged) > max_chars:
        return merged[-max_chars:]
    return merged


def _history_item_context_urls(item: dict[str, Any], max_urls: int = 3) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    raw_urls = item.get("citation_urls")
    if isinstance(raw_urls, list):
        for raw in raw_urls:
            url = str(raw or "").strip()
            normalized = normalize_url_for_match(url)
            if normalized and normalized not in seen:
                seen.add(normalized)
                urls.append(url)
                if len(urls) >= max_urls:
                    return urls
    for raw in extract_urls(_message_text(item)):
        normalized = normalize_url_for_match(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(raw)
            if len(urls) >= max_urls:
                return urls
    return urls


def _fallback_final_text(user_message: str, state: AgentGraphState) -> str:
    if str((state.get("intent") or {}).get("route") or "") == "direct_answer":
        if re.search(r"你好|您好|hello|hi", user_message, re.IGNORECASE):
            return "你好，我可以帮助你检索和分析科技新闻，包括趋势、对比、时间线、来源差异和行业格局。"
        return "我可以帮助你基于新闻证据做科技情报分析。你可以直接提出公司、产品、主题、时间范围或要比较的对象。"
    urls = list(state.get("evidence_urls") or [])
    if urls:
        return f"已找到相关证据，但当前模型暂时无法完成完整综合。建议先查看核心来源：{urls[0]}"
    return "目前没有足够可靠的证据生成结论。请补充更具体的时间范围、实体或来源。"


_INTENT_ROUTER_SYSTEM_PROMPT = (
    "You classify user requests for a tech-news intelligence agent. "
    "Return JSON only with route, intent_type, reason, confidence, requires_tools, "
    "analysis_depth, entities, time_window, risk_flags. "
    "route must be one of direct_answer, needs_clarification, needs_tools."
)

_TOOL_WORKER_SYSTEM_PROMPT = (
    "You are a tool-planning worker. Return JSON only: "
    '{"tool_calls":[{"name":"tool_name","args":{}}]}. '
    "Use only selected tools. Do not answer the user."
)

_FINAL_SYSTEM_PROMPT = (
    SYSTEM_INSTRUCTION
    + "\n\nYou are now the final synthesis node. Do not call tools. "
    "Use only the provided ToolEnvelope summaries and evidence brief. "
    "If evidence is insufficient, say so clearly. "
    "When evidence URLs are provided, include at least one exact raw URL in the answer body."
)
