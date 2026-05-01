"""Node implementations for the custom LangGraph agent."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
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
    record_graph_model_event,
    record_graph_node_event,
    record_tool_policy_block,
    set_request_token_usage,
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


class GraphNodeRunner:
    def __init__(self, deps: GraphDependencies) -> None:
        self.deps = deps
        self.registry = build_default_registry()

    def prepare_context(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("prepare_context", "start")
        emit_graph_progress("understanding", "正在理解问题", detail="")
        messages = _history_to_messages(state.get("history") or [])
        user_message = str(state.get("user_message") or "")
        messages.append(HumanMessage(content=user_message))
        llm_input = build_llm_input_messages(messages)
        _audit("prepare_context", "finish", {"message_count": len(llm_input)})
        return {
            "llm_input_messages": llm_input,
            "tool_results": list(state.get("tool_results") or []),
            "tool_round": int(state.get("tool_round") or 0),
            "max_tool_rounds": int(state.get("max_tool_rounds") or self.deps.config.max_tool_rounds),
            "node_audit": _append_node_audit(state, "prepare_context", "finish"),
        }

    def intent_router(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("intent_router", "start")
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
        _audit("intent_router", "finish", {"route": intent.get("route"), "intent_type": intent.get("intent_type")})
        return {
            "intent": intent,
            "node_audit": _append_node_audit(state, "intent_router", "finish", {"route": intent.get("route")}),
        }

    def tool_selection(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("tool_selection", "start")
        intent = state.get("intent") or {}
        selected = _select_tools(intent)
        emit_graph_progress("selecting_tools", "正在调用工具")
        _audit("tool_selection", "finish", {"selected_tools": selected})
        return {
            "selected_tools": selected,
            "node_audit": _append_node_audit(state, "tool_selection", "finish", {"selected_tools": selected}),
        }

    def tool_worker(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("tool_worker", "start")
        selected_tools = list(state.get("selected_tools") or [])
        user_message = str(state.get("user_message") or "")
        context_snippet = _recent_context_snippet(state.get("history") or [])
        tool_results = list(state.get("tool_results") or [])
        existing_calls = _normalize_tool_calls({"tool_calls": state.get("pending_tool_calls") or []})
        if existing_calls:
            emit_graph_progress("selecting_tools", "正在调用工具")
            _audit("tool_worker", "finish", {"tool_calls": existing_calls, "source": "state"})
            return {
                "pending_tool_calls": existing_calls,
                "node_audit": _append_node_audit(
                    state,
                    "tool_worker",
                    "finish",
                    {"tool_calls": existing_calls, "source": "state"},
                ),
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
        _audit("tool_worker", "finish", {"tool_calls": calls})
        return {
            "pending_tool_calls": calls,
            "node_audit": _append_node_audit(state, "tool_worker", "finish", {"tool_calls": calls}),
        }

    def tool_policy(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("tool_policy", "start")
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
        if decision.allowed:
            _audit("tool_policy", "finish", {"decision": "allow"})
            return {
                "node_audit": _append_node_audit(state, "tool_policy", "finish", {"decision": "allow"}),
            }

        record_tool_policy_block(reason=decision.reason, details=decision.details)
        clarification = build_clarification_payload(
            str(state.get("user_message") or ""),
            reason=infer_clarification_reason(str(state.get("user_message") or "")),
            context={"policy_reason": decision.reason, "policy_details": decision.details or {}},
        ).to_dict()
        _audit("tool_policy", "blocked", {"reason": decision.reason})
        return {
            "clarification": clarification,
            "node_audit": _append_node_audit(state, "tool_policy", "blocked", {"reason": decision.reason}),
        }

    def tool_executor(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("tool_executor", "start")
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

        _audit("tool_executor", "finish", {"tool_count": len(results), "evidence_count": len(evidence_urls)})
        return {
            "tool_results": results,
            "evidence_urls": evidence_urls,
            "tool_round": int(state.get("tool_round") or 0) + 1,
            "pending_tool_calls": [],
            "node_audit": _append_node_audit(
                state,
                "tool_executor",
                "finish",
                {"tool_count": len(results), "evidence_count": len(evidence_urls)},
            ),
        }

    def evidence_normalizer(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("evidence_normalizer", "start")
        results = list(state.get("tool_results") or [])
        evidence_urls, brief = _normalize_evidence(results)
        items = evidence_status_items(results, limit=self.deps.config.max_status_items)
        if items:
            emit_graph_progress("retrieving", f"已找到 {len(evidence_urls)} 篇相关报道", items=items)
        emit_graph_progress("analyzing", "正在整理信息")
        _audit("evidence_normalizer", "finish", {"evidence_count": len(evidence_urls)})
        return {
            "evidence_urls": evidence_urls,
            "valid_urls": evidence_urls,
            "evidence_brief": brief,
            "node_audit": _append_node_audit(
                state,
                "evidence_normalizer",
                "finish",
                {"evidence_count": len(evidence_urls)},
            ),
        }

    def tool_loop_decider(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("tool_loop_decider", "start")
        results = list(state.get("tool_results") or [])
        evidence_urls = list(state.get("evidence_urls") or [])
        max_rounds = int(state.get("max_tool_rounds") or self.deps.config.max_tool_rounds)
        tool_round = int(state.get("tool_round") or 0)
        if not evidence_urls and _requires_evidence(state):
            next_step = "insufficient_evidence"
        elif tool_round < max_rounds and _should_read_after_search(results, state):
            next_step = "more_tools"
            read_calls = _fulltext_calls_from_evidence(evidence_urls, state)
            return {
                "next_step": next_step,
                "pending_tool_calls": read_calls,
                "node_audit": _append_node_audit(state, "tool_loop_decider", "finish", {"next_step": next_step}),
            }
        else:
            next_step = "enough_evidence"
        _audit("tool_loop_decider", "finish", {"next_step": next_step})
        return {
            "next_step": next_step,
            "node_audit": _append_node_audit(state, "tool_loop_decider", "finish", {"next_step": next_step}),
        }

    def final_synthesizer(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("final_synthesizer", "start")
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
        _audit("final_synthesizer", "finish", {"chars": len(text)})
        return {
            "final_text": text,
            "node_audit": _append_node_audit(state, "final_synthesizer", "finish", {"chars": len(text)}),
        }

    def output_guard(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("output_guard", "start")
        text = str(state.get("final_text") or "").strip()
        valid_urls = list(state.get("evidence_urls") or state.get("valid_urls") or [])
        guarded_text, guard_metadata = _guard_output_urls(text, valid_urls)
        _audit("output_guard", "finish", {"valid_url_count": len(valid_urls), **guard_metadata})
        return {
            "final_text": guarded_text,
            "valid_urls": valid_urls,
            "node_audit": _append_node_audit(
                state,
                "output_guard",
                "finish",
                {"valid_url_count": len(valid_urls), **guard_metadata},
            ),
        }

    def clarification_response(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("clarification_response", "start")
        payload = state.get("clarification")
        if not isinstance(payload, dict):
            payload = build_clarification_payload(str(state.get("user_message") or "")).to_dict()
        emit_graph_progress("clarification_required", "需要补充信息", detail=str(payload.get("question") or ""))
        _audit("clarification_response", "finish", {"reason": payload.get("reason")})
        return {
            "clarification": payload,
            "final_text": str(payload.get("question") or ""),
            "valid_urls": [],
            "node_audit": _append_node_audit(state, "clarification_response", "finish", {"reason": payload.get("reason")}),
        }

    def insufficient_evidence_response(self, state: AgentGraphState) -> dict[str, Any]:
        _audit("insufficient_evidence_response", "start")
        text = (
            "目前没有找到足够可靠的相关新闻证据支撑这个分析结论。"
            "请缩小时间范围、补充更具体的实体或提供相关 URL 后我再继续。"
        )
        emit_graph_progress("analyzing", "正在整理信息", detail="当前证据不足")
        _audit("insufficient_evidence_response", "finish")
        return {
            "final_text": text,
            "valid_urls": [],
            "node_audit": _append_node_audit(state, "insufficient_evidence_response", "finish"),
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


def _audit(node: str, status: str, metadata: dict[str, Any] | None = None) -> None:
    record_graph_node_event(node=node, status=status, metadata=metadata)


def _append_node_audit(
    state: AgentGraphState,
    node: str,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    items = list(state.get("node_audit") or [])
    entry: dict[str, Any] = {"node": node, "status": status}
    if metadata:
        entry["metadata"] = metadata
    items.append(entry)
    return items


def _invoke_json_model(handle: GraphModelHandle, *, node: str, messages: list[BaseMessage]) -> dict[str, Any] | None:
    text = _invoke_text_model(handle, node=node, messages=messages)
    if not text:
        return None
    return _extract_json_object(text)


def _invoke_text_model(handle: GraphModelHandle, *, node: str, messages: list[BaseMessage]) -> str:
    record_graph_model_event(
        node=node,
        provider=handle.provider,
        model=handle.model,
        fallback=handle.fallback,
        error=handle.error,
    )
    if handle.client is None:
        return ""
    try:
        result = handle.client.invoke(messages)
        usage = extract_token_usage([result])
        if usage:
            set_request_token_usage(usage)
        return _coerce_to_text(getattr(result, "content", result)).strip()
    except Exception as exc:  # noqa: BLE001
        record_graph_model_event(
            node=node,
            provider=handle.provider,
            model=handle.model,
            fallback=True,
            error=str(exc),
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
    if _compare_topics_hit(lowered):
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


def _compare_topics_hit(lowered: str) -> bool:
    return bool(re.search(r"\bvs\b|versus|compare|comparison", lowered) or "对比" in lowered or "比较" in lowered)


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
        "topic_comparison": ["compare_topics", "search_news", "fulltext_batch"],
        "source_comparison": ["compare_sources", "search_news", "fulltext_batch"],
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
    for pattern in (r"(.+?)\s+(?:vs|versus)\s+(.+)", r"(.+?)(?:对比|比较)(.+)"):
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return _clean_topic(match.group(1)), _clean_topic(match.group(2))
    entities = _extract_entity_hints(raw)
    if len(entities) >= 2:
        return entities[0], entities[1]
    return _topic_from_message(raw), "competitors"


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
    return intent_type in {"news_analysis", "trend", "roundup_listing", "article_read"}


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
