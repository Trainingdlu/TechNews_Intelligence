"""LangGraph node orchestration for Router -> Miner -> Analyst -> Formatter."""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, TypedDict

from langgraph.graph import END, StateGraph

try:
    from mcp.client import build_default_mcp_client
    from prompts import get_role_system_instruction
    from core.role_policy import assert_skill_allowed
    from core.skill_contracts import SkillEnvelope, build_error_envelope
    from core.skill_registry import SkillRegistry
    from core.tool_hooks import ToolHookRunner
    from tools import (
        QueryNewsSkillInput,
        TrendAnalysisSkillInput,
        SearchNewsSkillInput,
        CompareSourcesSkillInput,
        CompareTopicsSkillInput,
        BuildTimelineSkillInput,
        AnalyzeLandscapeSkillInput,
        FulltextBatchSkillInput,
        query_news_skill,
        trend_analysis_skill,
        search_news_skill,
        compare_sources_skill,
        compare_topics_skill,
        build_timeline_skill,
        analyze_landscape_skill,
        fulltext_batch_skill,
    )
except ImportError:  # package-style import fallback
    from ..mcp.client import build_default_mcp_client
    from ..prompts import get_role_system_instruction
    from ..core.role_policy import assert_skill_allowed
    from ..core.skill_contracts import SkillEnvelope, build_error_envelope
    from ..core.skill_registry import SkillRegistry
    from ..core.tool_hooks import ToolHookRunner
    from ..tools import (
        QueryNewsSkillInput,
        TrendAnalysisSkillInput,
        SearchNewsSkillInput,
        CompareSourcesSkillInput,
        CompareTopicsSkillInput,
        BuildTimelineSkillInput,
        AnalyzeLandscapeSkillInput,
        FulltextBatchSkillInput,
        query_news_skill,
        trend_analysis_skill,
        search_news_skill,
        compare_sources_skill,
        compare_topics_skill,
        build_timeline_skill,
        analyze_landscape_skill,
        fulltext_batch_skill,
    )


class WorkflowState(TypedDict, total=False):
    user_message: str
    history: list[dict[str, Any]]
    intent: str
    selected_skill: str
    miner_transport: str
    role_prompts: dict[str, str]
    miner_payload: dict[str, Any]
    miner_result: SkillEnvelope
    analyst_result: dict[str, Any]
    final_text: str
    evidence_urls: list[str]
    node_audit: list[dict[str, Any]]
    analyst_denied: bool
    # LLM integration fields (Phase 3)
    router_llm_output: dict[str, Any]   # Raw LLM routing decision JSON
    router_confidence: str               # Router classification confidence
    analyst_llm_output: str              # Raw LLM analysis text
    analyst_confidence: str              # Analyst confidence assessment
    formatter_llm_output: str            # Raw LLM formatted text


@dataclass(frozen=True)
class _WorkflowRuntime:
    registry: SkillRegistry
    hooks: ToolHookRunner
    miner_transport: str
    mcp_client: Any | None
    role_allowlists: dict[str, set[str]]


_DEFAULT_REGISTRY: SkillRegistry | None = None
_DEFAULT_HOOK_RUNNER: ToolHookRunner | None = None
_GRAPH_CACHE_MAX_SIZE = 16
_GRAPH_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_GRAPH_CACHE_LOCK = threading.RLock()

MCP_SKILL_MAP: dict[str, str] = {
    "query_news": "mcp__newsdb__query_news_vector",
    "trend_analysis": "mcp__newsdb__trend_analysis",
    "search_news": "mcp__newsdb__search_news",
    "compare_sources": "mcp__newsdb__compare_sources",
    "compare_topics": "mcp__newsdb__compare_topics",
    "build_timeline": "mcp__newsdb__build_timeline",
    "analyze_landscape": "mcp__newsdb__analyze_landscape",
    "fulltext_batch": "mcp__newsdb__fulltext_batch",
}


def build_default_registry() -> SkillRegistry:
    """Build the default in-process skill registry."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    registry = SkillRegistry()
    registry.register(
        name="query_news",
        input_model=QueryNewsSkillInput,
        handler=lambda payload: query_news_skill(payload),
        description="Structured news retrieval",
    )
    registry.register(
        name="trend_analysis",
        input_model=TrendAnalysisSkillInput,
        handler=lambda payload: trend_analysis_skill(payload),
        description="Structured trend momentum analysis",
    )
    registry.register(
        name="search_news",
        input_model=SearchNewsSkillInput,
        handler=lambda payload: search_news_skill(payload),
        description="Hybrid semantic+keyword news search",
    )
    registry.register(
        name="compare_sources",
        input_model=CompareSourcesSkillInput,
        handler=lambda payload: compare_sources_skill(payload),
        description="HackerNews vs TechCrunch source comparison",
    )
    registry.register(
        name="compare_topics",
        input_model=CompareTopicsSkillInput,
        handler=lambda payload: compare_topics_skill(payload),
        description="A-vs-B entity comparison with evidence",
    )
    registry.register(
        name="build_timeline",
        input_model=BuildTimelineSkillInput,
        handler=lambda payload: build_timeline_skill(payload),
        description="Chronological event timeline construction",
    )
    registry.register(
        name="analyze_landscape",
        input_model=AnalyzeLandscapeSkillInput,
        handler=lambda payload: analyze_landscape_skill(payload),
        description="Competitive landscape analysis with entity stats",
    )
    registry.register(
        name="fulltext_batch",
        input_model=FulltextBatchSkillInput,
        handler=lambda payload: fulltext_batch_skill(payload),
        description="Batch full-text article reading",
    )
    _DEFAULT_REGISTRY = registry
    return registry


def build_default_hook_runner() -> ToolHookRunner:
    """Build the default shared hook runner to maximize graph reuse."""

    global _DEFAULT_HOOK_RUNNER
    if _DEFAULT_HOOK_RUNNER is not None:
        return _DEFAULT_HOOK_RUNNER
    _DEFAULT_HOOK_RUNNER = ToolHookRunner()
    return _DEFAULT_HOOK_RUNNER


def _contains_cjk(text: str) -> bool:
    return bool(
        re.search(
            r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]",
            text or "",
        )
    )


def _extract_days(user_message: str, default: int = 21, maximum: int = 365) -> int:
    text = user_message or ""
    m = re.search(r"(\d{1,3})\s*(?:day|days|天|日)", text, flags=re.IGNORECASE)
    if not m:
        return default
    days = int(m.group(1))
    return max(1, min(maximum, days))


def _route_intent(user_message: str) -> tuple[str, str]:
    text = (user_message or "").lower()
    trend_markers = [
        "trend",
        "momentum",
        "变化",
        "趋势",
        "过去",
        "recent",
        "last",
    ]
    if any(token in text for token in trend_markers):
        return "trend_analysis", "trend_analysis"
    return "fact_retrieval", "query_news"


def _build_payload(intent: str, user_message: str) -> dict[str, Any]:
    if intent == "trend_analysis":
        return {
            "topic": _extract_trend_topic(user_message),
            "window": max(3, min(60, _extract_days(user_message, default=7, maximum=60))),
        }

    return {
        "query": user_message.strip(),
        "days": _extract_days(user_message, default=21, maximum=365),
        "source": "all",
        "sort": "time_desc",
        "limit": 8,
    }


def _extract_trend_topic(user_message: str) -> str:
    """Extract likely topic/entity phrase from trend-style user prompts."""

    raw = (user_message or "").strip()
    if not raw:
        return raw

    text = raw
    text = re.sub(
        r"(?i)\b(?:in\s+)?(?:the\s+)?(?:last|past|recent)\s+\d{1,3}\s*(?:day|days|week|weeks|month|months)\b",
        " ",
        text,
    )
    text = re.sub(r"\d{1,3}\s*(?:天|日|周|个月|月)", " ", text, flags=re.IGNORECASE)

    text = re.sub(
        r"(?i)\b(?:trend|momentum|analysis|analyze|analyse|compare|changes?)\b",
        " ",
        text,
    )
    text = re.sub(r"(趋势|变化|分析|对比|最近|过去|近)", " ", text)
    text = re.sub(r"(?i)\b(?:of|for|about|on|in|the|please)\b", " ", text)

    text = re.sub(r"[\s,、，。:：;；!?！？]+", " ", text).strip()
    return text or raw


def _extract_urls(envelope: SkillEnvelope) -> list[str]:
    urls: list[str] = []
    for item in envelope.evidence:
        url = str(item.url or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _resolve_miner_transport(miner_transport: str | None = None) -> str:
    raw = (miner_transport or os.getenv("AGENT_MINER_TRANSPORT", "mcp")).strip().lower()
    if raw in {"mcp", "mcp_stdio", "remote_mcp"}:
        return "mcp"
    return "local"


def _mcp_local_fallback_enabled() -> bool:
    raw = str(os.getenv("AGENT_MCP_FALLBACK_LOCAL", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _is_mcp_fallback_error(error_code: str | None) -> bool:
    return str(error_code or "").strip().lower() in {
        "mcp_client_transport_error",
        "mcp_client_protocol_error",
        "mcp_unknown_namespaced_tool",
        "mcp_server_unavailable",
        "mcp_server_call_failed",
        "mcp_unknown_tool",
        "mcp_tool_execution_failed",
    }


def _execute_miner_skill(
    selected_skill: str,
    payload: dict[str, Any],
    skill_registry: SkillRegistry,
    miner_transport: str,
    mcp_client: Any | None = None,
) -> SkillEnvelope:
    if miner_transport != "mcp":
        result = skill_registry.execute(selected_skill, payload)
        result.diagnostics["miner_transport"] = "local"
        return result

    qualified_name = MCP_SKILL_MAP.get(selected_skill, f"mcp__newsdb__{selected_skill}")
    try:
        client = mcp_client or build_default_mcp_client()
        result = client.call_tool(qualified_name, payload)
    except Exception as exc:  # noqa: BLE001
        result = build_error_envelope(
            tool=selected_skill,
            request=payload,
            error="mcp_client_transport_error",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "mcp_qualified_name": qualified_name,
            },
        )

    if result.tool != selected_skill:
        result.diagnostics["upstream_tool"] = result.tool
        result.tool = selected_skill

    result.diagnostics["miner_transport"] = "mcp"
    result.diagnostics["mcp_qualified_name"] = qualified_name

    if result.status == "error" and _mcp_local_fallback_enabled() and _is_mcp_fallback_error(result.error):
        local_result = skill_registry.execute(selected_skill, payload)
        local_result.diagnostics.update(
            {
                "miner_transport": "local_fallback",
                "fallback_from_transport": "mcp",
                "fallback_reason": result.error,
                "mcp_error_diagnostics": result.diagnostics,
            }
        )
        return local_result

    return result


def _analyst_summarize(envelope: SkillEnvelope, user_message: str) -> dict[str, Any]:
    if envelope.status == "error":
        return {
            "facts": [],
            "inference": [],
            "final": f"Skill execution failed: {envelope.error or 'unknown_error'}",
        }

    if envelope.status == "empty":
        if _contains_cjk(user_message):
            final = "抱歉，当前时间窗内未检索到匹配数据。"
        else:
            final = "No matching records were found in the current time window."
        return {"facts": [], "inference": [], "final": final}

    if envelope.tool == "query_news":
        records = []
        if isinstance(envelope.data, dict):
            records = list(envelope.data.get("records", []))

        facts: list[str] = []
        for idx, row in enumerate(records[:3], 1):
            title = row.get("title_cn") or row.get("title") or "(untitled)"
            source = row.get("source") or "unknown"
            points = row.get("points", 0)
            facts.append(f"{idx}. [{source}] {title} (points={points})")

        if _contains_cjk(user_message):
            final = (
                "\n".join(["检索摘要：", *facts])
                if facts
                else "检索完成，但无可展示记录。"
            )
        else:
            final = "\n".join(["Retrieval summary:", *facts]) if facts else "Retrieval completed with no displayable records."

        return {"facts": facts, "inference": [], "final": final}

    if envelope.tool == "trend_analysis":
        summary = envelope.data if isinstance(envelope.data, dict) else {}
        topic = str(summary.get("topic", ""))
        recent_cnt = int(summary.get("recent_count", 0))
        prev_cnt = int(summary.get("previous_count", 0))
        count_delta = str(summary.get("count_delta", ""))

        if _contains_cjk(user_message):
            final = (
                f"趋势摘要：topic={topic}，recent={recent_cnt}，previous={prev_cnt}，"
                f"delta={count_delta}。"
            )
        else:
            final = (
                f"Trend summary: topic={topic}, recent={recent_cnt}, previous={prev_cnt}, "
                f"delta={count_delta}."
            )
        return {"facts": [final], "inference": [], "final": final}

    return {"facts": [], "inference": [], "final": "No analyst formatter for this skill."}


def _build_role_prompts() -> dict[str, str]:
    return {
        "router": get_role_system_instruction("router"),
        "miner": get_role_system_instruction("miner"),
        "analyst": get_role_system_instruction("analyst"),
        "formatter": get_role_system_instruction("formatter"),
    }


def _normalize_role_allowlists(
    role_allowlists: dict[str, Iterable[str]] | None,
) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {}
    if not role_allowlists:
        return normalized

    for role, skills in role_allowlists.items():
        role_key = str(role).strip().lower()
        if not role_key:
            continue
        if isinstance(skills, str):
            skill_values: list[Any] = [skills]
        elif skills is None:
            skill_values = []
        else:
            try:
                skill_values = list(skills)
            except TypeError:
                skill_values = [skills]
        allowed = {
            str(skill).strip().lower()
            for skill in skill_values
            if str(skill).strip()
        }
        normalized[role_key] = allowed
    return normalized


def _load_role_allowlists_from_env() -> dict[str, set[str]]:
    raw_map: dict[str, list[str]] = {}
    for role in ("router", "miner", "analyst", "formatter"):
        raw = str(os.getenv(f"AGENT_ROLE_ALLOWLIST_{role.upper()}", "")).strip()
        if not raw:
            continue
        parts = [segment.strip() for segment in re.split(r"[,\s]+", raw) if segment.strip()]
        raw_map[role] = parts
    return _normalize_role_allowlists(raw_map)


def _role_allowlists_cache_key(role_allowlists: dict[str, set[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows: list[tuple[str, tuple[str, ...]]] = []
    for role, skills in sorted(role_allowlists.items(), key=lambda item: item[0]):
        rows.append((role, tuple(sorted(skills))))
    return tuple(rows)


def _callable_cache_key(fn: Any) -> tuple[str, str]:
    return (
        str(getattr(fn, "__module__", "")),
        str(getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__))),
    )


def _type_cache_key(tp: Any) -> tuple[str, str]:
    return (
        str(getattr(tp, "__module__", "")),
        str(getattr(tp, "__qualname__", getattr(tp, "__name__", type(tp).__name__))),
    )


def _registry_cache_key(registry: SkillRegistry) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for skill_name in registry.list_skills():
        try:
            spec = registry.get(skill_name)
        except KeyError:
            continue
        rows.append(
            (
                skill_name,
                _type_cache_key(spec.input_model),
                _callable_cache_key(spec.handler),
            )
        )
    return tuple(rows)


def _hook_runner_cache_key(hooks: ToolHookRunner) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    pre = tuple(_callable_cache_key(hook) for hook in getattr(hooks, "pre_hooks", []))
    post = tuple(_callable_cache_key(hook) for hook in getattr(hooks, "post_hooks", []))
    return pre, post


def _workflow_graph_cache_key(runtime: _WorkflowRuntime) -> tuple[Any, ...]:
    return (
        _registry_cache_key(runtime.registry),
        _hook_runner_cache_key(runtime.hooks),
        runtime.miner_transport,
        _role_allowlists_cache_key(runtime.role_allowlists),
    )


def _get_cached_graph(cache_key: tuple[Any, ...]) -> Any | None:
    with _GRAPH_CACHE_LOCK:
        cached = _GRAPH_CACHE.get(cache_key)
        if cached is not None:
            _GRAPH_CACHE.move_to_end(cache_key)
        return cached


def _put_cached_graph(cache_key: tuple[Any, ...], compiled_graph: Any) -> Any:
    with _GRAPH_CACHE_LOCK:
        _GRAPH_CACHE[cache_key] = compiled_graph
        _GRAPH_CACHE.move_to_end(cache_key)
        while len(_GRAPH_CACHE) > _GRAPH_CACHE_MAX_SIZE:
            _GRAPH_CACHE.popitem(last=False)
        return compiled_graph


def _resolve_role_permission(
    role: str,
    skill_name: str,
    role_allowlists: dict[str, set[str]],
) -> tuple[bool, str | None]:
    normalized_role = str(role).strip().lower()
    normalized_skill = str(skill_name).strip().lower()

    if normalized_role in role_allowlists:
        if normalized_skill in role_allowlists[normalized_role]:
            return True, None
        return (
            False,
            f"role:{normalized_role} cannot use skill:{normalized_skill} (override)",
        )
    return assert_skill_allowed(normalized_role, normalized_skill)


def _append_node_audit(
    state: WorkflowState,
    node: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trail = list(state.get("node_audit") or [])
    event: dict[str, Any] = {"node": str(node), "status": str(status)}
    if details:
        event["details"] = details
    trail.append(event)
    return trail


def _router_node(state: WorkflowState, runtime: _WorkflowRuntime) -> WorkflowState:
    user_message = str(state.get("user_message") or "")
    intent, selected_skill = _route_intent(user_message)
    payload = _build_payload(intent, user_message)

    allowed, deny_reason = _resolve_role_permission(
        role="router",
        skill_name=selected_skill,
        role_allowlists=runtime.role_allowlists,
    )
    if not allowed:
        denied = build_error_envelope(
            tool=selected_skill,
            request={"intent": intent, "payload": payload},
            error="role_policy_denied",
            diagnostics={"reason": deny_reason, "role": "router"},
        )
        return {
            "intent": intent,
            "selected_skill": selected_skill,
            "miner_payload": payload,
            "miner_result": denied,
            "node_audit": _append_node_audit(
                state,
                node="router",
                status="deny",
                details={"selected_skill": selected_skill, "reason": deny_reason},
            ),
        }

    return {
        "intent": intent,
        "selected_skill": selected_skill,
        "miner_payload": payload,
        "node_audit": _append_node_audit(
            state,
            node="router",
            status="allow",
            details={"intent": intent, "selected_skill": selected_skill},
        ),
    }


def _miner_node(state: WorkflowState, runtime: _WorkflowRuntime) -> WorkflowState:
    existing = state.get("miner_result")
    if isinstance(existing, SkillEnvelope) and existing.status == "error":
        return {
            "miner_transport": runtime.miner_transport,
            "miner_result": existing,
            "node_audit": _append_node_audit(
                state,
                node="miner",
                status="skip",
                details={"reason": "upstream_error", "error": existing.error},
            ),
        }

    selected_skill = str(state.get("selected_skill") or "").strip()
    payload = state.get("miner_payload")
    normalized_payload = payload if isinstance(payload, dict) else {}

    allowed, deny_reason = _resolve_role_permission(
        role="miner",
        skill_name=selected_skill,
        role_allowlists=runtime.role_allowlists,
    )
    if not allowed:
        return {
            "miner_transport": runtime.miner_transport,
            "miner_result": build_error_envelope(
                tool=selected_skill,
                request=normalized_payload,
                error="role_policy_denied",
                diagnostics={"reason": deny_reason, "role": "miner"},
            ),
            "node_audit": _append_node_audit(
                state,
                node="miner",
                status="deny",
                details={"selected_skill": selected_skill, "reason": deny_reason},
            ),
        }

    pre = runtime.hooks.pre_tool_use(selected_skill, normalized_payload)
    if pre.action == "deny":
        deny_payload = (
            pre.updated_payload if pre.updated_payload is not None else normalized_payload
        )
        deny_details: dict[str, Any] = {"phase": "pre_hook", "reason": pre.reason}
        if pre.diagnostics:
            deny_details["diagnostics"] = pre.diagnostics
        return {
            "miner_transport": runtime.miner_transport,
            "miner_payload": deny_payload,
            "miner_result": build_error_envelope(
                tool=selected_skill,
                request=deny_payload,
                error="pre_hook_denied",
                diagnostics={"reason": pre.reason, **pre.diagnostics},
            ),
            "node_audit": _append_node_audit(
                state,
                node="miner",
                status="deny",
                details=deny_details,
            ),
        }

    effective_payload = (
        pre.updated_payload if pre.updated_payload is not None else normalized_payload
    )
    miner_result = _execute_miner_skill(
        selected_skill=selected_skill,
        payload=effective_payload,
        skill_registry=runtime.registry,
        miner_transport=runtime.miner_transport,
        mcp_client=runtime.mcp_client,
    )

    post = runtime.hooks.post_tool_use(selected_skill, effective_payload, miner_result)
    if post.action == "deny":
        return {
            "miner_transport": runtime.miner_transport,
            "miner_payload": effective_payload,
            "miner_result": build_error_envelope(
                tool=selected_skill,
                request=effective_payload,
                error="post_hook_denied",
                diagnostics={"reason": post.reason, **post.diagnostics},
            ),
            "node_audit": _append_node_audit(
                state,
                node="miner",
                status="deny",
                details={"phase": "post_hook", "reason": post.reason},
            ),
        }

    if post.diagnostics:
        miner_result.diagnostics.update(post.diagnostics)
    if post.action == "warn" and post.reason:
        miner_result.diagnostics["post_hook_warning"] = post.reason

    return {
        "miner_transport": runtime.miner_transport,
        "miner_payload": effective_payload,
        "miner_result": miner_result,
        "node_audit": _append_node_audit(
            state,
            node="miner",
            status="allow",
            details={
                "selected_skill": selected_skill,
                "transport": runtime.miner_transport,
                "result_status": miner_result.status,
            },
        ),
    }


def _analyst_node(state: WorkflowState, runtime: _WorkflowRuntime) -> WorkflowState:
    allowed, deny_reason = _resolve_role_permission(
        role="analyst",
        skill_name="synthesize_findings",
        role_allowlists=runtime.role_allowlists,
    )
    if not allowed:
        miner_result = state.get("miner_result")
        if not isinstance(miner_result, SkillEnvelope):
            miner_result = build_error_envelope(
                tool=str(state.get("selected_skill") or "unknown"),
                request=state.get("miner_payload") if isinstance(state.get("miner_payload"), dict) else {},
                error="missing_miner_result",
            )

        denied = build_error_envelope(
            tool="synthesize_findings",
            request={"selected_skill": str(state.get("selected_skill") or "")},
            error="role_policy_denied",
            diagnostics={"reason": deny_reason, "role": "analyst"},
        )
        analyst = _analyst_summarize(denied, str(state.get("user_message") or ""))
        return {
            "miner_result": miner_result,
            "analyst_result": analyst,
            "analyst_denied": True,
            "node_audit": _append_node_audit(
                state,
                node="analyst",
                status="deny",
                details={"reason": deny_reason},
            ),
        }

    user_message = str(state.get("user_message") or "")
    miner_result = state.get("miner_result")
    if not isinstance(miner_result, SkillEnvelope):
        miner_result = build_error_envelope(
            tool=str(state.get("selected_skill") or "unknown"),
            request=state.get("miner_payload") if isinstance(state.get("miner_payload"), dict) else {},
            error="missing_miner_result",
        )
    analyst = _analyst_summarize(miner_result, user_message)
    return {
        "miner_result": miner_result,
        "analyst_result": analyst,
        "analyst_denied": False,
        "node_audit": _append_node_audit(
            state,
            node="analyst",
            status="allow",
            details={"result_status": miner_result.status},
        ),
    }


def _formatter_node(state: WorkflowState, runtime: _WorkflowRuntime) -> WorkflowState:
    allowed, deny_reason = _resolve_role_permission(
        role="formatter",
        skill_name="format_answer",
        role_allowlists=runtime.role_allowlists,
    )
    if not allowed:
        denied = build_error_envelope(
            tool="format_answer",
            request={"selected_skill": str(state.get("selected_skill") or "")},
            error="role_policy_denied",
            diagnostics={"reason": deny_reason, "role": "formatter"},
        )
        return {
            "final_text": _analyst_summarize(denied, str(state.get("user_message") or ""))["final"],
            "evidence_urls": [],
            "node_audit": _append_node_audit(
                state,
                node="formatter",
                status="deny",
                details={"reason": deny_reason},
            ),
        }

    analyst_result = state.get("analyst_result")
    final_text = ""
    if isinstance(analyst_result, dict):
        final_text = str(analyst_result.get("final") or "")

    miner_result = state.get("miner_result")
    evidence_urls: list[str] = []
    analyst_denied = bool(state.get("analyst_denied"))
    if isinstance(miner_result, SkillEnvelope) and miner_result.status != "error" and not analyst_denied:
        evidence_urls = _extract_urls(miner_result)

    return {
        "final_text": final_text,
        "evidence_urls": evidence_urls,
        "node_audit": _append_node_audit(
            state,
            node="formatter",
            status="allow",
            details={"final_len": len(final_text), "evidence_count": len(evidence_urls)},
        ),
    }


def build_workflow_graph(
    registry: SkillRegistry | None = None,
    hook_runner: ToolHookRunner | None = None,
    miner_transport: str | None = None,
    mcp_client: Any | None = None,
    role_allowlists: dict[str, Iterable[str]] | None = None,
):
    """Build and compile the LangGraph workflow for v2 orchestration."""

    effective_allowlists = (
        _normalize_role_allowlists(role_allowlists)
        if role_allowlists is not None
        else _load_role_allowlists_from_env()
    )
    runtime = _WorkflowRuntime(
        registry=registry or build_default_registry(),
        hooks=hook_runner or build_default_hook_runner(),
        miner_transport=_resolve_miner_transport(miner_transport),
        mcp_client=mcp_client,
        role_allowlists=effective_allowlists,
    )
    # Avoid retaining non-shareable client instances in global cache.
    use_cache = runtime.mcp_client is None
    if use_cache:
        cache_key = _workflow_graph_cache_key(runtime)
        cached = _get_cached_graph(cache_key)
        if cached is not None:
            return cached

    graph = StateGraph(WorkflowState)
    graph.add_node("router", lambda state: _router_node(state, runtime))
    graph.add_node("miner", lambda state: _miner_node(state, runtime))
    graph.add_node("analyst", lambda state: _analyst_node(state, runtime))
    graph.add_node("formatter", lambda state: _formatter_node(state, runtime))
    graph.set_entry_point("router")
    graph.add_edge("router", "miner")
    graph.add_edge("miner", "analyst")
    graph.add_edge("analyst", "formatter")
    graph.add_edge("formatter", END)
    compiled = graph.compile()
    if not use_cache:
        return compiled
    return _put_cached_graph(cache_key, compiled)


def run_workflow(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    registry: SkillRegistry | None = None,
    hook_runner: ToolHookRunner | None = None,
    miner_transport: str | None = None,
    mcp_client: Any | None = None,
    role_allowlists: dict[str, Iterable[str]] | None = None,
) -> WorkflowState:
    """Run Router -> Miner -> Analyst -> Formatter with a compiled StateGraph."""

    initial_state: WorkflowState = {
        "user_message": user_message,
        "history": history or [],
        "miner_transport": _resolve_miner_transport(miner_transport),
        "role_prompts": _build_role_prompts(),
        "node_audit": [],
        "analyst_denied": False,
    }

    effective_allowlists = (
        _normalize_role_allowlists(role_allowlists)
        if role_allowlists is not None
        else _load_role_allowlists_from_env()
    )

    app = build_workflow_graph(
        registry=registry,
        hook_runner=hook_runner,
        miner_transport=miner_transport,
        mcp_client=mcp_client,
        role_allowlists=effective_allowlists,
    )
    final_state = app.invoke(initial_state)
    if not isinstance(final_state, dict):
        raise RuntimeError("Workflow graph returned non-dict state")
    return final_state


def run_workflow_text(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, set[str]]:
    """Compatibility adapter for agent.py: returns final text and valid URL set."""

    state = run_workflow(user_message=user_message, history=history)
    return state.get("final_text", ""), set(state.get("evidence_urls", []))
