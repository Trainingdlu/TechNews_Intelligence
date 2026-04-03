"""LangGraph node orchestration for Router -> Miner -> Analyst -> Formatter."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, TypedDict

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
        query_news_skill,
        trend_analysis_skill,
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
        query_news_skill,
        trend_analysis_skill,
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


@dataclass(frozen=True)
class _WorkflowRuntime:
    registry: SkillRegistry
    hooks: ToolHookRunner
    miner_transport: str
    mcp_client: Any | None


_DEFAULT_REGISTRY: SkillRegistry | None = None
MCP_SKILL_MAP: dict[str, str] = {
    "query_news": "mcp__newsdb__query_news_vector",
    "trend_analysis": "mcp__newsdb__trend_analysis",
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
    _DEFAULT_REGISTRY = registry
    return registry


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _extract_days(user_message: str, default: int = 21, maximum: int = 365) -> int:
    text = user_message or ""
    m = re.search(r"(\d{1,3})\s*(?:day|days|\u5929|\u65e5)", text, flags=re.IGNORECASE)
    if not m:
        return default
    days = int(m.group(1))
    return max(1, min(maximum, days))


def _route_intent(user_message: str) -> tuple[str, str]:
    text = (user_message or "").lower()
    trend_markers = [
        "trend",
        "momentum",
        "\u53d8\u5316",
        "\u8d8b\u52bf",
        "\u8fc7\u53bb",
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
    text = re.sub(r"\d{1,3}\s*(?:\u5929|\u65e5|\u5468|\u4e2a\u6708|\u6708)", " ", text, flags=re.IGNORECASE)

    text = re.sub(
        r"(?i)\b(?:trend|momentum|analysis|analyze|analyse|compare|changes?)\b",
        " ",
        text,
    )
    text = re.sub(r"(\u8d8b\u52bf|\u53d8\u5316|\u5206\u6790|\u5bf9\u6bd4|\u6700\u8fd1|\u8fc7\u53bb|\u8fd1)", " ", text)
    text = re.sub(r"(?i)\b(?:of|for|about|on|in|the|please)\b", " ", text)

    text = re.sub(r"[\s,\u3001\uff0c\u3002:：;；!?！？]+", " ", text).strip()
    return text or raw


def _extract_urls(envelope: SkillEnvelope) -> list[str]:
    urls: list[str] = []
    for item in envelope.evidence:
        url = str(item.url or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _resolve_miner_transport(miner_transport: str | None = None) -> str:
    raw = (miner_transport or os.getenv("AGENT_MINER_TRANSPORT", "local")).strip().lower()
    if raw in {"mcp", "mcp_stdio", "remote_mcp"}:
        return "mcp"
    return "local"


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

    client = mcp_client or build_default_mcp_client()
    qualified_name = MCP_SKILL_MAP.get(selected_skill, f"mcp__newsdb__{selected_skill}")
    result = client.call_tool(qualified_name, payload)
    if result.tool != selected_skill:
        result.diagnostics["upstream_tool"] = result.tool
        result.tool = selected_skill
    result.diagnostics["miner_transport"] = "mcp"
    result.diagnostics["mcp_qualified_name"] = qualified_name
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
            final = "\u62b1\u6b49\uff0c\u5f53\u524d\u65f6\u95f4\u7a97\u5185\u672a\u68c0\u7d22\u5230\u5339\u914d\u6570\u636e\u3002"
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
                "\n".join(["\u68c0\u7d22\u6458\u8981\uff1a", *facts])
                if facts
                else "\u68c0\u7d22\u5b8c\u6210\uff0c\u4f46\u65e0\u53ef\u5c55\u793a\u8bb0\u5f55\u3002"
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
                f"\u8d8b\u52bf\u6458\u8981\uff1atopic={topic}\uff0crecent={recent_cnt}\uff0cprevious={prev_cnt}\uff0c"
                f"delta={count_delta}\u3002"
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


def _router_node(state: WorkflowState) -> WorkflowState:
    user_message = str(state.get("user_message") or "")
    intent, selected_skill = _route_intent(user_message)
    payload = _build_payload(intent, user_message)
    return {
        "intent": intent,
        "selected_skill": selected_skill,
        "miner_payload": payload,
    }


def _miner_node(state: WorkflowState, runtime: _WorkflowRuntime) -> WorkflowState:
    selected_skill = str(state.get("selected_skill") or "").strip()
    payload = state.get("miner_payload")
    normalized_payload = payload if isinstance(payload, dict) else {}

    allowed, deny_reason = assert_skill_allowed("miner", selected_skill)
    if not allowed:
        return {
            "miner_result": build_error_envelope(
                tool=selected_skill,
                request=normalized_payload,
                error="role_policy_denied",
                diagnostics={"reason": deny_reason},
            )
        }

    pre = runtime.hooks.pre_tool_use(selected_skill, normalized_payload)
    if pre.action == "deny":
        return {
            "miner_result": build_error_envelope(
                tool=selected_skill,
                request=normalized_payload,
                error="pre_hook_denied",
                diagnostics={"reason": pre.reason, **pre.diagnostics},
            )
        }

    effective_payload = pre.updated_payload or normalized_payload
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
            "miner_payload": effective_payload,
            "miner_result": build_error_envelope(
                tool=selected_skill,
                request=effective_payload,
                error="post_hook_denied",
                diagnostics={"reason": post.reason, **post.diagnostics},
            ),
        }

    if post.diagnostics:
        miner_result.diagnostics.update(post.diagnostics)
    if post.action == "warn" and post.reason:
        miner_result.diagnostics["post_hook_warning"] = post.reason

    return {
        "miner_payload": effective_payload,
        "miner_result": miner_result,
    }


def _analyst_node(state: WorkflowState) -> WorkflowState:
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
    }


def _formatter_node(state: WorkflowState) -> WorkflowState:
    analyst_result = state.get("analyst_result")
    final_text = ""
    if isinstance(analyst_result, dict):
        final_text = str(analyst_result.get("final") or "")

    miner_result = state.get("miner_result")
    evidence_urls: list[str] = []
    if isinstance(miner_result, SkillEnvelope) and miner_result.status != "error":
        evidence_urls = _extract_urls(miner_result)

    return {
        "final_text": final_text,
        "evidence_urls": evidence_urls,
    }


def build_workflow_graph(
    registry: SkillRegistry | None = None,
    hook_runner: ToolHookRunner | None = None,
    miner_transport: str | None = None,
    mcp_client: Any | None = None,
):
    """Build and compile the LangGraph workflow for v2 orchestration."""

    runtime = _WorkflowRuntime(
        registry=registry or build_default_registry(),
        hooks=hook_runner or ToolHookRunner(),
        miner_transport=_resolve_miner_transport(miner_transport),
        mcp_client=mcp_client,
    )

    graph = StateGraph(WorkflowState)
    graph.add_node("router", _router_node)
    graph.add_node("miner", lambda state: _miner_node(state, runtime))
    graph.add_node("analyst", _analyst_node)
    graph.add_node("formatter", _formatter_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "miner")
    graph.add_edge("miner", "analyst")
    graph.add_edge("analyst", "formatter")
    graph.add_edge("formatter", END)
    return graph.compile()


def run_workflow(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    registry: SkillRegistry | None = None,
    hook_runner: ToolHookRunner | None = None,
    miner_transport: str | None = None,
    mcp_client: Any | None = None,
) -> WorkflowState:
    """Run Router -> Miner -> Analyst -> Formatter with a compiled StateGraph."""

    initial_state: WorkflowState = {
        "user_message": user_message,
        "history": history or [],
        "miner_transport": _resolve_miner_transport(miner_transport),
        "role_prompts": _build_role_prompts(),
    }

    app = build_workflow_graph(
        registry=registry,
        hook_runner=hook_runner,
        miner_transport=miner_transport,
        mcp_client=mcp_client,
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
