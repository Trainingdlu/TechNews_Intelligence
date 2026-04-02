"""LangGraph-style workflow skeleton for Router/Miner/Analyst orchestration."""

from __future__ import annotations

import os
import re
from typing import Any, TypedDict

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


_DEFAULT_REGISTRY: SkillRegistry | None = None
MCP_SKILL_MAP: dict[str, str] = {
    "query_news": "mcp__newsdb__query_news_vector",
    "trend_analysis": "mcp__newsdb__trend_analysis",
}


def build_default_registry() -> SkillRegistry:
    """Build skill registry for v2 workflow."""

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
            "topic": user_message.strip(),
            "window": max(3, min(60, _extract_days(user_message, default=7, maximum=60))),
        }

    return {
        "query": user_message.strip(),
        "days": _extract_days(user_message, default=21, maximum=365),
        "source": "all",
        "sort": "time_desc",
        "limit": 8,
    }


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
            final = "\n".join(["检索摘要：", *facts]) if facts else "检索完成，但无可展示记录。"
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


def run_workflow(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    registry: SkillRegistry | None = None,
    hook_runner: ToolHookRunner | None = None,
    miner_transport: str | None = None,
    mcp_client: Any | None = None,
) -> WorkflowState:
    """Run Router -> Miner -> Analyst -> Formatter pipeline."""

    state: WorkflowState = {
        "user_message": user_message,
        "history": history or [],
        "role_prompts": {
            "router": get_role_system_instruction("router"),
            "miner": get_role_system_instruction("miner"),
            "analyst": get_role_system_instruction("analyst"),
            "formatter": get_role_system_instruction("formatter"),
        },
    }

    skill_registry = registry or build_default_registry()
    hooks = hook_runner or ToolHookRunner()
    resolved_transport = _resolve_miner_transport(miner_transport)
    state["miner_transport"] = resolved_transport

    intent, selected_skill = _route_intent(user_message)
    state["intent"] = intent
    state["selected_skill"] = selected_skill

    allowed, deny_reason = assert_skill_allowed("miner", selected_skill)
    if not allowed:
        denied = build_error_envelope(
            tool=selected_skill,
            request={},
            error="role_policy_denied",
            diagnostics={"reason": deny_reason},
        )
        state["miner_result"] = denied
        analyst = _analyst_summarize(denied, user_message)
        state["analyst_result"] = analyst
        state["final_text"] = analyst["final"]
        state["evidence_urls"] = []
        return state

    payload = _build_payload(intent, user_message)
    state["miner_payload"] = payload

    pre = hooks.pre_tool_use(selected_skill, payload)
    if pre.action == "deny":
        denied = build_error_envelope(
            tool=selected_skill,
            request=payload,
            error="pre_hook_denied",
            diagnostics={"reason": pre.reason, **pre.diagnostics},
        )
        state["miner_result"] = denied
        analyst = _analyst_summarize(denied, user_message)
        state["analyst_result"] = analyst
        state["final_text"] = analyst["final"]
        state["evidence_urls"] = []
        return state

    effective_payload = pre.updated_payload or payload
    miner_result = _execute_miner_skill(
        selected_skill=selected_skill,
        payload=effective_payload,
        skill_registry=skill_registry,
        miner_transport=resolved_transport,
        mcp_client=mcp_client,
    )

    post = hooks.post_tool_use(selected_skill, effective_payload, miner_result)
    if post.diagnostics:
        miner_result.diagnostics.update(post.diagnostics)
    if post.action == "warn" and post.reason:
        miner_result.diagnostics["post_hook_warning"] = post.reason

    state["miner_result"] = miner_result

    analyst = _analyst_summarize(miner_result, user_message)
    state["analyst_result"] = analyst
    state["final_text"] = analyst["final"]
    state["evidence_urls"] = _extract_urls(miner_result)

    return state


def run_workflow_text(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, set[str]]:
    """Compatibility adapter for agent.py: returns final text and valid URL set."""

    state = run_workflow(user_message=user_message, history=history)
    return state.get("final_text", ""), set(state.get("evidence_urls", []))
