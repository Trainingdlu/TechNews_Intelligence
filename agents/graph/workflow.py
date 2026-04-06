"""LangGraph node orchestration for Router -> Miner -> Analyst -> Formatter."""

from __future__ import annotations

import json
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
_VERTEX_MODEL_CACHE: dict[tuple[str, str, str, float], Any] = {}
_VERTEX_MODEL_CACHE_LOCK = threading.RLock()

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


def _workflow_llm_enabled() -> bool:
    # Safe default for production: v2 LLM path is opt-in.
    raw = str(os.getenv("AGENT_WORKFLOW_V2_LLM", "0")).strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    # auto: enable only when Vertex project context exists.
    return bool(
        str(os.getenv("VERTEX_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", ""))).strip()
    )


def _coerce_llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content).strip()


def _build_vertex_chat_model_for_workflow(
    *,
    model_name: str,
    temperature: float,
) -> Any:
    project = os.getenv("VERTEX_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip()
    if not project:
        raise ValueError(
            "VERTEX_PROJECT is not set. You can also use GOOGLE_CLOUD_PROJECT."
        )

    location = os.getenv(
        "VERTEX_LOCATION",
        os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    ).strip()
    model = str(model_name).strip()
    if not model:
        raise ValueError("VERTEX model name is empty.")

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if credentials_path and not os.path.exists(credentials_path):
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS points to a missing file: "
            f"{credentials_path}"
        )

    cache_key = (project, location, model, float(temperature))
    with _VERTEX_MODEL_CACHE_LOCK:
        cached = _VERTEX_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError as exc:
        raise RuntimeError(
            "Vertex provider requires 'langchain-google-vertexai'. "
            "Install dependencies with: pip install -r agents/requirements.txt"
        ) from exc

    built = ChatVertexAI(
        model=model,
        project=project,
        location=location,
        temperature=float(temperature),
    )
    with _VERTEX_MODEL_CACHE_LOCK:
        _VERTEX_MODEL_CACHE[cache_key] = built
    return built


def _invoke_vertex_text(
    *,
    system_instruction: str,
    user_prompt: str,
    model_name: str,
    temperature: float = 0.1,
) -> str:
    model = _build_vertex_chat_model_for_workflow(
        model_name=model_name,
        temperature=temperature,
    )
    merged_prompt = (
        f"{system_instruction.strip()}\n\n"
        f"# Task\n{user_prompt.strip()}\n"
    )
    response = model.invoke(merged_prompt)
    content = getattr(response, "content", response)
    return _coerce_llm_content_to_text(content)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_confidence(value: Any, default: str = "medium") -> str:
    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    if text in {"h", "strong"}:
        return "high"
    if text in {"m", "mid"}:
        return "medium"
    if text in {"l", "weak"}:
        return "low"
    return default


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


def _extract_trend_topic_legacy(user_message: str) -> str:
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


def _extract_days_v2(user_message: str, default: int = 21, maximum: int = 365) -> int:
    text = user_message or ""
    match = re.search(
        r"(\d{1,3})\s*(?:day|days|week|weeks|month|months|\u5929|\u65e5|\u5468|\u4e2a\u6708)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return default
    days = int(match.group(1))
    unit = match.group(0).lower()
    if "week" in unit or "\u5468" in unit:
        days *= 7
    elif "month" in unit or "\u6708" in unit:
        days *= 30
    return max(1, min(maximum, days))


def _extract_trend_topic(user_message: str) -> str:
    """Extract likely topic/entity phrase from trend-style prompts."""

    raw = (user_message or "").strip()
    if not raw:
        return raw

    text = raw
    text = re.sub(
        r"(?i)\b(?:in\s+)?(?:the\s+)?(?:last|past|recent)\s+\d{1,3}\s*(?:day|days|week|weeks|month|months)\b",
        " ",
        text,
    )
    text = re.sub(
        r"\d{1,3}\s*(?:\u5929|\u65e5|\u5468|\u4e2a\u6708|\u6708)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?i)\b(?:trend|momentum|analysis|analyze|analyse|compare|changes?)\b",
        " ",
        text,
    )
    text = re.sub(
        r"(?:\u8d8b\u52bf|\u53d8\u5316|\u5206\u6790|\u5bf9\u6bd4|\u6700\u8fd1|\u8fc7\u53bb)",
        " ",
        text,
    )
    text = re.sub(r"(?i)\b(?:of|for|about|on|in|the|please)\b", " ", text)
    text = re.sub(r"[\s,\u3001\uff0c\u3002\uff1a\uff1b\uff01\uff1f]+", " ", text).strip()
    return text or raw


def _extract_compare_topics(user_message: str) -> tuple[str, str] | None:
    text = (user_message or "").strip()
    if not text:
        return None

    patterns = [
        r"(?i)\b(.+?)\s+(?:vs\.?|versus|compare(?:\s+with)?|against)\s+(.+?)\s*$",
        r"(.+?)\s*(?:\u5bf9\u6bd4|\u5bf9\u7167|\u4e0e|\u548c|VS|vs)\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        left = re.sub(r"^[\W_]+|[\W_]+$", "", str(match.group(1)).strip())
        right = re.sub(r"^[\W_]+|[\W_]+$", "", str(match.group(2)).strip())
        if left and right and left.lower() != right.lower():
            return left, right
    return None


def _intent_from_skill(skill_name: str) -> str:
    skill = str(skill_name or "").strip().lower()
    mapping = {
        "query_news": "fact_retrieval",
        "search_news": "semantic_search",
        "trend_analysis": "trend_analysis",
        "compare_sources": "source_comparison",
        "compare_topics": "comparative_analysis",
        "build_timeline": "timeline_analysis",
        "analyze_landscape": "landscape_analysis",
        "fulltext_batch": "document_reading",
    }
    return mapping.get(skill, "fact_retrieval")


def _route_intent_v2(user_message: str) -> tuple[str, str]:
    text = (user_message or "").lower()
    if re.search(r"(?i)\b(vs|versus|compare)\b", text):
        return "comparative_analysis", "compare_topics"
    if any(token in text for token in ["timeline", "chronology", "milestone", "\u65f6\u95f4\u7ebf", "\u8109\u7edc"]):
        return "timeline_analysis", "build_timeline"
    if any(token in text for token in ["landscape", "ecosystem", "key players", "\u751f\u6001", "\u683c\u5c40"]):
        return "landscape_analysis", "analyze_landscape"
    if any(token in text for token in ["source", "hackernews", "techcrunch", "\u6765\u6e90"]):
        return "source_comparison", "compare_sources"
    if any(token in text for token in ["full text", "read article", "full article", "\u5168\u6587", "\u539f\u6587"]):
        return "document_reading", "fulltext_batch"
    if any(token in text for token in ["semantic", "similar", "related", "\u76f8\u5173", "\u76f8\u4f3c"]):
        return "semantic_search", "search_news"
    if any(token in text for token in ["trend", "momentum", "\u8d8b\u52bf", "\u53d8\u5316", "recent", "last"]):
        return "trend_analysis", "trend_analysis"
    return "fact_retrieval", "query_news"


def _build_payload_for_skill(
    skill_name: str,
    user_message: str,
    intent: str | None = None,
) -> dict[str, Any]:
    skill = str(skill_name or "").strip().lower()
    text = (user_message or "").strip()
    topic = _extract_trend_topic(text) or text

    if skill == "query_news":
        return {
            "query": text,
            "days": _extract_days_v2(text, default=21, maximum=365),
            "source": "all",
            "sort": "time_desc",
            "limit": 8,
        }
    if skill == "search_news":
        return {"query": text or topic or "AI", "days": _extract_days_v2(text, default=21, maximum=365)}
    if skill == "trend_analysis":
        return {
            "topic": topic or text or "AI",
            "window": max(3, min(60, _extract_days_v2(text, default=7, maximum=60))),
        }
    if skill == "compare_sources":
        return {
            "topic": topic or text or "AI",
            "days": max(1, min(90, _extract_days_v2(text, default=14, maximum=90))),
        }
    if skill == "compare_topics":
        pair = _extract_compare_topics(text)
        if pair:
            topic_a, topic_b = pair
        else:
            inferred = topic or text or "OpenAI"
            topic_a = inferred
            topic_b = "Anthropic" if inferred.lower() != "anthropic" else "OpenAI"
        return {
            "topic_a": topic_a,
            "topic_b": topic_b,
            "days": max(1, min(90, _extract_days_v2(text, default=14, maximum=90))),
        }
    if skill == "build_timeline":
        return {
            "topic": topic or text or "AI",
            "days": max(1, min(180, _extract_days_v2(text, default=30, maximum=180))),
            "limit": 12,
        }
    if skill == "analyze_landscape":
        return {
            "topic": topic or text or "AI",
            "days": max(7, min(180, _extract_days_v2(text, default=30, maximum=180))),
            "entities": "",
            "limit_per_entity": 3,
        }
    if skill == "fulltext_batch":
        return {"urls": text or topic or "OpenAI", "max_chars_per_article": 4000}
    # Defensive fallback to baseline retrieval.
    return {
        "query": text,
        "days": _extract_days_v2(text, default=21, maximum=365),
        "source": "all",
        "sort": "time_desc",
        "limit": 8,
    }


def _build_payload_v2(intent: str, user_message: str) -> dict[str, Any]:
    if intent == "trend_analysis":
        return _build_payload_for_skill("trend_analysis", user_message, intent)
    return _build_payload_for_skill("query_news", user_message, intent)


def _coerce_payload_with_registry(
    registry: SkillRegistry,
    skill_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        spec = registry.get(skill_name)
    except KeyError:
        return payload
    try:
        validated = spec.input_model.model_validate(payload or {})
        return validated.model_dump(mode="python")
    except Exception:
        return payload


def _plan_route_with_llm(
    user_message: str,
    runtime: _WorkflowRuntime,
    role_prompts: dict[str, str],
) -> tuple[str, str, dict[str, Any], dict[str, Any], str]:
    heuristic_intent, heuristic_skill = _route_intent_v2(user_message)
    heuristic_payload = _coerce_payload_with_registry(
        runtime.registry,
        heuristic_skill,
        _build_payload_for_skill(heuristic_skill, user_message, heuristic_intent),
    )

    llm_enabled = _workflow_llm_enabled()
    fallback_meta: dict[str, Any] = {
        "enabled": llm_enabled,
        "used": False,
        "fallback_skill": heuristic_skill,
        "fallback_intent": heuristic_intent,
    }
    if not llm_enabled:
        fallback_meta["reason"] = "llm_disabled_or_unconfigured"
        return heuristic_intent, heuristic_skill, heuristic_payload, fallback_meta, "medium"

    allowed_skills = runtime.registry.list_skills()
    router_system_prompt = role_prompts.get("router") or get_role_system_instruction("router")
    router_user_prompt = (
        "Classify user request and return ONLY valid JSON.\n"
        f"User query: {user_message}\n"
        f"Allowed skills: {', '.join(allowed_skills)}\n"
        "Schema: "
        '{"intent":"<string>","skill":"<one_allowed_skill>","params":{...},"confidence":"high|medium|low","reason":"<short>"}'
    )
    model_name = os.getenv(
        "AGENT_WORKFLOW_ROUTER_MODEL",
        os.getenv("VERTEX_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")),
    ).strip()
    temperature = float(os.getenv("AGENT_WORKFLOW_ROUTER_TEMPERATURE", "0.0"))

    try:
        raw = _invoke_vertex_text(
            system_instruction=router_system_prompt,
            user_prompt=router_user_prompt,
            model_name=model_name,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        fallback_meta["reason"] = "router_llm_call_failed"
        fallback_meta["exception_type"] = type(exc).__name__
        fallback_meta["exception"] = str(exc)
        return heuristic_intent, heuristic_skill, heuristic_payload, fallback_meta, "medium"

    parsed = _extract_json_object(raw)
    if not parsed:
        fallback_meta["reason"] = "router_llm_parse_failed"
        fallback_meta["raw_output"] = raw
        return heuristic_intent, heuristic_skill, heuristic_payload, fallback_meta, "medium"

    candidate_skill = str(parsed.get("skill") or "").strip().lower()
    if candidate_skill not in allowed_skills:
        fallback_meta["reason"] = "router_llm_unknown_skill"
        fallback_meta["raw_output"] = raw
        fallback_meta["parsed_skill"] = candidate_skill
        return heuristic_intent, heuristic_skill, heuristic_payload, fallback_meta, "medium"

    llm_params = parsed.get("params")
    params = llm_params if isinstance(llm_params, dict) else {}
    merged_payload = _build_payload_for_skill(candidate_skill, user_message, _intent_from_skill(candidate_skill))
    merged_payload.update(params)
    merged_payload = _coerce_payload_with_registry(runtime.registry, candidate_skill, merged_payload)

    candidate_intent = str(parsed.get("intent") or "").strip().lower() or _intent_from_skill(candidate_skill)
    confidence = _normalize_confidence(parsed.get("confidence"), default="medium")
    router_meta = {
        "enabled": True,
        "used": True,
        "raw_output": raw,
        "parsed": parsed,
        "reason": str(parsed.get("reason") or "").strip(),
    }
    return candidate_intent, candidate_skill, merged_payload, router_meta, confidence


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


def _analyst_summarize_v2(envelope: SkillEnvelope, user_message: str) -> dict[str, Any]:
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


def _envelope_for_llm(envelope: SkillEnvelope) -> dict[str, Any]:
    data: Any = envelope.data
    if isinstance(data, dict):
        compact_data = dict(data)
        records = compact_data.get("records")
        if isinstance(records, list):
            compact_data["records"] = records[:8]
    elif isinstance(data, list):
        compact_data = data[:8]
    else:
        compact_data = data

    evidence_rows: list[dict[str, Any]] = []
    for item in list(envelope.evidence or [])[:12]:
        evidence_rows.append(
            {
                "url": str(item.url or "").strip(),
                "title": str(item.title or "").strip(),
                "source": str(item.source or "").strip(),
                "score": item.score,
                "created_at": item.created_at,
            }
        )

    return {
        "tool": envelope.tool,
        "status": envelope.status,
        "request": envelope.request,
        "data": compact_data,
        "diagnostics": envelope.diagnostics,
        "error": envelope.error,
        "evidence": evidence_rows,
    }


def _heuristic_analyst_summarize(envelope: SkillEnvelope, user_message: str) -> dict[str, Any]:
    if envelope.status == "error":
        return {
            "facts": [],
            "inference": [],
            "final": f"Skill execution failed: {envelope.error or 'unknown_error'}",
            "confidence": "low",
        }

    if envelope.status == "empty":
        if _contains_cjk(user_message):
            final = "\u62b1\u6b49\uff0c\u5f53\u524d\u65f6\u95f4\u7a97\u53e3\u5185\u672a\u68c0\u7d22\u5230\u5339\u914d\u6570\u636e\u3002"
        else:
            final = "No matching records were found in the current time window."
        return {"facts": [], "inference": [], "final": final, "confidence": "low"}

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
        final = "\n".join(["Retrieval summary:", *facts]) if facts else "Retrieval completed with no displayable records."
        return {"facts": facts, "inference": [], "final": final, "confidence": "medium"}

    if envelope.tool == "trend_analysis":
        summary = envelope.data if isinstance(envelope.data, dict) else {}
        topic = str(summary.get("topic", ""))
        recent_cnt = int(summary.get("recent_count", 0))
        prev_cnt = int(summary.get("previous_count", 0))
        count_delta = str(summary.get("count_delta", ""))
        final = (
            f"Trend summary: topic={topic}, recent={recent_cnt}, previous={prev_cnt}, "
            f"delta={count_delta}."
        )
        return {"facts": [final], "inference": [], "final": final, "confidence": "medium"}

    if _contains_cjk(user_message):
        final_generic = "\u5206\u6790\u5df2\u5b8c\u6210\uff0c\u8bf7\u7ed3\u5408\u8bc1\u636e\u94fe\u63a5\u67e5\u770b\u8be6\u7ec6\u6570\u636e\u3002"
    else:
        final_generic = "Analysis completed. Please review evidence URLs for detailed records."
    return {"facts": [], "inference": [], "final": final_generic, "confidence": "medium"}


def _analyst_summarize_with_meta(
    envelope: SkillEnvelope,
    user_message: str,
    role_prompts: dict[str, str] | None = None,
) -> tuple[dict[str, Any], str, str, bool]:
    heuristic = _heuristic_analyst_summarize(envelope, user_message)
    if envelope.status != "ok":
        return heuristic, "", str(heuristic.get("confidence", "low")), False
    if not _workflow_llm_enabled():
        return heuristic, "", str(heuristic.get("confidence", "medium")), False

    prompts = role_prompts or {}
    analyst_system_prompt = prompts.get("analyst") or get_role_system_instruction("analyst")
    analyst_user_prompt = (
        "Use evidence-first reasoning. Return ONLY JSON.\n"
        f"User question: {user_message}\n"
        f"Miner envelope JSON: {json.dumps(_envelope_for_llm(envelope), ensure_ascii=False)}\n"
        "Schema: "
        '{"facts":["..."],"inference":["..."],"final":"...","confidence":"high|medium|low"}'
    )
    model_name = os.getenv(
        "AGENT_WORKFLOW_ANALYST_MODEL",
        os.getenv("VERTEX_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")),
    ).strip()
    temperature = float(os.getenv("AGENT_WORKFLOW_ANALYST_TEMPERATURE", "0.1"))

    try:
        raw = _invoke_vertex_text(
            system_instruction=analyst_system_prompt,
            user_prompt=analyst_user_prompt,
            model_name=model_name,
            temperature=temperature,
        )
    except Exception:
        return heuristic, "", str(heuristic.get("confidence", "medium")), False

    parsed = _extract_json_object(raw)
    if not parsed:
        return heuristic, raw, str(heuristic.get("confidence", "medium")), False

    facts = parsed.get("facts")
    inference = parsed.get("inference")
    final = str(parsed.get("final") or "").strip()
    confidence = _normalize_confidence(parsed.get("confidence"), default="medium")
    if not isinstance(facts, list):
        facts = []
    if not isinstance(inference, list):
        inference = []
    if not final:
        return heuristic, raw, confidence, False

    llm_result = {
        "facts": [str(item).strip() for item in facts if str(item).strip()],
        "inference": [str(item).strip() for item in inference if str(item).strip()],
        "final": final,
        "confidence": confidence,
    }
    return llm_result, raw, confidence, True


def _analyst_summarize(envelope: SkillEnvelope, user_message: str) -> dict[str, Any]:
    result, _, confidence, _ = _analyst_summarize_with_meta(envelope, user_message)
    out = dict(result)
    out.setdefault("confidence", confidence)
    return out


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
    role_prompts = state.get("role_prompts")
    normalized_prompts = role_prompts if isinstance(role_prompts, dict) else _build_role_prompts()
    intent, selected_skill, payload, router_llm_output, router_confidence = _plan_route_with_llm(
        user_message=user_message,
        runtime=runtime,
        role_prompts=normalized_prompts,
    )

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
            "router_llm_output": router_llm_output,
            "router_confidence": router_confidence,
            "node_audit": _append_node_audit(
                state,
                node="router",
                status="deny",
                details={
                    "intent": intent,
                    "selected_skill": selected_skill,
                    "reason": deny_reason,
                    "router_confidence": router_confidence,
                },
            ),
        }

    return {
        "intent": intent,
        "selected_skill": selected_skill,
        "miner_payload": payload,
        "router_llm_output": router_llm_output,
        "router_confidence": router_confidence,
        "node_audit": _append_node_audit(
            state,
            node="router",
            status="allow",
            details={
                "intent": intent,
                "selected_skill": selected_skill,
                "router_confidence": router_confidence,
                "llm_used": bool(router_llm_output.get("used")) if isinstance(router_llm_output, dict) else False,
            },
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
    role_prompts = state.get("role_prompts")
    normalized_prompts = role_prompts if isinstance(role_prompts, dict) else _build_role_prompts()
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
        analyst = _analyst_summarize_v2(denied, str(state.get("user_message") or ""))
        return {
            "miner_result": miner_result,
            "analyst_result": analyst,
            "analyst_denied": True,
            "analyst_llm_output": "",
            "analyst_confidence": str(analyst.get("confidence", "low")),
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
    analyst, raw_llm, analyst_confidence, analyst_llm_used = _analyst_summarize_with_meta(
        miner_result,
        user_message,
        role_prompts=normalized_prompts,
    )
    return {
        "miner_result": miner_result,
        "analyst_result": analyst,
        "analyst_denied": False,
        "analyst_llm_output": raw_llm,
        "analyst_confidence": analyst_confidence,
        "node_audit": _append_node_audit(
            state,
            node="analyst",
            status="allow",
            details={
                "result_status": miner_result.status,
                "analyst_confidence": analyst_confidence,
                "llm_used": analyst_llm_used,
            },
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
            "final_text": _analyst_summarize_v2(denied, str(state.get("user_message") or ""))["final"],
            "formatter_llm_output": "",
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
        "formatter_llm_output": final_text,
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
        "router_llm_output": {},
        "router_confidence": "medium",
        "analyst_llm_output": "",
        "analyst_confidence": "medium",
        "formatter_llm_output": "",
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
