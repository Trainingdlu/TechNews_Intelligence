"""Agent runtime abstraction.

Primary runtime: LangChain/LangGraph standard agent loop.
Fallback runtime: native Gemini SDK (legacy) for compatibility.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, TypedDict

import requests
from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

try:
    from prompts import SYSTEM_INSTRUCTION
    from core.evidence import (
        contains_cjk as _contains_cjk,
        decorate_response_with_sources as _decorate_response_with_sources_core,
        ensure_evidence_section as _ensure_evidence_section,
        extract_urls as _extract_urls,
    )
    from core.metrics import (
        emit_route_metrics as _emit_route_metrics,
        get_route_metrics_snapshot,
        metrics_inc as _metrics_inc,
        reset_route_metrics,
    )
    from core.router import (
        count_timeline_items as _count_timeline_items,
        extract_compare_request as _extract_compare_request,
        extract_fulltext_request as _extract_fulltext_request,
        extract_landscape_request as _extract_landscape_request,
        extract_query_request as _extract_query_request,
        extract_timeline_request as _extract_timeline_request,
        extract_timeline_request_with_confidence as _extract_timeline_request_with_confidence,
        extract_trend_request as _extract_trend_request,
        extract_source_compare_request as _extract_source_compare_request,
    )
    from core.pipelines import (
        run_compare_pipeline,
        run_fulltext_pipeline,
        run_landscape_pipeline,
        run_query_pipeline,
        run_runtime_pipeline,
        run_source_compare_pipeline,
        run_timeline_pipeline,
        run_trend_pipeline,
    )
    from tools import (
        analyze_landscape,
        analyze_ai_landscape,
        build_timeline,
        compare_topics,
        compare_sources,
        fulltext_batch,
        get_db_stats,
        list_topics,
        lookup_url_titles,
        query_news,
        read_news_content,
        search_news,
        trend_analysis,
    )
except ImportError:  # package-style import fallback
    from .prompts import SYSTEM_INSTRUCTION
    from .core.evidence import (
        contains_cjk as _contains_cjk,
        decorate_response_with_sources as _decorate_response_with_sources_core,
        ensure_evidence_section as _ensure_evidence_section,
        extract_urls as _extract_urls,
    )
    from .core.metrics import (
        emit_route_metrics as _emit_route_metrics,
        get_route_metrics_snapshot,
        metrics_inc as _metrics_inc,
        reset_route_metrics,
    )
    from .core.router import (
        count_timeline_items as _count_timeline_items,
        extract_compare_request as _extract_compare_request,
        extract_fulltext_request as _extract_fulltext_request,
        extract_landscape_request as _extract_landscape_request,
        extract_query_request as _extract_query_request,
        extract_timeline_request as _extract_timeline_request,
        extract_timeline_request_with_confidence as _extract_timeline_request_with_confidence,
        extract_trend_request as _extract_trend_request,
        extract_source_compare_request as _extract_source_compare_request,
    )
    from .core.pipelines import (
        run_compare_pipeline,
        run_fulltext_pipeline,
        run_landscape_pipeline,
        run_query_pipeline,
        run_runtime_pipeline,
        run_source_compare_pipeline,
        run_timeline_pipeline,
        run_trend_pipeline,
    )
    from .tools import (
        analyze_landscape,
        analyze_ai_landscape,
        build_timeline,
        compare_topics,
        compare_sources,
        fulltext_batch,
        get_db_stats,
        list_topics,
        lookup_url_titles,
        query_news,
        read_news_content,
        search_news,
        trend_analysis,
    )


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------
@tool("search_news")
def search_news_tool(query: str, days: int = 21) -> str:
    """Search related news with hybrid retrieval (semantic + keyword)."""
    return search_news(query=query, days=days)


@tool("read_news_content")
def read_news_content_tool(url: str) -> str:
    """Read full article content by URL from stored raw logs."""
    return read_news_content(url=url)


@tool("get_db_stats")
def get_db_stats_tool() -> str:
    """Get database freshness stats and total article count."""
    return get_db_stats()


@tool("list_topics")
def list_topics_tool() -> str:
    """Get article volume distribution over recent days."""
    return list_topics()


@tool("query_news")
def query_news_tool(
    query: str = "",
    source: str = "all",
    days: int = 21,
    category: str = "",
    sentiment: str = "",
    sort: str = "time_desc",
    limit: int = 8,
) -> str:
    """Query news with filterable retrieval (source/days/category/sentiment/sort)."""
    return query_news(
        query=query,
        source=source,
        days=days,
        category=category,
        sentiment=sentiment,
        sort=sort,
        limit=limit,
    )


@tool("trend_analysis")
def trend_analysis_tool(topic: str, window: int = 7) -> str:
    """Analyze topic momentum in recent window vs previous window."""
    return trend_analysis(topic=topic, window=window)


@tool("compare_sources")
def compare_sources_tool(topic: str, days: int = 14) -> str:
    """Compare HackerNews vs TechCrunch coverage/sentiment/heat for a topic."""
    return compare_sources(topic=topic, days=days)


@tool("compare_topics")
def compare_topics_tool(topic_a: str, topic_b: str, days: int = 14) -> str:
    """Compare two entities/topics (e.g., OpenAI vs Anthropic) with DB evidence."""
    return compare_topics(topic_a=topic_a, topic_b=topic_b, days=days)


@tool("build_timeline")
def build_timeline_tool(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build chronological timeline for topic/company/product."""
    return build_timeline(topic=topic, days=days, limit=limit)


@tool("analyze_ai_landscape")
def analyze_ai_landscape_tool(days: int = 30, entities: str = "", limit_per_entity: int = 3) -> str:
    """Analyze global AI landscape with DB-backed entity stats and evidence URLs."""
    return analyze_ai_landscape(days=days, entities=entities, limit_per_entity=limit_per_entity)


@tool("analyze_landscape")
def analyze_landscape_tool(
    topic: str = "",
    days: int = 30,
    entities: str = "",
    limit_per_entity: int = 3,
) -> str:
    """Analyze cross-domain landscape with DB-backed entity stats and evidence URLs."""
    return analyze_landscape(topic=topic, days=days, entities=entities, limit_per_entity=limit_per_entity)


@tool("fulltext_batch")
def fulltext_batch_tool(urls: str, max_chars_per_article: int = 4000) -> str:
    """Batch read multiple article full-text contents by URLs."""
    return fulltext_batch(urls=urls, max_chars_per_article=max_chars_per_article)


LANGCHAIN_TOOLS = [
    search_news_tool,
    read_news_content_tool,
    get_db_stats_tool,
    list_topics_tool,
    query_news_tool,
    trend_analysis_tool,
    compare_sources_tool,
    compare_topics_tool,
    build_timeline_tool,
    analyze_landscape_tool,
    analyze_ai_landscape_tool,
    fulltext_batch_tool,
]


# ---------------------------------------------------------------------------
# Legacy Gemini (fallback only)
# ---------------------------------------------------------------------------
LEGACY_TOOLS = [
    search_news,
    read_news_content,
    get_db_stats,
    list_topics,
    query_news,
    trend_analysis,
    compare_sources,
    compare_topics,
    build_timeline,
    analyze_landscape,
    analyze_ai_landscape,
    fulltext_batch,
]


def _build_legacy_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        tools=LEGACY_TOOLS,
        system_instruction=SYSTEM_INSTRUCTION,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=50,
        ),
    )


def _generate_legacy(history: list[dict], user_message: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        config=_build_legacy_config(),
        history=history,
    )
    response = chat.send_message(user_message)
    if getattr(response, "candidates", None) is None or not response.candidates:
        return "[Error] No candidate generated by model."

    parts_texts = [
        part.text
        for part in response.candidates[0].content.parts
        if hasattr(part, "text") and part.text
    ]
    if not parts_texts:
        return "[Error] Model returned no valid text."
    return parts_texts[-1]


# ---------------------------------------------------------------------------
# LangGraph runtime
# ---------------------------------------------------------------------------
_langgraph_agent: Any | None = None
_analysis_model: Any | None = None
## NOTE:
## Route metrics, evidence formatting, and deterministic request extractors
## have been moved to agents/core/metrics.py, agents/core/evidence.py,
## and agents/core/router.py.


def _history_to_messages(history: list[dict]) -> list[Any]:
    messages: list[Any] = []
    for item in history or []:
        role = item.get("role", "")
        text = ""
        for part in item.get("parts", []):
            if isinstance(part, dict) and part.get("text"):
                text += str(part["text"])
        if not text:
            continue

        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def _coerce_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                txt = part.get("text")
                if txt:
                    chunks.append(str(txt))
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content)


def _parse_tool_json(output: str, expected_tool: str = "") -> dict[str, Any] | None:
    raw = (output or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    tool_name = str(payload.get("tool", "")).strip().lower()
    if expected_tool and tool_name and tool_name != expected_tool.strip().lower():
        return None
    return payload


def _format_query_ground_truth(query_output: str) -> str:
    payload = _parse_tool_json(query_output, expected_tool="query_news")
    if not payload:
        return query_output

    records = payload.get("records") or []
    if not records:
        return "No matching records."

    lines = [f"Query records: {len(records)}"]
    for item in records[:20]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title_cn") or item.get("title") or "").strip()
        source = str(item.get("source") or "").strip()
        created_at = str(item.get("created_at") or "").replace("T", " ")[:16]
        sentiment = str(item.get("sentiment") or "").strip()
        points = item.get("points", 0)
        url = str(item.get("url") or "").strip()
        summary = str(item.get("summary") or "").strip()
        rank = item.get("rank", 0)
        lines.append(
            f"{rank}. [{source}] {title}\n"
            f"   time={created_at}, sentiment={sentiment}, points={points}\n"
            f"   url={url}\n"
            f"   summary={summary[:220]}"
        )
    return "\n".join(lines)


def _format_fulltext_ground_truth(fulltext_output: str) -> str:
    payload = _parse_tool_json(fulltext_output, expected_tool="fulltext_batch")
    if not payload:
        return fulltext_output

    status = str(payload.get("status", "ok")).lower()
    if status != "ok":
        return str(payload.get("error") or "No candidate articles found.")

    selected = payload.get("selected") or []
    articles = payload.get("articles") or []
    lines: list[str] = [f"Selected articles: {len(selected)}"]

    for idx, item in enumerate(selected, 1):
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("source_type") or "").strip()
        headline = str(item.get("headline") or "").strip()
        points = item.get("points", 0)
        score = item.get("score")
        created_at = str(item.get("created_at") or "").replace("T", " ")[:16]
        url = str(item.get("url") or "").strip()
        if score is None:
            lines.append(
                f"{idx}. [{source_type}] {headline} | points={points} | {created_at} | {url}"
            )
        else:
            lines.append(
                f"{idx}. [{source_type}] {headline} | points={points} | score={float(score):.3f} | {created_at} | {url}"
            )

    for idx, item in enumerate(articles, 1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "")
        lines.append(f"=== [{idx}] {url} ===\n{content}")
    return "\n".join(lines).strip()


_DATE_TOKEN_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?P<y>\d{4})\s*(?:-|/|\.|年)\s*"
    r"(?P<m>\d{1,2})\s*(?:-|/|\.|月)\s*"
    r"(?P<d>\d{1,2})"
    r"(?:\s*日)?"
)


def _extract_normalized_dates(text: str) -> set[str]:
    out: set[str] = set()
    for m in _DATE_TOKEN_PATTERN.finditer(text or ""):
        try:
            y = int(m.group("y"))
            mo = int(m.group("m"))
            d = int(m.group("d"))
            # Use datetime constructor for strict calendar validation.
            datetime(y, mo, d)
        except Exception:
            continue
        out.add(f"{y:04d}-{mo:02d}-{d:02d}")
    return out


def _format_raw_snapshot_for_user(source_output: str) -> str:
    raw = str(source_output or "").strip()
    if not raw:
        return ""

    payload = _parse_tool_json(raw)
    if payload:
        tool_name = str(payload.get("tool", "")).strip().lower()
        if tool_name == "query_news":
            return _format_query_ground_truth(raw)
        if tool_name == "fulltext_batch":
            return _format_fulltext_ground_truth(raw)

    lines: list[str] = []
    for ln in raw.splitlines():
        stripped = str(ln).strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        pretty = stripped
        if " | " in pretty and pretty.count(" | ") >= 2:
            pretty = pretty.replace(" | ", " · ")
        pretty = re.sub(r"(?i)\btime=", "time: ", pretty)
        pretty = re.sub(r"(?i)\burl=", "url: ", pretty)
        lines.append(pretty)

    return "\n".join(lines).strip()


_GENERIC_ANALYSIS_LEADIN_PATTERNS = (
    re.compile(
        r"^(?:好(?:的)?|当然|没问题|可以)[，,\s]*"
        r"(?:作为一名[^。:\n]*分析师[^。:\n]*"
        r"|(?:这是|以下是|下面是)基于[^。:\n]*(?:数据|输出)[^。:\n]*(?:分析|解读|梳理)"
        r"|(?:我们|我|让我)来看[^。:\n]*(?:分析|解读|梳理|格局)"
        r"|(?:我|我们)来[^。:\n]*(?:分析|解读|梳理))"
        r"[\s。!！:：]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:这是|以下是|下面是)基于[^。:\n]*(?:数据|输出)[^。:\n]*(?:分析|解读|梳理)[\s。!！:：]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:okay|sure|got it)[,\s]+(?:here(?:'s| is)|below is).*(?:analysis|summary)[\s.]*$",
        re.IGNORECASE,
    ),
)
_GENERIC_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
_ENTITY_ROW_REF_RE = re.compile(r"\[(?P<entity>[^\]\n]{1,80})\]\s*#(?P<rank>\d{1,2})")


def _strip_generic_analysis_leadin(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw.strip()

    lines = raw.splitlines()
    first_idx = 0
    while first_idx < len(lines) and not lines[first_idx].strip():
        first_idx += 1
    if first_idx >= len(lines):
        return raw.strip()

    first_line = lines[first_idx].strip()
    if not any(p.match(first_line) for p in _GENERIC_ANALYSIS_LEADIN_PATTERNS):
        return raw.strip()

    start_idx = first_idx + 1
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    if start_idx < len(lines) and _GENERIC_SEPARATOR_RE.match(lines[start_idx].strip()):
        start_idx += 1
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1

    stripped = "\n".join(lines[start_idx:]).strip()
    return stripped or raw.strip()


def _normalize_entity_row_ref(entity: str, rank: str) -> str:
    normalized_entity = re.sub(r"\s+", " ", str(entity or "").strip()).lower()
    try:
        normalized_rank = int(rank)
    except Exception:
        normalized_rank = 0
    return f"{normalized_entity}#{normalized_rank}"


def _build_entity_row_ref_url_map(source_output: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in str(source_output or "").splitlines():
        match = _ENTITY_ROW_REF_RE.search(line)
        if not match:
            continue
        urls = _extract_urls(line)
        if not urls:
            continue
        key = _normalize_entity_row_ref(match.group("entity"), match.group("rank"))
        if key and key not in mapping:
            mapping[key] = urls[0]
    return mapping


def _remap_entity_row_refs_to_global_sources(answer: str, source_output: str) -> str:
    body = str(answer or "")
    if not body:
        return body

    ref_to_url = _build_entity_row_ref_url_map(source_output)
    if not ref_to_url:
        return body

    ordered_urls: list[str] = []
    seen: set[str] = set()
    for url in _extract_urls(source_output):
        if url in seen:
            continue
        seen.add(url)
        ordered_urls.append(url)
    if not ordered_urls:
        return body

    url_to_index = {url: idx for idx, url in enumerate(ordered_urls, 1)}
    ref_to_index: dict[str, int] = {}
    for ref, url in ref_to_url.items():
        index = url_to_index.get(url)
        if index is not None:
            ref_to_index[ref] = index
    if not ref_to_index:
        return body

    def _replace(match: re.Match[str]) -> str:
        key = _normalize_entity_row_ref(match.group("entity"), match.group("rank"))
        index = ref_to_index.get(key)
        if index is None:
            return match.group(0)
        return f"[{index}]"

    return _ENTITY_ROW_REF_RE.sub(_replace, body)


def _extract_final_text(result: Any) -> str:
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                text = _coerce_to_text(msg.content)
                if text:
                    return text
        if messages:
            tail = messages[-1]
            text = _coerce_to_text(getattr(tail, "content", tail))
            if text:
                return text
    return _coerce_to_text(result)


def _decorate_response_with_sources(text: str, user_message: str) -> tuple[str, dict[str, str]]:
    """Attach numbered citations and source section via shared evidence helper."""
    return _decorate_response_with_sources_core(
        text=text,
        user_message=user_message,
        lookup_url_titles=lookup_url_titles,
    )


def _is_unstable_landscape_synthesis(text: str) -> bool:
    """Detect unstable synthesis patterns and fallback to DB raw snapshot."""
    content = (text or "").strip()
    if not content:
        return True
    lower = content.lower()
    suspicious_markers = [
        "工具当前遇到技术问题",
        "无法生成全球ai态势",
        "无法生成完整报告",
        "请问您最关心哪两个公司",
        "which two companies",
        "cannot generate a complete",
    ]
    return any(marker in content or marker in lower for marker in suspicious_markers)


def _format_landscape_no_data_response(
    user_message: str,
    topic: str,
    days: int,
    entities: list[str],
    raw_landscape: str,
) -> str:
    topic_text = topic or "全局"
    entity_text = ", ".join(entities) if entities else "默认实体集合"
    raw_text = (raw_landscape or "").strip()
    is_tool_error = raw_text.lower().startswith("analyze_landscape failed:")
    if _contains_cjk(user_message):
        if is_tool_error:
            return (
                f"当前无法在最近 {days} 天内为“{topic_text}”生成稳定的格局分析（实体范围：{entity_text}）。\n"
                "这次是工具执行错误，不是样本不足导致。\n\n"
                f"数据库返回：{raw_text}\n\n"
                "建议：\n"
                "- 先修复工具时间戳比较/时区处理\n"
                "- 修复后再重试同一问题验证\n"
                "- 若仍无结果，再考虑扩大窗口或补充实体\n"
                "置信度：低"
            )
        return (
            f"当前无法在最近 {days} 天内为“{topic_text}”生成稳定的格局分析（实体范围：{entity_text}）。\n"
            "这通常意味着样本不足或主题与已入库新闻匹配度较低。\n\n"
            f"数据库返回：{raw_text}\n\n"
            "建议：\n"
            "- 扩大时间窗口（例如最近 60 天）\n"
            "- 明确实体（例如 OpenAI, Google, Microsoft）\n"
            "- 使用更宽主题（例如“科技格局”而非过窄关键词）\n"
            "置信度：低"
        )
    if is_tool_error:
        return (
            f"Unable to produce a stable landscape answer for topic='{topic or 'all'}' "
            f"in the last {days} days (entities: {entity_text}).\n"
            "This is a tool execution failure (not a sparse-data issue).\n\n"
            f"DB output: {raw_text}\n\n"
            "Try:\n"
            "- fix timestamp/timezone comparison in the landscape tool\n"
            "- retry the same query after the fix\n"
            "- only then broaden window/entities if needed.\n"
            "Confidence: Low"
        )
    return (
        f"Unable to produce a stable landscape answer for topic='{topic or 'all'}' "
        f"in the last {days} days (entities: {entity_text}).\n"
        "This usually means sparse coverage or weak topic-entity matching.\n\n"
        f"DB output: {raw_text}\n\n"
        "Try:\n"
        "- a wider time window (e.g., 60 days)\n"
        "- explicit entities (e.g., OpenAI, Google, Microsoft)\n"
        "- a broader topic query.\n"
        "Confidence: Low"
    )


def _get_langgraph_agent():
    global _langgraph_agent
    if _langgraph_agent is not None:
        return _langgraph_agent

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        google_api_key=api_key,
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
    )

    try:
        _langgraph_agent = create_react_agent(
            model=model,
            tools=LANGCHAIN_TOOLS,
            prompt=SYSTEM_INSTRUCTION,
        )
    except TypeError:
        _langgraph_agent = create_react_agent(
            model=model,
            tools=LANGCHAIN_TOOLS,
        )

    return _langgraph_agent


def _get_analysis_model():
    global _analysis_model
    if _analysis_model is not None:
        return _analysis_model

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    _analysis_model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        google_api_key=api_key,
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
    )
    return _analysis_model


def _planner_enabled(runtime: str) -> bool:
    if runtime == "legacy":
        return False
    flag = os.getenv("AGENT_PLANNER_ENABLED", "true").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def _planner_min_confidence() -> float:
    try:
        raw = float(os.getenv("AGENT_PLANNER_MIN_CONFIDENCE", "0.75"))
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.75


def _planner_timeout_sec() -> float:
    try:
        raw = float(os.getenv("AGENT_PLANNER_TIMEOUT_SEC", "8"))
        return max(2.0, min(30.0, raw))
    except Exception:
        return 8.0


def _planner_endpoint() -> str:
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").strip().rstrip("/")
    lower = base.lower()
    if lower.endswith("/chat/completions"):
        return base
    if lower.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _strip_json_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_planner_payload(text: str) -> dict[str, Any] | None:
    raw = _strip_json_fence(text)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _to_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _to_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        num = int(value)
    except Exception:
        num = default
    return max(minimum, min(maximum, num))


def _planner_entities(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out[:12]
    text = str(value or "").strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"[,\n;/|，、]+", text) if s.strip()][:12]


def _plan_route_with_deepseek(history: list[dict], user_message: str) -> dict[str, Any] | None:
    del history  # Planner is stateless for predictable routing decisions.

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("DEEPSEEK_PLANNER_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    endpoint = _planner_endpoint()
    system_prompt = (
        "You are a routing planner for a tech-news agent.\n"
        "Your output MUST be strict JSON only with keys: intent, confidence, params.\n"
        "No markdown, no prose.\n"
        "Allowed intent values:\n"
        "- source_compare\n"
        "- compare\n"
        "- timeline\n"
        "- landscape\n"
        "- trend\n"
        "- fulltext\n"
        "- query\n"
        "- runtime\n"
        "Param schema:\n"
        "- source_compare: {topic, days}\n"
        "- compare: {topic_a, topic_b, days}\n"
        "- timeline: {topic, days, limit}\n"
        "- landscape: {topic, days, entities}\n"
        "- trend: {topic, window}\n"
        "- fulltext: {query, max_chars}\n"
        "- query: {query, source, days, sort, limit}\n"
        "- runtime: {}\n"
        "Rules:\n"
        "1) If uncertain, choose runtime with low confidence.\n"
        "2) Keep confidence in [0,1].\n"
        "3) Never fabricate URLs or facts."
    )

    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=_planner_timeout_sec(),
        )
        resp.raise_for_status()
        payload = resp.json()
        choices = payload.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content") or "")
        parsed = _parse_planner_payload(content)
        return parsed
    except Exception as exc:
        print(f"[Warn] DeepSeek planner failed, fallback to runtime: {exc}")
        return None


def _dispatch_planner_route(
    *,
    runtime: str,
    strict_mode: bool,
    history: list[dict],
    user_message: str,
) -> str | None:
    if not _planner_enabled(runtime):
        return None

    _metrics_inc("planner_attempts", 1)
    plan = _plan_route_with_deepseek(history, user_message)
    if not isinstance(plan, dict):
        _metrics_inc("planner_empty", 1)
        return None

    intent = _to_str(plan.get("intent"), "").lower()
    params = plan.get("params")
    if not isinstance(params, dict):
        params = {}

    try:
        confidence = float(plan.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    if confidence < _planner_min_confidence():
        _metrics_inc("planner_low_confidence", 1)
        return None

    if intent in {"", "runtime"}:
        _metrics_inc("planner_runtime", 1)
        return None

    _metrics_inc("planner_routed", 1)

    if intent == "source_compare":
        _metrics_inc("source_compare_forced")
        topic = _to_str(params.get("topic"), "AI")
        days = _to_int(params.get("days"), 14, 1, 90)
        return run_source_compare_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            days=days,
            compare_sources_fn=compare_sources,
            analyze_fn=_analyze_source_compare_output,
            ensure_evidence_fn=_ensure_source_compare_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "compare":
        topic_a = _to_str(params.get("topic_a"), "")
        topic_b = _to_str(params.get("topic_b"), "")
        if not topic_a or not topic_b:
            _metrics_inc("planner_invalid_params", 1)
            return None
        _metrics_inc("compare_forced")
        days = _to_int(params.get("days"), 14, 1, 90)
        return run_compare_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic_a=topic_a,
            topic_b=topic_b,
            days=days,
            compare_topics_fn=compare_topics,
            analyze_fn=_analyze_compare_output,
            ensure_evidence_fn=_ensure_compare_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "timeline":
        topic = _to_str(params.get("topic"), "")
        if not topic:
            _metrics_inc("planner_invalid_params", 1)
            return None
        _metrics_inc("timeline_forced")
        days = _to_int(params.get("days"), 30, 1, 180)
        limit = _to_int(params.get("limit"), 12, 3, 40)
        return run_timeline_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            days=days,
            limit=limit,
            build_timeline_fn=build_timeline,
            count_timeline_items_fn=_count_timeline_items,
            format_low_sample_fn=_format_low_sample_timeline_response,
            analyze_fn=_analyze_timeline_output,
            ensure_evidence_fn=_ensure_timeline_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "landscape":
        _metrics_inc("landscape_forced")
        topic = _to_str(params.get("topic"), "")
        days = _to_int(params.get("days"), 30, 7, 180)
        entities = _planner_entities(params.get("entities"))
        return run_landscape_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            days=days,
            entities=entities,
            analyze_landscape_fn=analyze_landscape,
            is_no_data_fn=_is_landscape_no_data,
            format_no_data_fn=_format_landscape_no_data_response,
            is_evidence_sufficient_fn=_is_landscape_evidence_sufficient,
            metrics_inc_fn=_metrics_inc,
            format_low_evidence_fn=_format_low_evidence_landscape_response,
            analyze_output_fn=_analyze_landscape_output,
            is_unstable_synthesis_fn=_is_unstable_landscape_synthesis,
            ensure_evidence_fn=_ensure_landscape_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "trend":
        topic = _to_str(params.get("topic"), "")
        if not topic:
            _metrics_inc("planner_invalid_params", 1)
            return None
        _metrics_inc("trend_forced")
        window = _to_int(params.get("window"), 7, 3, 60)
        return run_trend_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            window=window,
            trend_analysis_fn=trend_analysis,
            analyze_fn=_analyze_trend_output,
            ensure_evidence_fn=_ensure_trend_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "fulltext":
        request_query = _to_str(params.get("query"), "")
        if not request_query:
            _metrics_inc("planner_invalid_params", 1)
            return None
        _metrics_inc("fulltext_forced")
        max_chars = _to_int(params.get("max_chars"), 4000, 800, 12000)
        return run_fulltext_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            request_query=request_query,
            max_chars=max_chars,
            fulltext_batch_fn=fulltext_batch,
            analyze_fn=_analyze_fulltext_output,
            ensure_evidence_fn=_ensure_fulltext_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "query":
        query = _to_str(params.get("query"), "")
        if not query:
            _metrics_inc("planner_invalid_params", 1)
            return None
        _metrics_inc("query_forced")
        source = _to_str(params.get("source"), "all")
        days = _to_int(params.get("days"), 21, 1, 365)
        sort = _to_str(params.get("sort"), "time_desc")
        limit = _to_int(params.get("limit"), 8, 1, 30)
        return run_query_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            query=query,
            source=source,
            days=days,
            sort=sort,
            limit=limit,
            query_news_fn=query_news,
            analyze_fn=_analyze_query_output,
            ensure_evidence_fn=_ensure_query_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    _metrics_inc("planner_unknown_intent", 1)
    return None


def _is_answer_grounded_in_source(answer: str, source_output: str) -> bool:
    answer_urls = set(_extract_urls(answer))
    source_urls = set(_extract_urls(source_output))
    if answer_urls and source_urls and any(url not in source_urls for url in answer_urls):
        return False

    source_dates = _extract_normalized_dates(source_output or "")
    answer_dates = _extract_normalized_dates(answer or "")
    if source_dates and answer_dates and any(d not in source_dates for d in answer_dates):
        return False
    return True


def _ensure_grounded_evidence_or_fallback(
    *,
    answer: str,
    source_output: str,
    user_message: str,
    max_urls: int,
    route_label: str,
) -> str:
    if not answer:
        return answer
    if not _is_answer_grounded_in_source(answer, source_output):
        print(f"[Warn] {route_label} synthesis failed grounding gate; fallback to raw tool output.")
        formatted_snapshot = _format_raw_snapshot_for_user(source_output)
        if _contains_cjk(user_message):
            return (
                "检测到回答包含未在数据库工具输出中出现的事实字段，已回退为数据库原始结果。\n\n"
                "数据库事实快照（已格式化）：\n"
                f"{formatted_snapshot}"
            )
        return (
            "Detected unsupported facts not present in tool output; falling back to formatted DB/tool snapshot.\n\n"
            f"{formatted_snapshot}"
        )
    answer = _remap_entity_row_refs_to_global_sources(answer, source_output)
    return _ensure_evidence_section(
        answer=answer,
        source_output=source_output,
        user_message=user_message,
        max_urls=max_urls,
    )


def _analyze_compare_output(
    user_message: str,
    topic_a: str,
    topic_b: str,
    days: int,
    compare_output: str,
) -> str:
    """Turn DB comparison output into analyst-friendly answer without adding new facts."""
    model = _get_analysis_model()
    is_cjk = _contains_cjk(user_message)
    section_hint = (
        "- 对比结论\n"
        "- 关键变量（算力/算法/数据）\n"
        "- 证据\n"
        "- 工程优化 vs 范式转移\n"
        "- 未来6-18个月影响\n"
        "- 事实 / 推断 / 情景（分别列出）"
        if is_cjk
        else "- Comparison Conclusions\n"
        "- Key Variables (compute/algorithm/data)\n"
        "- Evidence\n"
        "- Engineering Optimization vs Paradigm Shift\n"
        "- Next 6-18 Months Implications\n"
        "- Facts / Inference / Scenarios (separated)"
    )
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive DB comparison output.\n"
        "Rules:\n"
        "1) Use ONLY facts/URLs/numbers that already exist in the provided DB output.\n"
        "2) Do NOT invent any company event, model release, personnel change, or URL.\n"
        "3) Every major conclusion must be tied to explicit evidence in the DB output.\n"
        "4) Decompose major differences into compute/cost, algorithm/efficiency, data/moat when evidence exists.\n"
        "5) Do not ignore accumulation effects: repeated small deltas can indicate structural change.\n"
        "6) Explicitly label whether each major change is engineering optimization or paradigm shift.\n"
        "7) Provide conditional 6-18 month implications; avoid deterministic prophecy.\n"
        "8) Separate facts vs inference vs scenarios; do not mix them.\n"
        "9) If evidence is weak or missing for any section, write evidence is insufficient.\n"
        "10) Reply in the user's language.\n"
        "11) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Comparison target: {topic_a} vs {topic_b}, window={days} days\n\n"
        "DB comparison output (ground truth):\n"
        f"{compare_output}\n\n"
        "Now produce an analysis answer based only on the DB output above.\n"
        "Use this section order:\n"
        f"{section_hint}"
    )
    result = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    text = _coerce_to_text(getattr(result, "content", result))
    return text.strip()


def _ensure_compare_evidence(answer: str, compare_output: str, user_message: str) -> str:
    """Ensure compare answer always includes explicit evidence URL section."""
    return _ensure_grounded_evidence_or_fallback(
        answer=answer,
        source_output=compare_output,
        user_message=user_message,
        max_urls=6,
        route_label="compare",
    )


def _ensure_timeline_evidence(answer: str, timeline_output: str, user_message: str) -> str:
    """Ensure timeline answer always includes explicit evidence URL section."""
    return _ensure_grounded_evidence_or_fallback(
        answer=answer,
        source_output=timeline_output,
        user_message=user_message,
        max_urls=8,
        route_label="timeline",
    )


def _ensure_trend_evidence(answer: str, trend_output: str, user_message: str) -> str:
    """Ensure trend answer includes evidence URL section when URLs are available."""
    return _ensure_grounded_evidence_or_fallback(
        answer=answer,
        source_output=trend_output,
        user_message=user_message,
        max_urls=6,
        route_label="trend",
    )


def _ensure_source_compare_evidence(answer: str, source_output: str, user_message: str) -> str:
    """Ensure source-compare answer includes evidence URL section."""
    return _ensure_grounded_evidence_or_fallback(
        answer=answer,
        source_output=source_output,
        user_message=user_message,
        max_urls=8,
        route_label="source_compare",
    )


def _ensure_query_evidence(answer: str, query_output: str, user_message: str) -> str:
    """Ensure query answer always includes evidence URL section."""
    return _ensure_grounded_evidence_or_fallback(
        answer=answer,
        source_output=query_output,
        user_message=user_message,
        max_urls=10,
        route_label="query",
    )


def _ensure_fulltext_evidence(answer: str, fulltext_output: str, user_message: str) -> str:
    """Ensure fulltext answer always includes evidence URL section."""
    return _ensure_grounded_evidence_or_fallback(
        answer=answer,
        source_output=fulltext_output,
        user_message=user_message,
        max_urls=10,
        route_label="fulltext",
    )


def _analyze_trend_output(
    user_message: str,
    topic: str,
    window: int,
    trend_output: str,
) -> str:
    """Turn trend tool output into analyst-friendly answer without adding new facts."""
    model = _get_analysis_model()
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive trend tool output.\n"
        "Rules:\n"
        "1) Use ONLY facts/URLs/numbers from tool output.\n"
        "2) Do NOT invent events/URLs.\n"
        "3) Explicitly separate observed trend vs hypothesis.\n"
        "4) Reply in user's language.\n"
        "5) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Trend target: {topic}, window={window} days\n\n"
        "Trend tool output (ground truth):\n"
        f"{trend_output}\n\n"
        "Now produce concise analysis with sections:\n"
        "- 趋势结论\n"
        "- 关键数据\n"
        "- 驱动因素（证据不足则写证据不足）\n"
        "- 后续关注"
    )
    result = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return _coerce_to_text(getattr(result, "content", result)).strip()


def _analyze_source_compare_output(
    user_message: str,
    topic: str,
    days: int,
    source_compare_output: str,
) -> str:
    """Turn source-compare output into analyst-friendly answer without adding new facts."""
    model = _get_analysis_model()
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive source comparison output.\n"
        "Rules:\n"
        "1) Use ONLY tool facts/URLs/numbers.\n"
        "2) Do NOT invent facts/URLs.\n"
        "3) Explain differences by evidence-backed dimensions.\n"
        "4) Reply in user's language.\n"
        "5) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Source compare target: topic={topic}, window={days} days\n\n"
        "Source compare tool output (ground truth):\n"
        f"{source_compare_output}\n\n"
        "Now produce concise analysis with sections:\n"
        "- 对比结论\n"
        "- 维度差异（热度/情绪/覆盖）\n"
        "- 证据\n"
        "- 对决策的影响"
    )
    result = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return _coerce_to_text(getattr(result, "content", result)).strip()


def _analyze_query_output(
    user_message: str,
    query: str,
    source: str,
    days: int,
    sort: str,
    query_output: str,
) -> str:
    """Turn query tool output into concise analyst summary without adding new facts."""
    model = _get_analysis_model()
    normalized_ground_truth = _format_query_ground_truth(query_output)
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive filtered retrieval output.\n"
        "Rules:\n"
        "1) Use ONLY facts/URLs/numbers from tool output.\n"
        "2) Do NOT invent facts or URLs.\n"
        "3) If data is sparse, state it explicitly.\n"
        "4) Reply in user's language.\n"
        "5) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Query target: query={query}, source={source}, window={days}d, sort={sort}\n\n"
        "Query tool output (ground truth):\n"
        f"{normalized_ground_truth}\n\n"
        "Now produce concise analysis with sections:\n"
        "- 检索结果摘要\n"
        "- 关键信号\n"
        "- 后续跟进建议"
    )
    result = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return _coerce_to_text(getattr(result, "content", result)).strip()


def _analyze_fulltext_output(
    user_message: str,
    request_query: str,
    fulltext_output: str,
) -> str:
    """Turn batch fulltext output into evidence-grounded synthesis."""
    model = _get_analysis_model()
    normalized_ground_truth = _format_fulltext_ground_truth(fulltext_output)
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive batch fulltext output.\n"
        "Rules:\n"
        "1) Use ONLY facts/quotes/URLs from tool output.\n"
        "2) Do NOT invent claims or URLs.\n"
        "3) Separate consensus vs disagreement.\n"
        "4) Reply in user's language.\n"
        "5) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Fulltext batch target: {request_query}\n\n"
        "Fulltext tool output (ground truth):\n"
        f"{normalized_ground_truth}\n\n"
        "Now produce concise analysis with sections:\n"
        "- 核心结论\n"
        "- 争议焦点\n"
        "- 证据摘录（仅基于原文）\n"
        "- 不确定性与后续验证"
    )
    result = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return _coerce_to_text(getattr(result, "content", result)).strip()


def _analyze_landscape_output(
    user_message: str,
    topic: str,
    days: int,
    entities: list[str],
    landscape_output: str,
) -> str:
    """Turn DB landscape output into analyst-friendly answer without adding new facts."""
    model = _get_analysis_model()
    entity_text = ", ".join(entities) if entities else "default tracked companies"
    topic_text = topic or "all"
    is_cjk = _contains_cjk(user_message)
    section_hint = (
        "- 格局结论\n"
        "- 关键变量与拐点\n"
        "- 公司角色（基于样本）\n"
        "- 供需-生态位分析\n"
        "- 工程优化 vs 范式转移\n"
        "- 18个月前瞻（条件式）\n"
        "- 事实 / 推断 / 情景（分别列出）"
        if is_cjk
        else "- Landscape Conclusions\n"
        "- Key Variables and Turning Points\n"
        "- Company Roles (sample-based)\n"
        "- Supply-Demand-Ecosystem Analysis\n"
        "- Engineering Optimization vs Paradigm Shift\n"
        "- Next 18 Months (conditional)\n"
        "- Facts / Inference / Scenarios (separated)"
    )
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive DB landscape output.\n"
        "Rules:\n"
        "1) Use ONLY facts/URLs/numbers from the provided DB output.\n"
        "2) Do NOT invent company events, product launches, investments, or URLs.\n"
        "3) Explain each company role using explicit evidence (count/share/sentiment/heat/momentum/source mix).\n"
        "4) Identify balance-changing variables and map evidence to compute/cost, algorithm/efficiency, data/moat.\n"
        "5) Evaluate supply barriers, demand quality, and ecosystem layer (protocol/platform/application) when supported.\n"
        "6) Use proxy metrics when direct finance is unavailable (pricing trend, API demand, hiring density, CAPEX intensity).\n"
        "7) Do not ignore accumulation effects: repeated small deltas can signal a coming turning point.\n"
        "8) Distinguish engineering optimization vs paradigm shift with explicit justification.\n"
        "9) Add conditional 6-18 month implications grounded in evidence.\n"
        "10) Separate facts vs inference vs scenarios; do not mix them.\n"
        "11) If evidence is sparse, say evidence is insufficient and avoid strong claims.\n"
        "12) Reply in the user's language.\n"
        "13) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Landscape target: topic={topic_text}, entities={entity_text}, window={days} days\n\n"
        "DB landscape output (ground truth):\n"
        f"{landscape_output}\n\n"
        "Now produce an analysis answer based only on the DB output above.\n"
        "Use this section order:\n"
        f"{section_hint}"
    )
    result = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    text = _coerce_to_text(getattr(result, "content", result))
    return text.strip()


def _ensure_landscape_evidence(answer: str, landscape_output: str, user_message: str) -> str:
    """Ensure landscape answer includes explicit evidence URL section."""
    return _ensure_grounded_evidence_or_fallback(
        answer=answer,
        source_output=landscape_output,
        user_message=user_message,
        max_urls=10,
        route_label="landscape",
    )


def _is_landscape_no_data(raw_landscape: str) -> bool:
    low_landscape = (raw_landscape or "").lower()
    return (
        raw_landscape.startswith("analyze_landscape failed:")
        or raw_landscape.startswith("No landscape data")
        or ("no tracked entities matched" in low_landscape)
    )


def _extract_landscape_coverage(landscape_output: str) -> dict[str, int]:
    url_count = len(_extract_urls(landscape_output))
    matched = 0
    active = 0

    m = re.search(r"matched_entity_articles=(\d+)", landscape_output)
    if m:
        matched = int(m.group(1))

    m = re.search(r"active_entities=(\d+)\s*/\s*(\d+)", landscape_output)
    if m:
        active = int(m.group(1))

    return {"url_count": url_count, "matched": matched, "active": active}


def _landscape_thresholds() -> dict[str, int]:
    def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 999) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except Exception:
            value = default
        return max(minimum, min(maximum, value))

    return {
        "min_urls": _env_int("LANDSCAPE_MIN_URLS", 4),
        "min_matched": _env_int("LANDSCAPE_MIN_MATCHED_ARTICLES", 6),
        "min_active": _env_int("LANDSCAPE_MIN_ACTIVE_ENTITIES", 2),
    }


def _is_landscape_evidence_sufficient(landscape_output: str) -> bool:
    cov = _extract_landscape_coverage(landscape_output)
    th = _landscape_thresholds()
    return (
        cov["url_count"] >= th["min_urls"]
        and cov["matched"] >= th["min_matched"]
        and cov["active"] >= th["min_active"]
    )


def _format_low_evidence_landscape_response(
    user_message: str,
    topic: str,
    days: int,
    entities: list[str],
    raw_landscape: str,
) -> str:
    cov = _extract_landscape_coverage(raw_landscape)
    th = _landscape_thresholds()
    topic_text = topic or "all"
    entity_text = ", ".join(entities) if entities else "default entities"

    if _contains_cjk(user_message):
        return (
            f"检测到这是格局类问题（topic={topic_text}, window={days}天, entities={entity_text}），"
            "但当前证据不足，已触发保守降级。\n"
            f"- URL 数：{cov['url_count']} / 阈值 {th['min_urls']}\n"
            f"- 命中文章数：{cov['matched']} / 阈值 {th['min_matched']}\n"
            f"- 活跃实体数：{cov['active']} / 阈值 {th['min_active']}\n"
            "为避免模型补全，我仅返回数据库事实快照，不输出强结论或角色推断。\n\n"
            f"{raw_landscape}\n\n"
            "建议：扩大时间窗口、补充实体列表，或放宽主题关键词后重试。\n"
            "置信度：低"
        )

    return (
        f"Landscape question detected (topic={topic_text}, window={days}d, entities={entity_text}), "
        "but evidence is insufficient, so conservative fallback is applied.\n"
        f"- URLs: {cov['url_count']} / threshold {th['min_urls']}\n"
        f"- Matched articles: {cov['matched']} / threshold {th['min_matched']}\n"
        f"- Active entities: {cov['active']} / threshold {th['min_active']}\n"
        "To avoid model fabrication, returning factual DB snapshot only (no strong role inference).\n\n"
        f"{raw_landscape}\n\n"
        "Try a wider time window, explicit entities, or a broader topic query.\n"
        "Confidence: Low"
    )


_TIMELINE_EVENT_LINE_RE = re.compile(
    r"^\s*(?P<rank>\d+)\.\s*"
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*"
    r"(?P<source>[^|]+?)\s*\|\s*(?P<sentiment>[^|]+?)\s*\|\s*"
    r"points=(?P<points>-?\d+(?:\.\d+)?)\s*$"
)


def _timeline_analysis_mode() -> str:
    raw = os.getenv("TIMELINE_ANALYSIS_MODE", "deterministic").strip().lower()
    if raw in {"model", "llm"}:
        return "model"
    return "deterministic"


def _safe_parse_datetime(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M")
    except Exception:
        return None


def _format_points(points: float) -> str:
    rounded = round(points)
    if abs(points - rounded) < 1e-9:
        return str(int(rounded))
    return f"{points:.1f}"


def _parse_timeline_records(timeline_output: str) -> list[dict[str, Any]]:
    lines = (timeline_output or "").splitlines()
    records: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _TIMELINE_EVENT_LINE_RE.match(line)
        if not m:
            i += 1
            continue

        rank = int(m.group("rank"))
        ts = m.group("ts").strip()
        source = m.group("source").strip()
        sentiment = m.group("sentiment").strip()
        try:
            points = float(m.group("points"))
        except Exception:
            points = 0.0

        block: list[str] = []
        j = i + 1
        while j < len(lines):
            if _TIMELINE_EVENT_LINE_RE.match(lines[j]):
                break
            block.append(lines[j].strip())
            j += 1

        headline = ""
        url = ""
        for entry in block:
            if not entry:
                continue
            if not headline and not entry.lower().startswith("http"):
                headline = entry
            if not url:
                u = re.search(r"https?://[^\s)\]]+", entry)
                if u:
                    url = u.group(0)

        records.append(
            {
                "rank": rank,
                "timestamp": ts,
                "source": source,
                "sentiment": sentiment,
                "points": points,
                "headline": headline,
                "url": url,
                "dt": _safe_parse_datetime(ts),
            }
        )
        i = j

    return records


def _is_timeline_analysis_grounded(answer: str, timeline_output: str) -> bool:
    answer_urls = set(_extract_urls(answer))
    source_urls = set(_extract_urls(timeline_output))
    if answer_urls and source_urls and any(url not in source_urls for url in answer_urls):
        return False

    source_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", timeline_output or ""))
    answer_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", answer or ""))
    if source_dates and answer_dates and any(d not in source_dates for d in answer_dates):
        return False
    return True


def _format_grounded_timeline_response(
    user_message: str,
    topic: str,
    days: int,
    timeline_output: str,
) -> str:
    records = _parse_timeline_records(timeline_output)
    if not records:
        return timeline_output

    cjk = _contains_cjk(user_message)
    total = len(records)
    peak = max(records, key=lambda x: float(x.get("points", 0.0)))
    peak_points = _format_points(float(peak.get("points", 0.0)))

    source_counts: dict[str, int] = {}
    sentiment_counts: dict[str, int] = {}
    parsed_records = [r for r in records if isinstance(r.get("dt"), datetime)]
    for rec in records:
        source = str(rec.get("source", "")).strip() or "Unknown"
        sentiment = str(rec.get("sentiment", "")).strip() or "Unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    top_source = max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else ""
    top_source_count = source_counts.get(top_source, 0) if top_source else 0
    top_source_share = (float(top_source_count) / float(total) * 100.0) if total else 0.0

    recent_line = ""
    if parsed_records:
        latest_dt = max(rec["dt"] for rec in parsed_records)
        recent_days = min(7, max(1, days // 4))
        cutoff = latest_dt - timedelta(days=recent_days)
        recent_count = sum(1 for rec in parsed_records if rec["dt"] >= cutoff)
        previous_count = max(0, len(parsed_records) - recent_count)
        if previous_count == 0:
            delta_text = "+new" if recent_count > 0 else "0.0%"
        else:
            delta = ((float(recent_count) - float(previous_count)) / float(previous_count)) * 100.0
            delta_text = f"{delta:+.1f}%"
        if cjk:
            recent_line = f"- 节奏变化：最近{recent_days}天 {recent_count} 条，之前 {previous_count} 条（{delta_text}）。"
        else:
            recent_line = (
                f"- Pace shift: recent {recent_days}d={recent_count}, "
                f"previous={previous_count} ({delta_text})."
            )

    pos = sentiment_counts.get("Positive", 0)
    neu = sentiment_counts.get("Neutral", 0)
    neg = sentiment_counts.get("Negative", 0)

    lines: list[str] = []
    if cjk:
        lines.append("以下内容仅基于数据库时间线记录生成，未使用外部知识补全。")
        lines.append("")
        lines.append(f"## 事件时间线（{topic}，最近 {days} 天）")
    else:
        lines.append("This answer is generated only from DB timeline records (no external knowledge fill-in).")
        lines.append("")
        lines.append(f"## Timeline Events ({topic}, last {days} days)")

    for rec in records:
        p = _format_points(float(rec.get("points", 0.0)))
        lines.append(
            f"{rec.get('rank')}. {rec.get('timestamp')} | {rec.get('source')} | "
            f"{rec.get('sentiment')} | points={p}"
        )
        headline = str(rec.get("headline", "")).strip()
        url = str(rec.get("url", "")).strip()
        if headline:
            lines.append(f"   {headline}")
        if url:
            lines.append(f"   {url}")

    lines.append("")
    lines.append("## 拐点" if cjk else "## Inflection Points")
    if cjk:
        lines.append(
            f"- 热度峰值：{peak.get('timestamp')} | {peak.get('source')} | "
            f"points={peak_points} | {peak.get('headline')}"
        )
        lines.append(
            f"- 来源集中度：{top_source} 占比 {top_source_share:.1f}% "
            f"（{top_source_count}/{total}）。"
        )
        lines.append(f"- 情绪结构：Positive/Neutral/Negative = {pos}/{neu}/{neg}。")
        if recent_line:
            lines.append(recent_line)
    else:
        lines.append(
            f"- Peak heat: {peak.get('timestamp')} | {peak.get('source')} | "
            f"points={peak_points} | {peak.get('headline')}"
        )
        lines.append(
            f"- Source concentration: {top_source} share {top_source_share:.1f}% "
            f"({top_source_count}/{total})."
        )
        lines.append(f"- Sentiment mix: Positive/Neutral/Negative = {pos}/{neu}/{neg}.")
        if recent_line:
            lines.append(recent_line)

    lines.append("")
    lines.append("## 后续关注" if cjk else "## What To Watch")
    if cjk:
        lines.append(f"- 是否出现新的高热事件（当前峰值 points={peak_points}）。")
        lines.append("- 来源是否继续向单一渠道集中。")
        lines.append("- 情绪结构是否从中性转向负面或正面。")
    else:
        lines.append(f"- Whether a new high-heat event appears (current peak points={peak_points}).")
        lines.append("- Whether source mix keeps concentrating into a single channel.")
        lines.append("- Whether sentiment mix shifts away from neutral.")

    return "\n".join(lines).strip()


def _analyze_timeline_output(
    user_message: str,
    topic: str,
    days: int,
    timeline_output: str,
) -> str:
    """Turn DB timeline output into analyst-friendly timeline answer."""
    deterministic = _format_grounded_timeline_response(
        user_message=user_message,
        topic=topic,
        days=days,
        timeline_output=timeline_output,
    )
    mode = _timeline_analysis_mode()
    if mode != "model":
        return deterministic

    model = _get_analysis_model()
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive DB timeline output.\n"
        "Rules:\n"
        "1) Use ONLY facts/URLs/dates from the provided DB output.\n"
        "2) Do NOT invent events or URLs.\n"
        "3) If data is sparse, explicitly say data is limited.\n"
        "4) Reply in the user's language.\n"
        "5) Structure output with:\n"
        "   - 事件时间线\n"
        "   - 拐点\n"
        "   - 后续关注\n"
        "6) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Timeline target: {topic}, window={days} days\n\n"
        "DB timeline output (ground truth):\n"
        f"{timeline_output}\n\n"
        "Now produce an analysis answer based only on the DB output above."
    )
    result = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    text = _coerce_to_text(getattr(result, "content", result))
    cleaned = text.strip()
    if not cleaned:
        return deterministic
    if not _is_timeline_analysis_grounded(cleaned, timeline_output):
        print("[Warn] Timeline model synthesis contains out-of-DB facts; fallback to grounded timeline response.")
        return deterministic
    return cleaned


def _format_low_sample_timeline_response(
    user_message: str,
    topic: str,
    days: int,
    event_count: int,
    min_events: int,
    raw_timeline: str,
) -> str:
    if _contains_cjk(user_message):
        return (
            f"检测到 `{topic}` 在最近 {days} 天仅有 {event_count} 条时间线记录（阈值 {min_events}）。\n"
            "为避免过度解读，本次仅返回事实时间线，不输出高强度推断。\n\n"
            f"{raw_timeline}\n\n"
            "数据提示：样本偏少，建议扩大时间窗口或放宽关键词后重试。"
        )
    return (
        f"Only {event_count} timeline records found for '{topic}' in the last {days} days "
        f"(threshold: {min_events}).\n"
        "To avoid over-interpretation, returning factual timeline only.\n\n"
        f"{raw_timeline}\n\n"
        "Data note: sample is small; try a wider time window or broader query."
    )


def _generate_langgraph(history: list[dict], user_message: str) -> str:
    agent = _get_langgraph_agent()
    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=user_message))

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": int(os.getenv("AGENT_MAX_ITERATIONS", "16"))},
    )
    text = _extract_final_text(result)
    if not text:
        raise RuntimeError("LangGraph returned empty response.")
    return text


def _forced_route_min_confidence() -> float:
    try:
        raw = float(os.getenv("AGENT_ROUTE_MIN_CONFIDENCE", "0.72"))
    except Exception:
        raw = 0.72
    return max(0.0, min(1.0, raw))


def _build_safe_query_params(
    *,
    user_message: str,
    fallback_query: str = "",
    fallback_days: int = 21,
) -> dict[str, Any]:
    query_req = _extract_query_request(user_message)
    if query_req:
        query, source, days, sort, limit = query_req
        return {
            "query": query,
            "source": source,
            "days": days,
            "sort": sort,
            "limit": limit,
        }

    query = str(fallback_query or "").strip()
    if not query:
        m = re.search(r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,24})", user_message or "")
        if m:
            query = str(m.group(0)).strip()
    if not query:
        query = "AI"

    return {
        "query": query,
        "source": "all",
        "days": max(1, min(365, int(fallback_days))),
        "sort": "time_desc",
        "limit": 8,
    }


def _run_safe_query_fallback(*, strict_mode: bool, user_message: str, fallback_query: str, fallback_days: int) -> str:
    _metrics_inc("route_low_confidence_fallback")
    params = _build_safe_query_params(
        user_message=user_message,
        fallback_query=fallback_query,
        fallback_days=fallback_days,
    )
    return run_query_pipeline(
        strict_mode=strict_mode,
        user_message=user_message,
        query=str(params["query"]),
        source=str(params["source"]),
        days=int(params["days"]),
        sort=str(params["sort"]),
        limit=int(params["limit"]),
        query_news_fn=query_news,
        analyze_fn=_analyze_query_output,
        ensure_evidence_fn=_ensure_query_evidence,
        emit_metrics_fn=_emit_route_metrics,
    )


def _extract_timeline_route_request(user_message: str) -> tuple[str, int, int, float] | None:
    base_req = _extract_timeline_request(user_message)
    if base_req is None:
        return None

    topic, days, limit = base_req
    conf_req = _extract_timeline_request_with_confidence(user_message)
    if conf_req is None:
        return topic, days, limit, 0.9

    c_topic, c_days, c_limit, confidence = conf_req
    if (
        str(topic).strip().lower() == str(c_topic).strip().lower()
        and int(days) == int(c_days)
        and int(limit) == int(c_limit)
    ):
        return topic, days, limit, float(confidence)
    return topic, days, limit, 0.9


def _detect_forced_route(user_message: str) -> tuple[str, dict[str, Any], float] | None:
    source_compare_req = _extract_source_compare_request(user_message)
    if source_compare_req:
        topic, days = source_compare_req
        return "source_compare", {"topic": topic, "days": days}, 0.95

    compare_req = _extract_compare_request(user_message)
    if compare_req:
        topic_a, topic_b, days = compare_req
        return "compare", {"topic_a": topic_a, "topic_b": topic_b, "days": days}, 0.95

    timeline_req = _extract_timeline_route_request(user_message)
    if timeline_req:
        topic, days, limit, confidence = timeline_req
        return "timeline", {"topic": topic, "days": days, "limit": limit}, float(confidence)

    landscape_req = _extract_landscape_request(user_message)
    if landscape_req:
        topic, days, entities = landscape_req
        return "landscape", {"topic": topic, "days": days, "entities": entities}, 0.92

    trend_req = _extract_trend_request(user_message)
    if trend_req:
        topic, window = trend_req
        return "trend", {"topic": topic, "window": window}, 0.92

    fulltext_req = _extract_fulltext_request(user_message)
    if fulltext_req:
        request_query, max_chars = fulltext_req
        return "fulltext", {"request_query": request_query, "max_chars": max_chars}, 0.95

    query_req = _extract_query_request(user_message)
    if query_req:
        query, source, days, sort, limit = query_req
        return (
            "query",
            {
                "query": query,
                "source": source,
                "days": days,
                "sort": sort,
                "limit": limit,
            },
            0.9,
        )
    return None


def _run_forced_route(*, intent: str, params: dict[str, Any], strict_mode: bool, user_message: str) -> str:
    if intent == "source_compare":
        _metrics_inc("source_compare_forced")
        return run_source_compare_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=str(params["topic"]),
            days=int(params["days"]),
            compare_sources_fn=compare_sources,
            analyze_fn=_analyze_source_compare_output,
            ensure_evidence_fn=_ensure_source_compare_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "compare":
        _metrics_inc("compare_forced")
        return run_compare_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic_a=str(params["topic_a"]),
            topic_b=str(params["topic_b"]),
            days=int(params["days"]),
            compare_topics_fn=compare_topics,
            analyze_fn=_analyze_compare_output,
            ensure_evidence_fn=_ensure_compare_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "timeline":
        _metrics_inc("timeline_forced")
        return run_timeline_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=str(params["topic"]),
            days=int(params["days"]),
            limit=int(params["limit"]),
            build_timeline_fn=build_timeline,
            count_timeline_items_fn=_count_timeline_items,
            format_low_sample_fn=_format_low_sample_timeline_response,
            analyze_fn=_analyze_timeline_output,
            ensure_evidence_fn=_ensure_timeline_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "landscape":
        _metrics_inc("landscape_forced")
        entities = params.get("entities")
        if not isinstance(entities, list):
            entities = []
        return run_landscape_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=str(params["topic"]),
            days=int(params["days"]),
            entities=[str(x) for x in entities if str(x).strip()],
            analyze_landscape_fn=analyze_landscape,
            is_no_data_fn=_is_landscape_no_data,
            format_no_data_fn=_format_landscape_no_data_response,
            is_evidence_sufficient_fn=_is_landscape_evidence_sufficient,
            metrics_inc_fn=_metrics_inc,
            format_low_evidence_fn=_format_low_evidence_landscape_response,
            analyze_output_fn=_analyze_landscape_output,
            is_unstable_synthesis_fn=_is_unstable_landscape_synthesis,
            ensure_evidence_fn=_ensure_landscape_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "trend":
        _metrics_inc("trend_forced")
        return run_trend_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=str(params["topic"]),
            window=int(params["window"]),
            trend_analysis_fn=trend_analysis,
            analyze_fn=_analyze_trend_output,
            ensure_evidence_fn=_ensure_trend_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "fulltext":
        _metrics_inc("fulltext_forced")
        return run_fulltext_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            request_query=str(params["request_query"]),
            max_chars=int(params["max_chars"]),
            fulltext_batch_fn=fulltext_batch,
            analyze_fn=_analyze_fulltext_output,
            ensure_evidence_fn=_ensure_fulltext_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    if intent == "query":
        _metrics_inc("query_forced")
        return run_query_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            query=str(params["query"]),
            source=str(params["source"]),
            days=int(params["days"]),
            sort=str(params["sort"]),
            limit=int(params["limit"]),
            query_news_fn=query_news,
            analyze_fn=_analyze_query_output,
            ensure_evidence_fn=_ensure_query_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    raise ValueError(f"Unknown forced route intent: {intent}")


def _generate_with_graph_dispatch(
    *,
    history: list[dict],
    user_message: str,
    runtime: str,
    strict_mode: bool,
) -> str:
    class _DispatchState(TypedDict, total=False):
        history: list[dict]
        user_message: str
        runtime: str
        strict_mode: bool
        intent: str
        confidence: float
        params: dict[str, Any]
        response: str

    forced_intents = {"source_compare", "compare", "timeline", "landscape", "trend", "fulltext", "query"}

    def _node_detect(state):
        forced = _detect_forced_route(str(state.get("user_message", "")))
        if forced:
            intent, params, confidence = forced
            if intent == "timeline" and confidence < _forced_route_min_confidence():
                _metrics_inc("route_low_confidence_fallback")
                safe = _build_safe_query_params(
                    user_message=str(state.get("user_message", "")),
                    fallback_query=str(params.get("topic", "")),
                    fallback_days=int(params.get("days", 21)),
                )
                return {"intent": "query", "params": safe, "confidence": confidence}
            return {"intent": intent, "params": params, "confidence": confidence}
        return {"intent": "planner"}

    def _after_detect(state):
        intent = str(state.get("intent", "planner"))
        return "forced" if intent in forced_intents else "planner"

    def _node_forced(state):
        intent = str(state.get("intent", ""))
        params = state.get("params")
        if not isinstance(params, dict):
            params = {}
        text = _run_forced_route(
            intent=intent,
            params=params,
            strict_mode=bool(state.get("strict_mode", False)),
            user_message=str(state.get("user_message", "")),
        )
        return {"response": text}

    def _node_planner(state):
        planned = _dispatch_planner_route(
            runtime=str(state.get("runtime", "langchain")),
            strict_mode=bool(state.get("strict_mode", False)),
            history=state.get("history") if isinstance(state.get("history"), list) else [],
            user_message=str(state.get("user_message", "")),
        )
        if planned is not None:
            return {"response": planned, "intent": "done"}
        return {"intent": "runtime"}

    def _after_planner(state):
        if str(state.get("intent", "")) == "done" and str(state.get("response", "")).strip():
            return "end"
        return "runtime"

    def _node_runtime(state):
        text = run_runtime_pipeline(
            strict_mode=bool(state.get("strict_mode", False)),
            runtime=str(state.get("runtime", "langchain")),
            history=state.get("history") if isinstance(state.get("history"), list) else [],
            user_message=str(state.get("user_message", "")),
            metrics_inc_fn=_metrics_inc,
            emit_metrics_fn=_emit_route_metrics,
            generate_legacy_fn=_generate_legacy,
            generate_langgraph_fn=_generate_langgraph,
        )
        return {"response": text}

    graph = StateGraph(_DispatchState)
    graph.add_node("detect", _node_detect)
    graph.add_node("forced", _node_forced)
    graph.add_node("planner", _node_planner)
    graph.add_node("runtime", _node_runtime)
    graph.add_edge(START, "detect")
    graph.add_conditional_edges("detect", _after_detect, {"forced": "forced", "planner": "planner"})
    graph.add_conditional_edges("planner", _after_planner, {"runtime": "runtime", "end": END})
    graph.add_edge("forced", END)
    graph.add_edge("runtime", END)
    compiled = graph.compile()
    result = compiled.invoke(
        {
            "history": history,
            "user_message": user_message,
            "runtime": runtime,
            "strict_mode": strict_mode,
        }
    )
    text = str(result.get("response", "")).strip()
    if not text:
        raise RuntimeError("Graph dispatch produced empty response.")
    return text


def generate_response(history: list[dict], user_message: str) -> str:
    """Stateless generation entrypoint used by API/Bot."""
    runtime = os.getenv("AGENT_RUNTIME", "langchain").strip().lower()
    strict_mode = os.getenv("AGENT_RUNTIME_STRICT", "false").strip().lower() == "true"
    dispatch_mode = os.getenv("AGENT_DISPATCH_MODE", "linear").strip().lower()
    _metrics_inc("requests_total")

    if dispatch_mode == "graph":
        return _generate_with_graph_dispatch(
            history=history,
            user_message=user_message,
            runtime=runtime,
            strict_mode=strict_mode,
        )

    # Deterministic guardrail: source-comparison intent must use compare_sources tool first.
    source_compare_req = _extract_source_compare_request(user_message)
    if source_compare_req:
        _metrics_inc("source_compare_forced")
        topic, days = source_compare_req
        return run_source_compare_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            days=days,
            compare_sources_fn=compare_sources,
            analyze_fn=_analyze_source_compare_output,
            ensure_evidence_fn=_ensure_source_compare_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    # Deterministic guardrail: for explicit A-vs-B comparisons, force DB-backed compare tool.
    compare_req = _extract_compare_request(user_message)
    if compare_req:
        _metrics_inc("compare_forced")
        topic_a, topic_b, days = compare_req
        return run_compare_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic_a=topic_a,
            topic_b=topic_b,
            days=days,
            compare_topics_fn=compare_topics,
            analyze_fn=_analyze_compare_output,
            ensure_evidence_fn=_ensure_compare_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    # Deterministic guardrail: timeline-style queries must use DB timeline tool.
    timeline_req = _extract_timeline_route_request(user_message)
    if timeline_req:
        topic, days, limit, confidence = timeline_req
        if confidence < _forced_route_min_confidence():
            return _run_safe_query_fallback(
                strict_mode=strict_mode,
                user_message=user_message,
                fallback_query=str(topic),
                fallback_days=int(days),
            )
        _metrics_inc("timeline_forced")
        return run_timeline_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            days=days,
            limit=limit,
            build_timeline_fn=build_timeline,
            count_timeline_items_fn=_count_timeline_items,
            format_low_sample_fn=_format_low_sample_timeline_response,
            analyze_fn=_analyze_timeline_output,
            ensure_evidence_fn=_ensure_timeline_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    # Deterministic guardrail: cross-domain landscape/role questions must use DB landscape tool.
    landscape_req = _extract_landscape_request(user_message)
    if landscape_req:
        _metrics_inc("landscape_forced")
        topic, days, entities = landscape_req
        return run_landscape_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            days=days,
            entities=entities,
            analyze_landscape_fn=analyze_landscape,
            is_no_data_fn=_is_landscape_no_data,
            format_no_data_fn=_format_landscape_no_data_response,
            is_evidence_sufficient_fn=_is_landscape_evidence_sufficient,
            metrics_inc_fn=_metrics_inc,
            format_low_evidence_fn=_format_low_evidence_landscape_response,
            analyze_output_fn=_analyze_landscape_output,
            is_unstable_synthesis_fn=_is_unstable_landscape_synthesis,
            ensure_evidence_fn=_ensure_landscape_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    # Deterministic guardrail: trend questions should use trend_analysis before synthesis.
    trend_req = _extract_trend_request(user_message)
    if trend_req:
        _metrics_inc("trend_forced")
        topic, window = trend_req
        return run_trend_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            topic=topic,
            window=window,
            trend_analysis_fn=trend_analysis,
            analyze_fn=_analyze_trend_output,
            ensure_evidence_fn=_ensure_trend_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    # Deterministic guardrail: fulltext/deep-read questions should use fulltext_batch first.
    fulltext_req = _extract_fulltext_request(user_message)
    if fulltext_req:
        _metrics_inc("fulltext_forced")
        request_query, max_chars = fulltext_req
        return run_fulltext_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            request_query=request_query,
            max_chars=max_chars,
            fulltext_batch_fn=fulltext_batch,
            analyze_fn=_analyze_fulltext_output,
            ensure_evidence_fn=_ensure_fulltext_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    # Deterministic guardrail: filter-style retrieval requests should use query_news first.
    query_req = _extract_query_request(user_message)
    if query_req:
        _metrics_inc("query_forced")
        query, source, days, sort, limit = query_req
        return run_query_pipeline(
            strict_mode=strict_mode,
            user_message=user_message,
            query=query,
            source=source,
            days=days,
            sort=sort,
            limit=limit,
            query_news_fn=query_news,
            analyze_fn=_analyze_query_output,
            ensure_evidence_fn=_ensure_query_evidence,
            emit_metrics_fn=_emit_route_metrics,
        )

    planned = _dispatch_planner_route(
        runtime=runtime,
        strict_mode=strict_mode,
        history=history,
        user_message=user_message,
    )
    if planned is not None:
        return planned

    return run_runtime_pipeline(
        strict_mode=strict_mode,
        runtime=runtime,
        history=history,
        user_message=user_message,
        metrics_inc_fn=_metrics_inc,
        emit_metrics_fn=_emit_route_metrics,
        generate_legacy_fn=_generate_legacy,
        generate_langgraph_fn=_generate_langgraph,
    )


_generate_response_core = generate_response


def generate_response(history: list[dict], user_message: str) -> str:
    """Public generation entrypoint with agent-side response post-processing."""
    core_text = _generate_response_core(history, user_message)
    core_text = _strip_generic_analysis_leadin(core_text)
    final_text, _ = _decorate_response_with_sources(core_text, user_message)
    return final_text


def generate_response_payload(history: list[dict], user_message: str) -> dict[str, Any]:
    """Structured response for transport layers (e.g., Telegram bot)."""
    core_text = _generate_response_core(history, user_message)
    core_text = _strip_generic_analysis_leadin(core_text)
    final_text, title_map = _decorate_response_with_sources(core_text, user_message)
    return {
        "text": final_text,
        "url_title_map": title_map,
    }


@dataclass
class _ChatResponse:
    text: str


class _SessionChat:
    """Compatibility chat session for local CLI."""

    def __init__(self):
        self.history: list[dict] = []

    def send_message(self, user_message: str) -> _ChatResponse:
        reply = generate_response(self.history, user_message)
        self.history.append({"role": "user", "parts": [{"text": user_message}]})
        self.history.append({"role": "model", "parts": [{"text": reply}]})
        return _ChatResponse(text=reply)


def create_agent_chat():
    """Create a stateful chat session wrapper for CLI usage."""
    return _SessionChat()





