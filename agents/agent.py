"""Agent runtime abstraction.

Primary runtime: LangChain/LangGraph standard agent loop.
Fallback runtime: native Gemini SDK (legacy) for compatibility.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
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
    if _contains_cjk(user_message):
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
    return _ensure_evidence_section(answer=answer, source_output=compare_output, user_message=user_message, max_urls=6)


def _ensure_timeline_evidence(answer: str, timeline_output: str, user_message: str) -> str:
    """Ensure timeline answer always includes explicit evidence URL section."""
    return _ensure_evidence_section(answer=answer, source_output=timeline_output, user_message=user_message, max_urls=8)


def _ensure_trend_evidence(answer: str, trend_output: str, user_message: str) -> str:
    """Ensure trend answer includes evidence URL section when URLs are available."""
    return _ensure_evidence_section(answer=answer, source_output=trend_output, user_message=user_message, max_urls=6)


def _ensure_source_compare_evidence(answer: str, source_output: str, user_message: str) -> str:
    """Ensure source-compare answer includes evidence URL section."""
    return _ensure_evidence_section(answer=answer, source_output=source_output, user_message=user_message, max_urls=8)


def _ensure_query_evidence(answer: str, query_output: str, user_message: str) -> str:
    """Ensure query answer always includes evidence URL section."""
    return _ensure_evidence_section(answer=answer, source_output=query_output, user_message=user_message, max_urls=10)


def _ensure_fulltext_evidence(answer: str, fulltext_output: str, user_message: str) -> str:
    """Ensure fulltext answer always includes evidence URL section."""
    return _ensure_evidence_section(answer=answer, source_output=fulltext_output, user_message=user_message, max_urls=10)


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
    return _ensure_evidence_section(answer=answer, source_output=landscape_output, user_message=user_message, max_urls=10)


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


def _analyze_timeline_output(
    user_message: str,
    topic: str,
    days: int,
    timeline_output: str,
) -> str:
    """Turn DB timeline output into analyst-friendly timeline answer."""
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
    return text.strip()


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


def generate_response(history: list[dict], user_message: str) -> str:
    """Stateless generation entrypoint used by API/Bot."""
    runtime = os.getenv("AGENT_RUNTIME", "langchain").strip().lower()
    strict_mode = os.getenv("AGENT_RUNTIME_STRICT", "false").strip().lower() == "true"
    _metrics_inc("requests_total")

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
    timeline_req = _extract_timeline_request(user_message)
    if timeline_req:
        _metrics_inc("timeline_forced")
        topic, days, limit = timeline_req
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
    final_text, _ = _decorate_response_with_sources(core_text, user_message)
    return final_text


def generate_response_payload(history: list[dict], user_message: str) -> dict[str, Any]:
    """Structured response for transport layers (e.g., Telegram bot)."""
    core_text = _generate_response_core(history, user_message)
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





