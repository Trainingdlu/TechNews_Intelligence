"""Agent runtime abstraction.

Primary runtime: LangChain/LangGraph standard agent loop.
Fallback runtime: native Gemini SDK (legacy) for compatibility.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

try:
    from prompts import SYSTEM_INSTRUCTION
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
_route_metrics_lock = Lock()
_route_metrics: dict[str, int] = {
    "requests_total": 0,
    "compare_forced": 0,
    "timeline_forced": 0,
    "landscape_forced": 0,
    "landscape_low_evidence": 0,
    "legacy_direct": 0,
    "langchain_attempts": 0,
    "langchain_success": 0,
    "langchain_fallback": 0,
}


def _metrics_enabled() -> bool:
    return os.getenv("AGENT_ROUTE_METRICS", "true").strip().lower() not in {"0", "false", "no", "off"}


def _metrics_log_every() -> int:
    try:
        return max(1, int(os.getenv("AGENT_ROUTE_LOG_EVERY", "20")))
    except Exception:
        return 20


def _metrics_inc(key: str, amount: int = 1) -> None:
    if not _metrics_enabled():
        return
    with _route_metrics_lock:
        _route_metrics[key] = _route_metrics.get(key, 0) + amount


def _emit_route_metrics(route_event: str, force: bool = False) -> None:
    if not _metrics_enabled():
        return

    with _route_metrics_lock:
        snapshot = dict(_route_metrics)

    total = max(1, snapshot.get("requests_total", 0))
    attempts = snapshot.get("langchain_attempts", 0)
    fallback = snapshot.get("langchain_fallback", 0)
    success = snapshot.get("langchain_success", 0)
    compare_forced = snapshot.get("compare_forced", 0)
    timeline_forced = snapshot.get("timeline_forced", 0)
    landscape_forced = snapshot.get("landscape_forced", 0)
    landscape_low_evidence = snapshot.get("landscape_low_evidence", 0)
    legacy_direct = snapshot.get("legacy_direct", 0)

    should_log = force or (snapshot.get("requests_total", 0) % _metrics_log_every() == 0)
    if not should_log:
        return

    fallback_rate_total = fallback / total
    fallback_rate_attempt = (fallback / attempts) if attempts else 0.0
    langchain_success_rate = (success / attempts) if attempts else 0.0
    forced_route_rate = (compare_forced + timeline_forced + landscape_forced) / total
    landscape_low_evidence_rate = (landscape_low_evidence / landscape_forced) if landscape_forced else 0.0

    print(
        "[Metrics] "
        f"event={route_event} "
        f"total={snapshot.get('requests_total', 0)} "
        f"compare_forced={compare_forced} "
        f"timeline_forced={timeline_forced} "
        f"landscape_forced={landscape_forced} "
        f"landscape_low_evidence={landscape_low_evidence} "
        f"legacy_direct={legacy_direct} "
        f"langchain_attempts={attempts} "
        f"langchain_success={success} "
        f"langchain_fallback={fallback} "
        f"fallback_rate_total={fallback_rate_total:.1%} "
        f"fallback_rate_langchain={fallback_rate_attempt:.1%} "
        f"langchain_success_rate={langchain_success_rate:.1%} "
        f"forced_route_rate={forced_route_rate:.1%} "
        f"landscape_low_evidence_rate={landscape_low_evidence_rate:.1%}"
    )


def reset_route_metrics() -> None:
    """Reset in-memory route metrics counters."""
    with _route_metrics_lock:
        for k in list(_route_metrics.keys()):
            _route_metrics[k] = 0


def get_route_metrics_snapshot() -> dict[str, float]:
    """Return a snapshot of route metrics plus derived rates."""
    with _route_metrics_lock:
        snapshot: dict[str, float] = dict(_route_metrics)

    total = max(1, int(snapshot.get("requests_total", 0)))
    attempts = int(snapshot.get("langchain_attempts", 0))
    fallback = int(snapshot.get("langchain_fallback", 0))
    success = int(snapshot.get("langchain_success", 0))
    compare_forced = int(snapshot.get("compare_forced", 0))
    timeline_forced = int(snapshot.get("timeline_forced", 0))
    landscape_forced = int(snapshot.get("landscape_forced", 0))
    landscape_low_evidence = int(snapshot.get("landscape_low_evidence", 0))

    snapshot["fallback_rate_total"] = fallback / total
    snapshot["fallback_rate_langchain"] = (fallback / attempts) if attempts else 0.0
    snapshot["langchain_success_rate"] = (success / attempts) if attempts else 0.0
    snapshot["forced_route_rate"] = (compare_forced + timeline_forced + landscape_forced) / total
    snapshot["landscape_low_evidence_rate"] = (
        (landscape_low_evidence / landscape_forced) if landscape_forced else 0.0
    )
    return snapshot


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)\]]+", text)
    dedup: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup


_SOURCE_HEADER_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:来源|证据来源|source(?:s)?|evidence\s+sources?)\s*:?\s*$",
    re.IGNORECASE,
)


def _strip_existing_source_section(text: str) -> str:
    lines = (text or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if _SOURCE_HEADER_RE.match(line.strip()):
            start = i
            break
    if start is None:
        return (text or "").rstrip()
    return "\n".join(lines[:start]).rstrip()


def _apply_inline_citations(text: str, ordered_urls: list[str]) -> str:
    out = text or ""
    for idx, url in enumerate(ordered_urls, 1):
        cite = f"[{idx}]"
        out = out.replace(f"`{url}`", cite)
        out = out.replace(url, cite)
    return out


def _max_source_urls() -> int:
    try:
        # Backward-compatible fallback to BOT_MAX_CITATION_URLS.
        raw = os.getenv("AGENT_MAX_SOURCE_URLS", os.getenv("BOT_MAX_CITATION_URLS", "12"))
        return max(1, min(30, int(raw)))
    except Exception:
        return 12


def _build_source_section(ordered_urls: list[str], user_message: str) -> str:
    header = "## 来源" if _contains_cjk(user_message) else "## Sources"
    lines = [header]
    for idx, url in enumerate(ordered_urls, 1):
        lines.append(f"- [{idx}] {url}")
    return "\n".join(lines)


def _decorate_response_with_sources(text: str, user_message: str) -> tuple[str, dict[str, str]]:
    """Normalize output into citation style + source section in agent layer."""
    raw = (text or "").strip()
    if not raw:
        return raw, {}

    body = _strip_existing_source_section(raw)
    urls = _extract_urls(body) if body else []
    if not urls:
        # Preserve evidence URLs when response originally contains only a source section.
        urls = _extract_urls(raw)
    if not urls:
        return raw, {}

    ordered_urls = urls[:_max_source_urls()]
    title_map: dict[str, str] = {}
    try:
        title_map = lookup_url_titles(ordered_urls)
    except Exception as exc:
        print(f"[Warn] lookup_url_titles in agent failed: {exc}")
        title_map = {}

    render_body = body if body else raw
    cited_body = _apply_inline_citations(render_body, ordered_urls)
    source_section = _build_source_section(ordered_urls, user_message)
    merged = f"{cited_body.rstrip()}\n\n{source_section}".strip()
    return merged, title_map


def _count_timeline_items(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"^\s*\d+\.\s", text, flags=re.MULTILINE))


def _extract_days(text: str, default: int, maximum: int) -> int:
    m = re.search(r"(?:最近|过去|last)?\s*(\d{1,3})\s*(?:天|day|days)", text, flags=re.IGNORECASE)
    val = int(m.group(1)) if m else default
    return max(1, min(maximum, val))


def _extract_limit(text: str, default: int, maximum: int) -> int:
    m = re.search(r"(?:最多|max|limit)\s*[:=]?\s*(\d{1,2})\s*(?:条|items?)?", text, flags=re.IGNORECASE)
    val = int(m.group(1)) if m else default
    return max(1, min(maximum, val))


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


_LANDSCAPE_ENTITY_ALIASES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "microsoft": "Microsoft",
    "meta": "Meta",
    "amazon": "Amazon",
    "aws": "Amazon",
    "nvidia": "NVIDIA",
    "apple": "Apple",
    "tesla": "Tesla",
    "tsmc": "TSMC",
    "intel": "Intel",
    "amd": "AMD",
    "crowdstrike": "CrowdStrike",
    "palo alto": "Palo Alto Networks",
    "palo alto networks": "Palo Alto Networks",
    "cloudflare": "Cloudflare",
    "cisco": "Cisco",
    "xai": "xAI",
    "谷歌": "Google",
    "微软": "Microsoft",
    "亚马逊": "Amazon",
    "英伟达": "NVIDIA",
    "苹果": "Apple",
    "特斯拉": "Tesla",
    "台积电": "TSMC",
    "英特尔": "Intel",
}


_COMPARE_STOP_TOKENS = {
    "对比",
    "比较",
    "差异",
    "区别",
    "vs",
    "versus",
    "和",
    "与",
    "and",
    "the",
    "a",
    "an",
    "一下",
    "请",
    "请问",
    "我想",
    "想知道",
}


_LANDSCAPE_STOP_TOPICS = {
    "global",
    "world",
    "today",
    "current",
    "landscape",
    "ecosystem",
    "当今",
    "当前",
    "全球",
    "世界",
    "现在",
    "目前",
    "tech",
    "technology",
    "科技",
    "技术",
    "科技行业",
    "技术行业",
    "科技领域",
    "技术领域",
}


def _normalize_landscape_topic_candidate(raw: str) -> str:
    candidate = re.sub(r"\s+", " ", (raw or "").strip()).strip("：:，,。. ")
    candidate = re.sub(r"^(?:当今|当前|目前|全球|世界|现在|如今)+", "", candidate).strip("的之 ")
    return candidate


def _normalize_compare_entity(raw: str) -> str:
    token = re.sub(r"\s+", " ", (raw or "").strip()).strip("：:，,。.!?！？()[]{}\"'`")
    token = re.sub(r"^(?:请|请问|对比一下|比较一下|对比|比较)+", "", token).strip()
    return token


def _is_valid_compare_entity(token: str) -> bool:
    if not token:
        return False
    low = token.lower()
    if low in _COMPARE_STOP_TOKENS:
        return False
    if re.fullmatch(r"\d{1,3}", token):
        return False
    if len(token) < 2:
        return False
    return True


def _extract_compare_pair(text: str) -> tuple[str, str] | None:
    topic_pattern = r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9 ._&/-]{1,39})"
    patterns = [
        rf"(?:对比|比较|差异|区别)\s*(?:一下|下)?\s*(?P<a>{topic_pattern})\s*(?:和|与|vs|VS|Vs|versus|and|&)\s*(?P<b>{topic_pattern})",
        rf"(?P<a>{topic_pattern})\s*(?:和|与|vs|VS|Vs|versus|and|&)\s*(?P<b>{topic_pattern})",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        a = _normalize_compare_entity(m.group("a"))
        b = _normalize_compare_entity(m.group("b"))
        if not (_is_valid_compare_entity(a) and _is_valid_compare_entity(b)):
            continue
        if a.lower() == b.lower():
            continue
        return a, b
    return None


def _extract_landscape_topic(text: str, lower: str) -> str:
    if bool(re.search(r"\bai\b", lower)) or ("人工智能" in text) or ("大模型" in text) or ("llm" in lower):
        return "AI"

    if any(k in lower for k in ["business", "market", "finance", "commercial"]) or any(
        k in text for k in ["商业", "金融", "市场", "财经"]
    ):
        return "business"

    if any(k in lower for k in ["security", "cybersecurity", "cyber"]) or any(
        k in text for k in ["安全", "网络安全", "攻防", "威胁"]
    ):
        return "security"

    m = re.search(r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9 _/-]{1,20})\s*(?:领域|行业|赛道)?\s*(?:格局|版图|生态)", text)
    if m:
        normalized = _normalize_landscape_topic_candidate(m.group(1))
        candidate = normalized.lower()
        if candidate and candidate not in _LANDSCAPE_STOP_TOPICS:
            return normalized

    m = re.search(r"(?:landscape|ecosystem)\s*(?:of|for)?\s*([A-Za-z][A-Za-z0-9 _/-]{2,24})", lower)
    if m:
        candidate = _normalize_landscape_topic_candidate(m.group(1)).lower()
        if candidate and candidate not in _LANDSCAPE_STOP_TOPICS:
            return candidate

    return ""


def _extract_landscape_request(user_message: str) -> tuple[str, int, list[str]] | None:
    text = (user_message or "").strip()
    if not text:
        return None

    lower = text.lower()
    strong_landscape_keywords = [
        "格局",
        "版图",
        "生态位",
        "阵营",
        "角色",
        "玩家",
        "谁主导",
        "landscape",
        "power structure",
    ]
    weak_landscape_keywords = ["生态", "ecosystem"]
    has_strong = any((k in text) or (k in lower) for k in strong_landscape_keywords)
    has_weak = any((k in text) or (k in lower) for k in weak_landscape_keywords)
    if not (has_strong or has_weak):
        return None
    if has_weak and not has_strong:
        # Avoid false routing for product-ecosystem discussions unless role/structure intent exists.
        weak_context = ["格局", "版图", "角色", "玩家", "主导", "竞争", "地位", "who leads", "dominant"]
        if not any((k in text) or (k in lower) for k in weak_context):
            return None
    if any((k in text) or (k in lower) for k in ["时间线", "timeline", "里程碑"]):
        return None

    days = _extract_days(text, default=30, maximum=180)
    topic = _extract_landscape_topic(text, lower)

    entities: list[str] = []
    seen: set[str] = set()
    for alias, canonical in _LANDSCAPE_ENTITY_ALIASES.items():
        if re.fullmatch(r"[A-Za-z0-9. ]+", alias):
            if not re.search(rf"\b{re.escape(alias)}\b", lower):
                continue
        else:
            if alias not in text:
                continue
        key = canonical.lower()
        if key not in seen:
            seen.add(key)
            entities.append(canonical)

    return topic, days, entities


def _extract_compare_request(user_message: str) -> tuple[str, str, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None
    lower = text.lower()

    pair = _extract_compare_pair(text)
    if pair is None:
        return None

    has_explicit_marker = any(k in lower for k in ["对比", "比较", "差异", "区别", "vs", "versus"])
    has_comparative_question = any(
        k in lower
        for k in [
            "谁更",
            "哪个更",
            "哪家更",
            "谁强",
            "高于",
            "低于",
            "more than",
            "less than",
            "better",
            "hotter",
            "stronger",
        ]
    )
    score = 1  # extracted pair
    if has_explicit_marker:
        score += 2
    if has_comparative_question:
        score += 2
    if re.search(r"\b(?:vs|versus)\b", lower):
        score += 1

    if score < 3:
        return None

    topic_a, topic_b = pair
    days = _extract_days(text, default=14, maximum=90)
    return topic_a, topic_b, days


def _extract_timeline_request(user_message: str) -> tuple[str, int, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None

    lower = text.lower()
    explicit_timeline_markers = ["timeline", "时间线", "里程碑", "大事记", "发展历程"]
    action_markers = [
        "动作",
        "动态",
        "动向",
        "进展",
        "更新",
        "事件",
        "发生了什么",
        "都做了什么",
        "moves",
        "actions",
        "updates",
        "developments",
    ]
    has_explicit_marker = any(k in lower for k in explicit_timeline_markers)
    has_recent_window = bool(re.search(r"(最近|过去|近|last|recent|past)\s*\d{0,3}\s*(天|day|days)?", lower))
    has_action_intent = any(k in lower for k in action_markers)

    if not (has_explicit_marker or (has_recent_window and has_action_intent)):
        return None

    days = _extract_days(text, default=30, maximum=180)
    limit = _extract_limit(text, default=12, maximum=40)

    topic_pattern = r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,24})"
    patterns = [
        rf"(?:构建|生成|给我|做|列出|整理|build|make|create|show)\s+(?P<t>{topic_pattern})",
        rf"(?:最近|过去|近|last|recent|past)\s*\d{{0,3}}\s*(?:天|day|days)?\s*(?P<t>{topic_pattern})\s*(?:的)?\s*(?:动作|动态|动向|进展|更新|事件|moves?|actions?|updates?|developments?)",
        rf"(?P<t>{topic_pattern})\s*(?:最近|过去|近|last|recent|past)\s*\d{{0,3}}\s*(?:天|day|days)?\s*(?:的)?\s*(?:动作|动态|动向|进展|更新|事件|moves?|actions?|updates?|developments?)",
        rf"(?:最近|过去|近|last|recent|past)\s*(?P<t>{topic_pattern})\s*(?:的)?\s*(?:动作|动态|动向|进展|更新|事件|moves?|actions?|updates?|developments?)",
        rf"(?P<t>{topic_pattern})\s*(?:过去|最近|last)?\s*\d{{0,3}}\s*(?:天|day|days)?\s*(?:时间线|timeline)",
        rf"(?:时间线|timeline)\s*(?:关于|for)?\s*(?P<t>{topic_pattern})",
    ]

    topic = ""
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            topic = m.group("t").strip()
            break

    if not topic:
        stop = {
            "构建",
            "生成",
            "给我",
            "做",
            "列出",
            "整理",
            "时间线",
            "里程碑",
            "演进",
            "timeline",
            "build",
            "create",
            "show",
            "recent",
            "last",
            "最近",
            "过去",
            "天",
            "day",
            "days",
        }
        candidates = re.findall(topic_pattern, text)
        for c in candidates:
            lc = c.lower()
            if lc in stop:
                continue
            if re.fullmatch(r"\d{1,3}", c):
                continue
            topic = c
            break

    if not topic:
        return None
    # Normalize possessive/structural suffix for Chinese/English patterns.
    topic = re.sub(r"(?:的|之)$", "", topic).strip()
    topic = re.sub(r"(?:'s)$", "", topic, flags=re.IGNORECASE).strip()
    return topic, days, limit


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


def _ensure_evidence_section(answer: str, source_output: str, user_message: str, max_urls: int = 8) -> str:
    """Ensure answer includes an evidence URL section based on tool output."""
    if not answer:
        return answer
    source_urls = _extract_urls(source_output)
    if not source_urls:
        return answer

    answer_urls = _extract_urls(answer)
    has_section = bool(
        re.search(
            r"^\s{0,3}(?:#{1,6}\s*)?(?:来源|证据来源|source(?:s)?|evidence\s+sources?)\s*:?\s*$",
            answer,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    if has_section and answer_urls:
        return answer

    header = "## 证据来源" if _contains_cjk(user_message) else "## Evidence Sources"
    lines = [f"- {u}" for u in source_urls[:max_urls]]
    return f"{answer.rstrip()}\n\n{header}\n" + "\n".join(lines)


def _ensure_compare_evidence(answer: str, compare_output: str, user_message: str) -> str:
    """Ensure compare answer always includes explicit evidence URL section."""
    return _ensure_evidence_section(answer=answer, source_output=compare_output, user_message=user_message, max_urls=6)


def _ensure_timeline_evidence(answer: str, timeline_output: str, user_message: str) -> str:
    """Ensure timeline answer always includes explicit evidence URL section."""
    return _ensure_evidence_section(answer=answer, source_output=timeline_output, user_message=user_message, max_urls=8)


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

    # Deterministic guardrail: for explicit A-vs-B comparisons, force DB-backed compare tool.
    compare_req = _extract_compare_request(user_message)
    if compare_req:
        _metrics_inc("compare_forced")
        topic_a, topic_b, days = compare_req
        try:
            raw_compare = compare_topics(topic_a=topic_a, topic_b=topic_b, days=days)
            if raw_compare.startswith("compare_topics failed:"):
                _emit_route_metrics("compare_forced")
                return raw_compare
            try:
                analyzed = _analyze_compare_output(
                    user_message=user_message,
                    topic_a=topic_a,
                    topic_b=topic_b,
                    days=days,
                    compare_output=raw_compare,
                )
                if analyzed:
                    result = _ensure_compare_evidence(analyzed, raw_compare, user_message)
                    _emit_route_metrics("compare_forced")
                    return result
            except Exception as exc:
                if strict_mode:
                    _emit_route_metrics("compare_forced_strict_error", force=True)
                    raise
                print(f"[Warn] Compare analysis synthesis failed; fallback to raw compare: {exc}")
            _emit_route_metrics("compare_forced")
            return raw_compare
        except Exception as exc:
            if strict_mode:
                _emit_route_metrics("compare_forced_strict_error", force=True)
                raise
            _emit_route_metrics("compare_forced_error", force=True)
            return f"Compare request detected, but DB query failed: {exc}"

    # Deterministic guardrail: timeline-style queries must use DB timeline tool.
    timeline_req = _extract_timeline_request(user_message)
    if timeline_req:
        _metrics_inc("timeline_forced")
        topic, days, limit = timeline_req
        try:
            raw_timeline = build_timeline(topic=topic, days=days, limit=limit)
            if raw_timeline.startswith("build_timeline failed:") or raw_timeline.startswith("No timeline data"):
                _emit_route_metrics("timeline_forced")
                return raw_timeline

            min_events = max(1, int(os.getenv("TIMELINE_MIN_EVENTS", "5")))
            event_count = _count_timeline_items(raw_timeline)
            if event_count < min_events:
                result = _format_low_sample_timeline_response(
                    user_message=user_message,
                    topic=topic,
                    days=days,
                    event_count=event_count,
                    min_events=min_events,
                    raw_timeline=raw_timeline,
                )
                _emit_route_metrics("timeline_forced")
                return result

            try:
                analyzed = _analyze_timeline_output(
                    user_message=user_message,
                    topic=topic,
                    days=days,
                    timeline_output=raw_timeline,
                )
                if analyzed:
                    result = _ensure_timeline_evidence(analyzed, raw_timeline, user_message)
                    _emit_route_metrics("timeline_forced")
                    return result
            except Exception as exc:
                if strict_mode:
                    _emit_route_metrics("timeline_forced_strict_error", force=True)
                    raise
                print(f"[Warn] Timeline analysis synthesis failed; fallback to raw timeline: {exc}")
            _emit_route_metrics("timeline_forced")
            return raw_timeline
        except Exception as exc:
            if strict_mode:
                _emit_route_metrics("timeline_forced_strict_error", force=True)
                raise
            _emit_route_metrics("timeline_forced_error", force=True)
            return f"Timeline request detected, but DB query failed: {exc}"

    # Deterministic guardrail: cross-domain landscape/role questions must use DB landscape tool.
    landscape_req = _extract_landscape_request(user_message)
    if landscape_req:
        _metrics_inc("landscape_forced")
        topic, days, entities = landscape_req
        entities_csv = ",".join(entities)
        try:
            raw_landscape = analyze_landscape(topic=topic, days=days, entities=entities_csv, limit_per_entity=3)
            if _is_landscape_no_data(raw_landscape) and topic:
                retry_landscape = analyze_landscape(topic="", days=days, entities=entities_csv, limit_per_entity=3)
                if not _is_landscape_no_data(retry_landscape):
                    raw_landscape = retry_landscape
            if _is_landscape_no_data(raw_landscape):
                _emit_route_metrics("landscape_forced")
                return raw_landscape

            if not _is_landscape_evidence_sufficient(raw_landscape):
                _metrics_inc("landscape_low_evidence")
                result = _format_low_evidence_landscape_response(
                    user_message=user_message,
                    topic=topic,
                    days=days,
                    entities=entities,
                    raw_landscape=raw_landscape,
                )
                _emit_route_metrics("landscape_forced_low_evidence", force=True)
                return result

            try:
                analyzed = _analyze_landscape_output(
                    user_message=user_message,
                    topic=topic,
                    days=days,
                    entities=entities,
                    landscape_output=raw_landscape,
                )
                if analyzed:
                    result = _ensure_landscape_evidence(analyzed, raw_landscape, user_message)
                    _emit_route_metrics("landscape_forced")
                    return result
            except Exception as exc:
                if strict_mode:
                    _emit_route_metrics("landscape_forced_strict_error", force=True)
                    raise
                print(f"[Warn] Landscape analysis synthesis failed; fallback to raw landscape: {exc}")
            _emit_route_metrics("landscape_forced")
            return raw_landscape
        except Exception as exc:
            if strict_mode:
                _emit_route_metrics("landscape_forced_strict_error", force=True)
                raise
            _emit_route_metrics("landscape_forced_error", force=True)
            return f"Landscape analysis request detected, but DB query failed: {exc}"

    if runtime == "legacy":
        _metrics_inc("legacy_direct")
        _emit_route_metrics("legacy_direct")
        return _generate_legacy(history, user_message)

    _metrics_inc("langchain_attempts")
    try:
        result = _generate_langgraph(history, user_message)
        _metrics_inc("langchain_success")
        _emit_route_metrics("langchain_success")
        return result
    except Exception as exc:
        if strict_mode:
            _emit_route_metrics("langchain_strict_error", force=True)
            raise
        _metrics_inc("langchain_fallback")
        _emit_route_metrics("langchain_fallback", force=True)
        print(f"[Warn] LangChain runtime failed, fallback to legacy runtime: {exc}")
        return _generate_legacy(history, user_message)


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





