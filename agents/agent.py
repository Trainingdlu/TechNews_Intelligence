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
        build_timeline,
        compare_topics,
        compare_sources,
        fulltext_batch,
        get_db_stats,
        list_topics,
        query_news,
        read_news_content,
        search_news,
        trend_analysis,
    )
except ImportError:  # package-style import fallback
    from .prompts import SYSTEM_INSTRUCTION
    from .tools import (
        build_timeline,
        compare_topics,
        compare_sources,
        fulltext_batch,
        get_db_stats,
        list_topics,
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
    legacy_direct = snapshot.get("legacy_direct", 0)

    should_log = force or (snapshot.get("requests_total", 0) % _metrics_log_every() == 0)
    if not should_log:
        return

    fallback_rate_total = fallback / total
    fallback_rate_attempt = (fallback / attempts) if attempts else 0.0
    langchain_success_rate = (success / attempts) if attempts else 0.0
    forced_route_rate = (compare_forced + timeline_forced) / total

    print(
        "[Metrics] "
        f"event={route_event} "
        f"total={snapshot.get('requests_total', 0)} "
        f"compare_forced={compare_forced} "
        f"timeline_forced={timeline_forced} "
        f"legacy_direct={legacy_direct} "
        f"langchain_attempts={attempts} "
        f"langchain_success={success} "
        f"langchain_fallback={fallback} "
        f"fallback_rate_total={fallback_rate_total:.1%} "
        f"fallback_rate_langchain={fallback_rate_attempt:.1%} "
        f"langchain_success_rate={langchain_success_rate:.1%} "
        f"forced_route_rate={forced_route_rate:.1%}"
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

    snapshot["fallback_rate_total"] = fallback / total
    snapshot["fallback_rate_langchain"] = (fallback / attempts) if attempts else 0.0
    snapshot["langchain_success_rate"] = (success / attempts) if attempts else 0.0
    snapshot["forced_route_rate"] = (compare_forced + timeline_forced) / total
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


def _count_timeline_items(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"^\s*\d+\.\s", text, flags=re.MULTILINE))


def _extract_days(text: str, default: int, maximum: int) -> int:
    m = re.search(r"(?:最近|过去|last)?\s*(\d{1,3})\s*(?:天|day|days)", text, flags=re.IGNORECASE)
    if m:
        val = int(m.group(1))
    else:
        val = default
    return max(1, min(maximum, val))


def _extract_limit(text: str, default: int, maximum: int) -> int:
    m = re.search(r"(?:最多|max|limit)\s*[:=]?\s*(\d{1,2})\s*(?:条|items?)?", text, flags=re.IGNORECASE)
    if m:
        val = int(m.group(1))
    else:
        val = default
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


def _extract_compare_request(user_message: str) -> tuple[str, str, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None
    lower = text.lower()
    if not any(k in lower for k in ["对比", "比较", "差异", " vs ", "vs", "versus", " and ", "和", "与"]):
        return None

    # Example: "对比一下 OpenAI 和 Anthropic 最近14天差异"
    topic_pattern = r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,20})"
    m = re.search(
        rf"(?P<a>{topic_pattern})\s*(?:和|与|vs|VS|Vs|versus|and)\s*(?P<b>{topic_pattern})",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    topic_a = m.group("a").strip()
    topic_b = m.group("b").strip()
    days = _extract_days(text, default=14, maximum=90)
    return topic_a, topic_b, days


def _extract_timeline_request(user_message: str) -> tuple[str, int, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None

    lower = text.lower()
    if not any(k in lower for k in ["timeline", "时间线", "里程碑", "演进"]):
        return None

    days = _extract_days(text, default=30, maximum=180)
    limit = _extract_limit(text, default=12, maximum=40)

    topic_pattern = r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,24})"
    patterns = [
        rf"(?:构建|生成|给我|做|列出|整理|build|make|create|show)\s+(?P<t>{topic_pattern})",
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
    system_prompt = (
        "You are a strict tech-intelligence analyst.\n"
        "You will receive DB comparison output.\n"
        "Rules:\n"
        "1) Use ONLY facts/URLs/numbers that already exist in the provided DB output.\n"
        "2) Do NOT invent any company event, model release, personnel change, or URL.\n"
        "3) If evidence is weak, explicitly say evidence is insufficient.\n"
        "4) Reply in the user's language.\n"
        "5) Structure output with:\n"
        "   - 对比结论\n"
        "   - 证据\n"
        "   - 对决策的影响\n"
        "6) Keep concise and data-grounded."
    )
    human_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Comparison target: {topic_a} vs {topic_b}, window={days} days\n\n"
        "DB comparison output (ground truth):\n"
        f"{compare_output}\n\n"
        "Now produce an analysis answer based only on the DB output above."
    )
    result = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    text = _coerce_to_text(getattr(result, "content", result))
    return text.strip()


def _ensure_compare_evidence(answer: str, compare_output: str, user_message: str) -> str:
    """Ensure compare answer always includes explicit evidence URL section."""
    if not answer:
        return answer
    source_urls = _extract_urls(compare_output)
    if not source_urls:
        return answer

    answer_urls = _extract_urls(answer)
    low = answer.lower()
    has_section = ("证据来源" in answer) or ("evidence source" in low) or ("evidence urls" in low)
    if has_section and answer_urls:
        return answer

    header = "## 证据来源" if _contains_cjk(user_message) else "## Evidence Sources"
    lines = [f"- {u}" for u in source_urls[:6]]
    return f"{answer.rstrip()}\n\n{header}\n" + "\n".join(lines)


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
            return f"已识别为对比请求，但数据库查询失败：{exc}"

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
                    _emit_route_metrics("timeline_forced")
                    return analyzed
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
            return f"已识别为时间线请求，但数据库查询失败：{exc}"

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
