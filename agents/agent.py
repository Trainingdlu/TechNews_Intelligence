"""Agent runtime — ReAct-native architecture.

Single runtime: LangGraph ReAct agent with full tool autonomy.
No hardcoded routing, no secondary LLM analysis, no legacy fallback.
"""

from __future__ import annotations

import inspect
import os
import re
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

try:
    from prompts import SYSTEM_INSTRUCTION
    from core.evidence import (
        decorate_response_with_sources as _decorate_response_with_sources_core,
    )
    from core.metrics import (
        get_route_metrics_snapshot,
        metrics_inc as _metrics_inc,
        reset_route_metrics,
    )
    from tools import (
        analyze_landscape,
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
        decorate_response_with_sources as _decorate_response_with_sources_core,
    )
    from .core.metrics import (
        get_route_metrics_snapshot,
        metrics_inc as _metrics_inc,
        reset_route_metrics,
    )
    from .tools import (
        analyze_landscape,
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
# LangChain tool wrappers
# ---------------------------------------------------------------------------
@tool("search_news")
def search_news_tool(query: str, days: int = 21) -> str:
    """Search related news with hybrid retrieval (semantic + keyword).

    Best for exploratory queries. Returns up to 5 results ranked by relevance.
    If no results, try broader query or larger days window.
    """
    return search_news(query=query, days=days)


@tool("read_news_content")
def read_news_content_tool(url: str) -> str:
    """Read full article content by URL. Only works for URLs from other tools."""
    return read_news_content(url=url)


@tool("get_db_stats")
def get_db_stats_tool() -> str:
    """Get database freshness stats and total article count."""
    return get_db_stats()


@tool("list_topics")
def list_topics_tool() -> str:
    """Get daily article volume distribution for recent 21 days."""
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
    """Query news with structured filters — the primary retrieval tool.

    Supports filtering by source ('all'/'HackerNews'/'TechCrunch'), time window,
    sentiment, and sorting by time or heat (points). If no results, try broader
    query or larger time window.
    """
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
    """Analyze topic momentum: compare article count and points in recent
    N days vs previous N days. Includes daily breakdown."""
    return trend_analysis(topic=topic, window=window)


@tool("compare_sources")
def compare_sources_tool(topic: str, days: int = 14) -> str:
    """Compare HackerNews vs TechCrunch coverage and sentiment for a topic."""
    return compare_sources(topic=topic, days=days)


@tool("compare_topics")
def compare_topics_tool(topic_a: str, topic_b: str, days: int = 14) -> str:
    """Compare two entities side-by-side (e.g., OpenAI vs Anthropic) with
    DB-backed metrics, momentum, source mix, and evidence URLs."""
    return compare_topics(topic_a=topic_a, topic_b=topic_b, days=days)


@tool("build_timeline")
def build_timeline_tool(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build chronological event timeline for a topic/company/product.
    Auto-retries with wider window if no data found initially."""
    return build_timeline(topic=topic, days=days, limit=limit)


@tool("analyze_landscape")
def analyze_landscape_tool(
    topic: str = "",
    days: int = 30,
    entities: str = "",
    limit_per_entity: int = 3,
) -> str:
    """Analyze competitive landscape with entity-level stats and evidence.

    Use for landscape/structure/role questions. Set topic='AI' for AI landscape.
    If confidence is Low, try wider time window or more entities.
    """
    return analyze_landscape(
        topic=topic, days=days, entities=entities, limit_per_entity=limit_per_entity
    )


@tool("fulltext_batch")
def fulltext_batch_tool(urls: str, max_chars_per_article: int = 4000) -> str:
    """Batch read full article text. Accepts URLs or keyword query for auto-selection."""
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
    fulltext_batch_tool,
]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------
_react_agent: Any | None = None


class AgentGenerationError(Exception):
    """Business-level generation failure with user-safe message."""

    def __init__(self, message: str, code: str = "generation_failed"):
        super().__init__(message)
        self.message = str(message)
        self.code = str(code)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _build_react_prompt_kwargs() -> tuple[dict[str, str], str]:
    """Build prompt injection kwargs for the installed LangGraph version.

    We never allow creating an agent without SYSTEM_INSTRUCTION.
    """
    candidate_keys = ("prompt", "state_modifier", "messages_modifier")

    try:
        signature = inspect.signature(create_react_agent)
        parameters = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    # Prefer explicit signature-matched parameter to avoid silent drift.
    for key in candidate_keys:
        if key in parameters:
            return {key: SYSTEM_INSTRUCTION}, key

    # Some versions hide compatibility through **kwargs.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {"prompt": SYSTEM_INSTRUCTION}, "prompt(**kwargs)"

    raise RuntimeError(
        "Unable to inject SYSTEM_INSTRUCTION into create_react_agent; "
        "supported parameter not found (expected one of: prompt/state_modifier/messages_modifier)."
    )


def _get_model_provider() -> str:
    """Normalize AGENT_MODEL_PROVIDER into supported provider keys."""
    provider = os.getenv("AGENT_MODEL_PROVIDER", "gemini_api").strip().lower()
    if provider in {"gemini", "gemini_api", "google_ai_studio", "developer_api"}:
        return "gemini_api"
    if provider in {"vertex", "vertex_ai", "gcp"}:
        return "vertex"
    raise ValueError(
        "AGENT_MODEL_PROVIDER must be one of: "
        "gemini_api (default), vertex."
    )


def _build_chat_model() -> Any:
    """Create the chat model client from environment configuration."""
    temperature = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    provider = _get_model_provider()

    if provider == "gemini_api":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            google_api_key=api_key,
            temperature=temperature,
        )

    project = os.getenv("VERTEX_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip()
    if not project:
        raise ValueError(
            "VERTEX_PROJECT is not set. "
            "You can also use GOOGLE_CLOUD_PROJECT."
        )

    location = os.getenv("VERTEX_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global")).strip()
    model_name = os.getenv(
        "VERTEX_MODEL",
        os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"),
    ).strip()

    # Vertex Express Mode with API key:
    # this path requires langchain-google-genai >= 4.x (unified google-genai SDK).
    vertex_api_key = os.getenv("VERTEX_API_KEY", "").strip()
    if vertex_api_key:
        try:
            raw_ver = version("langchain-google-genai")
        except PackageNotFoundError:
            raw_ver = "0.0.0"
        major = int(raw_ver.split(".", maxsplit=1)[0] or "0")
        if major < 4:
            raise RuntimeError(
                "VERTEX_API_KEY mode requires langchain-google-genai>=4.0.0 "
                f"(current: {raw_ver}). "
                "Either upgrade dependencies or use ADC/service-account auth."
            )

        # Keep Google GenAI environment variables aligned with Vertex backend.
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=vertex_api_key,
            vertexai=True,
            project=project,
            location=location,
        )

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if credentials_path and not os.path.exists(credentials_path):
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS points to a missing file: "
            f"{credentials_path}"
        )

    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError as exc:
        raise RuntimeError(
            "Vertex provider requires 'langchain-google-vertexai'. "
            "Install dependencies with: pip install -r agents/requirements.txt"
        ) from exc

    return ChatVertexAI(
        model=model_name,
        project=project,
        location=location,
        temperature=temperature,
    )


def _get_react_agent():
    """Lazily initialize the LangGraph ReAct agent."""
    global _react_agent
    if _react_agent is not None:
        return _react_agent

    model = _build_chat_model()

    prompt_kwargs, prompt_key = _build_react_prompt_kwargs()
    try:
        _react_agent = create_react_agent(
            model=model,
            tools=LANGCHAIN_TOOLS,
            **prompt_kwargs,
        )
    except TypeError as exc:
        raise RuntimeError(
            f"create_react_agent rejected prompt injection via '{prompt_key}': {exc}"
        ) from exc

    return _react_agent


# ---------------------------------------------------------------------------
# Message conversion utilities
# ---------------------------------------------------------------------------
def _history_to_messages(history: list[dict]) -> list[Any]:
    """Convert API/bot history format to LangChain messages."""
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
    """Coerce various content types to plain text."""
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
    """Extract the final AI message text from a LangGraph result."""
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


# ---------------------------------------------------------------------------
# Lightweight post-processing (safety net for output stability)
# ---------------------------------------------------------------------------
_GENERIC_ANALYSIS_LEADIN_PATTERNS = (
    re.compile(
        r"^(?:好(?:的)?|当然|没问题|可以)[，,\s]*"
        r"(?:作为一名[^。:\n]*分析师[^。:\n]*"
        r"|(?:这是|以下是|下面是).*?(?:分析|解读|梳理|总结|动态)"
        r"|(?:我们|我|让我)来看[^。:\n]*(?:分析|解读|梳理|格局|动态)"
        r"|(?:我|我们)来[^。:\n]*(?:分析|解读|梳理|总结|动态))"
        r"[\s。!！:：]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:这是|以下是|下面是|为您梳理).*?(?:分析|解读|梳理|总结|动态)[\s。!！:：]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:okay|sure|got it)[,\s]+(?:here(?:'s| is)|below is).*(?:analysis|summary|update)[\s.]*$",
        re.IGNORECASE,
    ),
)


def _strip_generic_analysis_leadin(text: str) -> str:
    """Remove generic 'as an analyst...' opening lines that add no value."""
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

    stripped = "\n".join(lines[start_idx:]).strip()
    return stripped or raw.strip()


def _decorate_response_with_sources(text: str, user_message: str, valid_urls: set[str] | None = None) -> tuple[str, dict[str, str]]:
    """Attach numbered citations and source section via shared evidence helper."""
    return _decorate_response_with_sources_core(
        text=text,
        user_message=user_message,
        lookup_url_titles=lookup_url_titles,
        valid_urls=valid_urls,
    )


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------
def _generate_react(history: list[dict], user_message: str) -> tuple[str, set[str]]:
    """Run the ReAct agent loop."""
    agent = _get_react_agent()
    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=user_message))

    recursion_limit = int(os.getenv("AGENT_REACT_RECURSION_LIMIT", "25"))

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": recursion_limit},
    )
    text = _extract_final_text(result)
    if not text:
        raise RuntimeError("ReAct agent returned empty response.")
        
    valid_urls: set[str] = set()
    if isinstance(result, dict) and "messages" in result:
        from langchain_core.messages import ToolMessage
        try:
            from core.evidence import extract_urls
        except ImportError:
            from .core.evidence import extract_urls
            
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                content_str = _coerce_to_text(getattr(msg, "content", ""))
                valid_urls.update(extract_urls(content_str))

    return text, valid_urls


def _generate_response_core(history: list[dict], user_message: str) -> tuple[str, set[str]]:
    """Core generation: invoke ReAct agent with metrics tracking."""
    _metrics_inc("requests_total")

    try:
        _metrics_inc("react_attempts")
        result, valid_urls = _generate_react(history, user_message)
        
        if not valid_urls:
            raise AgentGenerationError(
                "抱歉，针对该问题，系统未能检索到相关的新闻。",
                code="react_empty_evidence_blocked"
            )
            
        _metrics_inc("react_success")
        return result, valid_urls
    except Exception as exc:
        _metrics_inc("react_error")
        if isinstance(exc, AgentGenerationError):
            raise exc
            
        exc_name = type(exc).__name__.lower()
        exc_str = str(exc).lower()
        recursion_hit = (
            "graphrecursionerror" in exc_name
            or "recursion" in exc_str
            or ("limit" in exc_str and "graph" in exc_str)
        )
        if recursion_hit:
            _metrics_inc("react_recursion_limit_hit")
            print(f"[Agent][Warn] recursion limit hit: {type(exc).__name__}: {exc}")
            raise AgentGenerationError(
                "抱歉，由于问题跨度较大，本次分析在多次检索后超时。"
                "请尝试缩小时间范围或换一个更具体的关键词。",
                code="react_recursion_limit_hit",
            )

        transient_markers = (
            "429",
            "rate limit",
            "resource exhausted",
            "quota",
            "deadline",
            "timeout",
            "timed out",
            "service unavailable",
            "unavailable",
        )
        if any(marker in exc_str for marker in transient_markers):
            print(f"[Agent][Warn] upstream/transient error: {type(exc).__name__}: {exc}")
            raise AgentGenerationError(
                "抱歉，当前模型或数据服务暂时不可用。"
                "请稍后重试，或先缩小问题范围重试。",
                code="react_upstream_unavailable",
            )

        print(f"[Agent][Error] unexpected runtime failure: {type(exc).__name__}: {exc}")
        raise AgentGenerationError(
            "抱歉，本次分析未能完成。请尝试更换关键词或缩小时间范围。",
            code="react_unexpected_runtime_error",
        )


def generate_response(history: list[dict], user_message: str) -> str:
    """Public generation entrypoint with post-processing safety net.

    Post-processing includes:
    1. Strip generic analysis lead-in phrases
    2. Normalize citations and attach source section
    """
    core_text, valid_urls = _generate_response_core(history, user_message)
    core_text = _strip_generic_analysis_leadin(core_text)
    final_text, _ = _decorate_response_with_sources(core_text, user_message, valid_urls)
    return final_text


def generate_response_payload(history: list[dict], user_message: str) -> dict[str, Any]:
    """Structured response for transport layers (e.g., Telegram bot)."""
    core_text, valid_urls = _generate_response_core(history, user_message)
    core_text = _strip_generic_analysis_leadin(core_text)
    final_text, title_map = _decorate_response_with_sources(core_text, user_message, valid_urls)
    return {
        "text": final_text,
        "url_title_map": title_map,
    }


# ---------------------------------------------------------------------------
# CLI compatibility
# ---------------------------------------------------------------------------
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
