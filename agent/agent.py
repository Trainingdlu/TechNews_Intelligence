"""Agent runtime — ReAct-native architecture with Skill infrastructure.

Single runtime: LangGraph ReAct agent with full tool autonomy.
Tool wrappers execute through SkillRegistry + ToolHookRunner for
structured validation, pre/post guards, and evidence tracking.
"""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .core.runtime_factories import build_default_hook_runner, build_default_registry
from .prompts import SYSTEM_INSTRUCTION
from .clarification import (
    ClarificationRequiredError,
    build_clarification_payload,
    detect_scope_or_conflict_reason,
    infer_clarification_reason,
)
from .core.evidence import (
    decorate_response_with_sources as _decorate_response_with_sources_core,
    has_inline_citation_in_body as _has_inline_citation_in_body_core,
)
from .core.intent import classify_user_intent
from .core.metrics import (
    get_route_metrics_snapshot,
    metrics_inc as _metrics_inc,
    reset_route_metrics,
)
from .core.run_context import (
    add_evidence_urls as _accumulate_evidence,
    add_tool_call as _accumulate_tool_call,
    agent_run_context,
    emit_progress as _emit_progress,
    get_evidence_urls as _get_accumulated_evidence,
    get_tool_calls as _get_accumulated_tool_calls,
    set_request_metadata as _set_request_metadata,
)
from .core.trace import (
    extract_token_usage as _extract_token_usage,
    finalize_request_trace as _finalize_request_trace,
    get_current_request_id as _get_current_request_id,
    get_current_thread_id as _get_current_thread_id,
    get_last_trace_summary as _get_last_trace_summary,
    request_trace_context as _request_trace_context,
    set_request_token_usage as _set_request_token_usage,
    trace_tool_finish_error as _trace_tool_finish_error,
    trace_tool_finish_with_envelope as _trace_tool_finish_with_envelope,
    trace_tool_start as _trace_tool_start,
)
from .core.skill_contracts import SkillEnvelope
from .skills.news_ops import lookup_url_titles


# ---------------------------------------------------------------------------
# Skill infrastructure (lazy-initialized singletons)
# ---------------------------------------------------------------------------
_registry = None
_hook_runner = None


def _get_registry():
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def _get_hook_runner():
    global _hook_runner
    if _hook_runner is None:
        _hook_runner = build_default_hook_runner()
    return _hook_runner


# ---------------------------------------------------------------------------
# Unified skill dispatch (SkillRegistry + ToolHookRunner)
# ---------------------------------------------------------------------------
def _envelope_to_tool_text(envelope: SkillEnvelope) -> str:
    """Convert SkillEnvelope to text for the ReAct LLM.

    For tools that returned raw text in their data, we pass that through.
    For structured data, we format a summary the LLM can reason over.
    """
    if envelope.status == "error":
        return f"[Error] {envelope.error or 'skill execution failed'}"

    if envelope.status == "empty":
        return "No matching records found."

    # If the skill stored raw_output text, use it directly (preserves URLs)
    data = envelope.data
    if isinstance(data, dict):
        raw = data.get("raw_output")
        if isinstance(raw, str) and raw.strip():
            return raw

        # For query_news / trend_analysis with structured data, format records
        records = data.get("records")
        if isinstance(records, list) and records:
            lines = []
            for idx, item in enumerate(records[:15], 1):
                title = item.get("title_cn") or item.get("title") or "(untitled)"
                url = item.get("url", "")
                source = item.get("source") or item.get("source_type") or ""
                points = item.get("points", 0)
                created = str(item.get("created_at", ""))[:16].replace("T", " ")
                summary = str(item.get("summary", ""))[:220]
                lines.append(
                    f"{idx}. [{source}] {title}\n"
                    f"   time={created}, points={points}\n"
                    f"   url={url}\n"
                    f"   summary={summary}"
                )
            return "\n".join(lines)

        # Trend analysis data
        if "topic" in data and "recent_count" in data:
            topic = data.get("topic", "")
            recent = data.get("recent_count", 0)
            prev = data.get("previous_count", 0)
            delta = data.get("count_delta", 0)
            daily = data.get("daily", [])
            lines = [
                f"Trend for topic: {topic}",
                f"Recent count: {recent}, Previous count: {prev}, Delta: {delta:+d}",
            ]
            if daily:
                lines.append("Daily breakdown:")
                for d in daily[:14]:
                    lines.append(f"  {d.get('day', '')}: count={d.get('count', 0)}")
            return "\n".join(lines)

    return str(data) if data else "No data returned."


def _execute_skill(skill_name: str, payload: dict[str, Any]) -> str:
    """Unified skill dispatch: Registry → Hooks → Execute → Evidence collect.

    Returns the text representation for the ReAct LLM to consume.
    """
    trace_event_index = _trace_tool_start(skill_name, payload)
    hooks = _get_hook_runner()

    try:
        # Pre-hook guard
        pre = hooks.pre_tool_use(skill_name, payload)
        if pre.action == "deny":
            blocked_reason = pre.reason or "pre-hook denied"
            _trace_tool_finish_error(
                trace_event_index,
                error_code="tool_pre_hook_denied",
                error_message=blocked_reason,
                error=RuntimeError(blocked_reason),
            )
            return f"[Blocked] {blocked_reason}"
        effective_payload = pre.updated_payload if pre.updated_payload is not None else payload

        # Execute via registry
        registry = _get_registry()
        if skill_name in {"search_news", "query_news", "trend_analysis", "fulltext_batch"}:
            _emit_progress("retrieving", skill_name)
        elif skill_name in {"compare_sources", "compare_topics", "build_timeline", "analyze_landscape"}:
            _emit_progress("analyzing", skill_name)
        else:
            _emit_progress("retrieving", skill_name)
        envelope = registry.execute(skill_name, effective_payload)
        _accumulate_tool_call(skill_name)

        # Post-hook guard
        post = hooks.post_tool_use(skill_name, effective_payload, envelope)
        if post.action == "deny":
            blocked_reason = post.reason or "post-hook denied"
            _trace_tool_finish_with_envelope(
                trace_event_index,
                envelope,
                status_override="blocked",
                error_code="tool_post_hook_denied",
                error_message=blocked_reason,
            )
            return f"[Blocked] {blocked_reason}"

        # Collect structured evidence URLs
        evidence_urls = [
            str(e.url).strip()
            for e in (envelope.evidence or [])
            if str(e.url or "").strip()
        ]
        _accumulate_evidence(evidence_urls)
        _trace_tool_finish_with_envelope(trace_event_index, envelope)

        # Convert to text for LLM
        return _envelope_to_tool_text(envelope)
    except Exception as exc:
        _trace_tool_finish_error(
            trace_event_index,
            error_code=f"tool_{type(exc).__name__.lower()}",
            error_message=str(exc),
            error=exc,
        )
        raise


# ---------------------------------------------------------------------------
# LangChain tool wrappers (LLM-facing interface unchanged)
# ---------------------------------------------------------------------------
@tool("search_news")
def search_news_tool(query: str, days: int = 21) -> str:
    """Search related news with hybrid retrieval (semantic + keyword).

    Best for exploratory queries. Returns up to 5 results ranked by relevance.
    If no results, try broader query or larger days window.
    """
    return _execute_skill("search_news", {"query": query, "days": days})


@tool("read_news_content")
def read_news_content_tool(url: str) -> str:
    """Read full article content by URL. Only works for URLs from other tools."""
    return _execute_skill("read_news_content", {"url": url})


@tool("get_db_stats")
def get_db_stats_tool() -> str:
    """Get database freshness stats and total article count."""
    return _execute_skill("get_db_stats", {})


@tool("list_topics")
def list_topics_tool() -> str:
    """Get daily article volume distribution for recent 21 days."""
    return _execute_skill("list_topics", {})


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
    return _execute_skill("query_news", {
        "query": query,
        "source": source,
        "days": days,
        "category": category,
        "sentiment": sentiment,
        "sort": sort,
        "limit": limit,
    })


@tool("trend_analysis")
def trend_analysis_tool(topic: str, window: int = 7) -> str:
    """Analyze topic momentum: compare article count and points in recent
    N days vs previous N days. Includes daily breakdown."""
    return _execute_skill("trend_analysis", {"topic": topic, "window": window})


@tool("compare_sources")
def compare_sources_tool(topic: str, days: int = 14) -> str:
    """Compare HackerNews vs TechCrunch coverage and sentiment for a topic."""
    return _execute_skill("compare_sources", {"topic": topic, "days": days})


@tool("compare_topics")
def compare_topics_tool(topic_a: str, topic_b: str, days: int = 14) -> str:
    """Compare two entities side-by-side (e.g., OpenAI vs Anthropic) with
    DB-backed metrics, momentum, source mix, and evidence URLs."""
    return _execute_skill("compare_topics", {
        "topic_a": topic_a,
        "topic_b": topic_b,
        "days": days,
    })


@tool("build_timeline")
def build_timeline_tool(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build chronological event timeline for a topic/company/product.
    Auto-retries with wider window if no data found initially."""
    return _execute_skill("build_timeline", {
        "topic": topic,
        "days": days,
        "limit": limit,
    })


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
    return _execute_skill("analyze_landscape", {
        "topic": topic,
        "days": days,
        "entities": entities,
        "limit_per_entity": limit_per_entity,
    })


@tool("fulltext_batch")
def fulltext_batch_tool(urls: str, max_chars_per_article: int = 4000) -> str:
    """Batch read full article text. Accepts URLs or keyword query for auto-selection."""
    return _execute_skill("fulltext_batch", {
        "urls": urls,
        "max_chars_per_article": max_chars_per_article,
    })


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

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if credentials_path and not os.path.isfile(credentials_path):
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS must point to an existing JSON file: "
            f"{credentials_path}"
        )

    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError as exc:
        raise RuntimeError(
            "Vertex provider requires 'langchain-google-vertexai'. "
            "Install dependencies with: pip install -r requirements.txt"
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


def _decorate_response_with_sources(
    text: str, user_message: str, valid_urls: list[str] | set[str] | None = None
) -> tuple[str, dict[str, str]]:
    """Attach numbered citations and source section via shared evidence helper."""
    return _decorate_response_with_sources_core(
        text=text,
        user_message=user_message,
        lookup_url_titles=lookup_url_titles,
        valid_urls=valid_urls,
    )



_SOURCE_BULLET_RE = re.compile(r"^\s*-\s*\[(\d{1,3})\]\s+(.+?)\s*$")


def _extract_citation_urls_from_text(text: str) -> list[str]:
    """Extract ordered citation URLs from rendered source bullet lines."""
    citation_map: dict[int, str] = {}
    for line in str(text or "").splitlines():
        match = _SOURCE_BULLET_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1))
        rest = match.group(2).strip()

        url_match = re.search(r"\]\((https?://[^\s)]+)\)", rest, flags=re.IGNORECASE)
        if not url_match:
            url_match = re.search(r"https?://[^\s)]+", rest, flags=re.IGNORECASE)
        if not url_match:
            continue

        url = str(url_match.group(1) if url_match.lastindex else url_match.group(0)).rstrip(".,;:!?)")
        if not url:
            continue
        citation_map[idx] = url

    if not citation_map:
        return []
    return [citation_map[idx] for idx in sorted(citation_map.keys())]

def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text or ""))


def _should_block_empty_evidence(
    user_message: str, valid_urls: list[str] | set[str], tool_calls: set[str]
) -> bool:
    """Decide whether empty-evidence outputs should be blocked."""
    if valid_urls:
        return False
    if tool_calls:
        return True
    user_intent = classify_user_intent(user_message)
    if user_intent == "smalltalk_or_capability":
        return False
    if user_intent in {"analysis", "conflict_resolution", "roundup_listing"}:
        return True
    return True


def _strict_inline_citations_enabled() -> bool:
    return os.getenv("AGENT_STRICT_INLINE_CITATIONS", "true").strip().lower() not in {"0", "false", "no", "off"}


def _enforce_inline_citation_guard(
    final_text: str, user_message: str, valid_urls: list[str] | set[str] | None
) -> None:
    """Block response when evidence exists but body-level [n] citations are missing."""
    if not _strict_inline_citations_enabled():
        return
    if not valid_urls:
        return
    if _has_inline_citation_in_body_core(final_text):
        return

    _metrics_inc("react_inline_citation_blocked")
    if _contains_cjk(user_message):
        raise AgentGenerationError(
            "抱歉，本次回答缺少准确引用。为避免无证据结论，已阻断该输出，请重试。",
            code="react_inline_citation_missing",
        )
    raise AgentGenerationError(
        "The response was blocked because inline citations ([1], [2], ...) were missing in the answer body.",
        code="react_inline_citation_missing",
    )


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def _generate_react(
    history: list[dict],
    user_message: str,
    *,
    invoke_metadata: dict[str, Any] | None = None,
    invoke_tags: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Run the ReAct agent loop with Skill infrastructure."""
    agent = _get_react_agent()
    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=user_message))

    recursion_limit = int(os.getenv("AGENT_REACT_RECURSION_LIMIT", "25"))
    invoke_config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if invoke_metadata:
        invoke_config["metadata"] = dict(invoke_metadata)
    if invoke_tags:
        invoke_config["tags"] = [str(tag).strip() for tag in invoke_tags if str(tag).strip()]

    result = agent.invoke(
        {"messages": messages},
        config=invoke_config,
    )

    if isinstance(result, dict) and "messages" in result:
        usage = _extract_token_usage(list(result.get("messages", [])))
        _set_request_token_usage(usage)

    text = _extract_final_text(result)
    if not text:
        raise RuntimeError("ReAct agent returned empty response.")

    # Collect evidence from two sources:
    # 1. Structured evidence from SkillEnvelope (accumulated during tool calls)
    valid_urls: list[str] = list(_get_accumulated_evidence())
    seen_urls = set(valid_urls)

    # 2. URLs parsed from ToolMessage content (fallback for non-skill tools)
    if isinstance(result, dict) and "messages" in result:
        from langchain_core.messages import ToolMessage
        from .core.evidence import extract_urls

        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage):
                for call in (getattr(msg, "tool_calls", None) or []):
                    if isinstance(call, dict):
                        _accumulate_tool_call(str(call.get("name", "")).strip())
            if isinstance(msg, ToolMessage):
                _accumulate_tool_call(str(getattr(msg, "name", "")).strip())
                content_str = _coerce_to_text(getattr(msg, "content", ""))
                for url in extract_urls(content_str):
                    if url not in seen_urls:
                        valid_urls.append(url)
                        seen_urls.add(url)

    return text, valid_urls


def _generate_response_core(
    history: list[dict],
    user_message: str,
    *,
    invoke_metadata: dict[str, Any] | None = None,
    invoke_tags: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Core generation: invoke ReAct agent with metrics tracking."""
    _metrics_inc("requests_total")

    try:
        _metrics_inc("react_attempts")
        result, valid_urls = _generate_react(
            history,
            user_message,
            invoke_metadata=invoke_metadata,
            invoke_tags=invoke_tags,
        )
        tool_calls = _get_accumulated_tool_calls()

        if _should_block_empty_evidence(user_message, valid_urls, tool_calls):
            reason = infer_clarification_reason(user_message)
            clarification = build_clarification_payload(
                user_message=user_message,
                reason=reason,
                context={"tool_calls": sorted(tool_calls)},
            )
            raise ClarificationRequiredError(clarification)

        risk_reason, risk_context = detect_scope_or_conflict_reason(
            user_message=user_message,
            candidate_answer=result,
            valid_urls=valid_urls,
            tool_calls=tool_calls,
        )
        if risk_reason:
            clarification = build_clarification_payload(
                user_message=user_message,
                reason=risk_reason,
                context=risk_context,
            )
            raise ClarificationRequiredError(clarification)

        _metrics_inc("react_success")
        return result, valid_urls
    except Exception as exc:
        _metrics_inc("react_error")
        if isinstance(exc, (AgentGenerationError, ClarificationRequiredError)):
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
                "抱歉，当前服务暂时不可用。"
                "请稍后重试。",
                code="react_upstream_unavailable",
            )

        print(f"[Agent][Error] unexpected runtime failure: {type(exc).__name__}: {exc}")
        raise AgentGenerationError(
            "抱歉，本次分析未能完成。请尝试更换关键词或缩小时间范围。",
            code="react_unexpected_runtime_error",
        )


def _extract_thread_id(history: list[dict] | None) -> str | None:
    """Best-effort thread id extraction from transport history payload."""
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        for key in ("thread_id", "threadId"):
            value = item.get(key)
            if value:
                return str(value)
    return None


def _resolve_trace_final_status(error: Exception) -> str:
    if isinstance(error, ClarificationRequiredError):
        return "clarification_required"
    if isinstance(error, AgentGenerationError):
        return "blocked"
    return "error"


def _resolve_trace_error_code(error: Exception) -> str:
    if isinstance(error, ClarificationRequiredError):
        return f"clarification_{error.clarification.reason}"
    code = getattr(error, "code", None)
    if code:
        return str(code)
    return f"runtime_{type(error).__name__.lower()}"


def _run_generation_core(
    history: list[dict],
    user_message: str,
    progress_callback: Callable[[dict[str, str]], None] | None = None,
) -> tuple[str, list[str]]:
    """Run core generation with optional progress callback lifecycle."""
    with agent_run_context(progress_callback=progress_callback):
        _set_request_metadata(
            request_id=_get_current_request_id(),
            thread_id=_get_current_thread_id(),
            user_message=user_message,
        )
        _emit_progress("understanding")
        core_text, valid_urls = _generate_response_core(history, user_message)
        _emit_progress("finalizing")
        return core_text, valid_urls


def generate_response(
    history: list[dict],
    user_message: str,
    progress_callback: Callable[[dict[str, str]], None] | None = None,
    request_id: str | None = None,
) -> str:
    """Public generation entrypoint with post-processing safety net.

    Post-processing includes:
    1. Strip generic analysis lead-in phrases
    2. Normalize citations and attach source section
    """
    thread_id = _extract_thread_id(history)
    with _request_trace_context(
        user_message=user_message,
        thread_id=thread_id,
        request_id=request_id,
    ):
        try:
            core_text, valid_urls = _run_generation_core(
                history,
                user_message,
                progress_callback=progress_callback,
            )
            core_text = _strip_generic_analysis_leadin(core_text)
            final_text, _ = _decorate_response_with_sources(core_text, user_message, valid_urls)
            _enforce_inline_citation_guard(final_text, user_message, valid_urls)
            _finalize_request_trace(
                final_status="success",
                evidence_count=len(valid_urls),
                final_answer_metadata={
                    "response_kind": "text",
                    "answer_chars": len(final_text),
                    "source_count": len(valid_urls),
                },
            )
            return final_text
        except ClarificationRequiredError as exc:
            payload = exc.clarification.to_dict()
            question_text = str(payload.get("question", "")).strip()
            _finalize_request_trace(
                final_status="clarification_required",
                evidence_count=0,
                error_code=f"clarification_{payload.get('reason', 'required')}",
                error_message=question_text,
                error=exc,
                final_answer_metadata={
                    "response_kind": "clarification_text",
                    "answer_chars": len(question_text),
                    "source_count": 0,
                },
            )
            return question_text
        except Exception as exc:
            evidence_count = len(valid_urls) if "valid_urls" in locals() else None
            _finalize_request_trace(
                final_status=_resolve_trace_final_status(exc),
                evidence_count=evidence_count,
                error_code=_resolve_trace_error_code(exc),
                error_message=str(exc),
                error=exc,
            )
            raise


def generate_response_payload(
    history: list[dict],
    user_message: str,
    progress_callback: Callable[[dict[str, str]], None] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Structured response for transport layers (e.g., Telegram bot)."""
    thread_id = _extract_thread_id(history)
    with _request_trace_context(
        user_message=user_message,
        thread_id=thread_id,
        request_id=request_id,
    ):
        try:
            core_text, valid_urls = _run_generation_core(
                history,
                user_message,
                progress_callback=progress_callback,
            )
            core_text = _strip_generic_analysis_leadin(core_text)
            final_text, title_map = _decorate_response_with_sources(core_text, user_message, valid_urls)
            _enforce_inline_citation_guard(final_text, user_message, valid_urls)
            _finalize_request_trace(
                final_status="success",
                evidence_count=len(valid_urls),
                final_answer_metadata={
                    "response_kind": "payload",
                    "answer_chars": len(final_text),
                    "source_count": len(valid_urls),
                    "title_map_count": len(title_map),
                },
            )
            return {
                "kind": "answer",
                "text": final_text,
                "url_title_map": title_map,
                "citation_urls": _extract_citation_urls_from_text(final_text),
            }
        except ClarificationRequiredError as exc:
            payload = exc.clarification.to_dict()
            question_text = str(payload.get("question", "")).strip()
            _finalize_request_trace(
                final_status="clarification_required",
                evidence_count=0,
                error_code=f"clarification_{payload.get('reason', 'required')}",
                error_message=question_text,
                error=exc,
                final_answer_metadata={
                    "response_kind": "clarification_payload",
                    "answer_chars": len(question_text),
                    "source_count": 0,
                    "title_map_count": 0,
                },
            )
            return {
                "kind": "clarification_required",
                "text": question_text,
                "url_title_map": {},
                "citation_urls": [],
                "clarification": payload,
            }
        except Exception as exc:
            evidence_count = len(valid_urls) if "valid_urls" in locals() else None
            _finalize_request_trace(
                final_status=_resolve_trace_final_status(exc),
                evidence_count=evidence_count,
                error_code=_resolve_trace_error_code(exc),
                error_message=str(exc),
                error=exc,
            )
            raise


def generate_response_eval_payload(
    history: list[dict],
    user_message: str,
    request_id: str | None = None,
    case_id: str | None = None,
    experiment_group: str | None = None,
    include_trace_summary: bool = False,
) -> dict[str, Any]:
    """Structured response for eval with tool trace and URL evidence."""
    thread_id = _extract_thread_id(history)
    with _request_trace_context(
        user_message=user_message,
        thread_id=thread_id,
        request_id=request_id,
    ):
        try:
            with agent_run_context():
                _set_request_metadata(
                    request_id=_get_current_request_id(),
                    thread_id=_get_current_thread_id(),
                    user_message=user_message,
                )
                eval_request_id = _get_current_request_id()
                invoke_metadata: dict[str, Any] = {
                    "entrypoint": "eval",
                    "request_id": eval_request_id,
                }
                invoke_tags: list[str] = ["eval"]
                if case_id:
                    invoke_metadata["case_id"] = str(case_id)
                    invoke_tags.append(f"case:{case_id}")
                if experiment_group:
                    invoke_metadata["experiment_group"] = str(experiment_group)
                    invoke_tags.append(f"exp:{experiment_group}")
                if thread_id:
                    invoke_metadata["thread_id"] = str(thread_id)

                core_text, valid_urls = _generate_response_core(
                    history,
                    user_message,
                    invoke_metadata=invoke_metadata,
                    invoke_tags=invoke_tags,
                )
                core_text = _strip_generic_analysis_leadin(core_text)
                final_text, _ = _decorate_response_with_sources(core_text, user_message, valid_urls)
                _enforce_inline_citation_guard(final_text, user_message, valid_urls)
                tool_calls = sorted(_get_accumulated_tool_calls())
                trace_summary = _finalize_request_trace(
                    final_status="success",
                    evidence_count=len(valid_urls),
                    final_answer_metadata={
                        "response_kind": "eval_payload",
                        "answer_chars": len(final_text),
                        "source_count": len(valid_urls),
                        "tool_count": len(tool_calls),
                    },
                )
                payload: dict[str, Any] = {
                    "text": final_text,
                    "valid_urls": sorted(valid_urls),
                    "tool_calls": tool_calls,
                    "request_id": eval_request_id,
                }
                if case_id:
                    payload["case_id"] = str(case_id)
                if experiment_group:
                    payload["experiment_group"] = str(experiment_group)
                if include_trace_summary:
                    payload["trace_summary"] = trace_summary
                return payload
        except Exception as exc:
            evidence_count = len(valid_urls) if "valid_urls" in locals() else None
            _finalize_request_trace(
                final_status=_resolve_trace_final_status(exc),
                evidence_count=evidence_count,
                error_code=_resolve_trace_error_code(exc),
                error_message=str(exc),
                error=exc,
            )
            raise


def get_last_tool_calls_snapshot() -> list[str]:
    """Return current request-local tool call set (best-effort for diagnostics)."""
    return sorted(_get_accumulated_tool_calls())


def get_last_request_trace_summary() -> dict[str, Any] | None:
    """Return the last finalized request-level trace summary."""
    return _get_last_trace_summary()


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
