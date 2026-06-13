"""Agent runtime backed by a custom LangGraph StateGraph.

Custom runtime: LangGraph StateGraph with project-owned ToolRuntime.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from services.llm_provider import build_agent_chat_model

from .graph.builder import invoke_custom_graph
from .clarification import (
    ClarificationPayload,
    ClarificationRequiredError,
    build_clarification_payload,
    detect_scope_or_conflict_reason,
    infer_clarification_reason,
)
from .core.evidence import (
    contains_cjk as _contains_cjk_core,
    contains_valid_url_in_body as _contains_valid_url_in_body_core,
    decorate_response_with_sources as _decorate_response_with_sources_core,
    extract_urls as _extract_urls_core,
    normalize_url_for_match as _normalize_url_for_match_core,
)
from .core.intent import classify_user_intent
from .core.metrics import (
    get_route_metrics_snapshot,
    metrics_inc as _metrics_inc,
    reset_route_metrics,
)
from .core.run_context import (
    agent_run_context,
    emit_progress as _emit_progress,
    get_tool_call_chain as _get_accumulated_tool_call_chain,
    get_tool_calls as _get_accumulated_tool_calls,
)
from .core.trace import (
    finalize_request_trace as _finalize_request_trace,
    get_current_request_id as _get_current_request_id,
    get_current_thread_id as _get_current_thread_id,
    get_last_trace_summary as _get_last_trace_summary,
    request_trace_context as _request_trace_context,
)
from .tools.news_ops import lookup_url_titles


class AgentGenerationError(Exception):
    """Business-level generation failure with user-safe message."""

    def __init__(self, message: str, code: str = "generation_failed"):
        super().__init__(message)
        self.message = str(message)
        self.code = str(code)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


_EMOJI_RE = re.compile(
    r"(?:[0-9#*]\ufe0f?\u20e3)"
    r"|(?:[\U0001F1E6-\U0001F1FF]{2})"
    r"|(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]\ufe0f?"
    r"(?:\u200d[\U0001F300-\U0001FAFF\u2600-\u27BF]\ufe0f?)*)"
    r"|[\ufe0e\ufe0f\u200d]"
)


def _strip_emoji(text: Any) -> str:
    """Remove emoji and emoji joiner artifacts from model-visible output."""
    return _EMOJI_RE.sub("", str(text or ""))


def _strip_emoji_from_title_map(title_map: dict[str, str] | None) -> dict[str, str]:
    if not isinstance(title_map, dict):
        return {}
    return {
        str(url).strip(): _strip_emoji(title).strip()
        for url, title in title_map.items()
        if str(url).strip() and _strip_emoji(title).strip()
    }


def _strip_emoji_from_clarification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload or {})
    cleaned["question"] = _strip_emoji(cleaned.get("question", "")).strip()
    cleaned["original_question"] = _strip_emoji(cleaned.get("original_question", "")).strip()
    hints = cleaned.get("hints", [])
    if isinstance(hints, list):
        cleaned["hints"] = [
            _strip_emoji(item).strip()
            for item in hints
            if _strip_emoji(item).strip()
        ]
    else:
        cleaned["hints"] = []
    return cleaned


def _build_chat_model() -> Any:
    """Create the chat model client from environment configuration."""
    temperature = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    return build_agent_chat_model(temperature=temperature)

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


# ---------------------------------------------------------------------------
# Lightweight post-processing (safety net for output stability)
# ---------------------------------------------------------------------------
_GENERIC_ANALYSIS_LEADIN_PATTERNS = (
    re.compile(
        r"^(?:好的|当然|可以|没问题|这是|以下是|下面是).{0,80}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:让我|我来|先来).{0,80}(?:分析|解读|总结).{0,40}$",
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


def _ordered_tool_calls_for_eval(trace_summary: dict[str, Any] | None) -> list[str]:
    if isinstance(trace_summary, dict):
        chain = trace_summary.get("tool_call_chain")
        if isinstance(chain, list):
            ordered = [str(item).strip() for item in chain if str(item).strip()]
            if ordered:
                return ordered
    chain = _get_accumulated_tool_call_chain()
    if chain:
        return [str(item).strip() for item in chain if str(item).strip()]
    return sorted(_get_accumulated_tool_calls())

def _contains_cjk(text: str) -> bool:
    return _contains_cjk_core(text)


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


def _strict_body_url_citations_enabled() -> bool:
    raw = os.getenv(
        "AGENT_STRICT_BODY_URL_CITATIONS",
        os.getenv("AGENT_STRICT_INLINE_CITATIONS", "true"),
    )
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _enforce_body_valid_url_guard(
    core_text: str,
    user_message: str,
    valid_urls: list[str] | set[str] | None,
) -> None:
    """Block response when evidence exists but body has no valid source URL citation."""
    if not _strict_body_url_citations_enabled():
        return
    if not valid_urls:
        return
    if _contains_valid_url_in_body_core(core_text, valid_urls):
        return

    _metrics_inc("graph_inline_citation_blocked")
    if _contains_cjk(user_message):
        raise AgentGenerationError(
            "抱歉，我没能基于现有证据给出有把握的回答。可能数据库中暂无与此直接相关的信息，"
            "建议换一个更具体的实体、时间范围或来源再试。",
            code="graph_inline_citation_missing",
        )
    raise AgentGenerationError(
        "Sorry, I couldn't produce a well-grounded answer from the available evidence. "
        "There may be no directly relevant information in the database; "
        "try a more specific entity, time range, or source.",
        code="graph_inline_citation_missing",
    )


def _normalize_url_for_guard(url: str) -> str:
    return _normalize_url_for_match_core(url)


def _enforce_output_urls_in_valid_set(
    text: str, user_message: str, valid_urls: list[str] | set[str] | None
) -> None:
    """Block when model output contains URLs outside current valid_urls set."""
    if not valid_urls:
        return

    allowed: set[str] = set()
    for item in valid_urls:
        normalized = _normalize_url_for_guard(str(item))
        if normalized:
            allowed.add(normalized)
    if not allowed:
        return

    unknown: list[str] = []
    seen_unknown: set[str] = set()
    for url in _extract_urls_core(str(text or "")):
        normalized = _normalize_url_for_guard(url)
        if not normalized or normalized in allowed:
            continue
        if normalized in seen_unknown:
            continue
        seen_unknown.add(normalized)
        unknown.append(normalized)

    if not unknown:
        return

    _metrics_inc("graph_url_outside_valid_set_blocked")
    preview = ", ".join(unknown[:3])
    if _contains_cjk(user_message):
        raise AgentGenerationError(
            f"抱歉，本次输出包含不在证据集合中的 URL，已拦截。异常 URL：{preview}",
            code="graph_url_outside_valid_set",
        )
    raise AgentGenerationError(
        f"Blocked: output contains URLs outside current valid_urls set: {preview}",
        code="graph_url_outside_valid_set",
    )


def _build_hitl_soft_followup(
    *,
    user_message: str,
    risk_reason: str,
    risk_context: dict[str, Any],
) -> str:
    """Generate a dynamic HITL follow-up question via the current model."""
    fallback = (
        "为提高本次结论可信度，请补充一个约束后我再继续："
        "时间范围、信息来源范围，或分析维度。你希望先收敛哪一项？"
    )
    context_preview = {
        "reason": str(risk_reason or "").strip(),
        "source_count": int(risk_context.get("source_count", 0) or 0),
        "url_count": int(risk_context.get("url_count", 0) or 0),
        "time_span_days": risk_context.get("time_span_days"),
        "entity_candidates": risk_context.get("entity_candidates", []),
        "tool_calls": risk_context.get("tool_calls", []),
        "ambiguous_scope_reasons": risk_context.get("ambiguous_scope_reasons", []),
    }
    prompt = (
        "You are a HITL clarification assistant for a news analysis system.\n"
        "Based on the user question, write a concise follow-up question in Chinese to collect one missing constraint.\n"
        "Requirements:\n"
        "1) Output only the final follow-up text, no reasoning.\n"
        "2) Keep it within 1-3 sentences and professional tone.\n"
        "3) Do not output template ids, URLs, or inline citation markers like [1].\n"
        "4) Follow-up must stay specific to the current question.\n"
        "5) Do not use emoji, emoticons, pictographs, or decorative reaction icons.\n\n"
        f"User question: {user_message}\n"
        f"Risk context: {context_preview}\n"
    )
    try:
        model = _build_chat_model()
        result = model.invoke([HumanMessage(content=prompt)])
        text = _coerce_to_text(getattr(result, "content", result)).strip()
        text = re.sub(r"https?://[^\s)\]]+", "", text).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return fallback
        return text
    except Exception as exc:
        print(f"[Agent][Warn] hitl soft follow-up generation failed: {type(exc).__name__}: {exc}")
        return fallback


def _dynamic_clarification_enabled() -> bool:
    raw = os.getenv("AGENT_DYNAMIC_CLARIFICATION_ENABLED", "true")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _sanitize_clarification_text(text: str, *, max_chars: int) -> str:
    cleaned = _strip_emoji(text).strip()
    cleaned = re.sub(r"https?://[^\s)\]]+", "", cleaned).strip()
    cleaned = re.sub(r"\[(?:\d{1,3}|[^\]\n]{1,80})\]", "", cleaned).strip()
    cleaned = re.sub(r"^[\-*•\d\.\s]+", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars].strip()


def _coerce_dynamic_hints(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    hints: list[str] = []
    seen: set[str] = set()
    for item in value:
        hint = _sanitize_clarification_text(str(item or ""), max_chars=90)
        key = hint.lower()
        if not hint or key in seen:
            continue
        seen.add(key)
        hints.append(hint)
        if len(hints) >= 3:
            break
    return hints


def _build_dynamic_clarification_payload(
    user_message: str,
    *,
    reason: str | None = None,
    context: dict[str, Any] | None = None,
) -> ClarificationPayload:
    """Build an LLM-guided clarification payload, with deterministic fallback."""
    fallback = build_clarification_payload(user_message, reason=reason, context=context)
    if fallback.reason != "insufficient_evidence" or not _dynamic_clarification_enabled():
        return fallback

    context_preview = {
        "reason": fallback.reason,
        "tool_calls": (context or {}).get("tool_calls", []),
        "policy_reason": (context or {}).get("policy_reason", ""),
        "candidate_answer_preview": str((context or {}).get("candidate_answer", ""))[:300],
        "fallback_hints": fallback.hints[:4],
    }
    prompt = (
        "You are a clarification assistant for a tech-news intelligence agent.\n"
        "The retrieval step found insufficient reliable evidence for the user's exact request.\n"
        "Generate a tailored clarification that helps the user continue successfully.\n\n"
        "Return JSON only with this schema:\n"
        '{"question":"...","hints":["...","..."]}\n\n'
        "Rules:\n"
        "1) Use the same language as the user.\n"
        "2) The question must directly mention the user's topic/entity/event when present.\n"
        "3) Do not use a generic template such as only asking for time/source/dimension.\n"
        "4) Ask for the smallest useful missing detail: timeframe, source, exact company/product/event, "
        "original URL, or whether to broaden the search terms.\n"
        "5) Hints must be concrete options tailored to the user's request; 2-3 hints max.\n"
        "6) Do not include URLs, citation markers, markdown tables, or explanations.\n"
        "7) Do not use emoji, emoticons, pictographs, or decorative reaction icons.\n\n"
        f"User question: {user_message}\n"
        f"Context: {context_preview}\n"
    )
    try:
        model = _build_chat_model()
        result = model.invoke([HumanMessage(content=prompt)])
        parsed = _extract_json_object(_coerce_to_text(getattr(result, "content", result)))
        if not parsed:
            return fallback
        question = _sanitize_clarification_text(str(parsed.get("question") or ""), max_chars=220)
        hints = _coerce_dynamic_hints(parsed.get("hints"))
        if not question:
            return fallback
        return ClarificationPayload(
            question=question,
            hints=hints or fallback.hints[:2],
            reason=fallback.reason,
            kind=fallback.kind,
            original_question=fallback.original_question,
        )
    except Exception as exc:
        print(f"[Agent][Warn] dynamic clarification generation failed: {type(exc).__name__}: {exc}")
        return fallback


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def _generate_response_core(
    history: list[dict],
    user_message: str,
) -> tuple[str, list[str], list[str]]:
    """Core generation: invoke the custom LangGraph agent with metrics tracking.

    Returns (text, valid_urls, citable_urls). valid_urls is this turn's retrieved
    evidence (drives empty-evidence/risk guards and evidence_count); citable_urls
    additionally includes context-pack sources the model was shown (drives the
    body-URL guard and source decoration).
    """
    _metrics_inc("requests_total")

    try:
        _metrics_inc("graph_attempts")
        graph_result = invoke_custom_graph(
            history,
            user_message,
            request_id=_get_current_request_id(),
            thread_id=_get_current_thread_id(),
        )
        if graph_result.clarification:
            payload = graph_result.clarification
            if str(payload.get("reason") or "").strip().lower() == "insufficient_evidence":
                raise ClarificationRequiredError(
                    _build_dynamic_clarification_payload(
                        user_message,
                        reason=str(payload.get("reason") or ""),
                        context={
                            "graph_payload": payload,
                            "policy_reason": payload.get("policy_reason", ""),
                        },
                    )
                )
            fallback = build_clarification_payload(user_message)
            raise ClarificationRequiredError(
                ClarificationPayload(
                    question=str(payload.get("question") or fallback.question).strip(),
                    hints=[
                        str(item).strip()
                        for item in (payload.get("hints") if isinstance(payload.get("hints"), list) else [])
                        if str(item).strip()
                    ],
                    reason=str(payload.get("reason") or fallback.reason),
                    kind=str(payload.get("kind") or "clarification_required"),
                    original_question=str(payload.get("original_question") or user_message),
                )
            )
        result = graph_result.text
        valid_urls = list(graph_result.urls)
        citable_urls = list(graph_result.citable_urls or valid_urls)
        tool_calls = _get_accumulated_tool_calls()

        if _should_block_empty_evidence(user_message, valid_urls, tool_calls):
            reason = infer_clarification_reason(user_message)
            print(
                "[Agent][Clarification] request_id=%s reason=%s stage=empty_evidence tool_calls=%s"
                % (
                    _get_current_request_id() or "-",
                    reason,
                    sorted(tool_calls),
                )
            )
            clarification = _build_dynamic_clarification_payload(
                user_message=user_message,
                reason=reason,
                context={
                    "tool_calls": sorted(tool_calls),
                    "candidate_answer": result,
                },
            )
            raise ClarificationRequiredError(clarification)

        risk_reason, risk_context = detect_scope_or_conflict_reason(
            user_message=user_message,
            candidate_answer=result,
            valid_urls=valid_urls,
            tool_calls=tool_calls,
        )
        if risk_reason:
            print(
                "[Agent][Clarification] request_id=%s reason=%s stage=risk_guard risk_context=%s"
                % (
                    _get_current_request_id() or "-",
                    risk_reason,
                    risk_context,
                )
            )
            _metrics_inc("graph_hitl_soft_prompt")
            followup = _build_hitl_soft_followup(
                user_message=user_message,
                risk_reason=risk_reason,
                risk_context=risk_context,
            )
            if followup:
                result = f"{result.rstrip()}\n\n{followup.strip()}".strip()

        _metrics_inc("graph_success")
        return result, valid_urls, citable_urls
    except Exception as exc:
        _metrics_inc("graph_error")
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
            _metrics_inc("graph_recursion_limit_hit")
            print(f"[Agent][Warn] recursion limit hit: {type(exc).__name__}: {exc}")
            raise AgentGenerationError(
                "抱歉，由于问题跨度较大，本次分析在多轮检索后超时。请缩小时间范围或换一个更具体的关键词再试。",
                code="graph_recursion_limit_hit",
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
                "抱歉，当前上游服务暂时不可用，请稍后重试。",
                code="graph_upstream_unavailable",
            )

        print(f"[Agent][Error] unexpected runtime failure: {type(exc).__name__}: {exc}")
        raise AgentGenerationError(
            "抱歉，本次分析未能完成，请稍后重试。",
            code="graph_unexpected_runtime_error",
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
) -> tuple[str, list[str], list[str]]:
    """Run core generation with optional progress callback lifecycle."""
    with agent_run_context(progress_callback=progress_callback):
        _emit_progress("understanding")
        core_text, valid_urls, citable_urls = _generate_response_core(
            history,
            user_message,
        )
        return core_text, valid_urls, citable_urls


def generate_response(
    history: list[dict],
    user_message: str,
    progress_callback: Callable[[dict[str, str]], None] | None = None,
    request_id: str | None = None,
    thread_id: str | None = None,
) -> str:
    """Public generation entrypoint with post-processing safety net.

    Post-processing includes:
    1. Strip generic analysis lead-in phrases
    2. Normalize citations and attach source section
    """
    thread_id = thread_id or _extract_thread_id(history)
    with _request_trace_context(
        user_message=user_message,
        thread_id=thread_id,
        request_id=request_id,
    ):
        try:
            core_text, valid_urls, citable_urls = _run_generation_core(
                history,
                user_message,
                progress_callback=progress_callback,
            )
            core_text = _strip_generic_analysis_leadin(core_text)
            core_text = _strip_emoji(core_text)
            _enforce_output_urls_in_valid_set(core_text, user_message, citable_urls)
            _enforce_body_valid_url_guard(core_text, user_message, citable_urls)
            final_text, _ = _decorate_response_with_sources(core_text, user_message, citable_urls)
            final_text = _strip_emoji(final_text)
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
            payload = _strip_emoji_from_clarification_payload(exc.clarification.to_dict())
            question_text = _strip_emoji(payload.get("question", "")).strip()
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
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Structured response for transport layers (e.g., Telegram bot)."""
    thread_id = thread_id or _extract_thread_id(history)
    with _request_trace_context(
        user_message=user_message,
        thread_id=thread_id,
        request_id=request_id,
    ):
        try:
            core_text, valid_urls, citable_urls = _run_generation_core(
                history,
                user_message,
                progress_callback=progress_callback,
            )
            core_text = _strip_generic_analysis_leadin(core_text)
            core_text = _strip_emoji(core_text)
            _enforce_output_urls_in_valid_set(core_text, user_message, citable_urls)
            _enforce_body_valid_url_guard(core_text, user_message, citable_urls)
            final_text, title_map = _decorate_response_with_sources(core_text, user_message, citable_urls)
            final_text = _strip_emoji(final_text)
            title_map = _strip_emoji_from_title_map(title_map)
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
            payload = _strip_emoji_from_clarification_payload(exc.clarification.to_dict())
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
    thread_id: str | None = None,
    case_id: str | None = None,
    experiment_group: str | None = None,
    include_trace_summary: bool = False,
) -> dict[str, Any]:
    """Structured response for eval with tool trace and URL evidence."""
    thread_id = thread_id or _extract_thread_id(history)
    with _request_trace_context(
        user_message=user_message,
        thread_id=thread_id,
        request_id=request_id,
    ):
        try:
            with agent_run_context():
                eval_request_id = _get_current_request_id()

                core_text, valid_urls, citable_urls = _generate_response_core(
                    history,
                    user_message,
                )
                core_text = _strip_generic_analysis_leadin(core_text)
                core_text = _strip_emoji(core_text)
                _enforce_output_urls_in_valid_set(core_text, user_message, citable_urls)
                _enforce_body_valid_url_guard(core_text, user_message, citable_urls)
                final_text, _ = _decorate_response_with_sources(core_text, user_message, citable_urls)
                final_text = _strip_emoji(final_text)
                tool_calls = _ordered_tool_calls_for_eval(None)
                trace_summary = _finalize_request_trace(
                    final_status="success",
                    evidence_count=len(valid_urls),
                    final_answer_metadata={
                        "response_kind": "eval_payload",
                        "answer_chars": len(final_text),
                        "source_count": len(valid_urls),
                        "tool_count": len(tool_calls),
                        "case_id": str(case_id or ""),
                        "experiment_group": str(experiment_group or ""),
                    },
                )
                tool_calls = _ordered_tool_calls_for_eval(trace_summary)
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
                final_answer_metadata={
                    "response_kind": "eval_payload",
                    "case_id": str(case_id or ""),
                    "experiment_group": str(experiment_group or ""),
                },
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

