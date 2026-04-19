"""Request-level tracing utilities for agent runtime."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator

from .skill_contracts import SkillEnvelope

_MAX_SUMMARY_DEPTH = 2
_MAX_STR_LEN = 240
_MAX_LIST_ITEMS = 6
_MAX_DICT_ITEMS = 12
_MAX_ERROR_CHAIN = 6
_MAX_CONTEXT_DOCS = 8
logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truncate_text(value: str, max_len: int = _MAX_STR_LEN) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def summarize_payload(value: Any, *, depth: int = _MAX_SUMMARY_DEPTH) -> Any:
    """Build a compact, JSON-safe summary for payload-like values."""
    if depth <= 0:
        return f"<{type(value).__name__}>"

    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)

    if isinstance(value, list):
        preview = [summarize_payload(item, depth=depth - 1) for item in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            preview.append(f"...(+{len(value) - _MAX_LIST_ITEMS} items)")
        return preview

    if isinstance(value, tuple):
        return summarize_payload(list(value), depth=depth)

    if isinstance(value, dict):
        items = list(value.items())
        summarized: dict[str, Any] = {}
        for key, item in items[:_MAX_DICT_ITEMS]:
            summarized[str(key)] = summarize_payload(item, depth=depth - 1)
        if len(items) > _MAX_DICT_ITEMS:
            summarized["__truncated_keys__"] = len(items) - _MAX_DICT_ITEMS
        return summarized

    return _truncate_text(repr(value))


def _extract_context_docs(data: Any) -> list[dict[str, Any]]:
    """Extract lightweight context docs for downstream eval use."""
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_doc(url: str, title: str, summary: str) -> None:
        key = f"{url}|{title}|{summary}"
        if key in seen:
            return
        seen.add(key)
        docs.append(
            {
                "url": _truncate_text(url, max_len=280),
                "title": _truncate_text(title, max_len=160),
                "summary": _truncate_text(summary, max_len=420),
            }
        )

    def _from_mapping(item: dict[str, Any]) -> None:
        url = str(item.get("url") or item.get("link") or "").strip()
        title = str(
            item.get("title")
            or item.get("title_cn")
            or item.get("headline")
            or item.get("name")
            or ""
        ).strip()
        summary = str(
            item.get("summary")
            or item.get("snippet")
            or item.get("content")
            or item.get("raw_output")
            or ""
        ).strip()
        if not (url or title or summary):
            return
        _append_doc(url, title, summary)

    if isinstance(data, dict):
        for key in ("records", "selected", "articles", "items"):
            value = data.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict):
                    _from_mapping(item)
                if len(docs) >= _MAX_CONTEXT_DOCS:
                    return docs

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _from_mapping(item)
            if len(docs) >= _MAX_CONTEXT_DOCS:
                return docs

    return docs


def summarize_envelope(envelope: SkillEnvelope) -> dict[str, Any]:
    """Build a compact summary from a SkillEnvelope."""
    summary: dict[str, Any] = {
        "status": envelope.status,
        "evidence_count": len(envelope.evidence),
    }

    data = envelope.data
    if isinstance(data, dict):
        summary["data_type"] = "dict"
        summary["data_keys"] = sorted(data.keys())[:_MAX_DICT_ITEMS]
        records = data.get("records")
        if isinstance(records, list):
            summary["records_count"] = len(records)
        raw_output = data.get("raw_output")
        if isinstance(raw_output, str):
            summary["raw_output_chars"] = len(raw_output)
    elif isinstance(data, list):
        summary["data_type"] = "list"
        summary["items_count"] = len(data)
    elif data is None:
        summary["data_type"] = "none"
    else:
        summary["data_type"] = type(data).__name__

    context_docs = _extract_context_docs(data)
    if context_docs:
        summary["context_docs"] = context_docs
        summary["context_count"] = len(context_docs)

    if envelope.error:
        summary["error"] = _truncate_text(envelope.error)
    if envelope.diagnostics:
        summary["diagnostics"] = summarize_payload(envelope.diagnostics)
    return summary


def build_exception_chain(error: BaseException | None) -> list[dict[str, str]]:
    """Capture causal/context exception chain in a bounded list."""
    if error is None:
        return []

    chain: list[dict[str, str]] = []
    visited: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in visited and len(chain) < _MAX_ERROR_CHAIN:
        visited.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": _truncate_text(str(current), max_len=400),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _normalize_usage_dict(payload: Any) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None

    def _to_int(candidate: Any) -> int | None:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    prompt = _to_int(
        payload.get("prompt_tokens")
        or payload.get("input_tokens")
        or payload.get("input_token_count")
    )
    completion = _to_int(
        payload.get("completion_tokens")
        or payload.get("output_tokens")
        or payload.get("output_token_count")
        or payload.get("candidates_token_count")
    )
    total = _to_int(payload.get("total_tokens") or payload.get("total_token_count"))

    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)

    normalized: dict[str, int] = {}
    if prompt is not None:
        normalized["prompt_tokens"] = prompt
    if completion is not None:
        normalized["completion_tokens"] = completion
    if total is not None:
        normalized["total_tokens"] = total
    return normalized or None


def _iter_usage_payloads(message: Any) -> Iterator[dict[str, Any]]:
    direct = getattr(message, "usage_metadata", None)
    if isinstance(direct, dict):
        yield direct

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("usage_metadata", "token_usage", "usage"):
            candidate = response_metadata.get(key)
            if isinstance(candidate, dict):
                yield candidate
        # Some integrations place fields directly in response_metadata.
        yield response_metadata

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        for key in ("usage_metadata", "token_usage", "usage"):
            candidate = additional_kwargs.get(key)
            if isinstance(candidate, dict):
                yield candidate


def extract_token_usage(messages: list[Any]) -> dict[str, int] | None:
    """Aggregate token usage from message metadata."""
    prompt_total = 0
    completion_total = 0
    total_total = 0
    seen = False

    for message in messages:
        for payload in _iter_usage_payloads(message):
            normalized = _normalize_usage_dict(payload)
            if not normalized:
                continue
            seen = True
            prompt_total += normalized.get("prompt_tokens", 0)
            completion_total += normalized.get("completion_tokens", 0)
            total_total += normalized.get("total_tokens", 0)

    if not seen:
        return None

    if total_total == 0:
        total_total = prompt_total + completion_total

    result: dict[str, int] = {}
    if prompt_total > 0:
        result["prompt_tokens"] = prompt_total
    if completion_total > 0:
        result["completion_tokens"] = completion_total
    if total_total > 0:
        result["total_tokens"] = total_total
    return result or None


@dataclass
class AgentToolEvent:
    """Trace entry for a single tool call."""

    event_index: int
    tool_name: str
    input_summary: Any
    started_at_ms: int
    status: str = "started"
    finished_at_ms: int | None = None
    latency_ms: int | None = None
    output_summary: Any = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    exception_chain: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "tool_name": self.tool_name,
            "status": self.status,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "exception_chain": copy.deepcopy(self.exception_chain),
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
        }


@dataclass
class AgentRequestTrace:
    """Request-level trace for one agent run."""

    request_id: str
    user_message: str
    thread_id: str | None = None
    started_at_ms: int = field(default_factory=_now_ms)
    finished_at_ms: int | None = None
    latency_ms: int | None = None
    final_status: str = "running"
    evidence_count: int = 0
    token_usage: dict[str, int] | None = None
    error_code: str | None = None
    error_message: str | None = None
    exception_chain: list[dict[str, str]] = field(default_factory=list)
    tool_events: list[AgentToolEvent] = field(default_factory=list)
    tool_call_chain: list[str] = field(default_factory=list)
    final_answer_metadata: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    _finalized: bool = False

    def start_tool(self, tool_name: str, payload: dict[str, Any]) -> int:
        event_index = len(self.tool_events) + 1
        event = AgentToolEvent(
            event_index=event_index,
            tool_name=str(tool_name or "").strip(),
            input_summary=summarize_payload(payload),
            started_at_ms=_now_ms(),
        )
        self.tool_events.append(event)
        self.tool_call_chain.append(event.tool_name)
        return event_index

    def finish_tool(
        self,
        event_index: int,
        *,
        status: str,
        output_summary: Any = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        if event_index <= 0 or event_index > len(self.tool_events):
            return
        event = self.tool_events[event_index - 1]
        event.finished_at_ms = _now_ms()
        event.latency_ms = max(0, event.finished_at_ms - event.started_at_ms)
        event.status = str(status or "unknown")
        event.output_summary = summarize_payload(output_summary)
        event.error_code = str(error_code) if error_code else None
        event.error_message = _truncate_text(error_message) if error_message else None
        event.exception_chain = build_exception_chain(error) if error is not None else []

        if isinstance(output_summary, dict):
            maybe_evidence = output_summary.get("evidence_count")
            try:
                if maybe_evidence is not None:
                    self.evidence_count += max(0, int(maybe_evidence))
            except (TypeError, ValueError):
                pass

    def finalize(
        self,
        *,
        final_status: str,
        evidence_count: int | None = None,
        token_usage: dict[str, int] | None = None,
        final_answer_metadata: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        if not self._finalized:
            self.finished_at_ms = _now_ms()
            self.latency_ms = max(0, self.finished_at_ms - self.started_at_ms)
            self.final_status = str(final_status or "unknown")
            if evidence_count is not None:
                self.evidence_count = max(0, int(evidence_count))
            if token_usage is not None:
                self.token_usage = dict(token_usage)
            if final_answer_metadata is not None:
                summarized_meta = summarize_payload(final_answer_metadata)
                if isinstance(summarized_meta, dict):
                    self.final_answer_metadata = summarized_meta
                else:
                    self.final_answer_metadata = {"value": summarized_meta}
            self.error_code = str(error_code) if error_code else None
            self.error_message = _truncate_text(error_message, max_len=400) if error_message else None
            self.exception_chain = build_exception_chain(error)
            self._finalized = True
        return self.to_summary()

    def to_summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "user_message": self.user_message,
            "latency_ms": self.latency_ms,
            "evidence_count": self.evidence_count,
            "final_status": self.final_status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "exception_chain": copy.deepcopy(self.exception_chain),
            "token_usage": copy.deepcopy(self.token_usage),
            "tool_call_chain": list(self.tool_call_chain),
            "tool_events": [event.to_dict() for event in self.tool_events],
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "final_answer_metadata": copy.deepcopy(self.final_answer_metadata),
            "runtime": copy.deepcopy(self.runtime),
        }


_TRACE_STATE: ContextVar[AgentRequestTrace | None] = ContextVar(
    "agent_request_trace_state",
    default=None,
)
_LAST_TRACE_LOCK = Lock()
_LAST_TRACE_SUMMARY: dict[str, Any] | None = None


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _build_runtime_metadata() -> dict[str, Any]:
    provider = os.getenv("AGENT_MODEL_PROVIDER", "gemini_api").strip().lower() or "gemini_api"
    if provider in {"vertex", "vertex_ai", "gcp"}:
        model = os.getenv(
            "VERTEX_GENERATION_MODEL",
            os.getenv("VERTEX_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")),
        ).strip()
    else:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro").strip()
    return {
        "route": "react",
        "provider": provider,
        "model": model,
    }


@contextmanager
def request_trace_context(
    *,
    user_message: str,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> Iterator[AgentRequestTrace]:
    """Bind a request-level trace to current execution context."""
    trace = AgentRequestTrace(
        request_id=str(request_id or _new_request_id()),
        user_message=str(user_message or ""),
        thread_id=str(thread_id) if thread_id else None,
        runtime=_build_runtime_metadata(),
    )
    token: Token[AgentRequestTrace | None] = _TRACE_STATE.set(trace)
    try:
        yield trace
    finally:
        _TRACE_STATE.reset(token)


def get_current_request_trace() -> AgentRequestTrace | None:
    return _TRACE_STATE.get()


def get_current_request_id() -> str | None:
    trace = get_current_request_trace()
    return trace.request_id if trace else None


def get_current_thread_id() -> str | None:
    trace = get_current_request_trace()
    return trace.thread_id if trace else None


def trace_tool_start(tool_name: str, payload: dict[str, Any]) -> int | None:
    trace = get_current_request_trace()
    if trace is None:
        return None
    return trace.start_tool(tool_name, payload)


def trace_tool_finish_with_envelope(
    event_index: int | None,
    envelope: SkillEnvelope,
    *,
    status_override: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    trace = get_current_request_trace()
    if trace is None or event_index is None:
        return

    status = status_override or ("success" if envelope.status == "ok" else envelope.status)
    trace.finish_tool(
        event_index,
        status=status,
        output_summary=summarize_envelope(envelope),
        error_code=error_code,
        error_message=error_message,
    )

    from .metrics import metrics_inc

    metrics_inc("trace_tool_calls_total")
    if status == "success":
        metrics_inc("trace_tool_success")
    elif status == "empty":
        metrics_inc("trace_tool_empty")
    elif status in {"error", "blocked"}:
        metrics_inc("trace_tool_error")


def trace_tool_finish_error(
    event_index: int | None,
    *,
    error_code: str,
    error_message: str,
    error: BaseException,
) -> None:
    trace = get_current_request_trace()
    if trace is None or event_index is None:
        return

    trace.finish_tool(
        event_index,
        status="error",
        output_summary={},
        error_code=error_code,
        error_message=error_message,
        error=error,
    )

    from .metrics import metrics_inc

    metrics_inc("trace_tool_calls_total")
    metrics_inc("trace_tool_error")


def set_request_token_usage(token_usage: dict[str, int] | None) -> None:
    trace = get_current_request_trace()
    if trace is None or token_usage is None:
        return
    trace.token_usage = dict(token_usage)


def finalize_request_trace(
    *,
    final_status: str,
    evidence_count: int | None = None,
    token_usage: dict[str, int] | None = None,
    final_answer_metadata: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    error: BaseException | None = None,
) -> dict[str, Any] | None:
    """Finalize and publish request-level summary for current context."""
    trace = get_current_request_trace()
    if trace is None:
        return None

    was_finalized = trace._finalized
    summary = trace.finalize(
        final_status=final_status,
        evidence_count=evidence_count,
        token_usage=token_usage,
        final_answer_metadata=final_answer_metadata,
        error_code=error_code,
        error_message=error_message,
        error=error,
    )

    with _LAST_TRACE_LOCK:
        global _LAST_TRACE_SUMMARY
        _LAST_TRACE_SUMMARY = copy.deepcopy(summary)

    if was_finalized:
        return summary

    from .metrics import metrics_inc

    metrics_inc("trace_runs_total")
    if final_status == "success":
        metrics_inc("trace_success")
    else:
        metrics_inc("trace_error")

    try:
        if not _persist_request_trace(summary):
            logger.warning(
                "request trace persistence skipped/failed: request_id=%s",
                summary.get("request_id"),
            )
    except Exception as exc:
        logger.warning(
            "request trace persistence exception: request_id=%s error=%s",
            summary.get("request_id"),
            exc,
        )

    print(f"[AgentTrace] {json.dumps(summary, ensure_ascii=False)}")
    return summary


def get_last_trace_summary() -> dict[str, Any] | None:
    with _LAST_TRACE_LOCK:
        if _LAST_TRACE_SUMMARY is None:
            return None
        return copy.deepcopy(_LAST_TRACE_SUMMARY)


def _persist_request_trace(summary: dict[str, Any]) -> bool:
    """Persist trace summary via storage adapter (isolated for tests)."""
    from services.agent_trace_store import persist_request_trace

    return persist_request_trace(summary)
