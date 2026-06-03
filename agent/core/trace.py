"""Request-level tracing utilities for agent runtime."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator

from services.llm_provider import agent_runtime_metadata

from .tool_contracts import ToolEnvelope

_MAX_SUMMARY_DEPTH = 2
_MAX_STR_LEN = 240
_MAX_LIST_ITEMS = 6
_MAX_DICT_ITEMS = 12
_MAX_ERROR_CHAIN = 6
_MAX_CONTEXT_DOCS = 8
logger = logging.getLogger(__name__)

_SPAN_TYPES = {"graph_node", "model_call", "tool_call", "guard", "postprocess", "context"}
_PROVIDER_INTERNAL_PAYLOAD_KEYS = {"thought_signature"}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[a-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)\b([a-z0-9_.-]*(?:api[_-]?key|x-api-key|access[_-]?token|refresh[_-]?token|smtp[_-]?password|password))"
        r"\s*[:=]\s*['\"]?([^'\"\s,;]{8,})"
    ),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s'\"<>]+"),
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def full_model_io_enabled() -> bool:
    """Return whether full model input/output persistence is enabled."""
    return _env_flag("AGENT_TRACE_FULL_MODEL_IO", default=True)


def secret_redaction_enabled() -> bool:
    """Return whether credential-like values should be redacted before storage."""
    return _env_flag("AGENT_TRACE_SECRET_REDACTION", default=True)


def _langsmith_metadata() -> dict[str, Any]:
    enabled = any(
        _env_flag(name, default=False)
        for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2")
    )
    return {
        "enabled": enabled,
        "project": os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "",
        "endpoint": os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "",
    }


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


def _json_safe_full(value: Any, *, seen: set[int] | None = None) -> Any:
    """Build a JSON-safe copy without truncating business text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return f"<circular:{type(value).__name__}>"
    seen.add(obj_id)

    if isinstance(value, dict):
        return {str(key): _json_safe_full(item, seen=seen) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_full(item, seen=seen) for item in value]

    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return repr(value)


def _provider_internal_placeholder(key: str, value: Any) -> str:
    try:
        length = len(str(value or ""))
    except Exception:
        length = 0
    return f"[provider_internal_{key}_omitted chars={length}]"


def _strip_provider_internal_payload(value: Any) -> Any:
    """Remove provider-internal binary/signature payloads from trace display/storage."""
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = str(key)
            if clean_key in _PROVIDER_INTERNAL_PAYLOAD_KEYS:
                stripped[clean_key] = _provider_internal_placeholder(clean_key, item)
            else:
                stripped[clean_key] = _strip_provider_internal_payload(item)
        return stripped
    if isinstance(value, list):
        return [_strip_provider_internal_payload(item) for item in value]
    return value


def _redact_text(text: str) -> str:
    redacted = text
    redacted = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[1].sub(r"\1=[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[2].sub("[REDACTED_DB_URL]", redacted)
    return redacted


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    return value


def _redact_if_enabled(value: Any) -> Any:
    safe = _json_safe_full(value)
    if not secret_redaction_enabled():
        return safe
    return _redact_value(safe)


def _serialize_message_full(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": str(getattr(message, "type", type(message).__name__)),
        "class": type(message).__name__,
        "content": _json_safe_full(getattr(message, "content", message)),
    }
    for attr in (
        "name",
        "id",
        "additional_kwargs",
        "response_metadata",
        "usage_metadata",
        "tool_calls",
        "invalid_tool_calls",
    ):
        if hasattr(message, attr):
            value = getattr(message, attr)
            if value not in (None, "", [], {}):
                payload[attr] = _json_safe_full(value)
    return _strip_provider_internal_payload(payload)


def _serialize_messages_full(messages: Any) -> Any:
    if isinstance(messages, (list, tuple)):
        return [_serialize_message_full(item) for item in messages]
    return _json_safe_full(messages)


def _serialize_raw_model_output(raw_output: Any) -> Any:
    if hasattr(raw_output, "content") or hasattr(raw_output, "response_metadata"):
        return _serialize_message_full(raw_output)
    return _strip_provider_internal_payload(_json_safe_full(raw_output))


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


def summarize_envelope(envelope: ToolEnvelope) -> dict[str, Any]:
    """Build a compact summary from a ToolEnvelope."""
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
class TraceSpan:
    """Trace entry for one graph, model, tool, guard, or postprocess step."""

    span_id: str
    parent_span_id: str | None
    request_id: str
    span_type: str
    name: str
    started_at_ms: int
    status: str = "running"
    finished_at_ms: int | None = None
    latency_ms: int | None = None
    input_summary: Any = field(default_factory=dict)
    output_summary: Any = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    exception_chain: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_output(self, output_summary: Any) -> None:
        summarized = summarize_payload(output_summary)
        if isinstance(output_summary, dict) and isinstance(summarized, dict):
            evidence_urls = output_summary.get("evidence_urls")
            if isinstance(evidence_urls, list):
                summarized["evidence_urls"] = [
                    str(item).strip()
                    for item in evidence_urls
                    if str(item).strip()
                ]
            diagnostics = output_summary.get("diagnostics")
            if self.span_type == "tool_call" and isinstance(diagnostics, dict):
                summarized["diagnostics"] = _redact_if_enabled(diagnostics)
        self.output_summary = summarized

    def set_error(
        self,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.status = "error"
        self.error_code = str(error_code) if error_code else self.error_code
        message = error_message if error_message is not None else (str(error) if error else None)
        self.error_message = _truncate_text(message, max_len=400) if message else self.error_message
        if error is not None:
            self.exception_chain = build_exception_chain(error)

    def finish(self, *, status: str | None = None) -> None:
        if self.finished_at_ms is not None:
            return
        self.finished_at_ms = _now_ms()
        self.latency_ms = max(0, self.finished_at_ms - self.started_at_ms)
        if status:
            self.status = str(status)
        elif self.status == "running":
            self.status = "success"

    def set_model_io(
        self,
        *,
        node: str,
        provider: str,
        model: str,
        input_messages: Any,
        raw_output: Any,
        parsed_output: Any | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        set_model_io(
            span_id=self.span_id,
            node=node,
            provider=provider,
            model=model,
            input_messages=input_messages,
            raw_output=raw_output,
            parsed_output=parsed_output,
            token_usage=token_usage,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "request_id": self.request_id,
            "span_type": self.span_type,
            "name": self.name,
            "status": self.status,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "latency_ms": self.latency_ms,
            "input_summary": copy.deepcopy(self.input_summary),
            "output_summary": copy.deepcopy(self.output_summary),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "exception_chain": copy.deepcopy(self.exception_chain),
            "metadata": copy.deepcopy(self.metadata),
        }


class NullTraceSpan:
    """No-op span used when tracing is not bound."""

    span_id = ""
    status = "success"

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    def set_output(self, output_summary: Any) -> None:
        return None

    def set_error(
        self,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        return None

    def finish(self, *, status: str | None = None) -> None:
        return None

    def set_model_io(
        self,
        *,
        node: str,
        provider: str,
        model: str,
        input_messages: Any,
        raw_output: Any,
        parsed_output: Any | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        return None


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
    spans: list[TraceSpan] = field(default_factory=list)
    model_io: list[dict[str, Any]] = field(default_factory=list)
    tool_call_chain: list[str] = field(default_factory=list)
    final_answer_metadata: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    _finalized: bool = False

    def start_span(
        self,
        *,
        span_type: str,
        name: str,
        parent_span_id: str | None = None,
        input_summary: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        normalized_type = str(span_type or "").strip()
        if normalized_type not in _SPAN_TYPES:
            normalized_type = "postprocess"
        summarized_input = summarize_payload(input_summary)
        if (
            normalized_type == "tool_call"
            and isinstance(input_summary, dict)
            and isinstance(summarized_input, dict)
            and isinstance(input_summary.get("args"), dict)
        ):
            summarized_input["args"] = _redact_if_enabled(input_summary.get("args"))
        clean_name = str(name or "").strip() or normalized_type
        span = TraceSpan(
            span_id=uuid.uuid4().hex,
            parent_span_id=str(parent_span_id).strip() if parent_span_id else None,
            request_id=self.request_id,
            span_type=normalized_type,
            name=clean_name,
            started_at_ms=_now_ms(),
            input_summary=summarized_input,
            metadata=summarize_payload(metadata) if isinstance(metadata, dict) else {},
        )
        self.spans.append(span)
        if normalized_type == "tool_call" and clean_name:
            self.tool_call_chain.append(clean_name)
        return span

    def add_model_io(
        self,
        *,
        span_id: str,
        node: str,
        provider: str,
        model: str,
        input_messages: Any,
        raw_output: Any,
        parsed_output: Any | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        if not full_model_io_enabled():
            return
        row: dict[str, Any] = {
            "request_id": self.request_id,
            "span_id": str(span_id or "").strip(),
            "node": str(node or "").strip(),
            "provider": str(provider or "").strip(),
            "model": str(model or "").strip(),
            "input_messages": _redact_if_enabled(_serialize_messages_full(input_messages)),
            "raw_output": _redact_if_enabled(_serialize_raw_model_output(raw_output)),
            "parsed_output": _redact_if_enabled(parsed_output) if parsed_output is not None else None,
            "token_usage": copy.deepcopy(token_usage) if isinstance(token_usage, dict) else None,
            "created_at_ms": _now_ms(),
        }
        self.model_io.append(row)

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
            "spans": [span.to_dict() for span in self.spans],
            "model_io": copy.deepcopy(self.model_io),
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "final_answer_metadata": copy.deepcopy(self.final_answer_metadata),
            "runtime": copy.deepcopy(self.runtime),
        }


_TRACE_STATE: ContextVar[AgentRequestTrace | None] = ContextVar(
    "agent_request_trace_state",
    default=None,
)
_SPAN_STACK: ContextVar[tuple[str, ...]] = ContextVar("agent_trace_span_stack", default=())
_LAST_TRACE_LOCK = Lock()
_LAST_TRACE_SUMMARY: dict[str, Any] | None = None


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _build_runtime_metadata() -> dict[str, Any]:
    metadata = dict(agent_runtime_metadata())
    metadata["trace"] = {
        "full_model_io_enabled": full_model_io_enabled(),
        "secret_redaction_enabled": secret_redaction_enabled(),
    }
    metadata["langsmith"] = _langsmith_metadata()
    return metadata


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
    span_token: Token[tuple[str, ...]] = _SPAN_STACK.set(())
    try:
        yield trace
    finally:
        _SPAN_STACK.reset(span_token)
        _TRACE_STATE.reset(token)


def get_current_request_trace() -> AgentRequestTrace | None:
    return _TRACE_STATE.get()


def get_current_request_id() -> str | None:
    trace = get_current_request_trace()
    return trace.request_id if trace else None


def get_current_thread_id() -> str | None:
    trace = get_current_request_trace()
    return trace.thread_id if trace else None


def _is_control_flow_signal(exc: BaseException) -> bool:
    """LangGraph interrupt/command exceptions are control flow, not errors."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {"GraphBubbleUp", "GraphInterrupt", "NodeInterrupt", "ParentCommand"})


@contextmanager
def trace_span(
    span_type: str,
    name: str,
    *,
    input_summary: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[TraceSpan | NullTraceSpan]:
    """Create a child span under the current request trace."""
    trace = get_current_request_trace()
    if trace is None:
        yield NullTraceSpan()
        return

    stack = _SPAN_STACK.get()
    parent_span_id = stack[-1] if stack else None
    span = trace.start_span(
        span_type=span_type,
        name=name,
        parent_span_id=parent_span_id,
        input_summary=input_summary,
        metadata=metadata,
    )
    token = _SPAN_STACK.set((*stack, span.span_id))
    try:
        yield span
    except Exception as exc:
        if not _is_control_flow_signal(exc):
            span.set_error(
                error_code=f"{span.span_type}_{type(exc).__name__.lower()}",
                error_message=str(exc),
                error=exc,
            )
        raise
    finally:
        span.finish()
        _SPAN_STACK.reset(token)


def set_model_io(
    *,
    span_id: str,
    node: str,
    provider: str,
    model: str,
    input_messages: Any,
    raw_output: Any,
    parsed_output: Any | None = None,
    token_usage: dict[str, int] | None = None,
) -> None:
    """Persist full model input/output payload for the current request."""
    trace = get_current_request_trace()
    if trace is None:
        return
    trace.add_model_io(
        span_id=span_id,
        node=node,
        provider=provider,
        model=model,
        input_messages=input_messages,
        raw_output=raw_output,
        parsed_output=parsed_output,
        token_usage=token_usage,
    )


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
    public_summary = _compact_trace_summary(summary)

    with _LAST_TRACE_LOCK:
        global _LAST_TRACE_SUMMARY
        _LAST_TRACE_SUMMARY = copy.deepcopy(public_summary)

    if was_finalized:
        return public_summary

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

    print(f"[AgentTrace] {json.dumps(public_summary, ensure_ascii=False)}")
    return public_summary


def get_last_trace_summary() -> dict[str, Any] | None:
    with _LAST_TRACE_LOCK:
        if _LAST_TRACE_SUMMARY is None:
            return None
        return copy.deepcopy(_LAST_TRACE_SUMMARY)


def _compact_trace_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a public trace summary without full model prompts/outputs."""
    compact = copy.deepcopy(summary)
    model_io = compact.get("model_io")
    if isinstance(model_io, list):
        compact["model_io"] = [
            {
                "span_id": item.get("span_id"),
                "node": item.get("node"),
                "provider": item.get("provider"),
                "model": item.get("model"),
                "token_usage": item.get("token_usage"),
            }
            for item in model_io
            if isinstance(item, dict)
        ]
    return compact


def _persist_request_trace(summary: dict[str, Any]) -> bool:
    """Persist trace summary via storage adapter (isolated for tests)."""
    from services.agent_trace_store import persist_request_trace

    return persist_request_trace(summary)
