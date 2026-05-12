"""Framework-independent tool runtime.

ToolRuntime is the single execution boundary for project tools. LangChain,
MCP, eval, and future LangGraph nodes should adapt to this runtime instead of
calling individual handlers directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .run_context import add_tool_call, emit_progress
from .tool_contracts import ToolEnvelope, build_tool_error_envelope
from .tool_registry import ToolRegistry
from .tool_runtime_hooks import ToolRuntimeHooks
from .metrics import metrics_inc
from .trace import (
    summarize_envelope,
    summarize_payload,
    trace_span,
)


@dataclass(frozen=True)
class ToolRuntimeContext:
    """Execution flags for adapters and graph nodes."""

    trace: bool = True
    emit_progress_events: bool = True
    accumulate_tool_call: bool = True


class ToolRuntime:
    """Execute tools through validation, hooks, tracing, and evidence capture."""

    def __init__(self, registry: ToolRegistry, hooks: ToolRuntimeHooks) -> None:
        self.registry = registry
        self.hooks = hooks

    def execute(
        self,
        tool_name: str,
        payload: dict[str, Any] | None = None,
        context: ToolRuntimeContext | None = None,
    ) -> ToolEnvelope:
        runtime_context = context or ToolRuntimeContext()
        request_payload = dict(payload or {})
        normalized_name = str(tool_name or "").strip()
        span_cm = (
            trace_span(
                "tool_call",
                normalized_name,
                input_summary={"tool": normalized_name, "args": request_payload},
                metadata={"tool_name": normalized_name},
            )
            if runtime_context.trace
            else None
        )
        tool_span = span_cm.__enter__() if span_cm is not None else None

        try:
            pre = self.hooks.pre_tool_use(normalized_name, request_payload)
            if pre.action == "deny":
                reason = pre.reason or "pre-hook denied"
                envelope = build_tool_error_envelope(
                    tool=normalized_name,
                    request=request_payload,
                    error=reason,
                    error_code="tool_pre_hook_denied",
                    diagnostics={
                        "hook": "pre_tool_use",
                        "hook_reason": reason,
                        **pre.diagnostics,
                    },
                )
                if tool_span is not None:
                    tool_span.set_output(_tool_span_output(envelope, status="blocked"))
                    tool_span.status = "blocked"
                _record_tool_trace_metrics("blocked", enabled=runtime_context.trace)
                return envelope

            effective_payload = pre.updated_payload if pre.updated_payload is not None else request_payload
            if tool_span is not None and effective_payload != request_payload:
                tool_span.metadata["effective_payload"] = summarize_payload(effective_payload)
            if runtime_context.emit_progress_events:
                emit_progress(
                    _progress_stage(normalized_name),
                    normalized_name,
                    title=f"调用{normalized_name}",
                    status="running",
                    event="progress",
                )

            envelope = self.registry.execute(normalized_name, effective_payload)
            if runtime_context.accumulate_tool_call:
                add_tool_call(normalized_name)

            post = self.hooks.post_tool_use(normalized_name, effective_payload, envelope)
            if post.action == "deny":
                reason = post.reason or "post-hook denied"
                blocked = build_tool_error_envelope(
                    tool=normalized_name,
                    request=effective_payload,
                    error=reason,
                    error_code="tool_post_hook_denied",
                    diagnostics={
                        "hook": "post_tool_use",
                        "hook_reason": reason,
                        "original_status": envelope.status,
                        **post.diagnostics,
                    },
                    data=envelope.data,
                )
                if tool_span is not None:
                    tool_span.set_output(_tool_span_output(blocked, status="blocked"))
                    tool_span.status = "blocked"
                _record_tool_trace_metrics("blocked", enabled=runtime_context.trace)
                return blocked

            if post.action == "warn":
                warnings = envelope.diagnostics.setdefault("hook_warnings", [])
                if isinstance(warnings, list) and post.reason:
                    warnings.append(post.reason)
                if post.diagnostics:
                    hook_diagnostics = envelope.diagnostics.setdefault("hook_diagnostics", {})
                    if isinstance(hook_diagnostics, dict):
                        hook_diagnostics.update(post.diagnostics)

            if tool_span is not None:
                tool_span.set_output(_tool_span_output(envelope))
                tool_span.status = "success" if envelope.status == "ok" else envelope.status
            _record_tool_trace_metrics(
                "success" if envelope.status == "ok" else envelope.status,
                enabled=runtime_context.trace,
            )
            return envelope
        except Exception as exc:  # noqa: BLE001
            if tool_span is not None:
                tool_span.set_error(
                    error_code=f"tool_{type(exc).__name__.lower()}",
                    error_message=str(exc),
                    error=exc,
                )
            envelope = build_tool_error_envelope(
                tool=normalized_name,
                request=request_payload,
                error="tool_runtime_failed",
                error_code="tool_runtime_failed",
                diagnostics={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            if tool_span is not None:
                tool_span.set_output(_tool_span_output(envelope, status="error"))
            _record_tool_trace_metrics("error", enabled=runtime_context.trace)
            return envelope
        finally:
            if span_cm is not None:
                span_cm.__exit__(None, None, None)


def _progress_stage(tool_name: str) -> str:
    if tool_name in {"compare_sources", "compare_topics", "build_timeline", "analyze_landscape"}:
        return "analyzing"
    return "retrieving"


def _record_tool_trace_metrics(status: str, *, enabled: bool) -> None:
    if not enabled:
        return
    metrics_inc("trace_tool_calls_total")
    if status == "success":
        metrics_inc("trace_tool_success")
    elif status == "empty":
        metrics_inc("trace_tool_empty")
    elif status in {"error", "blocked"}:
        metrics_inc("trace_tool_error")


def _tool_span_output(envelope: ToolEnvelope, *, status: str | None = None) -> dict[str, Any]:
    evidence_urls = [
        str(item.url).strip()
        for item in (envelope.evidence or [])
        if str(item.url or "").strip()
    ]
    summary = summarize_envelope(envelope)
    summary.update(
        {
            "tool": envelope.tool,
            "status": status or ("success" if envelope.status == "ok" else envelope.status),
            "error_code": envelope.error_code,
            "evidence_urls": evidence_urls,
            "diagnostics": envelope.diagnostics or {},
        }
    )
    return summary


def format_tool_envelope_for_model(envelope: ToolEnvelope) -> str:
    """Convert a ToolEnvelope to compact text for model-facing adapters."""

    if envelope.error_code in {"tool_pre_hook_denied", "tool_post_hook_denied"}:
        return f"[Blocked] {envelope.error or envelope.error_code}"
    if envelope.status == "error":
        return f"[Error] {envelope.error or 'tool execution failed'}"
    if envelope.status == "empty":
        return "No matching records found."

    data = envelope.data
    if isinstance(data, dict):
        raw = data.get("raw_output")
        if isinstance(raw, str) and raw.strip():
            return raw

        records = data.get("records")
        if isinstance(records, list) and records:
            lines = []
            for idx, item in enumerate(records[:15], 1):
                if not isinstance(item, dict):
                    continue
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
            if lines:
                return "\n".join(lines)

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
            if isinstance(daily, list) and daily:
                lines.append("Daily breakdown:")
                for item in daily[:14]:
                    if isinstance(item, dict):
                        lines.append(f"  {item.get('day', '')}: count={item.get('count', 0)}")
            return "\n".join(lines)

    return str(data) if data else "No data returned."


def format_tool_results_for_final_synthesis(envelopes: list[ToolEnvelope]) -> str:
    """Format several tool envelopes as grounded context for a final synthesis node."""

    blocks: list[str] = []
    for envelope in envelopes:
        blocks.append(
            f"Tool: {envelope.tool}\n"
            f"Status: {envelope.status}\n"
            f"Evidence count: {len(envelope.evidence)}\n"
            f"{format_tool_envelope_for_model(envelope)}"
        )
    return "\n\n".join(blocks).strip()
