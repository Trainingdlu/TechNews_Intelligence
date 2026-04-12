"""Route metrics helpers for agent runtime."""

from __future__ import annotations

import os
from threading import Lock

_route_metrics_lock = Lock()
_route_metrics: dict[str, int] = {
    "requests_total": 0,
    "react_attempts": 0,
    "react_success": 0,
    "react_error": 0,
    "react_recursion_limit_hit": 0,
    "trace_runs_total": 0,
    "trace_success": 0,
    "trace_error": 0,
    "trace_tool_calls_total": 0,
    "trace_tool_success": 0,
    "trace_tool_empty": 0,
    "trace_tool_error": 0,
}


def metrics_enabled() -> bool:
    return os.getenv("AGENT_ROUTE_METRICS", "true").strip().lower() not in {"0", "false", "no", "off"}


def metrics_log_every() -> int:
    try:
        return max(1, int(os.getenv("AGENT_ROUTE_LOG_EVERY", "20")))
    except Exception:
        return 20


def metrics_inc(key: str, amount: int = 1) -> None:
    if not metrics_enabled():
        return
    with _route_metrics_lock:
        _route_metrics[key] = _route_metrics.get(key, 0) + amount

    # Auto-emit on error events or every N requests
    if key in {"react_error", "react_recursion_limit_hit"}:
        emit_route_metrics(key, force=True)
    elif key == "requests_total":
        with _route_metrics_lock:
            total = _route_metrics.get("requests_total", 0)
        if total % metrics_log_every() == 0:
            emit_route_metrics("periodic")


def emit_route_metrics(route_event: str, force: bool = False) -> None:
    if not metrics_enabled():
        return

    with _route_metrics_lock:
        snapshot = dict(_route_metrics)

    total = max(1, snapshot.get("requests_total", 0))
    attempts = snapshot.get("react_attempts", 0)
    success = snapshot.get("react_success", 0)
    errors = snapshot.get("react_error", 0)
    recursion_hits = snapshot.get("react_recursion_limit_hit", 0)
    trace_runs = snapshot.get("trace_runs_total", 0)
    trace_errors = snapshot.get("trace_error", 0)
    trace_tools = snapshot.get("trace_tool_calls_total", 0)
    trace_tool_errors = snapshot.get("trace_tool_error", 0)
    success_rate = (success / attempts) if attempts else 0.0
    error_rate = (errors / total)
    trace_error_rate = (trace_errors / trace_runs) if trace_runs else 0.0
    trace_tool_error_rate = (trace_tool_errors / trace_tools) if trace_tools else 0.0

    print(
        "[Metrics] "
        f"event={route_event} "
        f"total={total} "
        f"react_attempts={attempts} "
        f"react_success={success} "
        f"react_error={errors} "
        f"react_recursion_limit_hit={recursion_hits} "
        f"trace_runs_total={trace_runs} "
        f"trace_error={trace_errors} "
        f"trace_tool_calls_total={trace_tools} "
        f"trace_tool_error={trace_tool_errors} "
        f"success_rate={success_rate:.1%} "
        f"error_rate={error_rate:.1%} "
        f"trace_error_rate={trace_error_rate:.1%} "
        f"trace_tool_error_rate={trace_tool_error_rate:.1%}"
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
    attempts = int(snapshot.get("react_attempts", 0))
    success = int(snapshot.get("react_success", 0))
    errors = int(snapshot.get("react_error", 0))
    recursion_hits = int(snapshot.get("react_recursion_limit_hit", 0))
    trace_runs = int(snapshot.get("trace_runs_total", 0))
    trace_errors = int(snapshot.get("trace_error", 0))
    trace_tools = int(snapshot.get("trace_tool_calls_total", 0))
    trace_tool_errors = int(snapshot.get("trace_tool_error", 0))

    snapshot["success_rate"] = (success / attempts) if attempts else 0.0
    snapshot["error_rate"] = errors / total
    snapshot["react_success_rate"] = (success / attempts) if attempts else 0.0
    snapshot["react_error_rate"] = (errors / attempts) if attempts else 0.0
    snapshot["react_recursion_limit_rate"] = (
        recursion_hits / attempts
    ) if attempts else 0.0
    snapshot["trace_error_rate"] = (
        trace_errors / trace_runs
    ) if trace_runs else 0.0
    snapshot["trace_tool_error_rate"] = (
        trace_tool_errors / trace_tools
    ) if trace_tools else 0.0
    return snapshot
