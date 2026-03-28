"""Route metrics helpers for agent runtime."""

from __future__ import annotations

import os
from threading import Lock

_route_metrics_lock = Lock()
_route_metrics: dict[str, int] = {
    "requests_total": 0,
    "source_compare_forced": 0,
    "compare_forced": 0,
    "timeline_forced": 0,
    "landscape_forced": 0,
    "trend_forced": 0,
    "fulltext_forced": 0,
    "query_forced": 0,
    "landscape_low_evidence": 0,
    "legacy_direct": 0,
    "langchain_attempts": 0,
    "langchain_success": 0,
    "langchain_fallback": 0,
}


def _forced_route_total(snapshot: dict[str, int]) -> int:
    return (
        snapshot.get("source_compare_forced", 0)
        + snapshot.get("compare_forced", 0)
        + snapshot.get("timeline_forced", 0)
        + snapshot.get("landscape_forced", 0)
        + snapshot.get("trend_forced", 0)
        + snapshot.get("fulltext_forced", 0)
        + snapshot.get("query_forced", 0)
    )


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


def emit_route_metrics(route_event: str, force: bool = False) -> None:
    if not metrics_enabled():
        return

    with _route_metrics_lock:
        snapshot = dict(_route_metrics)

    total = max(1, snapshot.get("requests_total", 0))
    attempts = snapshot.get("langchain_attempts", 0)
    fallback = snapshot.get("langchain_fallback", 0)
    success = snapshot.get("langchain_success", 0)
    source_compare_forced = snapshot.get("source_compare_forced", 0)
    compare_forced = snapshot.get("compare_forced", 0)
    timeline_forced = snapshot.get("timeline_forced", 0)
    landscape_forced = snapshot.get("landscape_forced", 0)
    trend_forced = snapshot.get("trend_forced", 0)
    fulltext_forced = snapshot.get("fulltext_forced", 0)
    query_forced = snapshot.get("query_forced", 0)
    landscape_low_evidence = snapshot.get("landscape_low_evidence", 0)
    legacy_direct = snapshot.get("legacy_direct", 0)

    should_log = force or (snapshot.get("requests_total", 0) % metrics_log_every() == 0)
    if not should_log:
        return

    fallback_rate_total = fallback / total
    fallback_rate_attempt = (fallback / attempts) if attempts else 0.0
    langchain_success_rate = (success / attempts) if attempts else 0.0
    forced_route_rate = _forced_route_total(snapshot) / total
    landscape_low_evidence_rate = (landscape_low_evidence / landscape_forced) if landscape_forced else 0.0

    print(
        "[Metrics] "
        f"event={route_event} "
        f"total={snapshot.get('requests_total', 0)} "
        f"source_compare_forced={source_compare_forced} "
        f"compare_forced={compare_forced} "
        f"timeline_forced={timeline_forced} "
        f"landscape_forced={landscape_forced} "
        f"trend_forced={trend_forced} "
        f"fulltext_forced={fulltext_forced} "
        f"query_forced={query_forced} "
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
    source_compare_forced = int(snapshot.get("source_compare_forced", 0))
    compare_forced = int(snapshot.get("compare_forced", 0))
    timeline_forced = int(snapshot.get("timeline_forced", 0))
    landscape_forced = int(snapshot.get("landscape_forced", 0))
    trend_forced = int(snapshot.get("trend_forced", 0))
    fulltext_forced = int(snapshot.get("fulltext_forced", 0))
    query_forced = int(snapshot.get("query_forced", 0))
    landscape_low_evidence = int(snapshot.get("landscape_low_evidence", 0))

    snapshot["fallback_rate_total"] = fallback / total
    snapshot["fallback_rate_langchain"] = (fallback / attempts) if attempts else 0.0
    snapshot["langchain_success_rate"] = (success / attempts) if attempts else 0.0
    snapshot["forced_route_rate"] = (
        source_compare_forced
        + compare_forced
        + timeline_forced
        + landscape_forced
        + trend_forced
        + fulltext_forced
        + query_forced
    ) / total
    snapshot["landscape_low_evidence_rate"] = (
        (landscape_low_evidence / landscape_forced) if landscape_forced else 0.0
    )
    return snapshot
