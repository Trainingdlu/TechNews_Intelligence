"""Reusable execution pipelines for deterministic agent routes."""

from __future__ import annotations

import json
import os
from typing import Any, Callable


def _json_status(raw: str, tool_name: str) -> str:
    text = (raw or "").strip()
    if not text.startswith("{"):
        return ""
    try:
        payload = json.loads(text)
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    if str(payload.get("tool", "")).strip().lower() != tool_name.strip().lower():
        return ""
    return str(payload.get("status", "")).strip().lower()


def run_source_compare_pipeline(
    *,
    strict_mode: bool,
    user_message: str,
    topic: str,
    days: int,
    compare_sources_fn: Callable[..., str],
    analyze_fn: Callable[..., str],
    ensure_evidence_fn: Callable[[str, str, str], str],
    emit_metrics_fn: Callable[..., None],
) -> str:
    try:
        raw_source_compare = compare_sources_fn(topic=topic, days=days)
        if raw_source_compare.startswith("compare_sources failed:") or raw_source_compare.startswith("No comparison data"):
            emit_metrics_fn("source_compare_forced")
            return raw_source_compare
        try:
            analyzed = analyze_fn(
                user_message=user_message,
                topic=topic,
                days=days,
                source_compare_output=raw_source_compare,
            )
            if analyzed:
                result = ensure_evidence_fn(analyzed, raw_source_compare, user_message)
                emit_metrics_fn("source_compare_forced")
                return result
        except Exception as exc:
            if strict_mode:
                emit_metrics_fn("source_compare_forced_strict_error", force=True)
                raise
            print(f"[Warn] Source compare synthesis failed; fallback to raw output: {exc}")
        emit_metrics_fn("source_compare_forced")
        return raw_source_compare
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("source_compare_forced_strict_error", force=True)
            raise
        emit_metrics_fn("source_compare_forced_error", force=True)
        return f"Source comparison request detected, but DB query failed: {exc}"


def run_compare_pipeline(
    *,
    strict_mode: bool,
    user_message: str,
    topic_a: str,
    topic_b: str,
    days: int,
    compare_topics_fn: Callable[..., str],
    analyze_fn: Callable[..., str],
    ensure_evidence_fn: Callable[[str, str, str], str],
    emit_metrics_fn: Callable[..., None],
) -> str:
    try:
        raw_compare = compare_topics_fn(topic_a=topic_a, topic_b=topic_b, days=days)
        if raw_compare.startswith("compare_topics failed:"):
            emit_metrics_fn("compare_forced")
            return raw_compare
        try:
            analyzed = analyze_fn(
                user_message=user_message,
                topic_a=topic_a,
                topic_b=topic_b,
                days=days,
                compare_output=raw_compare,
            )
            if analyzed:
                result = ensure_evidence_fn(analyzed, raw_compare, user_message)
                emit_metrics_fn("compare_forced")
                return result
        except Exception as exc:
            if strict_mode:
                emit_metrics_fn("compare_forced_strict_error", force=True)
                raise
            print(f"[Warn] Compare analysis synthesis failed; fallback to raw compare: {exc}")
        emit_metrics_fn("compare_forced")
        return raw_compare
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("compare_forced_strict_error", force=True)
            raise
        emit_metrics_fn("compare_forced_error", force=True)
        return f"Compare request detected, but DB query failed: {exc}"


def run_timeline_pipeline(
    *,
    strict_mode: bool,
    user_message: str,
    topic: str,
    days: int,
    limit: int,
    build_timeline_fn: Callable[..., str],
    count_timeline_items_fn: Callable[[str], int],
    format_low_sample_fn: Callable[..., str],
    analyze_fn: Callable[..., str],
    ensure_evidence_fn: Callable[[str, str, str], str],
    emit_metrics_fn: Callable[..., None],
) -> str:
    try:
        raw_timeline = build_timeline_fn(topic=topic, days=days, limit=limit)
        if raw_timeline.startswith("build_timeline failed:") or raw_timeline.startswith("No timeline data"):
            emit_metrics_fn("timeline_forced")
            return raw_timeline

        min_events = max(1, int(os.getenv("TIMELINE_MIN_EVENTS", "5")))
        event_count = count_timeline_items_fn(raw_timeline)
        if event_count < min_events:
            result = format_low_sample_fn(
                user_message=user_message,
                topic=topic,
                days=days,
                event_count=event_count,
                min_events=min_events,
                raw_timeline=raw_timeline,
            )
            emit_metrics_fn("timeline_forced")
            return result

        try:
            analyzed = analyze_fn(
                user_message=user_message,
                topic=topic,
                days=days,
                timeline_output=raw_timeline,
            )
            if analyzed:
                result = ensure_evidence_fn(analyzed, raw_timeline, user_message)
                emit_metrics_fn("timeline_forced")
                return result
        except Exception as exc:
            if strict_mode:
                emit_metrics_fn("timeline_forced_strict_error", force=True)
                raise
            print(f"[Warn] Timeline analysis synthesis failed; fallback to raw timeline: {exc}")
        emit_metrics_fn("timeline_forced")
        return raw_timeline
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("timeline_forced_strict_error", force=True)
            raise
        emit_metrics_fn("timeline_forced_error", force=True)
        return f"Timeline request detected, but DB query failed: {exc}"


def run_landscape_pipeline(
    *,
    strict_mode: bool,
    user_message: str,
    topic: str,
    days: int,
    entities: list[str],
    analyze_landscape_fn: Callable[..., str],
    is_no_data_fn: Callable[[str], bool],
    format_no_data_fn: Callable[..., str],
    is_evidence_sufficient_fn: Callable[[str], bool],
    metrics_inc_fn: Callable[[str, int], None],
    format_low_evidence_fn: Callable[..., str],
    analyze_output_fn: Callable[..., str],
    is_unstable_synthesis_fn: Callable[[str], bool],
    ensure_evidence_fn: Callable[[str, str, str], str],
    emit_metrics_fn: Callable[..., None],
) -> str:
    entities_csv = ",".join(entities)
    try:
        raw_landscape = analyze_landscape_fn(topic=topic, days=days, entities=entities_csv, limit_per_entity=3)
        if is_no_data_fn(raw_landscape) and topic:
            retry_landscape = analyze_landscape_fn(topic="", days=days, entities=entities_csv, limit_per_entity=3)
            if not is_no_data_fn(retry_landscape):
                raw_landscape = retry_landscape
        if is_no_data_fn(raw_landscape):
            emit_metrics_fn("landscape_forced")
            return format_no_data_fn(
                user_message=user_message,
                topic=topic,
                days=days,
                entities=entities,
                raw_landscape=raw_landscape,
            )

        if not is_evidence_sufficient_fn(raw_landscape):
            metrics_inc_fn("landscape_low_evidence", 1)
            result = format_low_evidence_fn(
                user_message=user_message,
                topic=topic,
                days=days,
                entities=entities,
                raw_landscape=raw_landscape,
            )
            emit_metrics_fn("landscape_forced_low_evidence", force=True)
            return result

        try:
            analyzed = analyze_output_fn(
                user_message=user_message,
                topic=topic,
                days=days,
                entities=entities,
                landscape_output=raw_landscape,
            )
            if analyzed:
                if is_unstable_synthesis_fn(analyzed):
                    print("[Warn] Landscape synthesis unstable; fallback to raw landscape snapshot.")
                    emit_metrics_fn("landscape_forced")
                    return raw_landscape
                result = ensure_evidence_fn(analyzed, raw_landscape, user_message)
                emit_metrics_fn("landscape_forced")
                return result
        except Exception as exc:
            if strict_mode:
                emit_metrics_fn("landscape_forced_strict_error", force=True)
                raise
            print(f"[Warn] Landscape analysis synthesis failed; fallback to raw landscape: {exc}")
        emit_metrics_fn("landscape_forced")
        return raw_landscape
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("landscape_forced_strict_error", force=True)
            raise
        emit_metrics_fn("landscape_forced_error", force=True)
        return f"Landscape analysis request detected, but DB query failed: {exc}"


def run_trend_pipeline(
    *,
    strict_mode: bool,
    user_message: str,
    topic: str,
    window: int,
    trend_analysis_fn: Callable[..., str],
    analyze_fn: Callable[..., str],
    ensure_evidence_fn: Callable[[str, str, str], str],
    emit_metrics_fn: Callable[..., None],
) -> str:
    try:
        raw_trend = trend_analysis_fn(topic=topic, window=window)
        if raw_trend.startswith("trend_analysis failed:") or raw_trend.startswith("trend_analysis requires"):
            emit_metrics_fn("trend_forced")
            return raw_trend
        try:
            analyzed = analyze_fn(
                user_message=user_message,
                topic=topic,
                window=window,
                trend_output=raw_trend,
            )
            if analyzed:
                result = ensure_evidence_fn(analyzed, raw_trend, user_message)
                emit_metrics_fn("trend_forced")
                return result
        except Exception as exc:
            if strict_mode:
                emit_metrics_fn("trend_forced_strict_error", force=True)
                raise
            print(f"[Warn] Trend synthesis failed; fallback to raw output: {exc}")
        emit_metrics_fn("trend_forced")
        return raw_trend
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("trend_forced_strict_error", force=True)
            raise
        emit_metrics_fn("trend_forced_error", force=True)
        return f"Trend request detected, but DB query failed: {exc}"


def run_fulltext_pipeline(
    *,
    strict_mode: bool,
    user_message: str,
    request_query: str,
    max_chars: int,
    fulltext_batch_fn: Callable[..., str],
    analyze_fn: Callable[..., str],
    ensure_evidence_fn: Callable[[str, str, str], str],
    emit_metrics_fn: Callable[..., None],
) -> str:
    try:
        try:
            raw_fulltext = fulltext_batch_fn(
                urls=request_query,
                max_chars_per_article=max_chars,
                response_format="json",
            )
        except TypeError:
            raw_fulltext = fulltext_batch_fn(urls=request_query, max_chars_per_article=max_chars)
        fulltext_status = _json_status(raw_fulltext, "fulltext_batch")
        if raw_fulltext.startswith("fulltext_batch requires") or raw_fulltext.startswith("No candidate articles found"):
            emit_metrics_fn("fulltext_forced")
            return raw_fulltext
        if fulltext_status == "empty":
            emit_metrics_fn("fulltext_forced")
            return "No candidate articles found."
        try:
            analyzed = analyze_fn(
                user_message=user_message,
                request_query=request_query,
                fulltext_output=raw_fulltext,
            )
            if analyzed:
                result = ensure_evidence_fn(analyzed, raw_fulltext, user_message)
                emit_metrics_fn("fulltext_forced")
                return result
        except Exception as exc:
            if strict_mode:
                emit_metrics_fn("fulltext_forced_strict_error", force=True)
                raise
            print(f"[Warn] Fulltext synthesis failed; fallback to raw output: {exc}")
        emit_metrics_fn("fulltext_forced")
        return raw_fulltext
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("fulltext_forced_strict_error", force=True)
            raise
        emit_metrics_fn("fulltext_forced_error", force=True)
        return f"Fulltext request detected, but tool execution failed: {exc}"


def run_query_pipeline(
    *,
    strict_mode: bool,
    user_message: str,
    query: str,
    source: str,
    days: int,
    sort: str,
    limit: int,
    query_news_fn: Callable[..., str],
    analyze_fn: Callable[..., str],
    ensure_evidence_fn: Callable[[str, str, str], str],
    emit_metrics_fn: Callable[..., None],
) -> str:
    try:
        try:
            raw_query = query_news_fn(
                query=query,
                source=source,
                days=days,
                category="",
                sentiment="",
                sort=sort,
                limit=limit,
                response_format="json",
            )
        except TypeError:
            raw_query = query_news_fn(
                query=query,
                source=source,
                days=days,
                category="",
                sentiment="",
                sort=sort,
                limit=limit,
            )
        query_status = _json_status(raw_query, "query_news")
        if raw_query.startswith("query_news failed:") or raw_query.startswith("No matching records"):
            emit_metrics_fn("query_forced")
            return raw_query
        if query_status == "empty":
            emit_metrics_fn("query_forced")
            return "No matching records."
        try:
            analyzed = analyze_fn(
                user_message=user_message,
                query=query,
                source=source,
                days=days,
                sort=sort,
                query_output=raw_query,
            )
            if analyzed:
                result = ensure_evidence_fn(analyzed, raw_query, user_message)
                emit_metrics_fn("query_forced")
                return result
        except Exception as exc:
            if strict_mode:
                emit_metrics_fn("query_forced_strict_error", force=True)
                raise
            print(f"[Warn] Query synthesis failed; fallback to raw output: {exc}")
        emit_metrics_fn("query_forced")
        return raw_query
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("query_forced_strict_error", force=True)
            raise
        emit_metrics_fn("query_forced_error", force=True)
        return f"Query request detected, but DB query failed: {exc}"


def run_runtime_pipeline(
    *,
    strict_mode: bool,
    runtime: str,
    history: list[dict],
    user_message: str,
    metrics_inc_fn: Callable[[str, int], None],
    emit_metrics_fn: Callable[..., None],
    generate_legacy_fn: Callable[[list[dict], str], str],
    generate_langgraph_fn: Callable[[list[dict], str], str],
) -> str:
    if runtime == "legacy":
        metrics_inc_fn("legacy_direct", 1)
        emit_metrics_fn("legacy_direct")
        return generate_legacy_fn(history, user_message)

    metrics_inc_fn("langchain_attempts", 1)
    try:
        result = generate_langgraph_fn(history, user_message)
        metrics_inc_fn("langchain_success", 1)
        emit_metrics_fn("langchain_success")
        return result
    except Exception as exc:
        if strict_mode:
            emit_metrics_fn("langchain_strict_error", force=True)
            raise
        metrics_inc_fn("langchain_fallback", 1)
        emit_metrics_fn("langchain_fallback", force=True)
        print(f"[Warn] LangChain runtime failed, fallback to legacy runtime: {exc}")
        return generate_legacy_fn(history, user_message)
