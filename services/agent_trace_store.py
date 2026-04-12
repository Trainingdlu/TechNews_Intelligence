"""Persistence adapter for request-level agent traces."""

from __future__ import annotations

import copy
import logging
from typing import Any

from psycopg2.extras import Json

from services.db import db_transaction

logger = logging.getLogger(__name__)


def persist_request_trace(summary: dict[str, Any] | None) -> bool:
    """Persist one request trace and its tool events.

    Returns:
        True if at least the `agent_runs` upsert succeeded, otherwise False.
    """
    if not isinstance(summary, dict):
        return False

    request_id = str(summary.get("request_id", "")).strip()
    if not request_id:
        return False

    try:
        with db_transaction() as (_, cur):
            run_params = _build_agent_run_params(summary)
            cur.execute(
                """
                INSERT INTO public.agent_runs (
                    request_id,
                    thread_id,
                    user_message,
                    final_status,
                    latency_ms,
                    evidence_count,
                    token_usage,
                    error_code,
                    error_message,
                    exception_chain,
                    tool_call_chain,
                    trace_payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE
                SET
                    thread_id = EXCLUDED.thread_id,
                    user_message = EXCLUDED.user_message,
                    final_status = EXCLUDED.final_status,
                    latency_ms = EXCLUDED.latency_ms,
                    evidence_count = EXCLUDED.evidence_count,
                    token_usage = EXCLUDED.token_usage,
                    error_code = EXCLUDED.error_code,
                    error_message = EXCLUDED.error_message,
                    exception_chain = EXCLUDED.exception_chain,
                    tool_call_chain = EXCLUDED.tool_call_chain,
                    trace_payload = EXCLUDED.trace_payload
                """,
                run_params,
            )

            tool_rows = _build_tool_event_params(summary)
            if tool_rows:
                cur.executemany(
                    """
                    INSERT INTO public.agent_tool_events (
                        request_id,
                        event_index,
                        tool_name,
                        status,
                        latency_ms,
                        input_summary,
                        output_summary,
                        error_code,
                        error_message,
                        exception_chain
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (request_id, event_index) DO UPDATE
                    SET
                        tool_name = EXCLUDED.tool_name,
                        status = EXCLUDED.status,
                        latency_ms = EXCLUDED.latency_ms,
                        input_summary = EXCLUDED.input_summary,
                        output_summary = EXCLUDED.output_summary,
                        error_code = EXCLUDED.error_code,
                        error_message = EXCLUDED.error_message,
                        exception_chain = EXCLUDED.exception_chain
                    """,
                    tool_rows,
                )
        return True
    except Exception as exc:
        logger.warning(
            "persist_request_trace failed: request_id=%s error=%s",
            request_id,
            exc,
        )
        return False


def _build_agent_run_params(summary: dict[str, Any]) -> tuple[Any, ...]:
    request_id = str(summary.get("request_id", "")).strip()
    thread_id = _optional_text(summary.get("thread_id"))
    user_message = str(summary.get("user_message", ""))
    final_status = str(summary.get("final_status", "unknown"))
    latency_ms = _to_non_negative_int(summary.get("latency_ms"))
    evidence_count = _to_non_negative_int(summary.get("evidence_count"))

    token_usage = summary.get("token_usage")
    if not isinstance(token_usage, dict):
        token_usage = None

    error_code = _optional_text(summary.get("error_code"))
    error_message = _optional_text(summary.get("error_message"))

    exception_chain = summary.get("exception_chain")
    if not isinstance(exception_chain, list):
        exception_chain = []

    tool_call_chain = summary.get("tool_call_chain")
    if not isinstance(tool_call_chain, list):
        tool_call_chain = []

    trace_payload = _build_trace_payload(summary)

    return (
        request_id,
        thread_id,
        user_message,
        final_status,
        latency_ms,
        evidence_count,
        Json(token_usage) if token_usage is not None else None,
        error_code,
        error_message,
        Json(exception_chain),
        Json(tool_call_chain),
        Json(trace_payload),
    )


def _build_tool_event_params(summary: dict[str, Any]) -> list[tuple[Any, ...]]:
    request_id = str(summary.get("request_id", "")).strip()
    rows: list[tuple[Any, ...]] = []
    for item in summary.get("tool_events", []) or []:
        if not isinstance(item, dict):
            continue
        event_index = _to_non_negative_int(item.get("event_index"))
        if event_index <= 0:
            continue
        rows.append(
            (
                request_id,
                event_index,
                str(item.get("tool_name", "")),
                str(item.get("status", "unknown")),
                _to_non_negative_int(item.get("latency_ms"), nullable=True),
                Json(_coerce_json_object(item.get("input_summary"))),
                Json(_coerce_json_object(item.get("output_summary"))),
                _optional_text(item.get("error_code")),
                _optional_text(item.get("error_message")),
                Json(_coerce_json_list(item.get("exception_chain"))),
            )
        )
    return rows


def _build_trace_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "summary": copy.deepcopy(summary),
        "final_answer_metadata": copy.deepcopy(summary.get("final_answer_metadata") or {}),
        "runtime": copy.deepcopy(summary.get("runtime") or {}),
    }
    return payload


def _to_non_negative_int(value: Any, *, nullable: bool = False) -> int | None:
    if value is None:
        return None if nullable else 0
    try:
        parsed = int(value)
    except Exception:
        return None if nullable else 0
    if parsed < 0:
        return None if nullable else 0
    return parsed


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if value is None:
        return {}
    return {"value": copy.deepcopy(value)}


def _coerce_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    if value is None:
        return []
    return [copy.deepcopy(value)]
