"""Persistence adapter for request-level agent traces."""

from __future__ import annotations

import copy
import logging
from typing import Any

from psycopg2.extras import Json

from services.db import db_transaction

logger = logging.getLogger(__name__)


def persist_request_trace(summary: dict[str, Any] | None) -> bool:
    """Persist one request trace and its span/model I/O rows.

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

        _persist_span_and_model_io(summary, request_id)
        return True
    except Exception as exc:
        logger.warning(
            "persist_request_trace failed: request_id=%s error=%s",
            request_id,
            exc,
        )
        return False


def _persist_span_and_model_io(summary: dict[str, Any], request_id: str) -> None:
    span_rows = _build_span_params(summary)
    model_io_rows = _build_model_io_params(summary)
    if not span_rows and not model_io_rows:
        return

    if span_rows:
        try:
            with db_transaction() as (_, cur):
                cur.executemany(
                    """
                    INSERT INTO public.agent_trace_spans (
                        request_id,
                        span_id,
                        parent_span_id,
                        span_type,
                        name,
                        status,
                        started_at_ms,
                        finished_at_ms,
                        latency_ms,
                        input_summary,
                        output_summary,
                        error_code,
                        error_message,
                        exception_chain,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (request_id, span_id) DO UPDATE
                    SET
                        parent_span_id = EXCLUDED.parent_span_id,
                        span_type = EXCLUDED.span_type,
                        name = EXCLUDED.name,
                        status = EXCLUDED.status,
                        started_at_ms = EXCLUDED.started_at_ms,
                        finished_at_ms = EXCLUDED.finished_at_ms,
                        latency_ms = EXCLUDED.latency_ms,
                        input_summary = EXCLUDED.input_summary,
                        output_summary = EXCLUDED.output_summary,
                        error_code = EXCLUDED.error_code,
                        error_message = EXCLUDED.error_message,
                        exception_chain = EXCLUDED.exception_chain,
                        metadata = EXCLUDED.metadata
                    """,
                    span_rows,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "persist_request_trace spans skipped: request_id=%s error=%s",
                request_id,
                exc,
            )
            return

    if model_io_rows:
        try:
            with db_transaction() as (_, cur):
                cur.executemany(
                    """
                    INSERT INTO public.agent_model_io (
                        request_id,
                        span_id,
                        node,
                        provider,
                        model,
                        input_messages,
                        raw_output,
                        parsed_output,
                        token_usage
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (request_id, span_id) DO UPDATE
                    SET
                        node = EXCLUDED.node,
                        provider = EXCLUDED.provider,
                        model = EXCLUDED.model,
                        input_messages = EXCLUDED.input_messages,
                        raw_output = EXCLUDED.raw_output,
                        parsed_output = EXCLUDED.parsed_output,
                        token_usage = EXCLUDED.token_usage
                    """,
                    model_io_rows,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "persist_request_trace model_io skipped: request_id=%s error=%s",
                request_id,
                exc,
            )


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


def _build_span_params(summary: dict[str, Any]) -> list[tuple[Any, ...]]:
    request_id = str(summary.get("request_id", "")).strip()
    rows: list[tuple[Any, ...]] = []
    for item in summary.get("spans", []) or []:
        if not isinstance(item, dict):
            continue
        span_id = _optional_text(item.get("span_id"))
        if not span_id:
            continue
        rows.append(
            (
                request_id,
                span_id,
                _optional_text(item.get("parent_span_id")),
                str(item.get("span_type", "postprocess")),
                str(item.get("name", "")),
                str(item.get("status", "unknown")),
                _to_non_negative_int(item.get("started_at_ms")),
                _to_non_negative_int(item.get("finished_at_ms"), nullable=True),
                _to_non_negative_int(item.get("latency_ms"), nullable=True),
                Json(_coerce_json_object(item.get("input_summary"))),
                Json(_coerce_json_object(item.get("output_summary"))),
                _optional_text(item.get("error_code")),
                _optional_text(item.get("error_message")),
                Json(_coerce_json_list(item.get("exception_chain"))),
                Json(_coerce_json_object(item.get("metadata"))),
            )
        )
    return rows


def _build_model_io_params(summary: dict[str, Any]) -> list[tuple[Any, ...]]:
    request_id = str(summary.get("request_id", "")).strip()
    rows: list[tuple[Any, ...]] = []
    for item in summary.get("model_io", []) or []:
        if not isinstance(item, dict):
            continue
        span_id = _optional_text(item.get("span_id"))
        if not span_id:
            continue
        rows.append(
            (
                request_id,
                span_id,
                str(item.get("node", "")),
                str(item.get("provider", "")),
                str(item.get("model", "")),
                Json(copy.deepcopy(item.get("input_messages") if item.get("input_messages") is not None else [])),
                Json(copy.deepcopy(item.get("raw_output"))),
                Json(copy.deepcopy(item.get("parsed_output"))) if item.get("parsed_output") is not None else None,
                Json(copy.deepcopy(item.get("token_usage"))) if isinstance(item.get("token_usage"), dict) else None,
            )
        )
    return rows


def _build_trace_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload_summary = copy.deepcopy(summary)
    payload_summary.pop("spans", None)
    payload_summary.pop("model_io", None)
    payload = {
        "summary": payload_summary,
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
