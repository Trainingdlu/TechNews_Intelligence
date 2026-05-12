"""Inspect self-hosted agent traces stored in PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.db import db_cursor  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query agent trace spans and full model I/O.")
    parser.add_argument("--request-id", help="Request id to inspect.")
    parser.add_argument("--case-id", help="Eval case id. The newest non-success run is preferred.")
    parser.add_argument("--model-span-id", help="Print full prompt/output for one model_call span.")
    parser.add_argument("--show-model-io", action="store_true", help="Print all full model I/O rows for the request.")
    parser.add_argument("--limit", type=int, default=20, help="Max rows for case lookup.")
    return parser.parse_args()


def _fetch_all(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with db_cursor() as (_, cur):
        cur.execute(sql, params)
        columns = [item[0] for item in cur.description or []]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _fetch_one(sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    rows = _fetch_all(sql, params)
    return rows[0] if rows else None


def _resolve_case_request(case_id: str, *, limit: int) -> str | None:
    rows = _fetch_all(
        """
        SELECT request_id, final_status, latency_ms, evidence_count, created_at
        FROM public.agent_runs
        WHERE trace_payload #>> '{final_answer_metadata,case_id}' = %s
           OR trace_payload #>> '{summary,final_answer_metadata,case_id}' = %s
        ORDER BY (final_status = 'success') ASC, created_at DESC
        LIMIT %s
        """,
        (case_id, case_id, max(1, int(limit))),
    )
    if not rows:
        print(f"No trace rows found for case_id={case_id}")
        return None
    print(f"Runs for case_id={case_id}:")
    for row in rows:
        print(
            f"- {row['request_id']} status={row['final_status']} "
            f"latency_ms={row['latency_ms']} evidence={row['evidence_count']} created_at={row['created_at']}"
        )
    return str(rows[0]["request_id"])


def _print_request_summary(request_id: str) -> None:
    row = _fetch_one(
        """
        SELECT final_status, latency_ms, evidence_count, error_code, trace_payload, created_at
        FROM public.agent_runs
        WHERE request_id = %s
        """,
        (request_id,),
    )
    if not row:
        print(f"No run found for request_id={request_id}")
        return
    payload = row.get("trace_payload") if isinstance(row.get("trace_payload"), dict) else {}
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    langsmith = runtime.get("langsmith") if isinstance(runtime.get("langsmith"), dict) else {}
    run_id = str(langsmith.get("langsmith_run_id") or "").strip()
    run_suffix = f" run_id={run_id}" if run_id else ""
    print(
        f"Run request_id={request_id} status={row['final_status']} "
        f"latency_ms={row['latency_ms']} evidence={row['evidence_count']} "
        f"error={row['error_code']} created_at={row['created_at']}"
    )
    print(
        "LangSmith "
        f"enabled={bool(langsmith.get('enabled'))} "
        f"project={langsmith.get('project') or ''} "
        f"endpoint={langsmith.get('endpoint') or ''}{run_suffix}"
    )


def _print_span_tree(request_id: str) -> None:
    spans = _fetch_all(
        """
        SELECT span_id, parent_span_id, span_type, name, status, latency_ms,
               input_summary, output_summary, error_code, error_message, metadata
        FROM public.agent_trace_spans
        WHERE request_id = %s
        ORDER BY started_at_ms ASC
        """,
        (request_id,),
    )
    if not spans:
        print(f"No spans found for request_id={request_id}")
        return

    children: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        parent = span.get("parent_span_id")
        children.setdefault(parent, []).append(span)

    def _walk(parent: str | None, depth: int) -> None:
        for span in children.get(parent, []):
            indent = "  " * depth
            suffix = ""
            if span.get("error_code"):
                suffix = f" error={span['error_code']}"
            print(
                f"{indent}- {span['name']} [{span['span_type']}] "
                f"status={span['status']} latency_ms={span['latency_ms']} span_id={span['span_id']}{suffix}"
            )
            _walk(str(span["span_id"]), depth + 1)

    print(f"Span tree for request_id={request_id}:")
    _walk(None, 0)


def _print_model_io(request_id: str, *, span_id: str | None = None) -> None:
    if span_id:
        rows = _fetch_all(
            """
            SELECT request_id, span_id, node, provider, model, input_messages,
                   raw_output, parsed_output, token_usage, created_at
            FROM public.agent_model_io
            WHERE request_id = %s AND span_id = %s
            ORDER BY created_at ASC
            """,
            (request_id, span_id),
        )
    else:
        rows = _fetch_all(
            """
            SELECT request_id, span_id, node, provider, model, input_messages,
                   raw_output, parsed_output, token_usage, created_at
            FROM public.agent_model_io
            WHERE request_id = %s
            ORDER BY created_at ASC
            """,
            (request_id,),
        )
    if not rows:
        target = f"{request_id}/{span_id}" if span_id else request_id
        print(f"No model I/O rows found for {target}")
        return
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    args = _parse_args()
    request_id = str(args.request_id or "").strip()
    if args.case_id and not request_id:
        resolved = _resolve_case_request(str(args.case_id).strip(), limit=int(args.limit))
        if not resolved:
            return 1
        request_id = resolved

    if not request_id:
        raise SystemExit("--request-id or --case-id is required.")

    _print_request_summary(request_id)
    _print_span_tree(request_id)
    if args.show_model_io or args.model_span_id:
        _print_model_io(request_id, span_id=str(args.model_span_id).strip() if args.model_span_id else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
