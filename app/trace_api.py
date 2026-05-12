"""Standalone Trace dashboard API and static frontend server."""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from services.db import close_db_pool, db_cursor, init_db_pool

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "agent" / ".env", override=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TRACE_STATIC_DIR = Path(
    os.getenv(
        "TRACE_DASHBOARD_STATIC_DIR",
        str(Path(__file__).resolve().parents[1] / "trace_dashboard" / "dist"),
    )
)

SPAN_TYPE_LABELS = {
    "graph_node": "流程节点",
    "model_call": "模型调用",
    "tool_call": "工具执行",
    "guard": "策略检查",
    "postprocess": "后处理",
}

SPAN_NAME_LABELS = {
    "prepare_context": "准备上下文",
    "intent_router": "判断问题类型",
    "tool_selection": "选择工具",
    "tool_worker": "规划工具调用",
    "tool_policy": "工具策略检查",
    "tool_executor": "执行工具",
    "evidence_normalizer": "归一化证据",
    "tool_loop_decider": "判断是否继续调用工具",
    "final_synthesizer": "生成最终回答",
    "output_guard": "输出检查",
    "clarification_response": "生成澄清回复",
    "insufficient_evidence_response": "证据不足回复",
}

_bearer_scheme = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db_pool()
        logger.info("trace dashboard database pool initialized")
    except Exception as exc:  # noqa: BLE001
        logger.warning("trace dashboard database pool initialization failed: %s", exc)
    try:
        yield
    finally:
        close_db_pool()
        logger.info("trace dashboard database pool closed")


app = FastAPI(title="TechNews Trace Dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)


def _expected_token() -> str:
    return os.getenv("TRACE_DASHBOARD_TOKEN", "").strip()


def _admin_email() -> str:
    return (os.getenv("TRACE_DASHBOARD_ADMIN_EMAIL") or os.getenv("ADMIN_EMAIL") or "").strip()


def _verify_trace_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, str]:
    expected = _expected_token()
    if not expected:
        raise HTTPException(status_code=503, detail="trace_dashboard_token_not_configured")

    supplied = credentials.credentials if credentials else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail="invalid_trace_dashboard_token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"admin_email": _admin_email()}


def _fetch_all(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with db_cursor() as (_, cur):
        cur.execute(sql, params)
        columns = [item[0] for item in cur.description or []]
        rows = cur.fetchall()
    return [_row_to_dict(columns, row) for row in rows]


def _fetch_one(sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    rows = _fetch_all(sql, params)
    return rows[0] if rows else None


def _row_to_dict(columns: list[str], row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return dict(zip(columns, row))


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _parse_datetime_filter(value: str | None, *, field_name: str) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid_{field_name}") from exc


def _langsmith_from_env() -> dict[str, Any]:
    enabled = any(
        str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2")
    )
    return {
        "enabled": enabled,
        "project": os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "",
        "endpoint": os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "",
    }


def _runtime_langsmith(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("trace_payload")
    runtime = payload.get("runtime") if isinstance(payload, dict) and isinstance(payload.get("runtime"), dict) else {}
    langsmith = runtime.get("langsmith") if isinstance(runtime.get("langsmith"), dict) else {}
    return {**_langsmith_from_env(), **langsmith}


def _public_run(row: dict[str, Any], *, include_payload: bool = False) -> dict[str, Any]:
    result = {
        "request_id": row.get("request_id"),
        "thread_id": row.get("thread_id"),
        "user_message": row.get("user_message"),
        "final_status": row.get("final_status"),
        "latency_ms": row.get("latency_ms"),
        "evidence_count": row.get("evidence_count"),
        "token_usage": row.get("token_usage"),
        "error_code": row.get("error_code"),
        "error_message": row.get("error_message"),
        "exception_chain": row.get("exception_chain") or [],
        "tool_call_chain": row.get("tool_call_chain") or [],
        "created_at": _json_safe(row.get("created_at")),
        "langsmith": _runtime_langsmith(row),
    }
    if include_payload:
        result["trace_payload"] = _json_safe(row.get("trace_payload") or {})
    return result


def _display_name(span_type: str, name: str) -> str:
    type_label = SPAN_TYPE_LABELS.get(span_type, "执行步骤")
    name_label = SPAN_NAME_LABELS.get(name, name)
    if span_type == "tool_call":
        return f"{type_label}：{name}"
    if span_type == "model_call":
        return f"{type_label}：{name_label}"
    return name_label if name_label != name else f"{type_label}：{name}"


def _public_span(row: dict[str, Any]) -> dict[str, Any]:
    span_type = str(row.get("span_type") or "")
    name = str(row.get("name") or "")
    result = {
        "span_id": row.get("span_id"),
        "parent_span_id": row.get("parent_span_id"),
        "span_type": span_type,
        "span_type_label": SPAN_TYPE_LABELS.get(span_type, "执行步骤"),
        "name": name,
        "display_name": _display_name(span_type, name),
        "status": row.get("status"),
        "started_at_ms": row.get("started_at_ms"),
        "finished_at_ms": row.get("finished_at_ms"),
        "latency_ms": row.get("latency_ms"),
        "input_summary": row.get("input_summary") or {},
        "output_summary": row.get("output_summary") or {},
        "error_code": row.get("error_code"),
        "error_message": row.get("error_message"),
        "exception_chain": row.get("exception_chain") or [],
        "metadata": row.get("metadata") or {},
        "created_at": _json_safe(row.get("created_at")),
    }
    return _json_safe(result)


def _build_span_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = [{**_public_span(span), "children": []} for span in spans]
    by_id = {str(node["span_id"]): node for node in nodes if node.get("span_id")}
    roots: list[dict[str, Any]] = []
    for node in nodes:
        parent_id = node.get("parent_span_id")
        parent = by_id.get(str(parent_id)) if parent_id else None
        if parent is None or parent is node:
            roots.append(node)
        else:
            parent["children"].append(node)
    return roots


@app.get("/trace-api/meta")
def get_meta(_: dict[str, str] = Depends(_verify_trace_token)) -> dict[str, Any]:
    return {
        "admin_email": _admin_email(),
        "langsmith": _langsmith_from_env(),
    }


@app.get("/trace-api/runs")
def list_runs(
    _: dict[str, str] = Depends(_verify_trace_token),
    status: str | None = Query(default=None),
    error_code: str | None = Query(default=None),
    q: str | None = Query(default=None),
    created_from: str | None = Query(default=None),
    created_to: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    filters: list[str] = []
    params: list[Any] = []

    if status and status != "all":
        filters.append("final_status = %s")
        params.append(status)
    if error_code:
        filters.append("error_code = %s")
        params.append(error_code)
    if q:
        term = f"%{q.strip()}%"
        filters.append(
            "("
            "request_id ILIKE %s OR "
            "COALESCE(thread_id, '') ILIKE %s OR "
            "user_message ILIKE %s OR "
            "COALESCE(error_message, '') ILIKE %s"
            ")"
        )
        params.extend([term, term, term, term])

    start_dt = _parse_datetime_filter(created_from, field_name="created_from")
    end_dt = _parse_datetime_filter(created_to, field_name="created_to")
    if start_dt is not None:
        filters.append("created_at >= %s")
        params.append(start_dt)
    if end_dt is not None:
        filters.append("created_at <= %s")
        params.append(end_dt)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    total_row = _fetch_one(
        f"SELECT COUNT(*) AS total FROM public.agent_runs {where_sql}",
        tuple(params),
    )
    rows = _fetch_all(
        f"""
        SELECT request_id, thread_id, user_message, final_status, latency_ms,
               evidence_count, token_usage, error_code, error_message,
               exception_chain, tool_call_chain, trace_payload, created_at
        FROM public.agent_runs
        {where_sql}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        (*params, int(limit), int(offset)),
    )
    return {
        "items": [_public_run(row) for row in rows],
        "total": int((total_row or {}).get("total") or 0),
        "limit": int(limit),
        "offset": int(offset),
    }


@app.get("/trace-api/runs/{request_id}")
def get_run(
    request_id: str,
    _: dict[str, str] = Depends(_verify_trace_token),
) -> dict[str, Any]:
    run = _fetch_one(
        """
        SELECT request_id, thread_id, user_message, final_status, latency_ms,
               evidence_count, token_usage, error_code, error_message,
               exception_chain, tool_call_chain, trace_payload, created_at
        FROM public.agent_runs
        WHERE request_id = %s
        """,
        (request_id,),
    )
    if not run:
        raise HTTPException(status_code=404, detail="trace_run_not_found")

    spans = _fetch_all(
        """
        SELECT request_id, span_id, parent_span_id, span_type, name, status,
               started_at_ms, finished_at_ms, latency_ms, input_summary,
               output_summary, error_code, error_message, exception_chain,
               metadata, created_at
        FROM public.agent_trace_spans
        WHERE request_id = %s
        ORDER BY started_at_ms ASC, id ASC
        """,
        (request_id,),
    )
    public_spans = [_public_span(span) for span in spans]
    return {
        "run": _public_run(run, include_payload=True),
        "spans": public_spans,
        "span_tree": _build_span_tree(spans),
    }


@app.get("/trace-api/spans/{span_id}")
def get_span(
    span_id: str,
    _: dict[str, str] = Depends(_verify_trace_token),
    request_id: str | None = Query(default=None),
) -> dict[str, Any]:
    if request_id:
        row = _fetch_one(
            """
            SELECT request_id, span_id, parent_span_id, span_type, name, status,
                   started_at_ms, finished_at_ms, latency_ms, input_summary,
                   output_summary, error_code, error_message, exception_chain,
                   metadata, created_at
            FROM public.agent_trace_spans
            WHERE request_id = %s AND span_id = %s
            """,
            (request_id, span_id),
        )
    else:
        row = _fetch_one(
            """
            SELECT request_id, span_id, parent_span_id, span_type, name, status,
                   started_at_ms, finished_at_ms, latency_ms, input_summary,
                   output_summary, error_code, error_message, exception_chain,
                   metadata, created_at
            FROM public.agent_trace_spans
            WHERE span_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (span_id,),
        )
    if not row:
        raise HTTPException(status_code=404, detail="trace_span_not_found")
    return {"span": _public_span(row)}


@app.get("/trace-api/spans/{span_id}/model-io")
def get_model_io(
    span_id: str,
    auth: dict[str, str] = Depends(_verify_trace_token),
    request_id: str | None = Query(default=None),
) -> dict[str, Any]:
    if request_id:
        span = _fetch_one(
            "SELECT request_id, span_id, span_type, name FROM public.agent_trace_spans WHERE request_id = %s AND span_id = %s",
            (request_id, span_id),
        )
    else:
        span = _fetch_one(
            """
            SELECT request_id, span_id, span_type, name
            FROM public.agent_trace_spans
            WHERE span_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (span_id,),
        )
    if not span:
        raise HTTPException(status_code=404, detail="trace_span_not_found")
    if span.get("span_type") != "model_call":
        raise HTTPException(status_code=400, detail="model_io_only_available_for_model_call")

    row = _fetch_one(
        """
        SELECT request_id, span_id, node, provider, model, input_messages,
               raw_output, parsed_output, token_usage, created_at
        FROM public.agent_model_io
        WHERE request_id = %s AND span_id = %s
        """,
        (span.get("request_id"), span_id),
    )
    if not row:
        raise HTTPException(status_code=404, detail="model_io_not_found")

    logger.info(
        "trace model io accessed admin=%s request_id=%s span_id=%s node=%s",
        auth.get("admin_email") or "",
        row.get("request_id"),
        row.get("span_id"),
        row.get("node"),
    )
    return {"model_io": _json_safe(row)}


def _render_missing_dashboard() -> HTMLResponse:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="zh-CN">
          <head><meta charset="utf-8"><title>Trace Dashboard</title></head>
          <body style="font-family: sans-serif; padding: 32px;">
            <h1>Trace Dashboard build missing</h1>
            <p>Run <code>npm install</code> and <code>npm run build</code> in <code>trace_dashboard/</code>,
            or build the Docker trace image.</p>
          </body>
        </html>
        """,
        status_code=503,
    )


if (TRACE_STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=TRACE_STATIC_DIR / "assets"), name="trace-assets")


@app.get("/{path:path}", response_class=HTMLResponse)
def serve_dashboard(path: str = "") -> HTMLResponse:
    if path.startswith("trace-api"):
        raise HTTPException(status_code=404, detail="trace_api_route_not_found")
    index_file = TRACE_STATIC_DIR / "index.html"
    if not index_file.exists():
        return _render_missing_dashboard()
    return HTMLResponse(index_file.read_text(encoding="utf-8"))
