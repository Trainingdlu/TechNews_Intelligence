"""Query-news skill implementation and structured adapter."""

from __future__ import annotations

import json
from typing import Any

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_records, _json_text, _source_to_db_label
from .schemas import QueryNewsSkillInput


def query_news(
    query: str = "",
    source: str = "all",
    days: int = 21,
    category: str = "",
    sentiment: str = "",
    sort: str = "time_desc",
    limit: int = 8,
    response_format: str = "text",
) -> str:
    """Query news with structured filters."""
    print(f"\n[Tool] query_news: query={query}, source={source}, days={days}, sort={sort}")
    days = _clamp_int(days, 1, 365)
    limit = _clamp_int(limit, 1, 30)
    source_label = _source_to_db_label(source)

    order_map = {
        "time_desc": "created_at DESC",
        "time_asc": "created_at ASC",
        "heat_desc": "points DESC NULLS LAST, created_at DESC",
        "heat_asc": "points ASC NULLS LAST, created_at DESC",
    }
    order_sql = order_map.get(sort.strip().lower(), order_map["time_desc"])

    where_parts = ["created_at > NOW() - %s::interval"]
    params: list[Any] = [f"{days} days"]

    if query.strip():
        where_parts.append("(title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)")
        query_like = f"%{query.strip()}%"
        params.extend([query_like, query_like, query_like])

    if source_label:
        where_parts.append("LOWER(COALESCE(source_type,'')) = LOWER(%s)")
        params.append(source_label)

    if category.strip():
        where_parts.append("(COALESCE(title_cn,'') ILIKE %s OR COALESCE(title_cn,'') ILIKE %s)")
        params.extend([f"%[{category.strip()}]%", f"%【{category.strip()}】%"])

    if sentiment.strip():
        where_parts.append("COALESCE(sentiment,'') ILIKE %s")
        params.append(sentiment.strip())

    sql = f"""
        SELECT title, title_cn, url, summary, sentiment, points, source_type, created_at
        FROM view_dashboard_news
        WHERE {' AND '.join(where_parts)}
        ORDER BY {order_sql}
        LIMIT %s
    """
    params.append(limit)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()

        records: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, 1):
            title, title_cn, url, summary, senti, points, src, created_at = row
            records.append(
                {
                    "rank": idx,
                    "source": src,
                    "title": title,
                    "title_cn": title_cn,
                    "url": url,
                    "summary": summary or "",
                    "sentiment": senti or "",
                    "points": int(points or 0),
                    "created_at": created_at.isoformat() if created_at else "",
                }
            )

        if response_format.strip().lower() == "json":
            payload = {
                "tool": "query_news",
                "status": "ok" if records else "empty",
                "request": {
                    "query": query,
                    "source": source,
                    "days": days,
                    "category": category,
                    "sentiment": sentiment,
                    "sort": sort,
                    "limit": limit,
                },
                "count": len(records),
                "records": records,
            }
            return _json_text(payload)

        if not rows:
            return "No matching records."

        lines = []
        for item in records:
            lines.append(
                f"{item['rank']}. [{item['source']}] {item.get('title_cn') or item.get('title')}\n"
                f"   time={item['created_at'].replace('T', ' ')[:16]}, sentiment={item['sentiment']}, points={item['points']}\n"
                f"   url={item['url']}\n"
                f"   summary={item['summary'][:220]}"
            )
        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] query_news failed: {exc}")
        return f"query_news failed: {exc}"
    finally:
        put_conn(conn)


def query_news_skill(payload: QueryNewsSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    raw_output = query_news(
        query=request.get("query", ""),
        source=request.get("source", "all"),
        days=int(request.get("days", 21)),
        category=request.get("category", ""),
        sentiment=request.get("sentiment", ""),
        sort=request.get("sort", "time_desc"),
        limit=int(request.get("limit", 8)),
        response_format="json",
    )

    try:
        parsed = json.loads(raw_output)
    except Exception as exc:
        return build_error_envelope(
            tool="query_news",
            request=request,
            error="query_news_json_parse_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "raw_preview": str(raw_output)[:500],
            },
        )

    status = str(parsed.get("status", "error")).lower()
    if status not in {"ok", "empty", "error"}:
        return build_error_envelope(
            tool="query_news",
            request=request,
            error="query_news_invalid_status",
            diagnostics={"status": parsed.get("status"), "raw_output": parsed},
        )

    if status == "error":
        return build_error_envelope(
            tool="query_news",
            request=request,
            error=str(parsed.get("error") or "query_news_failed"),
            diagnostics={"raw_output": parsed},
        )

    records_raw = parsed.get("records")
    records = records_raw if isinstance(records_raw, list) else []
    evidence = _evidence_from_records(records, max_items=int(request.get("limit", 8)))

    return SkillEnvelope(
        tool="query_news",
        status=status,
        request=request,
        data={"count": int(parsed.get("count", len(records))), "records": records},
        evidence=evidence,
        diagnostics={"source": request.get("source", "all"), "sort": request.get("sort", "time_desc")},
    )
