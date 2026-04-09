"""Trend-analysis skill implementation and structured adapter."""

from __future__ import annotations

import json
from typing import Any

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_records, _json_text
from .query_news import query_news
from .schemas import TrendAnalysisSkillInput
from .sql_builders import _build_topic_where_clause


def trend_analysis(topic: str, window: int = 7, response_format: str = "text") -> str:
    """Analyze topic momentum by comparing recent vs previous windows."""
    print(f"\n[Tool] trend_analysis: topic={topic}, window={window}")
    as_json = response_format.strip().lower() == "json"

    if not topic or not topic.strip():
        if as_json:
            return _json_text(
                {
                    "tool": "trend_analysis",
                    "status": "error",
                    "request": {"topic": topic, "window": window},
                    "data": None,
                    "error": "trend_analysis requires topic.",
                }
            )
        return "trend_analysis requires topic."

    topic = topic.strip()
    window = _clamp_int(window, 3, 60)
    topic_clause, topic_params = _build_topic_where_clause(topic)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            WITH matched AS (
                SELECT created_at, points
                FROM tech_news
                WHERE created_at >= NOW() - (%s::int * INTERVAL '2 day')
                  AND {topic_clause}
            ),
            recent AS (
                SELECT COUNT(*) AS cnt, COALESCE(AVG(points), 0) AS avg_points
                FROM matched
                WHERE created_at >= NOW() - (%s::int * INTERVAL '1 day')
            ),
            prev AS (
                SELECT COUNT(*) AS cnt, COALESCE(AVG(points), 0) AS avg_points
                FROM matched
                WHERE created_at < NOW() - (%s::int * INTERVAL '1 day')
            )
            SELECT recent.cnt, recent.avg_points, prev.cnt, prev.avg_points
            FROM recent, prev
            """,
            tuple([window] + topic_params + [window, window]),
        )
        recent_cnt, recent_avg, prev_cnt, prev_avg = cur.fetchone()

        cur.execute(
            f"""
            SELECT DATE(created_at) AS day, COUNT(*) AS cnt, ROUND(AVG(points)::numeric, 1) AS avg_points
            FROM tech_news
            WHERE created_at >= NOW() - %s::interval
              AND {topic_clause}
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """,
            tuple([f"{window} days"] + topic_params),
        )
        daily_rows = cur.fetchall()
        cur.close()

        recent_cnt_i = int(recent_cnt or 0)
        prev_cnt_i = int(prev_cnt or 0)
        recent_avg_f = float(recent_avg or 0.0)
        prev_avg_f = float(prev_avg or 0.0)

        change_cnt = recent_cnt_i - prev_cnt_i
        change_pct = ((change_cnt / prev_cnt_i) * 100.0) if prev_cnt_i > 0 else None
        if prev_cnt_i > 0:
            delta = f"{change_cnt:+d} ({change_pct:+.1f}%)"
        else:
            delta = f"{change_cnt:+d} (no previous baseline)"

        daily_records = [
            {"day": day.strftime("%Y-%m-%d"), "count": int(cnt or 0), "avg_points": float(avg_points or 0.0)}
            for day, cnt, avg_points in daily_rows
        ]

        if as_json:
            return _json_text(
                {
                    "tool": "trend_analysis",
                    "status": "ok" if (recent_cnt_i > 0 or prev_cnt_i > 0 or daily_records) else "empty",
                    "request": {"topic": topic, "window": window},
                    "data": {
                        "topic": topic,
                        "window": window,
                        "recent_count": recent_cnt_i,
                        "previous_count": prev_cnt_i,
                        "count_delta": change_cnt,
                        "count_delta_pct": change_pct,
                        "avg_points_recent": recent_avg_f,
                        "avg_points_previous": prev_avg_f,
                        "daily": daily_records,
                    },
                }
            )

        lines = [
            f"Trend for topic: {topic}",
            f"Window: recent {window} days vs previous {window} days",
            f"Article count: {recent_cnt_i} vs {prev_cnt_i} -> {delta}",
            f"Avg points: {recent_avg_f:.1f} vs {prev_avg_f:.1f}",
            "Daily breakdown:",
        ]
        if daily_rows:
            for day, cnt, avg_points in daily_rows:
                lines.append(f"  {day.strftime('%Y-%m-%d')}: count={cnt}, avg_points={avg_points}")
        else:
            lines.append("  no matched records")
        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] trend_analysis failed: {exc}")
        if as_json:
            return _json_text(
                {
                    "tool": "trend_analysis",
                    "status": "error",
                    "request": {"topic": topic, "window": window},
                    "data": None,
                    "error": f"trend_analysis failed: {exc}",
                }
            )
        return f"trend_analysis failed: {exc}"
    finally:
        put_conn(conn)


def trend_analysis_skill(payload: TrendAnalysisSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    raw_output = trend_analysis(
        topic=request.get("topic", ""),
        window=int(request.get("window", 7)),
        response_format="json",
    )

    try:
        parsed = json.loads(raw_output)
    except Exception as exc:
        return build_error_envelope(
            tool="trend_analysis",
            request=request,
            error="trend_analysis_json_parse_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "raw_preview": str(raw_output)[:500],
            },
        )

    status = str(parsed.get("status", "error")).lower()
    if status not in {"ok", "empty", "error"}:
        return build_error_envelope(
            tool="trend_analysis",
            request=request,
            error="trend_analysis_invalid_status",
            diagnostics={"status": parsed.get("status"), "raw_output": parsed},
        )

    if status == "error":
        return build_error_envelope(
            tool="trend_analysis",
            request=request,
            error=str(parsed.get("error") or "trend_analysis_failed"),
            diagnostics={"raw_output": parsed},
        )

    data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}

    evidence_records: list[dict[str, Any]] = []
    if status == "ok" and str(request.get("topic", "")).strip():
        evidence_limit = min(6, max(3, int(request.get("window", 7))))
        query_raw = query_news(
            query=str(request["topic"]),
            days=max(3, int(request.get("window", 7)) * 2),
            limit=evidence_limit,
            sort="heat_desc",
            response_format="json",
        )
        try:
            query_parsed = json.loads(query_raw)
            if isinstance(query_parsed, dict):
                records = query_parsed.get("records")
                if isinstance(records, list):
                    evidence_records = records
        except Exception:
            evidence_records = []

    evidence = _evidence_from_records(evidence_records, max_items=6)
    return SkillEnvelope(
        tool="trend_analysis",
        status=status,
        request=request,
        data=data,
        evidence=evidence,
        diagnostics={"evidence_backfill_count": len(evidence), "window": request.get("window")},
    )
