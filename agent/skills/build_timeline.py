"""Build-timeline skill implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_text_output
from .schemas import BuildTimelineSkillInput
from .sql_builders import _build_topic_where_clause


def build_timeline(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build a chronological event timeline for a topic, company, or product."""
    print(f"\n[Tool] build_timeline: topic={topic}, days={days}, limit={limit}")
    if not topic or not topic.strip():
        return "build_timeline requires topic."

    days = _clamp_int(days, 1, 180)
    limit = _clamp_int(limit, 3, 40)
    topic_clause, topic_params = _build_topic_where_clause(topic)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                created_at,
                source_type,
                COALESCE(title_cn, title) AS headline,
                sentiment,
                points,
                url
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
              AND {topic_clause}
            ORDER BY created_at ASC
            LIMIT %s
            """,
            tuple([f"{days} days"] + topic_params + [limit]),
        )
        rows = cur.fetchall()
        cur.close()

        if not rows and days < 90:
            retry_days = min(180, max(60, days * 2))
            print(f"[Tool] build_timeline: empty for {days}d, auto-retrying with {retry_days}d")
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT
                    created_at,
                    source_type,
                    COALESCE(title_cn, title) AS headline,
                    sentiment,
                    points,
                    url
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND {topic_clause}
                ORDER BY created_at ASC
                LIMIT %s
                """,
                tuple([f"{retry_days} days"] + topic_params + [limit]),
            )
            rows = cur.fetchall()
            cur.close()
            if rows:
                days = retry_days

        if not rows:
            return f"No timeline data for '{topic}' in {days} days."

        lines = [f"Timeline: {topic} (last {days} days, max {limit})"]
        for idx, (created_at, src, headline, senti, points, url) in enumerate(rows, 1):
            lines.append(
                f"{idx}. {created_at.strftime('%Y-%m-%d %H:%M')} | {src} | {senti} | points={points}\n"
                f"   {headline}\n"
                f"   {url}"
            )
        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] build_timeline failed: {exc}")
        return f"build_timeline failed: {exc}"
    finally:
        put_conn(conn)


def build_timeline_skill(payload: BuildTimelineSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    try:
        raw_output = build_timeline(
            topic=request["topic"],
            days=int(request.get("days", 30)),
            limit=int(request.get("limit", 12)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="build_timeline",
            request=request,
            error="build_timeline_execution_failed",
            diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    if raw_output.startswith("No timeline data") or raw_output.startswith("build_timeline"):
        is_error = "failed" in raw_output or "requires" in raw_output
        return SkillEnvelope(
            tool="build_timeline",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"topic": request["topic"]},
        )

    evidence = _evidence_from_text_output(raw_output, max_items=12)
    return SkillEnvelope(
        tool="build_timeline",
        status="ok",
        request=request,
        data={"raw_output": raw_output, "event_count": len(evidence)},
        evidence=evidence,
        diagnostics={"topic": request["topic"]},
    )
