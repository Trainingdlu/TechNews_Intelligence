"""Build-timeline skill implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_text_output
from .schemas import BuildTimelineSkillInput
from .semantic_pool import fetch_semantic_url_pool


def build_timeline(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build a chronological event timeline for a topic, company, or product."""
    print(f"\n[Tool] build_timeline: topic={topic}, days={days}, limit={limit}")
    if not topic or not topic.strip():
        return "build_timeline requires topic."

    days = _clamp_int(days, 1, 180)
    limit = _clamp_int(limit, 3, 40)

    # Semantic vector pool replaces the old ILIKE + hardcoded dictionary approach
    url_pool = fetch_semantic_url_pool(topic, days=days, limit=limit * 5)

    # Auto-retry with wider window if empty and original window is narrow
    if not url_pool and days < 90:
        retry_days = min(180, max(60, days * 2))
        print(f"[Tool] build_timeline: empty for {days}d, auto-retrying with {retry_days}d")
        url_pool = fetch_semantic_url_pool(topic, days=retry_days, limit=limit * 5)
        if url_pool:
            days = retry_days

    if not url_pool:
        return f"No timeline data for '{topic}' in {days} days."

    urls = [u for u, _ in url_pool]

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                created_at,
                source_type,
                COALESCE(title_cn, title) AS headline,
                sentiment,
                points,
                url
            FROM view_dashboard_news
            WHERE url = ANY(%s)
              AND created_at >= NOW() - %s::interval
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (urls, f"{days} days", limit),
        )
        rows = cur.fetchall()
        cur.close()

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
