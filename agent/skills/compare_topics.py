"""Compare-topics skill implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_text_output
from .schemas import CompareTopicsSkillInput


def compare_topics(topic_a: str, topic_b: str, days: int = 14) -> str:
    """Compare two entities or topics side-by-side with DB evidence."""
    print(f"\n[Tool] compare_topics: A={topic_a}, B={topic_b}, days={days}")
    if not topic_a or not topic_a.strip() or not topic_b or not topic_b.strip():
        return "compare_topics requires topic_a and topic_b."

    topic_a = topic_a.strip()
    topic_b = topic_b.strip()
    days = _clamp_int(days, 1, 90)
    split_days = max(3, days // 2)
    prev_days = max(1, days - split_days)
    query_a = f"%{topic_a}%"
    query_b = f"%{topic_b}%"

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    points,
                    sentiment
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT
                grp,
                COUNT(*) AS cnt,
                ROUND(AVG(points)::numeric, 1) AS avg_points,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neu_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp
            ORDER BY grp
            """,
            (query_a, query_a, query_a, query_b, query_b, query_b, f"{days} days"),
        )
        metric_rows = cur.fetchall()

        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    source_type
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT grp, source_type, COUNT(*) AS cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp, source_type
            ORDER BY grp, cnt DESC, source_type ASC
            """,
            (query_a, query_a, query_a, query_b, query_b, query_b, f"{days} days"),
        )
        source_rows = cur.fetchall()

        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    created_at
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT
                grp,
                COUNT(*) FILTER (WHERE created_at >= NOW() - %s::interval) AS recent_cnt,
                COUNT(*) FILTER (WHERE created_at < NOW() - %s::interval) AS prev_cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp
            ORDER BY grp
            """,
            (
                query_a,
                query_a,
                query_a,
                query_b,
                query_b,
                query_b,
                f"{days} days",
                f"{split_days} days",
                f"{split_days} days",
            ),
        )
        momentum_rows = cur.fetchall()

        cur.execute(
            """
            WITH candidates AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    source_type,
                    COALESCE(title_cn, title) AS headline,
                    url,
                    points,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            CASE
                                WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                                WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                                ELSE NULL
                            END
                        ORDER BY points DESC NULLS LAST, created_at DESC
                    ) AS rn
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT grp, source_type, headline, url, points, created_at, rn
            FROM candidates
            WHERE grp IS NOT NULL AND rn <= 3
            ORDER BY grp, rn
            """,
            (
                query_a,
                query_a,
                query_a,
                query_b,
                query_b,
                query_b,
                query_a,
                query_a,
                query_a,
                query_b,
                query_b,
                query_b,
                f"{days} days",
            ),
        )
        top_rows = cur.fetchall()
        cur.close()

        metric_map: dict[str, tuple | None] = {"A": None, "B": None}
        for row in metric_rows:
            metric_map[row[0]] = row

        source_map: dict[str, dict[str, int]] = {"A": {}, "B": {}}
        for grp, source_type, cnt in source_rows:
            bucket = source_map.setdefault(grp, {})
            bucket[source_type] = int(cnt)

        momentum_map: dict[str, tuple[int, int]] = {"A": (0, 0), "B": (0, 0)}
        for grp, recent_cnt, prev_cnt in momentum_rows:
            momentum_map[grp] = (int(recent_cnt or 0), int(prev_cnt or 0))

        def _fmt(name: str, row: tuple | None) -> str:
            if not row:
                return f"{name}: count=0, avg_points=0, sentiment(P/N/Ng)=0/0/0"
            _, cnt, avg_points, pos, neu, neg = row
            return f"{name}: count={cnt}, avg_points={avg_points}, sentiment(P/N/Ng)={pos}/{neu}/{neg}"

        lines = [f"Topic comparison: {topic_a} vs {topic_b} (last {days} days)", "Stats:"]
        lines.append("  " + _fmt(topic_a, metric_map["A"]))
        lines.append("  " + _fmt(topic_b, metric_map["B"]))
        lines.append(f"Time split: recent={split_days}d, previous={prev_days}d")
        lines.append("Momentum:")
        for grp, label in (("A", topic_a), ("B", topic_b)):
            recent_cnt, prev_cnt = momentum_map.get(grp, (0, 0))
            if prev_cnt == 0:
                delta_text = "+new" if recent_cnt > 0 else "0.0%"
            else:
                delta_text = f"{((float(recent_cnt) - float(prev_cnt)) / float(prev_cnt)) * 100:+.1f}%"
            lines.append(f"  {label}: recent={recent_cnt}, previous={prev_cnt}, delta={delta_text}")

        lines.append("Source mix:")
        for grp, label in (("A", topic_a), ("B", topic_b)):
            source_bucket = source_map.get(grp, {})
            total_cnt = metric_map[grp][1] if metric_map.get(grp) else 0
            if not source_bucket:
                lines.append(f"  {label}: no source records")
                continue
            mix_parts: list[str] = []
            for source_type, source_cnt in sorted(source_bucket.items(), key=lambda x: (-x[1], x[0])):
                source_share = (float(source_cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
                mix_parts.append(f"{source_type}={source_cnt}({source_share:.1f}%)")
            lines.append(f"  {label}: " + ", ".join(mix_parts))

        lines.append("Evidence URLs:")
        for grp, source_type, headline, url, points, created_at, rank in top_rows:
            label = topic_a if grp == "A" else topic_b
            lines.append(
                f"  [{label}] #{rank} [{source_type}] {headline} | points={points} | "
                f"{created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )

        count_a = metric_map["A"][1] if metric_map["A"] else 0
        count_b = metric_map["B"][1] if metric_map["B"] else 0
        url_count = len(top_rows)
        if count_a > 0 and count_b > 0 and url_count >= 2:
            confidence = "High"
        elif (count_a > 0 or count_b > 0) and url_count >= 1:
            confidence = "Medium"
        else:
            confidence = "Low"
        lines.append(f"Confidence: {confidence}")
        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] compare_topics failed: {exc}")
        return f"compare_topics failed: {exc}"
    finally:
        if conn is not None:
            put_conn(conn)


def compare_topics_skill(payload: CompareTopicsSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    try:
        raw_output = compare_topics(
            topic_a=request["topic_a"],
            topic_b=request["topic_b"],
            days=int(request.get("days", 14)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="compare_topics",
            request=request,
            error="compare_topics_execution_failed",
            diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    if raw_output.startswith("compare_topics requires") or raw_output.startswith("compare_topics failed"):
        return SkillEnvelope(
            tool="compare_topics",
            status="error",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output,
            diagnostics={},
        )

    confidence = None
    for line in raw_output.splitlines():
        if line.strip().startswith("Confidence:"):
            confidence = line.strip().split(":", 1)[1].strip()

    evidence = _evidence_from_text_output(raw_output, max_items=6)
    status = "ok" if evidence else "empty"
    return SkillEnvelope(
        tool="compare_topics",
        status=status,
        request=request,
        data={"raw_output": raw_output, "confidence": confidence},
        evidence=evidence,
        diagnostics={"topic_a": request["topic_a"], "topic_b": request["topic_b"], "confidence": confidence},
    )
