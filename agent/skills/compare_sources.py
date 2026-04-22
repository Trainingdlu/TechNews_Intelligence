"""Compare-sources skill implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_text_output
from .rerank_aggregation import format_reranked_evidence, retrieve_and_rerank
from .schemas import CompareSourcesSkillInput
from .semantic_pool import fetch_semantic_url_pool


def compare_sources(topic: str, days: int = 14) -> str:
    """Compare HackerNews vs TechCrunch coverage and sentiment for a topic."""
    print(f"\n[Tool] compare_sources: topic={topic}, days={days}")
    if not topic or not topic.strip():
        return "compare_sources requires topic."

    days = _clamp_int(days, 1, 90)

    # Semantic vector pool replaces the old ILIKE + hardcoded dictionary approach
    url_pool = fetch_semantic_url_pool(topic, days=days, limit=200)
    if not url_pool:
        return f"No comparison data for '{topic}' in {days} days."

    urls = [u for u, _ in url_pool]

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                source_type,
                COUNT(*) AS cnt,
                ROUND(AVG(points)::numeric, 1) AS avg_points,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neu_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
              AND url = ANY(%s)
            GROUP BY source_type
            ORDER BY source_type
            """,
            (f"{days} days", urls),
        )
        stats_rows = cur.fetchall()

        cur.execute(
            """
            WITH ranked AS (
                SELECT
                    source_type,
                    COALESCE(title_cn, title) AS headline,
                    url,
                    points,
                    created_at,
                    ROW_NUMBER() OVER (PARTITION BY source_type ORDER BY points DESC NULLS LAST, created_at DESC) AS rn
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT source_type, headline, url, points, created_at, rn
            FROM ranked
            WHERE rn <= 3
            ORDER BY source_type, rn
            """,
            (f"{days} days", urls),
        )
        top_rows = cur.fetchall()
        cur.close()

        if not stats_rows:
            return f"No comparison data for '{topic}' in {days} days."

        lines = [f"Source comparison: {topic} (last {days} days)", "Stats:"]
        for src, cnt, avg_points, pos, neu, neg in stats_rows:
            lines.append(f"  {src}: count={cnt}, avg_points={avg_points}, sentiment(P/N/Ng)={pos}/{neu}/{neg}")

        lines.append("Top evidence:")
        for src, headline, url, points, created_at, rank in top_rows:
            lines.append(
                f"  [{src}] #{rank} {headline} | points={points} | {created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )

        # --- Reranked Top Evidence (Top-5) ---
        try:
            reranked, _, rerank_meta = retrieve_and_rerank(
                topic, days=days, top_k=5, pool_limit=100,
            )
            evidence_block = format_reranked_evidence(
                reranked, header=f"Reranked Evidence: {topic}",
            )
            if evidence_block:
                lines.append(evidence_block)
        except Exception as rerank_exc:
            print(f"[Warn] compare_sources rerank failed (non-fatal): {rerank_exc}")

        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] compare_sources failed: {exc}")
        return f"compare_sources failed: {exc}"
    finally:
        put_conn(conn)


def compare_sources_skill(payload: CompareSourcesSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    try:
        raw_output = compare_sources(topic=request["topic"], days=int(request.get("days", 14)))
    except Exception as exc:
        return build_error_envelope(
            tool="compare_sources",
            request=request,
            error="compare_sources_execution_failed",
            diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    if raw_output.startswith("No comparison data") or raw_output.startswith("compare_sources"):
        is_error = "failed" in raw_output
        return SkillEnvelope(
            tool="compare_sources",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"topic": request["topic"]},
        )

    evidence = _evidence_from_text_output(raw_output, max_items=6)
    return SkillEnvelope(
        tool="compare_sources",
        status="ok",
        request=request,
        data={"raw_output": raw_output},
        evidence=evidence,
        diagnostics={"topic": request["topic"]},
    )
