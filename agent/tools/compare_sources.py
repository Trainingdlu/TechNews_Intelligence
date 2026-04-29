"""Compare-sources tool implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.tool_contracts import ToolEnvelope, build_tool_empty_envelope, build_tool_error_envelope
from .helpers import _clamp_int, _evidence_from_records
from .rerank_aggregation import format_reranked_evidence, retrieve_and_rerank
from .schemas import CompareSourcesToolInput
from .semantic_pool import fetch_semantic_url_pool


def _format_compare_sources_result(result: dict) -> str:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    topic = data.get("topic", "")
    days = data.get("days", 14)
    if result.get("status") == "empty":
        return f"No comparison data for '{topic}' in {days} days."
    if result.get("status") == "error":
        return str(result.get("error") or "compare_sources failed")

    lines = [f"Source comparison: {topic} (last {days} days)", "Stats:"]
    metrics_by_source = data.get("metrics_by_source") if isinstance(data.get("metrics_by_source"), dict) else {}
    for source, metrics in sorted(metrics_by_source.items()):
        lines.append(
            f"  {source}: count={metrics.get('count', 0)}, "
            f"avg_points={metrics.get('avg_points', 0)}, "
            f"sentiment(P/N/Ng)={metrics.get('positive_count', 0)}/"
            f"{metrics.get('neutral_count', 0)}/{metrics.get('negative_count', 0)}"
        )

    lines.append("Top evidence:")
    for item in data.get("top_evidence", []) or []:
        lines.append(
            f"  [{item.get('source')}] #{item.get('rank')} {item.get('title')} | "
            f"points={item.get('metadata', {}).get('points')} | {str(item.get('created_at') or '')[:16].replace('T', ' ')} | "
            f"{item.get('url')}"
        )

    reranked_text = data.get("reranked_output")
    if reranked_text:
        lines.append(str(reranked_text))
    return "\n".join(lines)


def _compare_sources_structured(topic: str, days: int = 14) -> dict:
    """Compare HackerNews vs TechCrunch coverage and sentiment for a topic."""
    print(f"\n[Tool] compare_sources: topic={topic}, days={days}")
    if not topic or not topic.strip():
        return {
            "status": "error",
            "error_code": "compare_sources_missing_topic",
            "error": "compare_sources requires topic.",
            "data": {"topic": topic, "days": days},
            "evidence": [],
            "diagnostics": {"topic": topic},
        }

    topic = topic.strip()
    days = _clamp_int(days, 1, 90)

    # Semantic vector pool replaces the old ILIKE + hardcoded dictionary approach
    url_pool = fetch_semantic_url_pool(topic, days=days, limit=200)
    if not url_pool:
        return {
            "status": "empty",
            "data": {"topic": topic, "days": days, "metrics_by_source": {}, "source_mix": {}, "top_evidence": []},
            "evidence": [],
            "diagnostics": {
                "topic": topic,
                "candidate_count": 0,
                "evidence_count": 0,
                "retrieval_mode": "semantic_url_pool",
                "fallback": False,
                "empty_reason": "no_semantic_matches",
            },
        }

    urls = [u for u, _ in url_pool]
    pool_scores = {u: float(score or 0.0) for u, score in url_pool}

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
                    ROW_NUMBER() OVER (
                        PARTITION BY source_type
                        ORDER BY array_position(%s::text[], url) ASC NULLS LAST, created_at DESC
                    ) AS rn
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT source_type, headline, url, points, created_at, rn
            FROM ranked
            WHERE rn <= 3
            ORDER BY source_type, rn
            """,
            (urls, f"{days} days", urls),
        )
        top_rows = cur.fetchall()
        cur.close()

        if not stats_rows:
            return {
                "status": "empty",
                "data": {"topic": topic, "days": days, "metrics_by_source": {}, "source_mix": {}, "top_evidence": []},
                "evidence": [],
                "diagnostics": {
                    "topic": topic,
                    "candidate_count": len(url_pool),
                    "evidence_count": 0,
                    "retrieval_mode": "semantic_url_pool",
                    "fallback": False,
                    "empty_reason": "no_source_rows",
                },
            }

        total_count = sum(int(row[1] or 0) for row in stats_rows)
        metrics_by_source: dict[str, dict] = {}
        source_mix: dict[str, dict] = {}
        for src, cnt, avg_points, pos, neu, neg in stats_rows:
            count = int(cnt or 0)
            metrics_by_source[str(src)] = {
                "count": count,
                "avg_points": float(avg_points or 0.0),
                "positive_count": int(pos or 0),
                "neutral_count": int(neu or 0),
                "negative_count": int(neg or 0),
            }
            source_mix[str(src)] = {
                "count": count,
                "share": (float(count) / float(total_count)) if total_count else 0.0,
            }

        top_evidence: list[dict] = []
        for src, headline, url, points, created_at, rank in top_rows:
            created_text = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or "")
            top_evidence.append(
                {
                    "rank": int(rank or 0),
                    "source": str(src or ""),
                    "title": str(headline or ""),
                    "url": str(url or ""),
                    "created_at": created_text,
                    "score": float(points or 0),
                    "match_score": pool_scores.get(str(url or "")),
                    "metadata": {"points": int(points or 0)},
                }
            )

        reranked_output = ""
        rerank_meta = {}
        try:
            reranked, _, rerank_meta = retrieve_and_rerank(
                topic, days=days, top_k=5,
            )
            reranked_output = format_reranked_evidence(
                reranked, header=f"Reranked Evidence: {topic}",
            )
        except Exception as rerank_exc:
            print(f"[Warn] compare_sources rerank failed (non-fatal): {rerank_exc}")

        data = {
            "topic": topic,
            "days": days,
            "metrics_by_source": metrics_by_source,
            "source_mix": source_mix,
            "top_evidence": top_evidence,
            "confidence": "High" if total_count >= 8 and len(top_evidence) >= 4 else "Medium" if top_evidence else "Low",
            "reranked_output": reranked_output,
        }
        data["raw_output"] = _format_compare_sources_result({"status": "ok", "data": data})
        return {
            "status": "ok",
            "data": data,
            "evidence": top_evidence,
            "diagnostics": {
                "topic": topic,
                "candidate_count": len(url_pool),
                "evidence_count": len(top_evidence),
                "retrieval_mode": "semantic_url_pool",
                "fallback": False,
                "rerank": rerank_meta,
            },
        }
    except Exception as exc:
        print(f"[Error] compare_sources failed: {exc}")
        return {
            "status": "error",
            "error_code": "compare_sources_execution_failed",
            "error": "compare_sources_execution_failed",
            "data": {"topic": topic, "days": days},
            "evidence": [],
            "diagnostics": {"exception_type": type(exc).__name__, "exception_message": str(exc), "topic": topic},
        }
    finally:
        put_conn(conn)


def compare_sources(topic: str, days: int = 14) -> str:
    """Compare HackerNews vs TechCrunch coverage and sentiment for a topic."""
    return _format_compare_sources_result(_compare_sources_structured(topic, days=days))


def compare_sources_tool(payload: CompareSourcesToolInput) -> ToolEnvelope:
    request = payload.model_dump(mode="python")
    result = _compare_sources_structured(topic=request["topic"], days=int(request.get("days", 14)))
    status = str(result.get("status") or "error")
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    if status == "error":
        return build_tool_error_envelope(
            tool="compare_sources",
            request=request,
            error=str(result.get("error_code") or "compare_sources_failed"),
            data={**data, "raw_output": _format_compare_sources_result(result)},
            diagnostics=diagnostics,
        )
    if status == "empty":
        return build_tool_empty_envelope(
            tool="compare_sources",
            request=request,
            empty_reason=str(diagnostics.get("empty_reason") or "no_comparison_data"),
            data={**data, "raw_output": _format_compare_sources_result(result)},
            diagnostics=diagnostics,
        )

    evidence = _evidence_from_records(result.get("evidence", []) or [], max_items=6)
    data = dict(data)
    data.setdefault("raw_output", _format_compare_sources_result(result))
    return ToolEnvelope(
        tool="compare_sources",
        status="ok",
        request=request,
        data=data,
        evidence=evidence,
        diagnostics={**diagnostics, "evidence_count": len(evidence)},
    )


