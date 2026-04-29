"""Compare-topics tool implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.tool_contracts import ToolEnvelope, build_tool_empty_envelope, build_tool_error_envelope
from .helpers import _clamp_int, _evidence_from_records
from .rerank_aggregation import format_reranked_evidence, retrieve_and_rerank
from .schemas import CompareTopicsToolInput
from .semantic_pool import fetch_semantic_url_pool


def _resolve_topic_pool_scores(
    topic_a: str, topic_b: str, *, days: int
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Build two non-overlapping URL lists from semantic pools.

    When a URL appears in both pools, it is assigned to the topic
    with the higher match_score (intersection arbitration).
    """
    pool_a = fetch_semantic_url_pool(topic_a, days=days, limit=200)
    pool_b = fetch_semantic_url_pool(topic_b, days=days, limit=200)

    sim_a = {url: sim for url, sim in pool_a}
    sim_b = {url: sim for url, sim in pool_b}

    assigned_a: list[tuple[str, float]] = []
    assigned_b: list[tuple[str, float]] = []

    for url in set(sim_a.keys()) | set(sim_b.keys()):
        sa = sim_a.get(url, -1.0)
        sb = sim_b.get(url, -1.0)
        if sa >= sb:
            assigned_a.append((url, sa))
        else:
            assigned_b.append((url, sb))

    return (
        sorted(assigned_a, key=lambda item: item[1], reverse=True),
        sorted(assigned_b, key=lambda item: item[1], reverse=True),
    )


def _resolve_topic_pools(
    topic_a: str, topic_b: str, *, days: int
) -> tuple[list[str], list[str]]:
    pool_a, pool_b = _resolve_topic_pool_scores(topic_a, topic_b, days=days)
    return [url for url, _score in pool_a], [url for url, _score in pool_b]


def _format_compare_topics_result(result: dict) -> str:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    topic_a = data.get("topic_a", "")
    topic_b = data.get("topic_b", "")
    days = data.get("days", 14)
    if result.get("status") == "empty":
        return f"No semantically matched articles for either '{topic_a}' or '{topic_b}' in the last {days} days."
    if result.get("status") == "error":
        return str(result.get("error") or "compare_topics failed")

    lines = [f"Topic comparison: {topic_a} vs {topic_b} (last {days} days)", "Stats:"]
    metrics = data.get("metrics_by_topic") if isinstance(data.get("metrics_by_topic"), dict) else {}
    for label in (topic_a, topic_b):
        item = metrics.get(label, {})
        lines.append(
            f"  {label}: count={item.get('count', 0)}, avg_points={item.get('avg_points', 0)}, "
            f"sentiment(P/N/Ng)={item.get('positive_count', 0)}/"
            f"{item.get('neutral_count', 0)}/{item.get('negative_count', 0)}"
        )
    lines.append("Momentum:")
    momentum = data.get("momentum") if isinstance(data.get("momentum"), dict) else {}
    for label in (topic_a, topic_b):
        item = momentum.get(label, {})
        lines.append(
            f"  {label}: recent={item.get('recent_count', 0)}, "
            f"previous={item.get('previous_count', 0)}, delta={item.get('delta_text', '0.0%')}"
        )
    lines.append("Source mix:")
    source_mix = data.get("source_mix") if isinstance(data.get("source_mix"), dict) else {}
    for label in (topic_a, topic_b):
        bucket = source_mix.get(label, {})
        if not bucket:
            lines.append(f"  {label}: no source records")
            continue
        parts = [
            f"{source}={item.get('count', 0)}({float(item.get('share', 0.0)) * 100:.1f}%)"
            for source, item in sorted(bucket.items())
        ]
        lines.append(f"  {label}: " + ", ".join(parts))
    lines.append("Evidence URLs:")
    for item in data.get("top_evidence", []) or []:
        lines.append(
            f"  [{item.get('topic')}] #{item.get('rank')} [{item.get('source')}] {item.get('title')} | "
            f"points={item.get('metadata', {}).get('points')} | {str(item.get('created_at') or '')[:16].replace('T', ' ')} | "
            f"{item.get('url')}"
        )
    lines.append(f"Confidence: {data.get('confidence')}")
    return "\n".join(lines)


def _compare_topics_structured(topic_a: str, topic_b: str, days: int = 14) -> dict:
    print(f"\n[Tool] compare_topics: A={topic_a}, B={topic_b}, days={days}")
    if not topic_a or not topic_a.strip() or not topic_b or not topic_b.strip():
        return {
            "status": "error",
            "error_code": "compare_topics_missing_topic",
            "error": "compare_topics requires topic_a and topic_b.",
            "data": {"topic_a": topic_a, "topic_b": topic_b, "days": days},
            "evidence": [],
            "diagnostics": {},
        }

    topic_a = topic_a.strip()
    topic_b = topic_b.strip()
    days = _clamp_int(days, 1, 90)
    split_days = max(3, days // 2)
    prev_days = max(1, days - split_days)
    pool_a, pool_b = _resolve_topic_pool_scores(topic_a, topic_b, days=days)
    urls_a = [url for url, _score in pool_a]
    urls_b = [url for url, _score in pool_b]
    score_by_url = {url: score for url, score in pool_a + pool_b}
    if not urls_a and not urls_b:
        return {
            "status": "empty",
            "data": {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "days": days,
                "metrics_by_topic": {},
                "momentum": {},
                "source_mix": {},
                "top_evidence": [],
                "confidence": "Low",
            },
            "evidence": [],
            "diagnostics": {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "candidate_count": 0,
                "evidence_count": 0,
                "retrieval_mode": "semantic_url_pool",
                "fallback": False,
                "empty_reason": "no_semantic_matches",
            },
        }

    urls_all = urls_a + urls_b
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE WHEN url = ANY(%s) THEN 'A' WHEN url = ANY(%s) THEN 'B' ELSE NULL END AS grp,
                    points,
                    sentiment
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT grp, COUNT(*) AS cnt, ROUND(AVG(points)::numeric, 1) AS avg_points,
                   COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                   COUNT(*) FILTER (WHERE sentiment = 'Neutral') AS neu_cnt,
                   COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp
            ORDER BY grp
            """,
            (urls_a, urls_b, f"{days} days", urls_all),
        )
        metric_rows = cur.fetchall()
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE WHEN url = ANY(%s) THEN 'A' WHEN url = ANY(%s) THEN 'B' ELSE NULL END AS grp,
                    source_type
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT grp, source_type, COUNT(*) AS cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp, source_type
            ORDER BY grp, cnt DESC, source_type ASC
            """,
            (urls_a, urls_b, f"{days} days", urls_all),
        )
        source_rows = cur.fetchall()
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE WHEN url = ANY(%s) THEN 'A' WHEN url = ANY(%s) THEN 'B' ELSE NULL END AS grp,
                    created_at
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT grp,
                   COUNT(*) FILTER (WHERE created_at >= NOW() - %s::interval) AS recent_cnt,
                   COUNT(*) FILTER (WHERE created_at < NOW() - %s::interval) AS prev_cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp
            ORDER BY grp
            """,
            (urls_a, urls_b, f"{days} days", urls_all, f"{split_days} days", f"{split_days} days"),
        )
        momentum_rows = cur.fetchall()
        cur.execute(
            """
            WITH candidates AS (
                SELECT
                    CASE WHEN url = ANY(%s) THEN 'A' WHEN url = ANY(%s) THEN 'B' ELSE NULL END AS grp,
                    source_type,
                    COALESCE(title_cn, title) AS headline,
                    url,
                    points,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY CASE WHEN url = ANY(%s) THEN 'A' WHEN url = ANY(%s) THEN 'B' ELSE NULL END
                        ORDER BY
                            CASE
                                WHEN url = ANY(%s) THEN array_position(%s::text[], url)
                                WHEN url = ANY(%s) THEN array_position(%s::text[], url)
                                ELSE NULL
                            END ASC NULLS LAST,
                            created_at DESC
                    ) AS rn
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT grp, source_type, headline, url, points, created_at, rn
            FROM candidates
            WHERE grp IS NOT NULL AND rn <= 3
            ORDER BY grp, rn
            """,
            (urls_a, urls_b, urls_a, urls_b, urls_a, urls_a, urls_b, urls_b, f"{days} days", urls_all),
        )
        top_rows = cur.fetchall()
        cur.close()

        label_by_grp = {"A": topic_a, "B": topic_b}
        metrics_by_topic = {
            topic_a: {"count": 0, "avg_points": 0.0, "positive_count": 0, "neutral_count": 0, "negative_count": 0},
            topic_b: {"count": 0, "avg_points": 0.0, "positive_count": 0, "neutral_count": 0, "negative_count": 0},
        }
        for grp, cnt, avg_points, pos, neu, neg in metric_rows:
            label = label_by_grp.get(grp, str(grp))
            metrics_by_topic[label] = {
                "count": int(cnt or 0),
                "avg_points": float(avg_points or 0.0),
                "positive_count": int(pos or 0),
                "neutral_count": int(neu or 0),
                "negative_count": int(neg or 0),
            }

        source_mix: dict[str, dict[str, dict[str, float | int]]] = {topic_a: {}, topic_b: {}}
        for grp, source_type, cnt in source_rows:
            label = label_by_grp.get(grp, str(grp))
            total = metrics_by_topic.get(label, {}).get("count", 0) or 0
            count = int(cnt or 0)
            source_mix.setdefault(label, {})[str(source_type)] = {
                "count": count,
                "share": (float(count) / float(total)) if total else 0.0,
            }

        momentum: dict[str, dict] = {}
        for label in (topic_a, topic_b):
            momentum[label] = {"recent_count": 0, "previous_count": 0, "delta_pct": None, "delta_text": "0.0%"}
        for grp, recent_cnt, prev_cnt in momentum_rows:
            label = label_by_grp.get(grp, str(grp))
            recent_i = int(recent_cnt or 0)
            prev_i = int(prev_cnt or 0)
            if prev_i == 0:
                delta_text = "+new" if recent_i > 0 else "0.0%"
                delta_pct = None
            else:
                delta_pct = ((float(recent_i) - float(prev_i)) / float(prev_i)) * 100
                delta_text = f"{delta_pct:+.1f}%"
            momentum[label] = {
                "recent_count": recent_i,
                "previous_count": prev_i,
                "delta_pct": delta_pct,
                "delta_text": delta_text,
            }

        top_evidence: list[dict] = []
        for grp, source_type, headline, url, points, created_at, rank in top_rows:
            label = label_by_grp.get(grp, str(grp))
            created_text = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or "")
            top_evidence.append(
                {
                    "rank": int(rank or 0),
                    "topic": label,
                    "source": str(source_type or ""),
                    "title": str(headline or ""),
                    "url": str(url or ""),
                    "created_at": created_text,
                    "score": float(points or 0),
                    "match_score": score_by_url.get(str(url or "")),
                    "metadata": {"points": int(points or 0), "group": str(grp or "")},
                }
            )

        count_a = metrics_by_topic[topic_a]["count"]
        count_b = metrics_by_topic[topic_b]["count"]
        if count_a > 0 and count_b > 0 and len(top_evidence) >= 2:
            confidence = "High"
        elif (count_a > 0 or count_b > 0) and top_evidence:
            confidence = "Medium"
        else:
            confidence = "Low"

        data = {
            "topic_a": topic_a,
            "topic_b": topic_b,
            "days": days,
            "time_split": {"recent_days": split_days, "previous_days": prev_days},
            "metrics_by_topic": metrics_by_topic,
            "momentum": momentum,
            "source_mix": source_mix,
            "top_evidence": top_evidence,
            "confidence": confidence,
        }
        data["raw_output"] = _format_compare_topics_result({"status": "ok", "data": data})
        return {
            "status": "ok" if top_evidence else "empty",
            "data": data,
            "evidence": top_evidence,
            "diagnostics": {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "candidate_count": len(urls_all),
                "evidence_count": len(top_evidence),
                "retrieval_mode": "semantic_url_pool",
                "fallback": False,
                "confidence": confidence,
                **({"empty_reason": "no_evidence_rows"} if not top_evidence else {}),
            },
        }
    except Exception as exc:
        print(f"[Error] compare_topics structured failed: {exc}")
        return {
            "status": "error",
            "error_code": "compare_topics_execution_failed",
            "error": "compare_topics_execution_failed",
            "data": {"topic_a": topic_a, "topic_b": topic_b, "days": days},
            "evidence": [],
            "diagnostics": {"exception_type": type(exc).__name__, "exception_message": str(exc)},
        }
    finally:
        if conn is not None:
            put_conn(conn)


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

    # Semantic vector pools with intersection arbitration
    urls_a, urls_b = _resolve_topic_pools(topic_a, topic_b, days=days)
    if not urls_a and not urls_b:
        return (
            f"No semantically matched articles for either '{topic_a}' or "
            f"'{topic_b}' in the last {days} days."
        )

    # Combined list for the outer WHERE filter
    urls_all = urls_a + urls_b

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()

        # --- 1. Core metrics per group ---
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN url = ANY(%s) THEN 'A'
                        WHEN url = ANY(%s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    points,
                    sentiment
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
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
            (urls_a, urls_b, f"{days} days", urls_all),
        )
        metric_rows = cur.fetchall()

        # --- 2. Source mix ---
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN url = ANY(%s) THEN 'A'
                        WHEN url = ANY(%s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    source_type
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT grp, source_type, COUNT(*) AS cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp, source_type
            ORDER BY grp, cnt DESC, source_type ASC
            """,
            (urls_a, urls_b, f"{days} days", urls_all),
        )
        source_rows = cur.fetchall()

        # --- 3. Momentum (recent vs previous) ---
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN url = ANY(%s) THEN 'A'
                        WHEN url = ANY(%s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    created_at
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
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
                urls_a,
                urls_b,
                f"{days} days",
                urls_all,
                f"{split_days} days",
                f"{split_days} days",
            ),
        )
        momentum_rows = cur.fetchall()

        # --- 4. Top evidence URLs ---
        cur.execute(
            """
            WITH candidates AS (
                SELECT
                    CASE
                        WHEN url = ANY(%s) THEN 'A'
                        WHEN url = ANY(%s) THEN 'B'
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
                                WHEN url = ANY(%s) THEN 'A'
                                WHEN url = ANY(%s) THEN 'B'
                                ELSE NULL
                            END
                        ORDER BY
                            CASE
                                WHEN url = ANY(%s) THEN array_position(%s::text[], url)
                                WHEN url = ANY(%s) THEN array_position(%s::text[], url)
                                ELSE NULL
                            END ASC NULLS LAST,
                            created_at DESC
                    ) AS rn
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND url = ANY(%s)
            )
            SELECT grp, source_type, headline, url, points, created_at, rn
            FROM candidates
            WHERE grp IS NOT NULL AND rn <= 3
            ORDER BY grp, rn
            """,
            (
                urls_a,
                urls_b,
                urls_a,
                urls_b,
                urls_a,
                urls_a,
                urls_b,
                urls_b,
                f"{days} days",
                urls_all,
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

        # --- Reranked Top Evidence (per-group Top-3) ---
        try:
            reranked_a, _, meta_a = retrieve_and_rerank(
                topic_a, days=days, top_k=3,
            )
            reranked_b, _, meta_b = retrieve_and_rerank(
                topic_b, days=days, top_k=3,
            )
            evidence_a = format_reranked_evidence(
                reranked_a, header=f"Reranked Evidence: {topic_a}",
            )
            evidence_b = format_reranked_evidence(
                reranked_b, header=f"Reranked Evidence: {topic_b}",
            )
            if evidence_a:
                lines.append(evidence_a)
            if evidence_b:
                lines.append(evidence_b)
        except Exception as rerank_exc:
            print(f"[Warn] compare_topics rerank failed (non-fatal): {rerank_exc}")

        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] compare_topics failed: {exc}")
        return f"compare_topics failed: {exc}"
    finally:
        if conn is not None:
            put_conn(conn)


def compare_topics_tool(payload: CompareTopicsToolInput) -> ToolEnvelope:
    request = payload.model_dump(mode="python")
    result = _compare_topics_structured(
        topic_a=request["topic_a"],
        topic_b=request["topic_b"],
        days=int(request.get("days", 14)),
    )
    status = str(result.get("status") or "error")
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    if status == "error":
        return build_tool_error_envelope(
            tool="compare_topics",
            request=request,
            error=str(result.get("error_code") or "compare_topics_failed"),
            data={**data, "raw_output": _format_compare_topics_result(result)},
            diagnostics=diagnostics,
        )
    if status == "empty":
        return build_tool_empty_envelope(
            tool="compare_topics",
            request=request,
            empty_reason=str(diagnostics.get("empty_reason") or "no_topic_comparison_data"),
            data={**data, "raw_output": _format_compare_topics_result(result)},
            diagnostics=diagnostics,
        )

    evidence = _evidence_from_records(result.get("evidence", []) or [], max_items=6)
    data = dict(data)
    data.setdefault("raw_output", _format_compare_topics_result(result))
    return ToolEnvelope(
        tool="compare_topics",
        status="ok",
        request=request,
        data=data,
        evidence=evidence,
        diagnostics={**diagnostics, "evidence_count": len(evidence)},
    )


