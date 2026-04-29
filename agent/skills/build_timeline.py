"""Build-timeline skill implementation and structured adapter."""

from __future__ import annotations

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_empty_envelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_records
from .rerank_aggregation import format_reranked_evidence, retrieve_and_rerank
from .schemas import BuildTimelineSkillInput
from .semantic_pool import fetch_semantic_url_pool


def _format_timeline_result(result: dict) -> str:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    topic = data.get("topic", "")
    days = data.get("days", 30)
    limit = data.get("limit", 12)
    if result.get("status") == "empty":
        return f"No timeline data for '{topic}' in {days} days."
    if result.get("status") == "error":
        return str(result.get("error") or "build_timeline failed")

    lines = [f"Timeline: {topic} (last {days} days, max {limit})"]
    context_block = data.get("reranked_output")
    if context_block:
        lines.append(str(context_block))
        lines.append("")
    for item in data.get("events", []) or []:
        lines.append(
            f"{item.get('rank')}. {str(item.get('created_at') or '')[:16].replace('T', ' ')} | "
            f"{item.get('source')} | {item.get('sentiment')} | points={item.get('metadata', {}).get('points')}\n"
            f"   {item.get('title')}\n"
            f"   {item.get('url')}"
        )
    return "\n".join(lines)


def _build_timeline_structured(topic: str, days: int = 30, limit: int = 12) -> dict:
    """Build a chronological event timeline for a topic, company, or product."""
    print(f"\n[Tool] build_timeline: topic={topic}, days={days}, limit={limit}")
    if not topic or not topic.strip():
        return {
            "status": "error",
            "error_code": "build_timeline_missing_topic",
            "error": "build_timeline requires topic.",
            "data": {"topic": topic, "days": days, "limit": limit, "events": [], "event_count": 0},
            "evidence": [],
            "diagnostics": {"topic": topic},
        }

    topic = topic.strip()
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
        return {
            "status": "empty",
            "data": {"topic": topic, "days": days, "limit": limit, "events": [], "event_count": 0},
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
            return {
                "status": "empty",
                "data": {"topic": topic, "days": days, "limit": limit, "events": [], "event_count": 0},
                "evidence": [],
                "diagnostics": {
                    "topic": topic,
                    "candidate_count": len(url_pool),
                    "evidence_count": 0,
                    "retrieval_mode": "semantic_url_pool",
                    "fallback": False,
                    "empty_reason": "no_timeline_rows",
                },
            }

        reranked_output = ""
        rerank_meta = {}
        try:
            reranked, _, rerank_meta = retrieve_and_rerank(
                topic, days=days, top_k=5,
            )
            reranked_output = format_reranked_evidence(
                reranked, header="Key Context (Reranked)",
            )
        except Exception as rerank_exc:
            print(f"[Warn] build_timeline rerank failed (non-fatal): {rerank_exc}")

        events: list[dict] = []
        for idx, (created_at, src, headline, senti, points, url) in enumerate(rows, 1):
            created_text = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or "")
            events.append(
                {
                    "rank": idx,
                    "created_at": created_text,
                    "source": str(src or ""),
                    "title": str(headline or ""),
                    "sentiment": str(senti or ""),
                    "url": str(url or ""),
                    "score": float(points or 0),
                    "match_score": pool_scores.get(str(url or "")),
                    "metadata": {"points": int(points or 0)},
                }
            )
        data = {
            "topic": topic,
            "days": days,
            "limit": limit,
            "events": events,
            "event_count": len(events),
            "reranked_output": reranked_output,
        }
        data["raw_output"] = _format_timeline_result({"status": "ok", "data": data})
        return {
            "status": "ok",
            "data": data,
            "evidence": events,
            "diagnostics": {
                "topic": topic,
                "candidate_count": len(url_pool),
                "evidence_count": len(events),
                "retrieval_mode": "semantic_url_pool",
                "fallback": False,
                "rerank": rerank_meta,
            },
        }
    except Exception as exc:
        print(f"[Error] build_timeline failed: {exc}")
        return {
            "status": "error",
            "error_code": "build_timeline_execution_failed",
            "error": "build_timeline_execution_failed",
            "data": {"topic": topic, "days": days, "limit": limit, "events": [], "event_count": 0},
            "evidence": [],
            "diagnostics": {"exception_type": type(exc).__name__, "exception_message": str(exc), "topic": topic},
        }
    finally:
        put_conn(conn)


def build_timeline(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build a chronological event timeline for a topic, company, or product."""
    return _format_timeline_result(_build_timeline_structured(topic, days=days, limit=limit))


def build_timeline_skill(payload: BuildTimelineSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    result = _build_timeline_structured(
        topic=request["topic"],
        days=int(request.get("days", 30)),
        limit=int(request.get("limit", 12)),
    )
    status = str(result.get("status") or "error")
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    if status == "error":
        return build_error_envelope(
            tool="build_timeline",
            request=request,
            error=str(result.get("error_code") or "build_timeline_failed"),
            data={**data, "raw_output": _format_timeline_result(result)},
            diagnostics=diagnostics,
        )
    if status == "empty":
        return build_empty_envelope(
            tool="build_timeline",
            request=request,
            empty_reason=str(diagnostics.get("empty_reason") or "no_timeline_data"),
            data={**data, "raw_output": _format_timeline_result(result)},
            diagnostics=diagnostics,
        )

    evidence = _evidence_from_records(result.get("evidence", []) or [], max_items=12)
    data = dict(data)
    data.setdefault("raw_output", _format_timeline_result(result))
    return SkillEnvelope(
        tool="build_timeline",
        status="ok",
        request=request,
        data=data,
        evidence=evidence,
        diagnostics={**diagnostics, "evidence_count": len(evidence)},
    )
