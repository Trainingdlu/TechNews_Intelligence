"""Search-news skill implementation and structured adapter."""

from __future__ import annotations

import json
from typing import Any

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_records, _json_text
from .rerank import RERANK_MODE_NONE, rerank_search_rows, resolve_rerank_mode
from .retrieval import _get_query_embedding
from .schemas import SearchNewsSkillInput


def _format_search_news_text(rows: list[tuple]) -> str:
    if not rows:
        return ""

    max_score = max(row[5] for row in rows)
    note = ""
    if max_score < 0.5:
        note = "[Note] Relevance is weak; these are nearest matches.\n\n"

    out: list[str] = []
    for title, url, summary, sentiment, pub_time, score in rows:
        out.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Summary: {summary}\n"
            f"Sentiment: {sentiment}\n"
            f"Time: {pub_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Score: {float(score):.3f}"
        )
    return note + "\n---\n".join(out)


def _records_from_rows(rows: list[tuple]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        title, url, summary, sentiment, pub_time, score = row
        records.append(
            {
                "rank": idx,
                "title": title,
                "url": url,
                "summary": summary or "",
                "sentiment": sentiment or "",
                "created_at": pub_time.isoformat() if pub_time else "",
                "score": float(score or 0.0),
            }
        )
    return records


def search_news(
    query: str,
    days: int = 21,
    rerank_mode: str | None = None,
    response_format: str = "text",
) -> str:
    """Search related news using hybrid retrieval (semantic + keyword)."""
    print(f"\n[Tool] search_news: query={query}, days={days}")
    as_json = response_format.strip().lower() == "json"
    days = _clamp_int(days, 1, 365)
    limit = 5
    time_filter = f"{days} days"
    resolved_rerank_mode = resolve_rerank_mode(
        rerank_mode, env_keys=("SEARCH_NEWS_RERANK_MODE", "NEWS_RERANK_MODE")
    )
    candidate_pool_limit = limit if resolved_rerank_mode == RERANK_MODE_NONE else min(limit * 6, 30)
    rerank_meta = {
        "rerank_mode": resolved_rerank_mode,
        "candidate_count": 0,
        "top_k": 0,
        "fallback": False,
    }

    conn = get_conn()
    try:
        cur = conn.cursor()
        query_vec = _get_query_embedding(query)
        used_semantic_recall = bool(query_vec)

        if query_vec:
            vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
            cur.execute(
                """
                WITH semantic AS (
                    SELECT
                        t.title, t.url, t.summary, t.sentiment, t.created_at,
                        1 - (e.embedding <=> %s::vector)
                            + 0.1 * EXP(-EXTRACT(EPOCH FROM (NOW() - t.created_at)) / 86400.0 / 21)
                            AS score
                    FROM tech_news t
                    JOIN news_embeddings e ON e.url = t.url
                    WHERE t.created_at > NOW() - %s::interval
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                ),
                keyword AS (
                    SELECT title, url, summary, sentiment, created_at, 1.0 AS score
                    FROM tech_news
                    WHERE (title ILIKE %s OR summary ILIKE %s)
                      AND created_at > NOW() - %s::interval
                    LIMIT %s
                ),
                combined AS (
                    SELECT * FROM semantic
                    UNION ALL
                    SELECT * FROM keyword
                )
                SELECT DISTINCT ON (url)
                    title, url, summary, sentiment, created_at, score
                FROM combined
                ORDER BY url, score DESC
                """,
                (
                    vec_str,
                    time_filter,
                    vec_str,
                    candidate_pool_limit,
                    f"%{query}%",
                    f"%{query}%",
                    time_filter,
                    candidate_pool_limit,
                ),
            )
            rows = cur.fetchall()
        else:
            cur.execute(
                """
                SELECT title, url, summary, sentiment, created_at, 1.0 AS score
                FROM tech_news
                WHERE (title ILIKE %s OR summary ILIKE %s)
                  AND created_at > NOW() - %s::interval
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (f"%{query}%", f"%{query}%", time_filter, candidate_pool_limit),
            )
            rows = cur.fetchall()

        if resolved_rerank_mode == RERANK_MODE_NONE:
            rerank_meta["candidate_count"] = len(rows)
            rerank_meta["top_k"] = min(limit, len(rows))
            if used_semantic_recall:
                rows.sort(key=lambda row: row[5], reverse=True)
                rows = rows[:limit]
            else:
                rows = list(rows)[:limit]
        else:
            rows, rerank_meta = rerank_search_rows(
                query=query,
                rows=rows,
                mode=resolved_rerank_mode,
                top_k=limit,
                env_keys=("SEARCH_NEWS_RERANK_MODE", "NEWS_RERANK_MODE"),
            )

        cur.close()
        if not rows:
            empty_text = f"No related news for '{query}' in last {days} days."
            if as_json:
                return _json_text(
                    {
                        "tool": "search_news",
                        "status": "empty",
                        "request": {"query": query, "days": days},
                        "rerank": rerank_meta,
                        "count": 0,
                        "records": [],
                        "raw_output": empty_text,
                    }
                )
            return empty_text

        raw_text = _format_search_news_text(rows)
        if as_json:
            return _json_text(
                {
                    "tool": "search_news",
                    "status": "ok",
                    "request": {"query": query, "days": days},
                    "rerank": rerank_meta,
                    "count": len(rows),
                    "records": _records_from_rows(rows),
                    "raw_output": raw_text,
                }
            )
        return raw_text
    except Exception as exc:
        print(f"[Error] search_news failed: {exc}")
        if as_json:
            return _json_text(
                {
                    "tool": "search_news",
                    "status": "error",
                    "request": {"query": query, "days": days},
                    "rerank": rerank_meta,
                    "error": f"search_news failed: {exc}",
                    "records": [],
                }
            )
        return f"search_news failed: {exc}"
    finally:
        put_conn(conn)


def search_news_skill(payload: SearchNewsSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    try:
        raw_output = search_news(
            query=request["query"],
            days=int(request.get("days", 21)),
            response_format="json",
        )
    except Exception as exc:
        return build_error_envelope(
            tool="search_news",
            request=request,
            error="search_news_execution_failed",
            diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    try:
        parsed = json.loads(raw_output)
    except Exception as exc:
        return build_error_envelope(
            tool="search_news",
            request=request,
            error="search_news_json_parse_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "raw_preview": str(raw_output)[:500],
            },
        )

    status = str(parsed.get("status", "error")).lower()
    rerank_meta = parsed.get("rerank", {})
    if status == "error":
        return SkillEnvelope(
            tool="search_news",
            status="error",
            request=request,
            data=parsed,
            evidence=[],
            error=str(parsed.get("error") or "search_news_failed"),
            diagnostics={"query": request["query"], "rerank": rerank_meta},
        )

    if status == "empty":
        return SkillEnvelope(
            tool="search_news",
            status="empty",
            request=request,
            data=parsed,
            evidence=[],
            diagnostics={"query": request["query"], "rerank": rerank_meta},
        )

    records_raw = parsed.get("records")
    records = records_raw if isinstance(records_raw, list) else []
    evidence = _evidence_from_records(records, max_items=5)
    return SkillEnvelope(
        tool="search_news",
        status="ok",
        request=request,
        data=parsed,
        evidence=evidence,
        diagnostics={
            "query": request["query"],
            "result_count": len(records),
            "rerank": rerank_meta,
        },
    )
