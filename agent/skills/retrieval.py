"""Low-level retrieval primitives for skills."""

from __future__ import annotations

from typing import Any

from services.db import get_conn, put_conn

from .embeddings import get_query_embedding as _get_query_embedding
from .helpers import _clamp_int
from .hybrid_retrieval import QUERY_PROFILE, candidate_from_row, fetch_hybrid_rows, retrieval_diagnostics
from .recall_profile import resolve_recall_profile
from .rerank import RERANK_MODE_NONE, rerank_candidates, resolve_rerank_mode


def _rows_to_rerank_candidates(rows: list[tuple]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = candidate_from_row(row)
        candidates.append(
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "summary": item.get("summary") or "",
                "sentiment": item.get("sentiment") or "",
                "source_type": item.get("source_type") or "",
                "created_at": item.get("created_at"),
                "points": item.get("points"),
                "score": item.get("score"),
                "match_score": item.get("match_score"),
                "payload": row,
            }
        )
    return candidates


def _finalize_candidate_rows(
    rows: list[tuple],
    *,
    query: str,
    limit: int,
    rerank_mode: str,
) -> tuple[list[tuple], dict[str, Any]]:
    default_meta: dict[str, Any] = {
        "rerank_mode": rerank_mode,
        "candidate_count": len(rows),
        "top_k": min(int(limit), len(rows)),
        "fallback": False,
    }
    if not rows:
        return [], default_meta
    if rerank_mode == RERANK_MODE_NONE:
        return list(rows)[:limit], default_meta

    reranked, rerank_meta = rerank_candidates(
        query,
        _rows_to_rerank_candidates(rows),
        mode=rerank_mode,
        top_k=limit,
        env_keys=("SEARCH_NEWS_RERANK_MODE", "FULLTEXT_BATCH_RERANK_MODE", "NEWS_RERANK_MODE"),
    )
    return [item["payload"] for item in reranked], rerank_meta


def _row_to_candidate(row: tuple) -> dict[str, Any]:
    return candidate_from_row(row)


def _fetch_legacy_keyword_rows(
    cur,
    *,
    query: str,
    days: int,
    limit: int,
) -> list[tuple]:
    q = f"%{query}%"
    cur.execute(
        """
        SELECT
            COALESCE(v.title_cn, v.title) AS title,
            v.url,
            COALESCE(v.summary, '') AS summary,
            COALESCE(v.sentiment, '') AS sentiment,
            v.source_type,
            v.created_at,
            COALESCE(v.points, 0) AS points,
            (
                CASE
                    WHEN (v.title ILIKE %s OR COALESCE(v.title_cn, '') ILIKE %s) THEN 1.3
                    ELSE 0.0
                END
                + CASE
                    WHEN COALESCE(v.summary, '') ILIKE %s THEN 0.6
                    ELSE 0.0
                END
            )::float AS score
        FROM view_dashboard_news v
        WHERE v.created_at >= NOW() - %s::interval
          AND (
              v.title ILIKE %s
              OR COALESCE(v.title_cn, '') ILIKE %s
              OR COALESCE(v.summary, '') ILIKE %s
          )
        ORDER BY score DESC, created_at DESC
        LIMIT %s
        """,
        (q, q, q, f"{days} days", q, q, q, limit),
    )
    return list(cur.fetchall())


def lookup_candidates_by_query(
    query: str,
    *,
    days: int = 14,
    limit: int = 5,
    rerank_mode: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return unified retrieval candidates for query-driven skills."""
    query_clean = (query or "").strip()
    recall_profile = resolve_recall_profile()
    resolved_mode = resolve_rerank_mode(
        rerank_mode,
        env_keys=("SEARCH_NEWS_RERANK_MODE", "FULLTEXT_BATCH_RERANK_MODE", "NEWS_RERANK_MODE"),
    )
    if not query_clean:
        return [], {
            "rerank_mode": resolved_mode,
            "candidate_count": 0,
            "top_k": 0,
            "fallback": False,
            "recall_profile": recall_profile.as_dict(),
        }

    days = _clamp_int(days, 1, 365)
    limit = _clamp_int(limit, 1, 12)
    if resolved_mode == RERANK_MODE_NONE:
        candidate_pool_limit = limit
    else:
        candidate_pool_limit = min(
            limit * int(recall_profile.query_cand_multiplier),
            int(recall_profile.query_cand_max),
        )
    query_vec = _get_query_embedding(query_clean)

    conn = get_conn()
    try:
        cur = conn.cursor()
        fallback_reason: str | None = None
        try:
            rows = fetch_hybrid_rows(
                cur,
                query=query_clean,
                days=days,
                limit=candidate_pool_limit,
                query_vec=query_vec,
                profile=QUERY_PROFILE,
            )
        except Exception as exc:
            fallback_reason = type(exc).__name__
            print(f"[Warn] hybrid candidate lookup failed; fallback to legacy keyword search: {exc}")
            try:
                conn.rollback()
            except Exception:
                pass
            cur = conn.cursor()
            rows = _fetch_legacy_keyword_rows(
                cur,
                query=query_clean,
                days=days,
                limit=candidate_pool_limit,
            )

        cur.close()
        ranked_rows, meta = _finalize_candidate_rows(
            rows,
            query=query_clean,
            limit=limit,
            rerank_mode=resolved_mode,
        )
        retrieval_meta = retrieval_diagnostics(
            profile=QUERY_PROFILE,
            query_vec=query_vec,
            candidate_count=len(rows),
            top_k=min(limit, len(rows)),
            fallback=bool(fallback_reason),
            fallback_reason=fallback_reason,
        )
        retrieval_meta["retrieval_fallback"] = retrieval_meta.pop("fallback")
        meta.update(retrieval_meta)
        meta["recall_profile"] = recall_profile.as_dict()
        return [_row_to_candidate(row) for row in ranked_rows], meta
    finally:
        put_conn(conn)
