"""Postgres-only hybrid retrieval primitives.

This module implements the shared SQL pipeline used by query-oriented and
analysis-oriented tools:

    FTS rank + pgvector rank + exact alias fallback + RRF fusion.
"""

from __future__ import annotations

import os
from typing import Any

from .helpers import _clamp_int
from .recall_profile import resolve_recall_profile


RRF_K = 60
LEXICAL_WEIGHT = 1.0
SEMANTIC_WEIGHT = 1.0
EXACT_WEIGHT = 0.7

QUERY_PROFILE = "query"
ANALYSIS_PROFILE = "analysis"

_POINTS_BOOST_WEIGHT = 0.03
_FRESHNESS_BOOST_WEIGHT = 0.0
_TRGM_THRESHOLD = 0.42


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "enabled"}


def _env_float(name: str, *, fallback: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(str(os.getenv(name, "")).strip())
    except Exception:
        return fallback
    return max(min_value, min(max_value, parsed))


def _vector_literal(query_vec: list[float] | None) -> str | None:
    if not query_vec:
        return None
    return "[" + ",".join(str(v) for v in query_vec) + "]"


def _profile_limits(profile: str, limit: int) -> tuple[int, int, int]:
    recall_profile = resolve_recall_profile()
    if profile == ANALYSIS_PROFILE:
        pool_limit = _clamp_int(limit, 1, 1000)
        channel_limit = max(pool_limit, int(pool_limit * float(recall_profile.oversample_ratio)))
        return pool_limit, channel_limit, channel_limit

    pool_limit = _clamp_int(limit, 1, 500)
    channel_limit = max(pool_limit, min(pool_limit * 4, int(recall_profile.query_cand_max)))
    semantic_limit = max(pool_limit, min(pool_limit * int(recall_profile.query_cand_multiplier), int(recall_profile.query_cand_max)))
    return pool_limit, channel_limit, semantic_limit


def retrieval_diagnostics(
    *,
    profile: str,
    query_vec: list[float] | None,
    candidate_count: int,
    top_k: int,
    fallback: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    channels = ["lexical", "exact"]
    if query_vec:
        channels.insert(1, "semantic")
    meta: dict[str, Any] = {
        "retrieval_mode": f"postgres_rrf_hybrid_{profile}",
        "fusion": "rrf",
        "rrf_k": RRF_K,
        "channels_used": channels,
        "candidate_count": int(candidate_count),
        "top_k": int(top_k),
        "fallback": bool(fallback),
        "points_boost": _env_flag("RETRIEVAL_POINTS_BOOST", default=False),
        "recall_profile": resolve_recall_profile().as_dict(),
    }
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    return meta


def fetch_hybrid_rows(
    cur,
    *,
    query: str,
    days: int,
    limit: int,
    query_vec: list[float] | None,
    profile: str,
    sim_floor: float | None = None,
) -> list[tuple]:
    """Execute the hybrid RRF SQL and return rows in retrieval.py shape."""
    query_clean = (query or "").strip()
    if not query_clean:
        return []

    days = _clamp_int(days, 1, 365)
    result_limit, lexical_limit, semantic_limit = _profile_limits(profile, limit)
    recall_profile = resolve_recall_profile()
    probes = _clamp_int(int(recall_profile.pgvector_probes), 1, 200)
    floor = sim_floor if sim_floor is not None else float(recall_profile.sim_floor)

    query_vec_literal = _vector_literal(query_vec)
    has_vector = bool(query_vec_literal)
    if has_vector:
        cur.execute("SET LOCAL ivfflat.probes = %s", (probes,))

    points_weight = _POINTS_BOOST_WEIGHT if _env_flag("RETRIEVAL_POINTS_BOOST", default=False) else 0.0
    freshness_weight = _env_float(
        "RETRIEVAL_FRESHNESS_BOOST",
        fallback=_FRESHNESS_BOOST_WEIGHT,
        min_value=0.0,
        max_value=0.05,
    )
    trgm_threshold = _env_float(
        "RETRIEVAL_TRGM_THRESHOLD",
        fallback=_TRGM_THRESHOLD,
        min_value=0.1,
        max_value=0.9,
    )

    semantic_cte = """
        semantic AS (
            SELECT
                v.url,
                ROW_NUMBER() OVER (ORDER BY e.embedding <=> %(query_vec)s::vector, v.created_at DESC) AS semantic_rank,
                (1 - (e.embedding <=> %(query_vec)s::vector))::float AS semantic_score
            FROM view_dashboard_news v
            JOIN news_embeddings e ON e.url = v.url
            WHERE v.created_at >= NOW() - %(days_interval)s::interval
            ORDER BY e.embedding <=> %(query_vec)s::vector
            LIMIT %(semantic_limit)s
        )
    """ if has_vector else """
        semantic AS (
            SELECT
                NULL::text AS url,
                NULL::bigint AS semantic_rank,
                NULL::float AS semantic_score
            WHERE FALSE
        )
    """

    sql = f"""
        WITH q AS (
            SELECT
                %(query)s::text AS raw_query,
                (websearch_to_tsquery('english', %(query)s)
                    || websearch_to_tsquery('simple', %(query)s)) AS tsq
        ),
        alias_terms_raw AS (
            SELECT raw_query AS term, 1.0::float AS term_weight, LENGTH(raw_query) AS term_len
            FROM q
            WHERE NULLIF(BTRIM(raw_query), '') IS NOT NULL
            UNION
            SELECT ea.alias AS term, COALESCE(ea.weight, 1.0)::float AS term_weight, LENGTH(ea.alias) AS term_len
            FROM entity_alias ea
            JOIN entity_registry er ON er.entity_id = ea.entity_id
            CROSS JOIN q
            WHERE ea.is_active = TRUE
              AND er.is_active = TRUE
              AND (
                    LOWER(ea.alias) = LOWER(q.raw_query)
                 OR LOWER(er.canonical_name) = LOWER(q.raw_query)
                 OR LOWER(q.raw_query) LIKE ('%%' || LOWER(ea.alias) || '%%')
                 OR LOWER(ea.alias) LIKE ('%%' || LOWER(q.raw_query) || '%%')
              )
        ),
        alias_terms AS (
            SELECT term, MAX(term_weight)::float AS term_weight
            FROM alias_terms_raw
            GROUP BY term
            ORDER BY MAX(term_weight) DESC, MAX(term_len) DESC
            LIMIT 16
        ),
        lexical_ranked AS (
            SELECT
                v.url,
                ts_rank_cd(si.search_tsv, q.tsq, 32)::float AS text_score,
                v.created_at
            FROM q
            JOIN news_search_index si ON numnode(q.tsq) > 0 AND si.search_tsv @@ q.tsq
            JOIN view_dashboard_news v ON v.url = si.url
            WHERE v.created_at >= NOW() - %(days_interval)s::interval
            ORDER BY text_score DESC, v.created_at DESC
            LIMIT %(lexical_limit)s
        ),
        lexical AS (
            SELECT
                url,
                ROW_NUMBER() OVER (ORDER BY text_score DESC, created_at DESC) AS lexical_rank,
                text_score
            FROM lexical_ranked
        ),
        {semantic_cte},
        exact_source AS (
            SELECT
                v.url,
                MAX(
                    CASE
                        WHEN v.title ILIKE ('%%' || t.term || '%%')
                          OR COALESCE(v.title_cn, '') ILIKE ('%%' || t.term || '%%')
                            THEN 2.0 * t.term_weight
                        WHEN similarity(LOWER(COALESCE(v.title, '')), LOWER(t.term)) >= %(trgm_threshold)s
                          OR similarity(LOWER(COALESCE(v.title_cn, '')), LOWER(t.term)) >= %(trgm_threshold)s
                            THEN 1.4 * t.term_weight
                        WHEN COALESCE(v.summary, '') ILIKE ('%%' || t.term || '%%')
                            THEN 0.8 * t.term_weight
                        ELSE 0.0
                    END
                )::float AS exact_score,
                MAX(v.created_at) AS created_at
            FROM view_dashboard_news v
            JOIN alias_terms t ON TRUE
            WHERE v.created_at >= NOW() - %(days_interval)s::interval
              AND (
                    v.title ILIKE ('%%' || t.term || '%%')
                 OR COALESCE(v.title_cn, '') ILIKE ('%%' || t.term || '%%')
                 OR COALESCE(v.summary, '') ILIKE ('%%' || t.term || '%%')
                 OR similarity(LOWER(COALESCE(v.title, '')), LOWER(t.term)) >= %(trgm_threshold)s
                 OR similarity(LOWER(COALESCE(v.title_cn, '')), LOWER(t.term)) >= %(trgm_threshold)s
              )
            GROUP BY v.url
            ORDER BY exact_score DESC, created_at DESC
            LIMIT %(lexical_limit)s
        ),
        exact AS (
            SELECT
                url,
                ROW_NUMBER() OVER (ORDER BY exact_score DESC, created_at DESC) AS exact_rank,
                exact_score
            FROM exact_source
        ),
        fused_urls AS (
            SELECT url FROM lexical
            UNION
            SELECT url FROM semantic
            UNION
            SELECT url FROM exact
        ),
        fused AS (
            SELECT
                u.url,
                (
                    COALESCE(%(lexical_weight)s::float / (%(rrf_k)s::float + l.lexical_rank::float), 0.0)
                  + COALESCE(%(semantic_weight)s::float / (%(rrf_k)s::float + s.semantic_rank::float), 0.0)
                  + COALESCE(%(exact_weight)s::float / (%(rrf_k)s::float + x.exact_rank::float), 0.0)
                )::float AS rrf_score,
                COALESCE(l.text_score, 0.0)::float AS text_score,
                COALESCE(s.semantic_score, 0.0)::float AS semantic_score,
                COALESCE(x.exact_score, 0.0)::float AS exact_score
            FROM fused_urls u
            LEFT JOIN lexical l ON l.url = u.url
            LEFT JOIN semantic s ON s.url = u.url
            LEFT JOIN exact x ON x.url = u.url
        )
        SELECT
            COALESCE(v.title_cn, v.title) AS title,
            v.url,
            COALESCE(v.summary, '') AS summary,
            COALESCE(v.sentiment, '') AS sentiment,
            v.source_type,
            v.created_at,
            COALESCE(v.points, 0) AS points,
            (
                f.rrf_score
                + %(freshness_weight)s::float
                    * EXP(-EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 86400.0 / GREATEST(%(days)s::float, 1.0))
                + %(points_weight)s::float
                    * LEAST(1.0, GREATEST(0.0, COALESCE(v.points, 0)::float / 500.0))
            )::float AS score,
            COALESCE(f.text_score, 0.0)::float AS text_score,
            COALESCE(f.semantic_score, 0.0)::float AS semantic_score,
            COALESCE(f.exact_score, 0.0)::float AS exact_score,
            GREATEST(
                COALESCE(f.semantic_score, 0.0)::float,
                CASE
                    WHEN COALESCE(f.exact_score, 0.0) > 0.0
                        THEN (COALESCE(f.exact_score, 0.0)::float / (1.0 + COALESCE(f.exact_score, 0.0)::float))
                    ELSE 0.0
                END,
                CASE
                    WHEN COALESCE(f.text_score, 0.0) > 0.0
                        THEN (COALESCE(f.text_score, 0.0)::float / (1.0 + COALESCE(f.text_score, 0.0)::float))
                    ELSE 0.0
                END
            )::float AS match_score
        FROM fused f
        JOIN view_dashboard_news v ON v.url = f.url
        WHERE (
            %(sim_floor)s::float <= 0.0
            OR f.semantic_score = 0.0
            OR f.semantic_score >= %(sim_floor)s::float
            OR f.text_score > 0.0
            OR f.exact_score > 0.0
        )
        ORDER BY score DESC, v.created_at DESC
        LIMIT %(result_limit)s
    """

    cur.execute(
        sql,
        {
            "query": query_clean,
            "days": days,
            "days_interval": f"{days} days",
            "query_vec": query_vec_literal,
            "lexical_limit": lexical_limit,
            "semantic_limit": semantic_limit,
            "result_limit": result_limit,
            "rrf_k": RRF_K,
            "lexical_weight": LEXICAL_WEIGHT,
            "semantic_weight": SEMANTIC_WEIGHT,
            "exact_weight": EXACT_WEIGHT,
            "points_weight": points_weight,
            "freshness_weight": freshness_weight,
            "trgm_threshold": trgm_threshold,
            "sim_floor": floor,
        },
    )
    return list(cur.fetchall())


def candidate_from_row(row: tuple) -> dict[str, Any]:
    title, url, summary, sentiment, source_type, created_at, points, score = row[:8]
    text_score = float(row[8] or 0.0) if len(row) > 8 else 0.0
    semantic_score = float(row[9] or 0.0) if len(row) > 9 else 0.0
    exact_score = float(row[10] or 0.0) if len(row) > 10 else 0.0
    if len(row) > 11:
        match_score = float(row[11] or 0.0)
    else:
        normalized_text_score = (text_score / (1.0 + text_score)) if text_score > 0.0 else 0.0
        normalized_exact_score = (exact_score / (1.0 + exact_score)) if exact_score > 0.0 else 0.0
        match_score = max(semantic_score, normalized_text_score, normalized_exact_score)
    return {
        "title": str(title or ""),
        "url": str(url or ""),
        "summary": str(summary or ""),
        "sentiment": str(sentiment or ""),
        "source_type": str(source_type or ""),
        "created_at": created_at,
        "points": int(points or 0),
        "score": float(score or 0.0),
        "final_score": float(score or 0.0),
        "rrf_score": float(score or 0.0),
        "text_score": text_score,
        "semantic_score": semantic_score,
        "exact_score": exact_score,
        "match_score": match_score,
    }
