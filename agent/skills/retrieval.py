"""Low-level retrieval primitives for skills."""

from __future__ import annotations

import os

import requests

from services.db import get_conn, put_conn

from .helpers import _clamp_int


JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"


def _get_query_embedding(query: str) -> list[float] | None:
    """Get query embedding via Jina API. Return None on failure."""
    jina_key = os.getenv("JINA_API_KEY", "")
    if not jina_key:
        print("[Error] JINA_API_KEY not set, skip vector search.")
        return None

    try:
        resp = requests.post(
            JINA_EMBED_URL,
            headers={
                "Authorization": f"Bearer {jina_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "task": "retrieval.query",
                "input": [query],
            },
            timeout=15,
        )
        resp.raise_for_status()
        emb = resp.json()["data"][0]["embedding"]
        return emb
    except Exception as exc:
        print(f"[Error] Embedding request failed, fallback to keyword search only: {exc}")
        return None


def _lookup_urls_by_query(query: str, days: int = 14, limit: int = 5) -> list[tuple]:
    """Return candidate URLs by fused semantic+keyword retrieval."""
    query_clean = (query or "").strip()
    if not query_clean:
        return []

    days = _clamp_int(days, 1, 180)
    limit = _clamp_int(limit, 1, 12)
    q = f"%{query_clean}%"
    query_vec = _get_query_embedding(query_clean)

    conn = get_conn()
    try:
        cur = conn.cursor()
        if query_vec:
            vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
            try:
                cur.execute(
                    """
                    WITH keyword AS (
                        SELECT
                            COALESCE(v.title_cn, v.title) AS headline,
                            v.url,
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
                                + LEAST(0.8, GREATEST(0.0, COALESCE(v.points, 0)::float / 220.0))
                                + 0.2 * EXP(-EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 86400.0 / 21.0)
                            )::float AS score
                        FROM view_dashboard_news v
                        WHERE v.created_at >= NOW() - %s::interval
                          AND (
                              v.title ILIKE %s
                              OR COALESCE(v.title_cn, '') ILIKE %s
                              OR COALESCE(v.summary, '') ILIKE %s
                          )
                        LIMIT %s
                    ),
                    semantic AS (
                        SELECT
                            COALESCE(v.title_cn, v.title) AS headline,
                            v.url,
                            v.source_type,
                            v.created_at,
                            COALESCE(v.points, 0) AS points,
                            (
                                (1 - (e.embedding <=> %s::vector))
                                + 0.2 * EXP(-EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 86400.0 / 21.0)
                                + LEAST(0.8, GREATEST(0.0, COALESCE(v.points, 0)::float / 280.0))
                            )::float AS score
                        FROM view_dashboard_news v
                        JOIN news_embeddings e ON e.url = v.url
                        WHERE v.created_at >= NOW() - %s::interval
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                    ),
                    combined AS (
                        SELECT * FROM keyword
                        UNION ALL
                        SELECT * FROM semantic
                    ),
                    dedup AS (
                        SELECT
                            headline,
                            url,
                            source_type,
                            created_at,
                            points,
                            score,
                            ROW_NUMBER() OVER (
                                PARTITION BY url
                                ORDER BY score DESC, points DESC NULLS LAST, created_at DESC
                            ) AS rn
                        FROM combined
                    )
                    SELECT headline, url, source_type, created_at, points, score
                    FROM dedup
                    WHERE rn = 1
                    ORDER BY score DESC, points DESC NULLS LAST, created_at DESC
                    LIMIT %s
                    """,
                    (
                        q,
                        q,
                        q,
                        f"{days} days",
                        q,
                        q,
                        q,
                        limit * 4,
                        vec_str,
                        f"{days} days",
                        vec_str,
                        limit * 6,
                        limit,
                    ),
                )
                rows = cur.fetchall()
                cur.close()
                return rows
            except Exception as exc:
                print(f"[Warn] semantic candidate lookup failed; fallback to keyword-only: {exc}")

        cur.execute(
            """
            SELECT
                COALESCE(v.title_cn, v.title) AS headline,
                v.url,
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
                    + LEAST(0.8, GREATEST(0.0, COALESCE(v.points, 0)::float / 220.0))
                    + 0.2 * EXP(-EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 86400.0 / 21.0)
                )::float AS score
            FROM view_dashboard_news v
            WHERE v.created_at >= NOW() - %s::interval
              AND (v.title ILIKE %s OR COALESCE(v.title_cn,'') ILIKE %s OR COALESCE(v.summary,'') ILIKE %s)
            ORDER BY score DESC, points DESC NULLS LAST, created_at DESC
            LIMIT %s
            """,
            (q, q, q, f"{days} days", q, q, q, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        put_conn(conn)

