"""Bulk semantic URL retrieval for aggregation-oriented skills.

Provides a large-batch vector recall channel that trades reranker precision
for throughput.  Downstream skills (trend_analysis, compare_topics, …) feed
the returned URL set into SQL aggregation queries, replacing the legacy
ILIKE + hardcoded-dictionary approach with true semantic matching.
"""

from __future__ import annotations

import os
import time
from typing import Any

from services.db import get_conn, put_conn

from .helpers import _clamp_int
from .retrieval import _get_query_embedding

# ---------------------------------------------------------------------------
# Tuning knobs (overridable via environment variables)
# ---------------------------------------------------------------------------

_DEFAULT_PROBES = 10          # IVFFlat scan breadth (env: PGVECTOR_PROBES)
_DEFAULT_SIM_FLOOR = 0.20    # Minimum cosine similarity to keep
_OVERSAMPLE_RATIO = 2.0      # Fetch extra rows to compensate for post-filter
_DEFAULT_RETRY_COUNT = 2     # Embedding API retry attempts
_DEFAULT_RETRY_DELAY = 0.8   # Base delay (seconds) between retries (exponential)

# ---------------------------------------------------------------------------
# In-memory embedding cache (short TTL)
# ---------------------------------------------------------------------------

_CACHE_TTL_SEC = 300  # 5 minutes

_embedding_cache: dict[str, tuple[list[float], float]] = {}
# key = query_text, value = (embedding_vector, timestamp)


def _cache_get(query: str) -> list[float] | None:
    """Return cached embedding if still within TTL, otherwise None."""
    entry = _embedding_cache.get(query)
    if entry is None:
        return None
    vec, ts = entry
    if time.time() - ts > _CACHE_TTL_SEC:
        _embedding_cache.pop(query, None)
        return None
    return vec


def _cache_set(query: str, vec: list[float]) -> None:
    """Store embedding vector with current timestamp."""
    _embedding_cache[query] = (vec, time.time())


def _evict_stale_cache() -> None:
    """Remove expired entries to prevent unbounded growth."""
    now = time.time()
    stale_keys = [k for k, (_, ts) in _embedding_cache.items() if now - ts > _CACHE_TTL_SEC]
    for k in stale_keys:
        _embedding_cache.pop(k, None)


# ---------------------------------------------------------------------------
# Embedding helper with retry + cache
# ---------------------------------------------------------------------------

def _get_embedding_with_retry(query: str) -> list[float] | None:
    """Get query embedding with in-memory caching and retry on failure.

    Flow:
    1. Check in-memory TTL cache.
    2. On miss, call ``_get_query_embedding`` with up to *retry_count* attempts.
    3. Cache successful results.
    4. Return ``None`` only if all attempts fail.
    """
    # 1. Cache hit
    cached = _cache_get(query)
    if cached is not None:
        return cached

    # Periodically evict stale entries
    _evict_stale_cache()

    retry_count = _clamp_int(
        int(os.getenv("SEMANTIC_POOL_RETRY_COUNT", str(_DEFAULT_RETRY_COUNT))),
        0, 5,
    )
    retry_delay = _DEFAULT_RETRY_DELAY

    # 2. Try with retries
    last_exc: Exception | None = None
    for attempt in range(1, retry_count + 2):  # +2 because first try is attempt 1
        vec = _get_query_embedding(query)
        if vec is not None:
            _cache_set(query, vec)
            return vec
        # _get_query_embedding already prints its own error – just retry
        if attempt <= retry_count:
            print(
                f"[Warn] semantic_pool: embedding attempt {attempt} failed, "
                f"retrying in {retry_delay:.1f}s …"
            )
            time.sleep(retry_delay)
            retry_delay *= 2  # exponential backoff

    print("[Error] semantic_pool: all embedding attempts exhausted.")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_semantic_url_pool(
    query: str,
    *,
    days: int = 30,
    limit: int = 300,
    sim_floor: float | None = None,
) -> list[tuple[str, float]]:
    """Large-batch vector recall for aggregation skills.

    Returns ``(url, similarity)`` tuples sorted by descending similarity.
    **No reranker.  No full payload.**  Designed for downstream SQL
    aggregation via ``WHERE url = ANY(…)``.

    Parameters
    ----------
    query:
        Natural-language query for semantic matching.
    days:
        Time window in days.
    limit:
        Maximum number of URLs to return.
    sim_floor:
        Minimum cosine similarity threshold.  Defaults to 0.20.

    Returns
    -------
    list[tuple[str, float]]
        ``(url, similarity_score)`` pairs, or an empty list on failure.
    """
    query_clean = (query or "").strip()
    if not query_clean:
        return []

    days = _clamp_int(days, 1, 365)
    limit = _clamp_int(limit, 1, 500)
    probes = _clamp_int(
        int(os.getenv("PGVECTOR_PROBES", str(_DEFAULT_PROBES))),
        1, 100,
    )
    floor = sim_floor if sim_floor is not None else _DEFAULT_SIM_FLOOR
    oversample_limit = int(limit * _OVERSAMPLE_RATIO)

    query_vec = _get_embedding_with_retry(query_clean)
    if query_vec is None:
        print("[Warn] fetch_semantic_url_pool: embedding unavailable, returning empty pool.")
        return []

    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    conn = get_conn()
    try:
        cur = conn.cursor()
        # SET LOCAL scopes the probes parameter to the current transaction only,
        # so it will not leak into other queries on the same connection.
        cur.execute("SET LOCAL ivfflat.probes = %s", (probes,))
        cur.execute(
            """
            WITH vec_pool AS (
                SELECT
                    e.url,
                    (1 - (e.embedding <=> %s::vector)) AS sim
                FROM news_embeddings e
                JOIN view_dashboard_news v ON v.url = e.url
                WHERE v.created_at >= NOW() - %s::interval
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
            )
            SELECT url, sim
            FROM vec_pool
            WHERE sim >= %s
            ORDER BY sim DESC
            LIMIT %s
            """,
            (vec_str, f"{days} days", vec_str, oversample_limit, floor, limit),
        )
        rows = cur.fetchall()
        cur.close()
        print(
            f"[semantic_pool] query={query_clean!r}, days={days}, "
            f"probes={probes}, requested={limit}, returned={len(rows)}"
        )
        return [(str(url), float(sim)) for url, sim in rows]
    except Exception as exc:
        print(f"[Error] fetch_semantic_url_pool failed: {exc}")
        try:
            conn.rollback()
        except Exception:
            pass
        return []
    finally:
        put_conn(conn)
