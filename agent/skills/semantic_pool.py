"""Bulk semantic URL retrieval for aggregation-oriented skills.

Provides a large-batch vector recall channel that trades reranker precision
for throughput.  Downstream skills (trend_analysis, compare_topics, …) feed
the returned URL set into SQL aggregation queries, replacing the legacy
ILIKE + hardcoded-dictionary approach with true semantic matching.

RAG 2.0 additions:
- ``fetch_semantic_candidates()`` returns rich candidate dicts (title, summary,
  created_at, sim_score, final_score) for downstream reranking.
- Dynamic time-decay scoring aligned with the additive model used in
  ``retrieval.py`` (weight-based bonus, not multiplicative penalty).
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Any

from services.db import get_conn, put_conn

from .helpers import _clamp_int
from .recall_profile import resolve_recall_profile
from .retrieval import _get_query_embedding

# ---------------------------------------------------------------------------
# Tuning knobs (overridable via environment variables)
# ---------------------------------------------------------------------------

_DEFAULT_PROBES = 10         # IVFFlat scan breadth (legacy fallback)
_DEFAULT_SIM_FLOOR = 0.20    # Minimum cosine similarity to keep
_OVERSAMPLE_RATIO = 2.0      # Fetch extra rows to compensate for post-filter
_DEFAULT_RETRY_COUNT = 2     # Embedding API retry attempts
_DEFAULT_RETRY_DELAY = 0.8   # Base delay (seconds) between retries (exponential)

# Time-decay tuning (additive bonus model, aligned with retrieval.py)
_DEFAULT_DECAY_WEIGHT = 0.2  # Maximum additive time bonus (env: SEMANTIC_DECAY_WEIGHT)
_DEFAULT_POINTS_WEIGHT = 0.8 # Maximum additive points bonus
_DEFAULT_POINTS_DIVISOR = 280.0  # Points normalization divisor

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
# Time-decay scoring (aligned with retrieval.py additive model)
# ---------------------------------------------------------------------------

def _compute_time_decay(
    age_days: float,
    window_days: float,
    *,
    weight: float | None = None,
) -> float:
    """Additive time bonus: newer articles get a higher bonus.

    Formula: ``weight * exp(-age_days / window_days)``

    This mirrors the pattern in ``retrieval.py`` (L169/191/272) where time
    is an additive scoring component rather than a multiplicative penalty.
    The key improvement here is using *dynamic* ``window_days`` (from user
    query context) instead of hardcoded 21 days.

    Parameters
    ----------
    age_days:
        Age of the article in days.
    window_days:
        The query's time window, used as the decay constant.
    weight:
        Maximum bonus value.  Defaults to ``SEMANTIC_DECAY_WEIGHT`` env
        or ``_DEFAULT_DECAY_WEIGHT`` (0.2).
    """
    if weight is None:
        weight = float(os.getenv("SEMANTIC_DECAY_WEIGHT", str(_DEFAULT_DECAY_WEIGHT)))
    if window_days <= 0:
        window_days = 1.0
    return weight * math.exp(-max(0.0, age_days) / window_days)


def _compute_points_bonus(points: int) -> float:
    """Additive points bonus (aligned with retrieval.py)."""
    return min(_DEFAULT_POINTS_WEIGHT, max(0.0, float(points) / _DEFAULT_POINTS_DIVISOR))


def _to_utc_naive(dt: Any) -> datetime:
    """Coerce a possibly-aware datetime to UTC-naive for arithmetic."""
    if dt is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_semantic_candidates(
    query: str,
    *,
    days: int = 30,
    limit: int = 200,
    sim_floor: float | None = None,
) -> list[dict[str, Any]]:
    """Rich candidate retrieval with dynamic time-decay scoring.

    Returns candidate dicts sorted by ``final_score`` descending, each
    containing: ``url``, ``title``, ``summary``, ``created_at``,
    ``points``, ``sim_score``, ``time_bonus``, ``points_bonus``,
    ``final_score``.

    Designed for downstream reranking via ``rerank_aggregation.py``.

    Parameters
    ----------
    query:
        Natural-language query for semantic matching.
    days:
        Time window in days (also used as decay constant).
    limit:
        Maximum number of candidates to return.
    sim_floor:
        Minimum cosine similarity threshold.  Defaults to 0.20.
    """
    query_clean = (query or "").strip()
    if not query_clean:
        return []

    days = _clamp_int(days, 1, 365)
    limit = _clamp_int(limit, 1, 500)
    recall_profile = resolve_recall_profile()
    probes = _clamp_int(int(recall_profile.pgvector_probes), 1, 200)
    floor = sim_floor if sim_floor is not None else float(recall_profile.sim_floor)
    oversample_ratio = max(1.0, float(recall_profile.oversample_ratio))
    oversample_limit = int(limit * oversample_ratio)

    query_vec = _get_embedding_with_retry(query_clean)
    if query_vec is None:
        print("[Warn] fetch_semantic_candidates: embedding unavailable, returning empty.")
        return []

    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SET LOCAL ivfflat.probes = %s", (probes,))
        cur.execute("SELECT NOW()")
        db_now = _to_utc_naive(cur.fetchone()[0])

        cur.execute(
            """
            WITH vec_pool AS (
                SELECT
                    e.url,
                    (1 - (e.embedding <=> %s::vector)) AS sim,
                    COALESCE(v.title_cn, v.title) AS title,
                    COALESCE(v.summary, v.title_cn, v.title, '') AS summary,
                    v.created_at,
                    COALESCE(v.points, 0) AS points
                FROM news_embeddings e
                JOIN view_dashboard_news v ON v.url = e.url
                WHERE v.created_at >= NOW() - %s::interval
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
            )
            SELECT url, sim, title, summary, created_at, points
            FROM vec_pool
            WHERE sim >= %s
            ORDER BY sim DESC
            LIMIT %s
            """,
            (vec_str, f"{days} days", vec_str, oversample_limit, floor, limit),
        )
        rows = cur.fetchall()
        cur.close()

        # Apply dynamic time-decay scoring in Python
        window_days = float(days)
        candidates: list[dict[str, Any]] = []
        for url, sim, title, summary, created_at, points in rows:
            created_naive = _to_utc_naive(created_at)
            age_days = max(0.0, (db_now - created_naive).total_seconds() / 86400.0)

            sim_score = float(sim)
            time_bonus = _compute_time_decay(age_days, window_days)
            points_bonus = _compute_points_bonus(int(points or 0))
            final_score = sim_score + time_bonus + points_bonus

            # Summary fallback: summary → title (already handled in SQL COALESCE)
            candidates.append({
                "url": str(url),
                "title": str(title or "").strip(),
                "summary": str(summary or "").strip(),
                "created_at": created_at,
                "points": int(points or 0),
                "sim_score": sim_score,
                "time_bonus": round(time_bonus, 4),
                "points_bonus": round(points_bonus, 4),
                "final_score": round(final_score, 4),
            })

        # Sort by final_score descending
        candidates.sort(key=lambda c: c["final_score"], reverse=True)

        print(
            f"[semantic_pool] query={query_clean!r}, days={days}, "
            f"profile={recall_profile.profile}, probes={probes}, floor={floor:.3f}, "
            f"oversample={oversample_ratio:.2f}, requested={limit}, returned={len(candidates)}"
        )
        return candidates
    except Exception as exc:
        print(f"[Error] fetch_semantic_candidates failed: {exc}")
        try:
            conn.rollback()
        except Exception:
            pass
        return []
    finally:
        put_conn(conn)


def fetch_semantic_url_pool(
    query: str,
    *,
    days: int = 30,
    limit: int = 300,
    sim_floor: float | None = None,
) -> list[tuple[str, float]]:
    """Large-batch vector recall for aggregation skills (backward-compatible).

    Returns ``(url, similarity)`` tuples sorted by descending final_score.
    Delegates to ``fetch_semantic_candidates()`` internally.

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
        ``(url, final_score)`` pairs, or an empty list on failure.
    """
    candidates = fetch_semantic_candidates(
        query, days=days, limit=limit, sim_floor=sim_floor,
    )
    return [(c["url"], c["final_score"]) for c in candidates]
