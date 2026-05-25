"""Bulk candidate retrieval for aggregation-oriented tools.

Runs the shared hybrid RRF SQL (``fetch_hybrid_rows``) with the analysis
recall profile and returns rich candidate dicts (title, summary, created_at,
sim_score, final_score) for downstream reranking. Used by the macro-analysis
tools (trend_analysis, compare_topics, compare_sources, build_timeline,
analyze_landscape), which feed the returned URL set into SQL aggregation.
"""

from __future__ import annotations

import os
import time
from typing import Any

from services.db import get_conn, put_conn

from .hybrid_retrieval import ANALYSIS_PROFILE, candidate_from_row, fetch_hybrid_rows, retrieval_diagnostics
from .embeddings import get_query_embedding as _get_query_embedding
from .helpers import _clamp_int

# ---------------------------------------------------------------------------
# Embedding retry knobs (overridable via environment variables)
# ---------------------------------------------------------------------------

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
    for attempt in range(1, retry_count + 2):  # +2 because first try is attempt 1
        vec = _get_query_embedding(query)
        if vec is not None:
            _cache_set(query, vec)
            return vec
        # _get_query_embedding already prints its own error – just retry
        if attempt <= retry_count:
            print(
                f"[Warn] hybrid_pool: embedding attempt {attempt} failed, "
                f"retrying in {retry_delay:.1f}s …"
            )
            time.sleep(retry_delay)
            retry_delay *= 2  # exponential backoff

    print("[Error] hybrid_pool: all embedding attempts exhausted.")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_hybrid_candidates(
    query: str,
    *,
    days: int = 30,
    limit: int = 200,
    sim_floor: float | None = None,
) -> list[dict[str, Any]]:
    """Rich candidate retrieval via the shared hybrid RRF SQL.

    Returns candidate dicts sorted by ``final_score`` descending, each
    containing: ``url``, ``title``, ``summary``, ``created_at``, ``points``,
    ``sim_score``, ``final_score``, ``match_score``.

    Designed for downstream reranking via ``rerank_aggregation.py``.

    Parameters
    ----------
    query:
        Natural-language query for semantic matching.
    days:
        Time window in days.
    limit:
        Maximum number of candidates to return.
    sim_floor:
        Minimum cosine similarity threshold.  Defaults to the recall profile.
    """
    query_clean = (query or "").strip()
    if not query_clean:
        return []

    days = _clamp_int(days, 1, 365)
    limit = _clamp_int(limit, 1, 500)

    query_vec = _get_embedding_with_retry(query_clean)
    if query_vec is None:
        print("[Warn] fetch_hybrid_candidates: embedding unavailable; using lexical/exact hybrid channels.")

    conn = get_conn()
    try:
        cur = conn.cursor()
        rows = fetch_hybrid_rows(
            cur,
            query=query_clean,
            days=days,
            limit=limit,
            query_vec=query_vec,
            profile=ANALYSIS_PROFILE,
            sim_floor=sim_floor,
        )
        cur.close()
        candidates = []
        for row in rows:
            item = candidate_from_row(row)
            item["sim_score"] = float(item.get("semantic_score", item.get("score", 0.0)) or 0.0)
            item["time_bonus"] = 0.0
            item["points_bonus"] = 0.0
            item["match_score"] = float(item.get("match_score", item.get("sim_score", 0.0)) or 0.0)
            candidates.append(item)

        meta = retrieval_diagnostics(
            profile=ANALYSIS_PROFILE,
            query_vec=query_vec,
            candidate_count=len(candidates),
            top_k=min(limit, len(candidates)),
        )
        print(
            f"[hybrid_pool] query={query_clean!r}, days={days}, "
            f"profile={meta['recall_profile']['profile']}, requested={limit}, returned={len(candidates)}"
        )
        return candidates
    except Exception as exc:
        print(f"[Warn] hybrid candidate pool failed: {exc}")
        try:
            conn.rollback()
        except Exception:
            pass
        return []
    finally:
        put_conn(conn)


def fetch_hybrid_url_pool(
    query: str,
    *,
    days: int = 30,
    limit: int = 300,
    sim_floor: float | None = None,
) -> list[tuple[str, float]]:
    """Large-batch recall for aggregation tools (URL + match_score pairs).

    Delegates to ``fetch_hybrid_candidates()`` and projects each candidate
    to ``(url, match_score)`` sorted by descending final_score.

    Parameters
    ----------
    query:
        Natural-language query for semantic matching.
    days:
        Time window in days.
    limit:
        Maximum number of URLs to return.
    sim_floor:
        Minimum cosine similarity threshold.

    Returns
    -------
    list[tuple[str, float]]
        ``(url, match_score)`` pairs, or an empty list on failure.
    """
    candidates = fetch_hybrid_candidates(
        query, days=days, limit=limit, sim_floor=sim_floor,
    )
    return [(c["url"], float(c.get("match_score", c.get("sim_score", c.get("final_score", 0.0))) or 0.0)) for c in candidates]
