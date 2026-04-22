"""Unified semantic-recall → time-decay → rerank pipeline for macro-analysis skills.

Provides a single entry-point ``retrieve_and_rerank()`` that combines:
1. ``fetch_semantic_candidates()`` — vector recall with dynamic time-decay.
2. Pre-rerank truncation (by ``final_score``).
3. ``rerank_candidates()`` — Jina reranker for precision Top-K.

Used by: compare_topics, compare_sources, trend_analysis,
         build_timeline, analyze_landscape.
"""

from __future__ import annotations

import time
from typing import Any

from .rerank import RERANK_MODE_LLM, rerank_candidates
from .semantic_pool import fetch_semantic_candidates


def retrieve_and_rerank(
    query: str,
    *,
    days: int = 14,
    pool_limit: int = 200,
    pre_rerank_limit: int = 30,
    top_k: int = 5,
    rerank_mode: str = RERANK_MODE_LLM,
    sim_floor: float | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Full retrieval-and-rerank pipeline for macro-analysis skills.

    Parameters
    ----------
    query:
        Natural-language query for semantic matching.
    days:
        Time window in days.  Also used as the decay constant for time
        scoring.  Each skill should pass its own business-appropriate
        default (e.g. 14 for compare_topics, 30 for build_timeline).
    pool_limit:
        Maximum candidates from the semantic pool.
    pre_rerank_limit:
        How many top candidates (by ``final_score``) to send to the
        reranker.  This bounds Jina API cost.
    top_k:
        Final number of candidates to return after reranking.
    rerank_mode:
        Rerank mode (``"llm_rerank"`` or ``"none"``).
    sim_floor:
        Minimum cosine similarity threshold for the semantic pool.

    Returns
    -------
    tuple of:
        - ``top_candidates`` — Top-K reranked candidate dicts (with
          ``title``, ``summary``, ``url``, ``rerank_score``).
        - ``all_urls`` — Full URL list from the semantic pool (for
          downstream SQL aggregation via ``WHERE url = ANY(…)``).
        - ``meta`` — Pipeline metadata (timing, pool_size, rerank info).
    """
    t0 = time.time()

    # Stage 1: Semantic recall with time-decay scoring
    candidates = fetch_semantic_candidates(
        query, days=days, limit=pool_limit, sim_floor=sim_floor,
    )
    all_urls = [c["url"] for c in candidates]
    pool_size = len(candidates)

    if not candidates:
        meta = {
            "pool_size": 0,
            "pre_rerank_size": 0,
            "top_k": top_k,
            "rerank_mode": rerank_mode,
            "pipeline_ms": round((time.time() - t0) * 1000, 1),
            "fallback": True,
        }
        return [], all_urls, meta

    # Stage 2: Truncate to pre_rerank_limit (already sorted by final_score)
    pre_rerank_candidates = candidates[:pre_rerank_limit]

    # Stage 3: Rerank via Jina (or passthrough if mode=none)
    reranked, rerank_meta = rerank_candidates(
        query,
        pre_rerank_candidates,
        mode=rerank_mode,
        top_k=top_k,
        env_keys=("MACRO_SKILL_RERANK_MODE", "NEWS_RERANK_MODE"),
    )

    pipeline_ms = round((time.time() - t0) * 1000, 1)
    meta = {
        "pool_size": pool_size,
        "pre_rerank_size": len(pre_rerank_candidates),
        "top_k": top_k,
        "rerank_mode": rerank_mode,
        "pipeline_ms": pipeline_ms,
        **rerank_meta,
    }

    return reranked, all_urls, meta


def format_reranked_evidence(
    candidates: list[dict[str, Any]],
    *,
    header: str = "Reranked Top Evidence",
    max_summary_len: int = 200,
) -> str:
    """Format reranked candidates into a human-readable evidence block.

    Used by macro-analysis skills to inject reranked context into the
    text output sent to the LLM.
    """
    if not candidates:
        return ""

    lines = [f"\n--- {header} ---"]
    for idx, c in enumerate(candidates, 1):
        title = str(c.get("title") or "").strip()
        summary = str(c.get("summary") or "").strip()
        url = str(c.get("url") or "").strip()
        rerank_score = c.get("rerank_score") or c.get("final_score") or 0.0

        if summary and len(summary) > max_summary_len:
            summary = summary[:max_summary_len] + "…"

        line = f"#{idx} [{title}]({url})"
        if summary and summary != title:
            line += f"\n   摘要: {summary}"
        line += f"\n   Relevance: {float(rerank_score):.3f}"
        lines.append(line)

    return "\n".join(lines)
