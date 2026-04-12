"""Second-stage reranking utilities for retrieval-oriented skills."""

from __future__ import annotations

import os
from typing import Any, Sequence

import requests


RERANK_MODE_NONE = "none"
RERANK_MODE_CROSS_ENCODER = "cross_encoder"
RERANK_MODE_LLM = "llm_rerank"
SUPPORTED_RERANK_MODES = {RERANK_MODE_NONE, RERANK_MODE_CROSS_ENCODER, RERANK_MODE_LLM}

DEFAULT_RERANK_MODE_ENV = "NEWS_RERANK_MODE"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_CROSS_ENCODER_MODEL = "jina-reranker-v2-base-multilingual"
JINA_LLM_RERANK_MODEL = "jina-reranker-v3"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_text(candidate: dict[str, Any]) -> str:
    title = str(candidate.get("title") or "").strip()
    summary = str(candidate.get("summary") or "").strip()
    source_type = str(candidate.get("source_type") or "").strip()
    url = str(candidate.get("url") or "").strip()

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if source_type:
        parts.append(f"Source: {source_type}")
    if url:
        parts.append(f"URL: {url}")
    return "\n".join(parts) if parts else url


def resolve_rerank_mode(mode: str | None = None, *, env_keys: Sequence[str] | None = None) -> str:
    """Resolve effective rerank mode from explicit arg or env."""
    raw = (mode or "").strip().lower()
    if not raw:
        for key in env_keys or ():
            env_val = os.getenv(key, "").strip().lower()
            if env_val:
                raw = env_val
                break
    if not raw:
        raw = os.getenv(DEFAULT_RERANK_MODE_ENV, "").strip().lower()

    alias = {
        "": RERANK_MODE_NONE,
        "off": RERANK_MODE_NONE,
        "disabled": RERANK_MODE_NONE,
        "disable": RERANK_MODE_NONE,
        RERANK_MODE_NONE: RERANK_MODE_NONE,
        "cross": RERANK_MODE_CROSS_ENCODER,
        "cross-encoder": RERANK_MODE_CROSS_ENCODER,
        RERANK_MODE_CROSS_ENCODER: RERANK_MODE_CROSS_ENCODER,
        "llm": RERANK_MODE_LLM,
        "llm-rerank": RERANK_MODE_LLM,
        RERANK_MODE_LLM: RERANK_MODE_LLM,
    }
    resolved = alias.get(raw, RERANK_MODE_NONE)
    if raw and raw not in alias:
        print(f"[Warn] unknown rerank mode '{raw}', fallback to '{RERANK_MODE_NONE}'.")
    return resolved


def _call_jina_rerank(
    *,
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_k: int,
    model: str,
) -> list[dict[str, Any]]:
    jina_key = os.getenv("JINA_API_KEY", "").strip()
    if not jina_key:
        raise RuntimeError("JINA_API_KEY not set.")

    if not model:
        raise RuntimeError("rerank model is empty.")

    docs = [_candidate_text(item) for item in candidates]
    timeout_sec = _safe_float(os.getenv("JINA_RERANK_TIMEOUT_SEC", "15"), 15.0)
    timeout_sec = max(3.0, min(timeout_sec, 60.0))

    resp = requests.post(
        JINA_RERANK_URL,
        headers={
            "Authorization": f"Bearer {jina_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": top_k,
        },
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("invalid rerank response: missing results.")

    ordered: list[dict[str, Any]] = []
    used_indexes: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
            continue
        if idx in used_indexes:
            continue
        used_indexes.add(idx)
        ranked = dict(candidates[idx])
        ranked["rerank_score"] = _safe_float(item.get("relevance_score"), 0.0)
        ordered.append(ranked)
        if len(ordered) >= top_k:
            break

    if not ordered:
        raise RuntimeError("rerank response has no valid result indices.")
    return ordered


def _rerank_meta(
    *,
    mode: str,
    candidate_count: int,
    top_k: int,
    fallback: bool,
    model: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "rerank_mode": mode,
        "candidate_count": int(candidate_count),
        "top_k": int(top_k),
        "fallback": bool(fallback),
    }
    if model:
        meta["model"] = model
    if error:
        meta["error"] = error
    return meta


def rerank_candidates(
    query: str,
    candidates: Sequence[dict[str, Any]],
    *,
    mode: str | None = None,
    top_k: int | None = None,
    env_keys: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank candidate dictionaries and return top-k plus metadata."""
    resolved_mode = resolve_rerank_mode(mode, env_keys=env_keys)
    ranked = list(candidates)
    if not ranked:
        return [], _rerank_meta(mode=resolved_mode, candidate_count=0, top_k=0, fallback=False)

    if top_k is None:
        top_k = len(ranked)
    top_k = max(1, min(int(top_k), len(ranked)))

    if resolved_mode == RERANK_MODE_NONE or len(ranked) <= 1:
        return ranked[:top_k], _rerank_meta(
            mode=resolved_mode,
            candidate_count=len(ranked),
            top_k=top_k,
            fallback=False,
        )

    model = (
        os.getenv("JINA_LLM_RERANK_MODEL", JINA_LLM_RERANK_MODEL).strip()
        if resolved_mode == RERANK_MODE_LLM
        else os.getenv("JINA_CROSS_ENCODER_MODEL", JINA_CROSS_ENCODER_MODEL).strip()
    )
    try:
        reranked = _call_jina_rerank(query=query, candidates=ranked, top_k=top_k, model=model)
        return reranked, _rerank_meta(
            mode=resolved_mode,
            candidate_count=len(ranked),
            top_k=top_k,
            fallback=False,
            model=model,
        )
    except Exception as exc:
        print(f"[Warn] rerank failed in mode '{resolved_mode}', fallback to recall order: {exc}")
        return ranked[:top_k], _rerank_meta(
            mode=resolved_mode,
            candidate_count=len(ranked),
            top_k=top_k,
            fallback=True,
            model=model,
            error=type(exc).__name__,
        )


def rerank_lookup_rows(
    query: str,
    rows: Sequence[tuple],
    *,
    mode: str | None = None,
    top_k: int | None = None,
    env_keys: Sequence[str] | None = None,
) -> tuple[list[tuple], dict[str, Any]]:
    """Rerank rows shaped as (headline, url, source_type, created_at, points, score)."""
    candidates: list[dict[str, Any]] = []
    for row in rows:
        headline, url, source_type, created_at, points, score = row
        candidates.append(
            {
                "title": headline or "",
                "summary": "",
                "url": url or "",
                "source_type": source_type or "",
                "created_at": created_at,
                "points": points,
                "score": score,
                "payload": row,
            }
        )

    reranked, meta = rerank_candidates(query, candidates, mode=mode, top_k=top_k, env_keys=env_keys)
    return [item["payload"] for item in reranked], meta


def rerank_search_rows(
    query: str,
    rows: Sequence[tuple],
    *,
    mode: str | None = None,
    top_k: int | None = None,
    env_keys: Sequence[str] | None = None,
) -> tuple[list[tuple], dict[str, Any]]:
    """Rerank rows shaped as (title, url, summary, sentiment, created_at, score)."""
    candidates: list[dict[str, Any]] = []
    for row in rows:
        title, url, summary, sentiment, created_at, score = row
        candidates.append(
            {
                "title": title or "",
                "summary": summary or "",
                "url": url or "",
                "sentiment": sentiment or "",
                "created_at": created_at,
                "score": score,
                "points": 0.0,
                "payload": row,
            }
        )

    reranked, meta = rerank_candidates(query, candidates, mode=mode, top_k=top_k, env_keys=env_keys)
    return [item["payload"] for item in reranked], meta

