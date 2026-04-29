"""Fulltext-batch skill implementation and structured adapter."""

from __future__ import annotations

import json
from typing import Any

from ..core.skill_contracts import SkillEnvelope, build_empty_envelope, build_error_envelope
from .helpers import (
    _clamp_int,
    _extract_time_window_days,
    _is_probable_url,
    _json_text,
    _safe_float,
    _split_urls,
)
from .news_ops import read_news_content
from .rerank import resolve_rerank_mode
from .retrieval import lookup_candidates_by_query
from .schemas import FulltextBatchSkillInput


def fulltext_batch(
    urls: str,
    max_chars_per_article: int = 4000,
    response_format: str = "text",
    rerank_mode: str | None = None,
) -> str:
    """Batch-read full article text by URL list or keyword auto-selection."""
    print("\n[Tool] fulltext_batch")
    max_chars_per_article = _clamp_int(max_chars_per_article, 800, 12000)
    as_json = response_format.strip().lower() == "json"
    resolved_rerank_mode = resolve_rerank_mode(
        rerank_mode, env_keys=("FULLTEXT_BATCH_RERANK_MODE", "NEWS_RERANK_MODE")
    )
    rerank_meta: dict[str, Any] = {
        "rerank_mode": resolved_rerank_mode,
        "candidate_count": 0,
        "top_k": 0,
        "fallback": False,
    }

    raw_items = _split_urls(urls)
    direct_urls = [item for item in raw_items if _is_probable_url(item)]

    selected: list[tuple[str, str, dict[str, Any]]] = []
    prefix_lines: list[str] = []
    selected_meta: list[dict[str, Any]] = []

    if direct_urls:
        for url in direct_urls[:12]:
            meta = {"selection_mode": "direct", "url": url, "rerank_mode": resolved_rerank_mode}
            selected.append(("direct", url, meta))
            selected_meta.append(meta)
        rerank_meta = {
            "rerank_mode": resolved_rerank_mode,
            "candidate_count": len(selected),
            "top_k": len(selected),
            "fallback": False,
        }
    else:
        query = (urls or "").strip()
        if not query:
            return "fulltext_batch requires URLs or a keyword query."

        days = _extract_time_window_days(query, default=14, maximum=120)
        candidates, rerank_meta = lookup_candidates_by_query(
            query=query,
            days=days,
            limit=6,
            rerank_mode=resolved_rerank_mode,
        )
        if not candidates:
            if as_json:
                return _json_text(
                    {
                        "tool": "fulltext_batch",
                        "status": "empty",
                        "request": {
                            "urls_or_query": urls,
                            "query": query,
                            "days": days,
                            "max_chars_per_article": max_chars_per_article,
                            "rerank_mode": resolved_rerank_mode,
                        },
                        "rerank": rerank_meta,
                        "selected": [],
                        "articles": [],
                        "error": f"No candidate articles found for query '{query}'.",
                    }
                )
            return f"No candidate articles found for query '{query}'."

        prefix_lines.append(
            f"No URLs provided. Auto-selected Top {len(candidates)} articles for query '{query}' (window={days}d):"
        )
        for rank, row in enumerate(candidates, 1):
            headline = str(row.get("title") or "")
            url = str(row.get("url") or "")
            source_type = str(row.get("source_type") or "")
            created_at = row.get("created_at")
            points = int(row.get("points") or 0)
            score = float(row.get("score") or 0.0)
            match_score = _safe_float(row.get("match_score"))
            created_at_text = created_at.strftime("%Y-%m-%d %H:%M") if hasattr(created_at, "strftime") else ""
            prefix_lines.append(
                f"{rank}. [{source_type}] {headline} | points={points} | "
                f"score={float(score):.3f} | {created_at_text} | {url}"
            )
            meta = {
                "selection_mode": "query",
                "query": query,
                "window_days": days,
                "rerank_mode": resolved_rerank_mode,
                "rank": rank,
                "headline": headline,
                "source_type": source_type,
                "points": points,
                "score": score,
                "match_score": match_score,
                "text_score": _safe_float(row.get("text_score")),
                "semantic_score": _safe_float(row.get("semantic_score")),
                "exact_score": _safe_float(row.get("exact_score")),
                "final_score": _safe_float(row.get("final_score")),
                "created_at": created_at.isoformat() if created_at else "",
                "url": url,
            }
            selected.append(("query", url, meta))
            selected_meta.append(meta)

    chunks: list[str] = []
    article_rows: list[dict[str, Any]] = []
    for idx, (_, url, meta) in enumerate(selected, 1):
        content = read_news_content(url)
        truncated = False
        if len(content) > max_chars_per_article:
            content = content[:max_chars_per_article] + "\n...[truncated]"
            truncated = True
        chunks.append(f"=== [{idx}] {url} ===\n{content}")
        article_rows.append(
            {
                "index": idx,
                "url": url,
                "content": content,
                "truncated": truncated,
                "meta": meta,
            }
        )

    if as_json:
        return _json_text(
            {
                "tool": "fulltext_batch",
                "status": "ok",
                "request": {
                    "urls_or_query": urls,
                    "max_chars_per_article": max_chars_per_article,
                    "rerank_mode": resolved_rerank_mode,
                },
                "rerank": rerank_meta,
                "selected": selected_meta,
                "articles": article_rows,
            }
        )

    if prefix_lines:
        return "\n".join(prefix_lines) + "\n\n" + "\n\n".join(chunks)
    return "\n\n".join(chunks)


def fulltext_batch_skill(payload: FulltextBatchSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    try:
        raw_output = fulltext_batch(
            urls=request["urls"],
            max_chars_per_article=int(request.get("max_chars_per_article", 4000)),
            response_format="json",
        )
    except Exception as exc:
        return build_error_envelope(
            tool="fulltext_batch",
            request=request,
            error="fulltext_batch_execution_failed",
            diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    try:
        parsed = json.loads(raw_output)
    except Exception as exc:
        return build_error_envelope(
            tool="fulltext_batch",
            request=request,
            error="fulltext_batch_json_parse_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "raw_preview": str(raw_output)[:500],
            },
        )

    status = str(parsed.get("status", "ok")).lower()
    if status == "empty":
        rerank_meta = parsed.get("rerank", {})
        if not isinstance(rerank_meta, dict):
            rerank_meta = {}
        return build_empty_envelope(
            tool="fulltext_batch",
            request=request,
            empty_reason="no_candidate_articles",
            data=parsed,
            diagnostics={
                "rerank": rerank_meta,
                "candidate_count": int(rerank_meta.get("candidate_count") or 0),
                "selected_count": 0,
                "article_count": 0,
                "fallback": bool(rerank_meta.get("retrieval_fallback") or rerank_meta.get("fallback") or False),
            },
        )
    if status == "error":
        return build_error_envelope(
            tool="fulltext_batch",
            request=request,
            error=str(parsed.get("error") or "fulltext_batch_failed"),
            data=parsed,
            diagnostics={"rerank": parsed.get("rerank", {}) if isinstance(parsed.get("rerank"), dict) else {}},
        )

    selected = parsed.get("selected", [])
    articles = parsed.get("articles", [])
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in selected:
        url = str(item.get("url") or "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            evidence.append(
                {
                    "url": url,
                    "title": item.get("headline"),
                    "source": item.get("source_type"),
                    "created_at": item.get("created_at"),
                    "score": _safe_float(item.get("score")),
                    "rank": int(item.get("rank")) if str(item.get("rank") or "").isdigit() else None,
                    "match_score": _safe_float(item.get("match_score")),
                    "score_components": {
                        "text_score": _safe_float(item.get("text_score")),
                        "semantic_score": _safe_float(item.get("semantic_score")),
                        "exact_score": _safe_float(item.get("exact_score")),
                        "final_score": _safe_float(item.get("final_score")),
                    },
                    "metadata": {
                        "selection_mode": item.get("selection_mode"),
                        "points": item.get("points"),
                        "rerank_mode": item.get("rerank_mode"),
                    },
                }
            )

    if not evidence:
        for article in articles:
            url = str(article.get("url") or "").strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                evidence.append(
                    {
                        "url": url,
                        "title": None,
                        "source": None,
                        "created_at": None,
                        "score": None,
                        "rank": int(article.get("index")) if str(article.get("index") or "").isdigit() else None,
                    }
                )
            if len(evidence) >= 12:
                break

    return SkillEnvelope(
        tool="fulltext_batch",
        status="ok",
        request=request,
        data=parsed,
        evidence=evidence[:12],
        diagnostics={
            "selected_count": len(selected) if isinstance(selected, list) else 0,
            "article_count": len(articles) if isinstance(articles, list) else 0,
            "candidate_count": len(selected) if isinstance(selected, list) else 0,
            "evidence_count": len(evidence[:12]),
            "retrieval_mode": (parsed.get("rerank") or {}).get("retrieval_mode") if isinstance(parsed.get("rerank"), dict) else None,
            "fallback": bool(
                ((parsed.get("rerank") or {}).get("retrieval_fallback") or (parsed.get("rerank") or {}).get("fallback") or False)
                if isinstance(parsed.get("rerank"), dict)
                else False
            ),
            "rerank": parsed.get("rerank", {}),
        },
    )
