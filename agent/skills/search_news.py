"""Search-news skill implementation and structured adapter."""

from __future__ import annotations

import json
from typing import Any

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_records, _json_text
from .retrieval import lookup_candidates_by_query
from .schemas import SearchNewsSkillInput


def _format_search_news_text(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""

    max_score = max(float(item.get("score", 0.0) or 0.0) for item in records)
    note = ""
    if max_score < 0.5:
        note = "[Note] Relevance is weak; these are nearest matches.\n\n"

    out: list[str] = []
    for item in records:
        title = str(item.get("title") or "")
        url = str(item.get("url") or "")
        summary = str(item.get("summary") or "")
        sentiment = str(item.get("sentiment") or "")
        pub_time = item.get("created_at")
        if hasattr(pub_time, "strftime"):
            pub_time_str = pub_time.strftime("%Y-%m-%d %H:%M")
        else:
            pub_time_str = str(pub_time or "").replace("T", " ")[:16]
        score = float(item.get("score") or 0.0)
        out.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Summary: {summary}\n"
            f"Sentiment: {sentiment}\n"
            f"Time: {pub_time_str}\n"
            f"Score: {float(score):.3f}"
        )
    return note + "\n---\n".join(out)


def _records_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates, 1):
        pub_time = item.get("created_at")
        records.append(
            {
                "rank": idx,
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "summary": str(item.get("summary") or ""),
                "sentiment": str(item.get("sentiment") or ""),
                "created_at": pub_time.isoformat() if pub_time else "",
                "score": float(item.get("score") or 0.0),
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
    try:
        candidates, rerank_meta = lookup_candidates_by_query(
            query=query,
            days=days,
            limit=5,
            rerank_mode=rerank_mode,
        )
        if not candidates:
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

        records = _records_from_candidates(candidates)
        raw_text = _format_search_news_text(records)
        if as_json:
            return _json_text(
                {
                    "tool": "search_news",
                    "status": "ok",
                    "request": {"query": query, "days": days},
                    "rerank": rerank_meta,
                    "count": len(records),
                    "records": records,
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
                    "rerank": {},
                    "error": f"search_news failed: {exc}",
                    "records": [],
                }
            )
        return f"search_news failed: {exc}"


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
