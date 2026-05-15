"""SSE event formatting for chat streaming responses."""

from __future__ import annotations

import json


def sse_event(event: str, data: dict) -> str:
    """Encode one SSE event payload."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def status_text_from_progress(payload: dict) -> str | None:
    """Map runtime progress payload to UI-facing status text."""
    title = str(payload.get("title", "")).strip()
    if title:
        return title
    stage = str(payload.get("stage", "")).strip().lower()
    if stage == "understanding":
        return "正在理解问题"
    if stage == "selecting_tools":
        return "正在调用工具"
    if stage == "retrieving":
        return "正在检索相关新闻"
    if stage == "analyzing":
        return "正在整理信息"
    if stage == "synthesizing":
        return "正在生成分析"
    if stage == "finalizing":
        return "正在生成分析"
    if stage == "clarification_required":
        return "需要补充信息"
    return None


def progress_event_payload(payload: dict) -> dict:
    return {
        "phase": str(payload.get("phase") or payload.get("stage") or "").strip(),
        "tool": str(payload.get("tool") or payload.get("tool_name") or "").strip(),
        "title": str(payload.get("title") or status_text_from_progress(payload) or "").strip(),
        "detail": str(payload.get("detail") or "").strip(),
        "article_title": str(payload.get("article_title") or "").strip(),
        "url": str(payload.get("url") or "").strip(),
        "index": payload.get("index"),
        "total": payload.get("total"),
        "items": [
            str(item).strip()
            for item in (payload.get("items") if isinstance(payload.get("items"), list) else [])
            if str(item).strip()
        ],
        "status": str(payload.get("status") or "running").strip(),
    }


def evidence_event_payload(payload: dict) -> dict:
    return {
        "tool": str(payload.get("tool") or "").strip(),
        "title": str(payload.get("title") or "").strip(),
        "source": str(payload.get("source") or payload.get("detail") or "").strip(),
        "url": str(payload.get("url") or "").strip(),
        "created_at": str(payload.get("created_at") or "").strip(),
        "rank": payload.get("rank"),
        "match_score": payload.get("match_score"),
    }
