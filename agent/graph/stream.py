"""Stable user-facing progress events for the custom graph."""

from __future__ import annotations

from typing import Any

from agent.core.run_context import emit_progress
from agent.core.tool_contracts import ToolEvidence, ToolEnvelope


def emit_graph_progress(
    phase: str,
    title: str,
    *,
    detail: str = "",
    items: list[str] | None = None,
    status: str = "running",
) -> None:
    emit_progress(
        phase,
        phase=phase,
        title=title,
        detail=detail,
        items=items,
        status=status,
        event="progress",
    )


def emit_graph_evidence(envelope: ToolEnvelope, *, limit: int = 3) -> None:
    for item in list(envelope.evidence or [])[: max(0, int(limit))]:
        payload = _evidence_payload(item, tool=envelope.tool)
        emit_progress(
            "evidence",
            tool_name=envelope.tool,
            title=str(payload.get("title") or ""),
            detail=str(payload.get("source") or ""),
            status="found",
            event="evidence",
            extra=payload,
        )


def evidence_status_items(envelopes: list[ToolEnvelope], *, limit: int = 3) -> list[str]:
    out: list[str] = []
    for envelope in envelopes:
        for item in envelope.evidence or []:
            label = _source_title_label(item)
            if label and label not in out:
                out.append(label)
            if len(out) >= limit:
                return out
    return out


def _evidence_payload(item: ToolEvidence, *, tool: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "title": item.title or "",
        "source": item.source or "",
        "url": item.url,
        "created_at": item.created_at or "",
        "rank": item.rank,
        "match_score": item.match_score,
    }


def _source_title_label(item: ToolEvidence) -> str:
    source = str(item.source or "").strip()
    title = str(item.title or item.url or "").strip()
    if not title:
        return ""
    if len(title) > 72:
        title = f"{title[:69]}..."
    return f"{source} · {title}" if source else title
