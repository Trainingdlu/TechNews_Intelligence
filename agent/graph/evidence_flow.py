"""Evidence normalization and retrieval fallback helpers for graph nodes."""

from __future__ import annotations

import re
from typing import Any

from agent.context_manager import active_question
from agent.core.evidence import normalize_url_for_match
from agent.core.intent import extract_user_intent_text as _extract_user_intent_text
from agent.core.tool_contracts import ToolEnvelope

from .intent_heuristics import _extract_days, _extract_entity_hints


def _normalize_evidence(results: list[ToolEnvelope]) -> tuple[list[str], str]:
    urls: list[str] = []
    seen: set[str] = set()
    lines: list[str] = []
    for envelope in results:
        for item in envelope.evidence or []:
            normalized = normalize_url_for_match(item.url)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            urls.append(item.url)
            source = str(item.source or "").strip()
            title = str(item.title or item.url).strip()
            created = str(item.created_at or "").strip()
            lines.append(f"- {source} · {title} · {created} · {item.url}".strip())
    return urls, "\n".join(lines[:12])


def _empty_evidence_fallback_calls(state: dict[str, Any]) -> list[dict[str, Any]]:
    selected = list(state.get("selected_tools") or [])
    executed = {item.tool for item in (state.get("tool_results") or [])}
    active = active_question(state.get("context_pack"), str(state.get("user_message") or ""))
    text = _extract_user_intent_text(active) or active
    query = _fallback_retrieval_query(text)
    days = _extract_days(text)
    calls: list[dict[str, Any]] = []
    if "search_news" in selected and "search_news" not in executed:
        calls.append({"name": "search_news", "args": {"query": query, "days": days}})
    if "query_news" in selected and "query_news" not in executed:
        calls.append({"name": "query_news", "args": {"query": query, "days": days, "limit": 8}})
    if calls:
        return calls
    if "fulltext_batch" in selected and "fulltext_batch" not in executed:
        return [{"name": "fulltext_batch", "args": {"urls": query, "max_chars_per_article": 4000}}]
    return []


def _fallback_retrieval_query(text: str) -> str:
    raw = str(text or "").strip()
    entities = _extract_entity_hints(raw)
    focus_terms = re.findall(
        r"企业市场|商业化|战略|策略|定价|开源|生态|产品|大模型|多模态|布局|差异|enterprise|commerciali[sz]ation|pricing|strategy|ecosystem",
        raw,
        flags=re.IGNORECASE,
    )
    parts: list[str] = []
    seen: set[str] = set()
    for item in [*entities[:4], *focus_terms]:
        normalized = str(item or "").strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            parts.append(normalized)
    if parts:
        return " ".join(parts)
    return raw[:180] or "technology news"


def _requires_evidence(state: dict[str, Any]) -> bool:
    intent = state.get("intent") or {}
    return str(intent.get("route") or "") == "needs_tools"


def _should_read_after_search(results: list[ToolEnvelope], state: dict[str, Any]) -> bool:
    selected = set(state.get("selected_tools") or [])
    if "fulltext_batch" not in selected:
        return False
    if any(item.tool == "fulltext_batch" for item in results):
        return False
    if not any(item.tool in {"search_news", "query_news"} and item.evidence for item in results):
        return False
    intent_type = str((state.get("intent") or {}).get("intent_type") or "")
    return intent_type in {
        "news_analysis",
        "trend",
        "roundup_listing",
        "article_read",
        "topic_comparison",
        "source_comparison",
        "landscape",
    }


def _fulltext_calls_from_evidence(evidence_urls: list[str], state: dict[str, Any]) -> list[dict[str, Any]]:
    if evidence_urls:
        return [
            {
                "name": "fulltext_batch",
                "args": {"urls": "\n".join(evidence_urls[:3]), "max_chars_per_article": 4000},
            }
        ]
    return [
        {
            "name": "fulltext_batch",
            "args": {
                "urls": _extract_user_intent_text(active_question(state.get("context_pack"), str(state.get("user_message") or ""))) or active_question(state.get("context_pack"), str(state.get("user_message") or "")),
                "max_chars_per_article": 4000,
            },
        }
    ]
