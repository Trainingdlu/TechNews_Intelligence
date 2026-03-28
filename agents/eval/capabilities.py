"""Capability registry for eval datasets and selection filters."""

from __future__ import annotations

from typing import Any

# Keep keys stable: they are used by dataset cases and report selection metadata.
CAPABILITY_CATALOG: dict[str, dict[str, Any]] = {
    "compare_topics": {
        "description": "Forced compare route (A vs B) using DB-backed compare_topics tool.",
        "route": "forced",
        "default_min_urls": 2,
    },
    "timeline": {
        "description": "Forced timeline route using DB-backed build_timeline tool.",
        "route": "forced",
        "default_min_urls": 1,
    },
    "landscape": {
        "description": "Forced landscape route using DB-backed analyze_landscape tool.",
        "route": "forced",
        "default_min_urls": 2,
    },
    "trend_analysis": {
        "description": "LangChain tool call for topic momentum trend analysis.",
        "route": "tool",
        "default_min_urls": 0,
    },
    "compare_sources": {
        "description": "LangChain tool call for HackerNews vs TechCrunch source comparison.",
        "route": "tool",
        "default_min_urls": 1,
    },
    "query_news": {
        "description": "LangChain tool call for filterable query retrieval.",
        "route": "tool",
        "default_min_urls": 1,
    },
    "fulltext_batch": {
        "description": "LangChain tool call for batch fulltext read (URLs or keyword fallback).",
        "route": "tool",
        "default_min_urls": 1,
    },
    "general_qa": {
        "description": "General QA path (LangChain + retrieval), no fixed tool contract.",
        "route": "runtime",
        "default_min_urls": 0,
    },
}


CATEGORY_DEFAULT_CAPABILITY: dict[str, str] = {
    "compare": "compare_topics",
    "timeline": "timeline",
    "landscape": "landscape",
    "trend": "trend_analysis",
    "source_compare": "compare_sources",
    "query": "query_news",
    "fulltext": "fulltext_batch",
    "brief": "general_qa",
    "general": "general_qa",
}


def supported_capabilities() -> set[str]:
    return set(CAPABILITY_CATALOG.keys())


def resolve_capability(category: str, capability: str | None) -> str:
    cap = (capability or "").strip().lower()
    if cap:
        return cap
    return CATEGORY_DEFAULT_CAPABILITY.get(category.strip().lower(), "general_qa")

