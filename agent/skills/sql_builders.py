"""SQL-related builders and topic term expansion."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_DEFAULT_TOPIC_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "ai": ["AI", "人工智能", "大模型", "LLM", "智能体", "GPT", "Gemini", "Claude", "Copilot"],
    "business": ["business", "commercial", "market", "finance", "enterprise", "商业", "市场", "金融", "营收", "盈利", "IPO", "并购"],
    "security": ["security", "cyber", "cybersecurity", "vulnerability", "breach", "安全", "网络安全", "漏洞", "威胁", "攻防", "勒索"],
}

TOPIC_QUERY_EXPANSIONS: dict[str, list[str]] = dict(_DEFAULT_TOPIC_QUERY_EXPANSIONS)


def _default_topic_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "topic_query_expansions.json"


def _normalize_topic_config(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return dict(_DEFAULT_TOPIC_QUERY_EXPANSIONS)

    out: dict[str, list[str]] = {}
    for key, value in payload.items():
        topic = str(key or "").strip().lower()
        if not topic:
            continue
        if not isinstance(value, list):
            continue
        terms: list[str] = []
        for item in value:
            term = str(item or "").strip()
            if term:
                terms.append(term)
        if terms:
            out[topic] = terms

    if not out:
        return dict(_DEFAULT_TOPIC_QUERY_EXPANSIONS)
    return out


def load_topic_query_expansions(force_reload: bool = False) -> dict[str, list[str]]:
    global TOPIC_QUERY_EXPANSIONS
    if TOPIC_QUERY_EXPANSIONS and not force_reload:
        return TOPIC_QUERY_EXPANSIONS

    path_str = os.getenv("AGENT_TOPIC_EXPANSIONS_PATH", "").strip()
    config_path = Path(path_str) if path_str else _default_topic_config_path()

    try:
        if config_path.exists():
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            TOPIC_QUERY_EXPANSIONS = _normalize_topic_config(parsed)
            return TOPIC_QUERY_EXPANSIONS
    except Exception as exc:
        print(f"[Warn] load_topic_query_expansions failed, fallback to defaults: {exc}")

    TOPIC_QUERY_EXPANSIONS = dict(_DEFAULT_TOPIC_QUERY_EXPANSIONS)
    return TOPIC_QUERY_EXPANSIONS


def _expand_topic_terms(topic: str) -> list[str]:
    base = (topic or "").strip()
    if not base:
        return []

    normalized = base.lower()
    canonical = normalized
    if normalized in {"ai", "人工智能", "大模型", "模型", "llm", "智能体"}:
        canonical = "ai"
    elif normalized in {"business", "商业", "市场", "金融", "财经"}:
        canonical = "business"
    elif normalized in {"security", "cybersecurity", "cyber", "安全", "网络安全"}:
        canonical = "security"

    expansions = load_topic_query_expansions()
    terms = [base]
    terms.extend(expansions.get(canonical, []))

    dedup: list[str] = []
    seen: set[str] = set()
    for term in terms:
        t = str(term).strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)
    return dedup[:12]


def _build_topic_where_clause(topic: str, table_alias: str = "") -> tuple[str, list[Any]]:
    terms = _expand_topic_terms(topic)
    if not terms:
        return "TRUE", []

    prefix = f"{table_alias}." if table_alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    for term in terms:
        like = f"%{term}%"
        clauses.append(
            f"({prefix}title ILIKE %s OR COALESCE({prefix}title_cn,'') ILIKE %s OR COALESCE({prefix}summary,'') ILIKE %s)"
        )
        params.extend([like, like, like])
    return "(" + " OR ".join(clauses) + ")", params


# Load once at startup (with fallback safety).
load_topic_query_expansions(force_reload=True)

