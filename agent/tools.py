"""Database and retrieval tools for the agent runtime."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from pydantic import BaseModel, Field

from services.db import get_conn, put_conn
from .core.skill_contracts import SkillEnvelope, build_error_envelope


JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"

DEFAULT_LANDSCAPE_ENTITIES = [
    "OpenAI",
    "Anthropic",
    "Google",
    "Microsoft",
    "Meta",
    "Amazon",
    "Apple",
    "NVIDIA",
    "Tesla",
    "TSMC",
    "Intel",
    "AMD",
    "CrowdStrike",
    "Palo Alto Networks",
    "Cloudflare",
    "Cisco",
]

LANDSCAPE_ENTITY_ALIASES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "谷歌": "Google",
    "microsoft": "Microsoft",
    "微软": "Microsoft",
    "meta": "Meta",
    "amazon": "Amazon",
    "aws": "Amazon",
    "亚马逊": "Amazon",
    "nvidia": "NVIDIA",
    "英伟达": "NVIDIA",
    "apple": "Apple",
    "苹果": "Apple",
    "tesla": "Tesla",
    "特斯拉": "Tesla",
    "tsmc": "TSMC",
    "台积电": "TSMC",
    "intel": "Intel",
    "英特尔": "Intel",
    "amd": "AMD",
    "crowdstrike": "CrowdStrike",
    "palo alto": "Palo Alto Networks",
    "palo alto networks": "Palo Alto Networks",
    "cloudflare": "Cloudflare",
    "cisco": "Cisco",
    "xai": "xAI",
    "x.ai": "xAI",
}

LANDSCAPE_SIGNAL_KEYWORDS = {
    "compute_cost": [
        "gpu",
        "tpu",
        "chip",
        "semiconductor",
        "datacenter",
        "data center",
        "server",
        "compute",
        "training cost",
        "inference cost",
        "capex",
        "算力",
        "芯片",
        "电力",
        "能耗",
    ],
    "algorithm_efficiency": [
        "model",
        "benchmark",
        "reasoning",
        "architecture",
        "transformer",
        "agent",
        "inference",
        "算法",
        "模型",
        "推理",
        "架构",
        "蒸馏",
        "微调",
    ],
    "data_moat": [
        "dataset",
        "data",
        "corpus",
        "licensing",
        "proprietary",
        "copyright",
        "privacy",
        "数据",
        "语料",
        "授权",
        "版权",
        "隐私",
    ],
    "go_to_market": [
        "enterprise",
        "customer",
        "pricing",
        "revenue",
        "subscription",
        "partnership",
        "adoption",
        "sales",
        "商业化",
        "客户",
        "定价",
        "收入",
        "订阅",
        "合作",
        "落地",
    ],
    "policy_security": [
        "regulation",
        "compliance",
        "antitrust",
        "lawsuit",
        "security",
        "breach",
        "vulnerability",
        "policy",
        "military",
        "监管",
        "合规",
        "诉讼",
        "安全",
        "漏洞",
        "军方",
    ],
}

LANDSCAPE_SIGNAL_LABELS = {
    "compute_cost": "Compute/Cost",
    "algorithm_efficiency": "Algorithm/Efficiency",
    "data_moat": "Data/Moat",
    "go_to_market": "Go-to-Market",
    "policy_security": "Policy/Security",
}

TOPIC_QUERY_EXPANSIONS = {
    "ai": [
        "AI",
        "人工智能",
        "大模型",
        "LLM",
        "智能体",
        "GPT",
        "Gemini",
        "Claude",
        "Copilot",
    ],
    "business": [
        "business",
        "commercial",
        "market",
        "finance",
        "enterprise",
        "商业",
        "市场",
        "金融",
        "营收",
        "盈利",
        "IPO",
        "并购",
    ],
    "security": [
        "security",
        "cyber",
        "cybersecurity",
        "vulnerability",
        "breach",
        "安全",
        "网络安全",
        "漏洞",
        "威胁",
        "攻防",
        "勒索",
    ],
}


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = minimum
    return max(minimum, min(maximum, n))


def _source_to_db_label(source: str) -> str | None:
    if not source:
        return None
    norm = source.strip().lower()
    if norm in {"hn", "hackernews", "hacker_news"}:
        return "HackerNews"
    if norm in {"tc", "techcrunch", "tech_crunch"}:
        return "TechCrunch"
    if norm in {"all", "*"}:
        return None
    # Allow direct filtering by any source name in view_dashboard_news.source_type.
    return source.strip()


def _split_urls(urls: str) -> list[str]:
    if not urls:
        return []

    text = urls.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    parts = re.split(r"[\n,\s]+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_probable_url(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("http://") or t.startswith("https://")


def _extract_time_window_days(text: str, default: int = 14, maximum: int = 180) -> int:
    m = re.search(
        r"(?:最近|过去|近|last|recent|past)?\s*(\d{1,3})\s*(天|日|周|星期|月|day|days|week|weeks|month|months)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return _clamp_int(default, 1, maximum)
    value = int(m.group(1))
    unit = str(m.group(2)).lower()
    if unit in {"周", "星期", "week", "weeks"}:
        value *= 7
    elif unit in {"月", "month", "months"}:
        value *= 30
    return _clamp_int(value, 1, maximum)


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

    terms = [base]
    terms.extend(TOPIC_QUERY_EXPANSIONS.get(canonical, []))

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


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _to_utc_naive_datetime(value: Any) -> datetime | None:
    """Normalize timestamp-like values into UTC-naive datetime for safe comparison."""
    if value is None:
        return None

    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        # datetime.fromisoformat doesn't accept trailing "Z".
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except Exception:
            return None
    else:
        return None

    if dt.tzinfo is not None:
        try:
            dt = dt.astimezone(timezone.utc)
        except Exception:
            pass
        dt = dt.replace(tzinfo=None)
    return dt


def _is_recent_timestamp(value: Any, cutoff: Any) -> bool:
    """Compare timestamps safely across naive/aware/string timestamp mixes."""
    lhs = _to_utc_naive_datetime(value)
    rhs = _to_utc_naive_datetime(cutoff)
    if lhs is None or rhs is None:
        return False
    try:
        return lhs >= rhs
    except Exception:
        return False


def lookup_url_titles(urls: list[str]) -> dict[str, str]:
    """Lookup URL -> title_cn/title in DB for presentation layer."""
    unique_urls = [u.strip() for u in (urls or []) if str(u).strip()]
    if not unique_urls:
        return {}

    dedup_urls = list(dict.fromkeys(unique_urls))
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT url, COALESCE(NULLIF(title_cn, ''), NULLIF(title, ''), '')
            FROM tech_news
            WHERE url = ANY(%s)
            """,
            (dedup_urls,),
        )
        rows = cur.fetchall()
        cur.close()
        out: dict[str, str] = {}
        for url, title in rows:
            key = str(url or "").strip()
            val = str(title or "").strip()
            if key and val:
                out[key] = val
        return out
    except Exception as e:
        print(f"[Warn] lookup_url_titles failed: {e}")
        return {}
    finally:
        if conn is not None:
            put_conn(conn)


def _normalize_landscape_entities(entities: str | list[str] | None) -> list[str]:
    raw_items: list[str] = []
    if isinstance(entities, str):
        raw_items = [x.strip() for x in re.split(r"[,\n;/|，、]+", entities) if x.strip()]
    elif isinstance(entities, list):
        raw_items = [str(x).strip() for x in entities if str(x).strip()]

    if not raw_items:
        return list(DEFAULT_LANDSCAPE_ENTITIES)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        alias_key = item.strip().lower()
        name = LANDSCAPE_ENTITY_ALIASES.get(alias_key, item.strip())
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)

    if not normalized:
        return list(DEFAULT_LANDSCAPE_ENTITIES)
    return normalized[:12]


def _landscape_signal_counts(rows: list[tuple]) -> dict[str, int]:
    """Keyword-proxy variable counts from (headline + summary) rows."""
    counts = {k: 0 for k in LANDSCAPE_SIGNAL_KEYWORDS.keys()}
    for _, _, headline, summary, _ in rows:
        text = f"{headline or ''} {summary or ''}".lower()
        for key, tokens in LANDSCAPE_SIGNAL_KEYWORDS.items():
            if any(token in text for token in tokens):
                counts[key] += 1
    return counts


def _get_query_embedding(query: str) -> list[float] | None:
    """Get query embedding via Jina API. Return None on failure."""
    jina_key = os.getenv("JINA_API_KEY", "")
    if not jina_key:
        print("[Error] JINA_API_KEY not set, skip vector search.")
        return None

    try:
        resp = requests.post(
            JINA_EMBED_URL,
            headers={
                "Authorization": f"Bearer {jina_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "task": "retrieval.query",
                "input": [query],
            },
            timeout=15,
        )
        resp.raise_for_status()
        emb = resp.json()["data"][0]["embedding"]
        return emb
    except Exception as e:
        print(f"[Error] Embedding request failed, fallback to keyword search only: {e}")
        return None


def _lookup_urls_by_query(query: str, days: int = 14, limit: int = 5) -> list[tuple]:
    """Return candidate URLs by fused semantic+keyword retrieval."""
    query_clean = (query or "").strip()
    if not query_clean:
        return []

    days = _clamp_int(days, 1, 180)
    limit = _clamp_int(limit, 1, 12)
    q = f"%{query_clean}%"
    query_vec = _get_query_embedding(query_clean)

    conn = get_conn()
    try:
        cur = conn.cursor()
        if query_vec:
            vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
            try:
                cur.execute(
                    """
                    WITH keyword AS (
                        SELECT
                            COALESCE(v.title_cn, v.title) AS headline,
                            v.url,
                            v.source_type,
                            v.created_at,
                            COALESCE(v.points, 0) AS points,
                            (
                                CASE
                                    WHEN (v.title ILIKE %s OR COALESCE(v.title_cn, '') ILIKE %s) THEN 1.3
                                    ELSE 0.0
                                END
                                + CASE
                                    WHEN COALESCE(v.summary, '') ILIKE %s THEN 0.6
                                    ELSE 0.0
                                END
                                + LEAST(0.8, GREATEST(0.0, COALESCE(v.points, 0)::float / 220.0))
                                + 0.2 * EXP(-EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 86400.0 / 21.0)
                            )::float AS score
                        FROM view_dashboard_news v
                        WHERE v.created_at >= NOW() - %s::interval
                          AND (
                              v.title ILIKE %s
                              OR COALESCE(v.title_cn, '') ILIKE %s
                              OR COALESCE(v.summary, '') ILIKE %s
                          )
                        LIMIT %s
                    ),
                    semantic AS (
                        SELECT
                            COALESCE(v.title_cn, v.title) AS headline,
                            v.url,
                            v.source_type,
                            v.created_at,
                            COALESCE(v.points, 0) AS points,
                            (
                                (1 - (e.embedding <=> %s::vector))
                                + 0.2 * EXP(-EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 86400.0 / 21.0)
                                + LEAST(0.8, GREATEST(0.0, COALESCE(v.points, 0)::float / 280.0))
                            )::float AS score
                        FROM view_dashboard_news v
                        JOIN news_embeddings e ON e.url = v.url
                        WHERE v.created_at >= NOW() - %s::interval
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                    ),
                    combined AS (
                        SELECT * FROM keyword
                        UNION ALL
                        SELECT * FROM semantic
                    ),
                    dedup AS (
                        SELECT
                            headline,
                            url,
                            source_type,
                            created_at,
                            points,
                            score,
                            ROW_NUMBER() OVER (
                                PARTITION BY url
                                ORDER BY score DESC, points DESC NULLS LAST, created_at DESC
                            ) AS rn
                        FROM combined
                    )
                    SELECT headline, url, source_type, created_at, points, score
                    FROM dedup
                    WHERE rn = 1
                    ORDER BY score DESC, points DESC NULLS LAST, created_at DESC
                    LIMIT %s
                    """,
                    (
                        q,
                        q,
                        q,
                        f"{days} days",
                        q,
                        q,
                        q,
                        limit * 4,
                        vec_str,
                        f"{days} days",
                        vec_str,
                        limit * 6,
                        limit,
                    ),
                )
                rows = cur.fetchall()
                cur.close()
                return rows
            except Exception as exc:
                # Keep fallback deterministic even when vector table/query is temporarily unavailable.
                print(f"[Warn] semantic candidate lookup failed; fallback to keyword-only: {exc}")

        cur.execute(
            """
            SELECT
                COALESCE(v.title_cn, v.title) AS headline,
                v.url,
                v.source_type,
                v.created_at,
                COALESCE(v.points, 0) AS points,
                (
                    CASE
                        WHEN (v.title ILIKE %s OR COALESCE(v.title_cn, '') ILIKE %s) THEN 1.3
                        ELSE 0.0
                    END
                    + CASE
                        WHEN COALESCE(v.summary, '') ILIKE %s THEN 0.6
                        ELSE 0.0
                    END
                    + LEAST(0.8, GREATEST(0.0, COALESCE(v.points, 0)::float / 220.0))
                    + 0.2 * EXP(-EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 86400.0 / 21.0)
                )::float AS score
            FROM view_dashboard_news v
            WHERE v.created_at >= NOW() - %s::interval
              AND (v.title ILIKE %s OR COALESCE(v.title_cn,'') ILIKE %s OR COALESCE(v.summary,'') ILIKE %s)
            ORDER BY score DESC, points DESC NULLS LAST, created_at DESC
            LIMIT %s
            """,
            (q, q, q, f"{days} days", q, q, q, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        put_conn(conn)


class QueryNewsSkillInput(BaseModel):
    """Typed input contract for structured query_news skill."""

    query: str = ""
    source: str = "all"
    days: int = Field(default=21, ge=1, le=365)
    category: str = ""
    sentiment: str = ""
    sort: str = "time_desc"
    limit: int = Field(default=8, ge=1, le=30)


class TrendAnalysisSkillInput(BaseModel):
    """Typed input contract for structured trend_analysis skill."""

    topic: str = Field(min_length=1)
    window: int = Field(default=7, ge=3, le=60)


class SearchNewsSkillInput(BaseModel):
    """Typed input contract for structured search_news skill."""

    query: str = Field(min_length=1)
    days: int = Field(default=21, ge=1, le=365)


class CompareSourcesSkillInput(BaseModel):
    """Typed input contract for structured compare_sources skill."""

    topic: str = Field(min_length=1)
    days: int = Field(default=14, ge=1, le=90)


class CompareTopicsSkillInput(BaseModel):
    """Typed input contract for structured compare_topics skill."""

    topic_a: str = Field(min_length=1)
    topic_b: str = Field(min_length=1)
    days: int = Field(default=14, ge=1, le=90)


class BuildTimelineSkillInput(BaseModel):
    """Typed input contract for structured build_timeline skill."""

    topic: str = Field(min_length=1)
    days: int = Field(default=30, ge=1, le=180)
    limit: int = Field(default=12, ge=3, le=40)


class AnalyzeLandscapeSkillInput(BaseModel):
    """Typed input contract for structured analyze_landscape skill."""

    topic: str = ""
    days: int = Field(default=30, ge=7, le=180)
    entities: str = ""
    limit_per_entity: int = Field(default=3, ge=1, le=5)


class FulltextBatchSkillInput(BaseModel):
    """Typed input contract for structured fulltext_batch skill."""

    urls: str = Field(min_length=1)
    max_chars_per_article: int = Field(default=4000, ge=800, le=12000)


def get_db_stats() -> str:
    """Get database freshness statistics and total article count.

    Use this tool first when you need to understand how recent the data is
    or how many articles are available before performing deeper analysis.

    Returns:
        A one-line summary: total article count and timestamp of the most
        recent article (e.g. 'DB stats: total=12345, latest_created_at=2025-03-28 14:30').
    """
    print("\n[Tool] get_db_stats")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(created_at), COUNT(*) FROM tech_news")
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            return f"DB stats: total={row[1]}, latest_created_at={row[0].strftime('%Y-%m-%d %H:%M')}"
        return "DB stats: no articles found."
    except Exception as e:
        print(f"[Error] get_db_stats failed: {e}")
        return f"get_db_stats failed: {e}"
    finally:
        put_conn(conn)


def list_topics() -> str:
    """Get daily article volume distribution for the most recent 21 days.

    Use this tool to understand data density and recency before choosing
    time windows for other queries.

    Returns:
        Daily date and article count, one line per day, sorted newest first.
    """
    print("\n[Tool] list_topics")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DATE(created_at) AS d, COUNT(*) AS c
            FROM tech_news
            WHERE created_at > NOW() - INTERVAL '21 days'
            GROUP BY DATE(created_at)
            ORDER BY d DESC
            """
        )
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return "No articles in the last 21 days."

        lines = ["Recent 21-day daily counts:"]
        for d, c in rows:
            lines.append(f"  {d.strftime('%Y-%m-%d')}: {c}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error] list_topics failed: {e}")
        return f"list_topics failed: {e}"
    finally:
        put_conn(conn)


def search_news(query: str, days: int = 21) -> str:
    """Search related news using hybrid retrieval (semantic vector + keyword matching).

    Best for exploratory, open-ended queries where you want the most relevant
    articles regardless of source or sorting. Returns up to 5 results ranked
    by composite relevance score.

    Args:
        query: Free-text search query (entity name, topic, keyword).
        days: Lookback window in days. Default 21, max 365.

    Returns:
        Formatted list of matching articles with Title, URL, Summary,
        Sentiment, Time, and relevance Score.
        Returns 'No related news...' if nothing matches.

    Retry guidance:
        - If no results, try broadening the query or increasing days.
        - If relevance scores are all < 0.5, results are weak matches.
    """
    print(f"\n[Tool] search_news: query={query}, days={days}")
    days = _clamp_int(days, 1, 365)
    limit = 5
    time_filter = f"{days} days"

    conn = get_conn()
    try:
        cur = conn.cursor()
        query_vec = _get_query_embedding(query)

        if query_vec:
            vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
            cur.execute(
                """
                WITH semantic AS (
                    SELECT
                        t.title, t.url, t.summary, t.sentiment, t.created_at,
                        1 - (e.embedding <=> %s::vector)
                            + 0.1 * EXP(-EXTRACT(EPOCH FROM (NOW() - t.created_at)) / 86400.0 / 21)
                            AS score
                    FROM tech_news t
                    JOIN news_embeddings e ON e.url = t.url
                    WHERE t.created_at > NOW() - %s::interval
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                ),
                keyword AS (
                    SELECT title, url, summary, sentiment, created_at, 1.0 AS score
                    FROM tech_news
                    WHERE (title ILIKE %s OR summary ILIKE %s)
                      AND created_at > NOW() - %s::interval
                    LIMIT %s
                ),
                combined AS (
                    SELECT * FROM semantic
                    UNION ALL
                    SELECT * FROM keyword
                )
                SELECT DISTINCT ON (url)
                    title, url, summary, sentiment, created_at, score
                FROM combined
                ORDER BY url, score DESC
                """,
                (vec_str, time_filter, vec_str, limit, f"%{query}%", f"%{query}%", time_filter, limit),
            )
            rows = cur.fetchall()
            rows.sort(key=lambda r: r[5], reverse=True)
            rows = rows[:limit]
        else:
            cur.execute(
                """
                SELECT title, url, summary, sentiment, created_at, 1.0 AS score
                FROM tech_news
                WHERE (title ILIKE %s OR summary ILIKE %s)
                  AND created_at > NOW() - %s::interval
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (f"%{query}%", f"%{query}%", time_filter, limit),
            )
            rows = cur.fetchall()

        cur.close()
        if not rows:
            return f"No related news for '{query}' in last {days} days."

        max_score = max(r[5] for r in rows)
        note = ""
        if max_score < 0.5:
            note = "[Note] Relevance is weak; these are nearest matches.\n\n"

        out = []
        for title, url, summary, sentiment, pub_time, score in rows:
            out.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Summary: {summary}\n"
                f"Sentiment: {sentiment}\n"
                f"Time: {pub_time.strftime('%Y-%m-%d %H:%M')}\n"
                f"Score: {score:.3f}"
            )
        return note + "\n---\n".join(out)
    except Exception as e:
        print(f"[Error] search_news failed: {e}")
        return f"search_news failed: {e}"
    finally:
        put_conn(conn)


def read_news_content(url: str) -> str:
    """Read the full-text content of a single article by its URL.

    Only works for URLs that exist in the database. Use URLs returned by
    search_news, query_news, or other tools.

    Args:
        url: The exact article URL as returned by other tools.

    Returns:
        Full article text, or an error message if the URL is not found.
    """
    print(f"\n[Tool] read_news_content: {url}")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM tech_news WHERE url = %s LIMIT 1", (url,))
        if not cur.fetchone():
            cur.close()
            return (
                f"[Error] URL '{url}' not found in DB. "
                "Use URLs returned by search tools."
            )

        cur.execute("SELECT raw_content FROM jina_raw_logs WHERE url = %s LIMIT 1", (url,))
        row = cur.fetchone()
        cur.close()

        if row and row[0]:
            return f"Full content:\n{row[0]}"
        return "URL exists but full content is not available yet."
    except Exception as e:
        print(f"[Error] read_news_content failed: {e}")
        return f"read_news_content failed: {e}"
    finally:
        put_conn(conn)


def query_news(
    query: str = "",
    source: str = "all",
    days: int = 21,
    category: str = "",
    sentiment: str = "",
    sort: str = "time_desc",
    limit: int = 8,
    response_format: str = "text",
) -> str:
    """Query news with structured filters — the primary retrieval tool.

    Use this tool for most news retrieval needs. It supports filtering by
    source, time window, sentiment, and sorting by time or heat (points).

    Args:
        query: Keyword or entity name to search (e.g. 'OpenAI', 'GPU').
        source: 'all', 'HackerNews', or 'TechCrunch'.
        days: Lookback window in days. Default 21, max 365.
        category: Optional category filter.
        sentiment: Optional: 'Positive', 'Neutral', or 'Negative'.
        sort: 'time_desc' (default), 'time_asc', 'heat_desc', 'heat_asc'.
        limit: Max results to return. Default 8, max 30.
        response_format: 'text' (default) or 'json'.

    Returns:
        Ranked list of articles with title, URL, summary, sentiment,
        points, source, and timestamp.

    Retry guidance:
        - If 'No matching records', try broadening the query keyword
          or increasing the days window.
        - For comprehensive coverage, try sort='heat_desc' to find
          the most discussed articles.
    """
    print(f"\n[Tool] query_news: query={query}, source={source}, days={days}, sort={sort}")
    days = _clamp_int(days, 1, 365)
    limit = _clamp_int(limit, 1, 30)
    source_label = _source_to_db_label(source)

    order_map = {
        "time_desc": "created_at DESC",
        "time_asc": "created_at ASC",
        "heat_desc": "points DESC NULLS LAST, created_at DESC",
        "heat_asc": "points ASC NULLS LAST, created_at DESC",
    }
    order_sql = order_map.get(sort.strip().lower(), order_map["time_desc"])

    where_parts = ["created_at > NOW() - %s::interval"]
    params: list[Any] = [f"{days} days"]

    if query.strip():
        where_parts.append("(title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)")
        q = f"%{query.strip()}%"
        params.extend([q, q, q])

    if source_label:
        where_parts.append("LOWER(COALESCE(source_type,'')) = LOWER(%s)")
        params.append(source_label)

    if category.strip():
        where_parts.append("(COALESCE(title_cn,'') ILIKE %s OR COALESCE(title_cn,'') ILIKE %s)")
        params.extend([f"%[{category.strip()}]%", f"%【{category.strip()}】%"])

    if sentiment.strip():
        where_parts.append("COALESCE(sentiment,'') ILIKE %s")
        params.append(sentiment.strip())

    sql = f"""
        SELECT title, title_cn, url, summary, sentiment, points, source_type, created_at
        FROM view_dashboard_news
        WHERE {' AND '.join(where_parts)}
        ORDER BY {order_sql}
        LIMIT %s
    """
    params.append(limit)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()
        records: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, 1):
            title, title_cn, url, summary, senti, points, src, created_at = row
            records.append(
                {
                    "rank": idx,
                    "source": src,
                    "title": title,
                    "title_cn": title_cn,
                    "url": url,
                    "summary": summary or "",
                    "sentiment": senti or "",
                    "points": int(points or 0),
                    "created_at": created_at.isoformat() if created_at else "",
                }
            )

        if response_format.strip().lower() == "json":
            payload = {
                "tool": "query_news",
                "status": "ok" if records else "empty",
                "request": {
                    "query": query,
                    "source": source,
                    "days": days,
                    "category": category,
                    "sentiment": sentiment,
                    "sort": sort,
                    "limit": limit,
                },
                "count": len(records),
                "records": records,
            }
            return _json_text(payload)

        if not rows:
            return "No matching records."

        lines = []
        for item in records:
            lines.append(
                f"{item['rank']}. [{item['source']}] {item.get('title_cn') or item.get('title')}\n"
                f"   time={item['created_at'].replace('T', ' ')[:16]}, sentiment={item['sentiment']}, points={item['points']}\n"
                f"   url={item['url']}\n"
                f"   summary={item['summary'][:220]}"
            )
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error] query_news failed: {e}")
        return f"query_news failed: {e}"
    finally:
        put_conn(conn)


def trend_analysis(topic: str, window: int = 7, response_format: str = "text") -> str:
    """Analyze topic momentum by comparing recent vs previous time window.

    Compares article count and average points in the recent N days vs the
    preceding N days. Includes daily breakdown.

    Args:
        topic: Entity or keyword to track (e.g. 'OpenAI', 'cybersecurity').
        window: Window size in days. Default 7, range 3-60.
        response_format: 'text' (default) or 'json'.

    Returns:
        Count delta, average-points comparison, and per-day breakdown.
    """
    print(f"\n[Tool] trend_analysis: topic={topic}, window={window}")
    as_json = response_format.strip().lower() == "json"

    if not topic or not topic.strip():
        if as_json:
            return _json_text(
                {
                    "tool": "trend_analysis",
                    "status": "error",
                    "request": {"topic": topic, "window": window},
                    "data": None,
                    "error": "trend_analysis requires topic.",
                }
            )
        return "trend_analysis requires topic."

    topic = topic.strip()
    window = _clamp_int(window, 3, 60)
    topic_clause, topic_params = _build_topic_where_clause(topic)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            WITH matched AS (
                SELECT created_at, points
                FROM tech_news
                WHERE created_at >= NOW() - (%s::int * INTERVAL '2 day')
                  AND {topic_clause}
            ),
            recent AS (
                SELECT COUNT(*) AS cnt, COALESCE(AVG(points), 0) AS avg_points
                FROM matched
                WHERE created_at >= NOW() - (%s::int * INTERVAL '1 day')
            ),
            prev AS (
                SELECT COUNT(*) AS cnt, COALESCE(AVG(points), 0) AS avg_points
                FROM matched
                WHERE created_at < NOW() - (%s::int * INTERVAL '1 day')
            )
            SELECT recent.cnt, recent.avg_points, prev.cnt, prev.avg_points
            FROM recent, prev
            """,
            tuple([window] + topic_params + [window, window]),
        )
        recent_cnt, recent_avg, prev_cnt, prev_avg = cur.fetchone()

        cur.execute(
            f"""
            SELECT DATE(created_at) AS day, COUNT(*) AS cnt, ROUND(AVG(points)::numeric, 1) AS avg_points
            FROM tech_news
            WHERE created_at >= NOW() - %s::interval
              AND {topic_clause}
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """,
            tuple([f"{window} days"] + topic_params),
        )
        daily_rows = cur.fetchall()
        cur.close()

        recent_cnt_i = int(recent_cnt or 0)
        prev_cnt_i = int(prev_cnt or 0)
        recent_avg_f = float(recent_avg or 0.0)
        prev_avg_f = float(prev_avg or 0.0)

        change_cnt = recent_cnt_i - prev_cnt_i
        change_pct = ((change_cnt / prev_cnt_i) * 100.0) if prev_cnt_i > 0 else None
        if prev_cnt_i > 0:
            delta = f"{change_cnt:+d} ({change_pct:+.1f}%)"
        else:
            delta = f"{change_cnt:+d} (no previous baseline)"

        daily_records = [
            {
                "day": day.strftime("%Y-%m-%d"),
                "count": int(cnt or 0),
                "avg_points": float(avg_points or 0.0),
            }
            for day, cnt, avg_points in daily_rows
        ]

        if as_json:
            return _json_text(
                {
                    "tool": "trend_analysis",
                    "status": "ok" if (recent_cnt_i > 0 or prev_cnt_i > 0 or daily_records) else "empty",
                    "request": {"topic": topic, "window": window},
                    "data": {
                        "topic": topic,
                        "window": window,
                        "recent_count": recent_cnt_i,
                        "previous_count": prev_cnt_i,
                        "count_delta": change_cnt,
                        "count_delta_pct": change_pct,
                        "avg_points_recent": recent_avg_f,
                        "avg_points_previous": prev_avg_f,
                        "daily": daily_records,
                    },
                }
            )

        lines = [
            f"Trend for topic: {topic}",
            f"Window: recent {window} days vs previous {window} days",
            f"Article count: {recent_cnt_i} vs {prev_cnt_i} -> {delta}",
            f"Avg points: {recent_avg_f:.1f} vs {prev_avg_f:.1f}",
            "Daily breakdown:",
        ]
        if daily_rows:
            for day, cnt, avg_points in daily_rows:
                lines.append(f"  {day.strftime('%Y-%m-%d')}: count={cnt}, avg_points={avg_points}")
        else:
            lines.append("  no matched records")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error] trend_analysis failed: {e}")
        if as_json:
            return _json_text(
                {
                    "tool": "trend_analysis",
                    "status": "error",
                    "request": {"topic": topic, "window": window},
                    "data": None,
                    "error": f"trend_analysis failed: {e}",
                }
            )
        return f"trend_analysis failed: {e}"
    finally:
        put_conn(conn)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _evidence_from_records(records: list[dict[str, Any]], max_items: int = 8) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for row in records:
        url = str(row.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = row.get("title_cn") or row.get("title")
        score_value = row.get("score")
        if score_value is None:
            score_value = row.get("points")
        evidence.append(
            {
                "url": url,
                "title": str(title).strip() if title else None,
                "source": str(row.get("source") or "").strip() or None,
                "created_at": str(row.get("created_at") or "").strip() or None,
                "score": _safe_float(score_value),
            }
        )
        if len(evidence) >= max_items:
            break
    return evidence


def query_news_skill(payload: QueryNewsSkillInput) -> SkillEnvelope:
    """Structured skill adapter for query_news with strong output contract."""

    request = payload.model_dump(mode="python")
    raw_output = query_news(
        query=request.get("query", ""),
        source=request.get("source", "all"),
        days=int(request.get("days", 21)),
        category=request.get("category", ""),
        sentiment=request.get("sentiment", ""),
        sort=request.get("sort", "time_desc"),
        limit=int(request.get("limit", 8)),
        response_format="json",
    )

    try:
        parsed = json.loads(raw_output)
    except Exception as exc:
        return build_error_envelope(
            tool="query_news",
            request=request,
            error="query_news_json_parse_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "raw_preview": str(raw_output)[:500],
            },
        )

    status = str(parsed.get("status", "error")).lower()
    if status not in {"ok", "empty", "error"}:
        return build_error_envelope(
            tool="query_news",
            request=request,
            error="query_news_invalid_status",
            diagnostics={"status": parsed.get("status"), "raw_output": parsed},
        )

    if status == "error":
        return build_error_envelope(
            tool="query_news",
            request=request,
            error=str(parsed.get("error") or "query_news_failed"),
            diagnostics={"raw_output": parsed},
        )

    records_raw = parsed.get("records")
    records = records_raw if isinstance(records_raw, list) else []
    evidence = _evidence_from_records(records, max_items=request.get("limit", 8))

    return SkillEnvelope(
        tool="query_news",
        status=status,
        request=request,
        data={
            "count": int(parsed.get("count", len(records))),
            "records": records,
        },
        evidence=evidence,
        diagnostics={
            "source": request.get("source", "all"),
            "sort": request.get("sort", "time_desc"),
        },
    )


def trend_analysis_skill(payload: TrendAnalysisSkillInput) -> SkillEnvelope:
    """Structured skill adapter for trend_analysis with evidence backfill."""

    request = payload.model_dump(mode="python")
    raw_output = trend_analysis(
        topic=request.get("topic", ""),
        window=int(request.get("window", 7)),
        response_format="json",
    )

    try:
        parsed = json.loads(raw_output)
    except Exception as exc:
        return build_error_envelope(
            tool="trend_analysis",
            request=request,
            error="trend_analysis_json_parse_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "raw_preview": str(raw_output)[:500],
            },
        )

    status = str(parsed.get("status", "error")).lower()
    if status not in {"ok", "empty", "error"}:
        return build_error_envelope(
            tool="trend_analysis",
            request=request,
            error="trend_analysis_invalid_status",
            diagnostics={"status": parsed.get("status"), "raw_output": parsed},
        )

    if status == "error":
        return build_error_envelope(
            tool="trend_analysis",
            request=request,
            error=str(parsed.get("error") or "trend_analysis_failed"),
            diagnostics={"raw_output": parsed},
        )

    data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}

    # Trend stats alone have no URLs; backfill top topic evidence to support
    # downstream citation checks and keep analyst output auditable.
    evidence_records: list[dict[str, Any]] = []
    if status == "ok" and str(request.get("topic", "")).strip():
        evidence_limit = min(6, max(3, int(request.get("window", 7))))
        query_raw = query_news(
            query=str(request["topic"]),
            days=max(3, int(request.get("window", 7)) * 2),
            limit=evidence_limit,
            sort="heat_desc",
            response_format="json",
        )
        try:
            query_parsed = json.loads(query_raw)
            if isinstance(query_parsed, dict):
                records = query_parsed.get("records")
                if isinstance(records, list):
                    evidence_records = records
        except Exception:
            evidence_records = []

    evidence = _evidence_from_records(evidence_records, max_items=6)

    return SkillEnvelope(
        tool="trend_analysis",
        status=status,
        request=request,
        data=data,
        evidence=evidence,
        diagnostics={
            "evidence_backfill_count": len(evidence),
            "window": request.get("window"),
        },
    )


def _evidence_from_text_output(text: str, max_items: int = 8) -> list[dict[str, Any]]:
    """Extract evidence entries from structured text output containing URLs.

    Parses lines with URL patterns and optional metadata (title, source, points).
    Used by text-based skill adapters that don't have a JSON mode.
    """
    if not text:
        return []

    url_pattern = re.compile(r"https?://[^\s)\]]+")
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for line in text.splitlines():
        urls_in_line = url_pattern.findall(line)
        for url in urls_in_line:
            url = url.rstrip(".,;:!?")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Try to extract title from structured line patterns
            title = None
            # Pattern: "headline | points=N | time | URL"
            parts = line.split("|")
            if len(parts) >= 2:
                # First part after rank/source markers is usually the headline
                candidate = parts[0].strip()
                # Strip common prefixes like "  [Entity] #N [Source] "
                candidate = re.sub(
                    r"^\s*(?:\d+\.\s*)?(?:\[.*?\]\s*)*(?:#\d+\s*)?(?:\[.*?\]\s*)*",
                    "",
                    candidate,
                ).strip()
                if candidate and len(candidate) > 3:
                    title = candidate

            # Try to extract source from line
            source = None
            source_match = re.search(r"\[(HackerNews|TechCrunch)\]", line)
            if source_match:
                source = source_match.group(1)

            evidence.append({
                "url": url,
                "title": title,
                "source": source,
                "created_at": None,
                "score": None,
            })
            if len(evidence) >= max_items:
                return evidence
    return evidence


def search_news_skill(payload: SearchNewsSkillInput) -> SkillEnvelope:
    """Structured skill adapter for search_news."""

    request = payload.model_dump(mode="python")
    try:
        raw_output = search_news(
            query=request["query"],
            days=int(request.get("days", 21)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="search_news",
            request=request,
            error="search_news_execution_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    if raw_output.startswith("No related news") or raw_output.startswith("search_news failed"):
        is_error = raw_output.startswith("search_news failed")
        return SkillEnvelope(
            tool="search_news",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"query": request["query"]},
        )

    evidence = _evidence_from_text_output(raw_output, max_items=5)
    return SkillEnvelope(
        tool="search_news",
        status="ok",
        request=request,
        data={"raw_output": raw_output, "result_count": len(evidence)},
        evidence=evidence,
        diagnostics={"query": request["query"]},
    )


def compare_sources_skill(payload: CompareSourcesSkillInput) -> SkillEnvelope:
    """Structured skill adapter for compare_sources."""

    request = payload.model_dump(mode="python")
    try:
        raw_output = compare_sources(
            topic=request["topic"],
            days=int(request.get("days", 14)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="compare_sources",
            request=request,
            error="compare_sources_execution_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    if raw_output.startswith("No comparison data") or raw_output.startswith("compare_sources"):
        is_error = "failed" in raw_output
        return SkillEnvelope(
            tool="compare_sources",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"topic": request["topic"]},
        )

    evidence = _evidence_from_text_output(raw_output, max_items=6)
    return SkillEnvelope(
        tool="compare_sources",
        status="ok",
        request=request,
        data={"raw_output": raw_output},
        evidence=evidence,
        diagnostics={"topic": request["topic"]},
    )


def compare_topics_skill(payload: CompareTopicsSkillInput) -> SkillEnvelope:
    """Structured skill adapter for compare_topics."""

    request = payload.model_dump(mode="python")
    try:
        raw_output = compare_topics(
            topic_a=request["topic_a"],
            topic_b=request["topic_b"],
            days=int(request.get("days", 14)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="compare_topics",
            request=request,
            error="compare_topics_execution_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    if raw_output.startswith("compare_topics requires") or raw_output.startswith("compare_topics failed"):
        return SkillEnvelope(
            tool="compare_topics",
            status="error",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output,
            diagnostics={},
        )

    # Extract confidence from output
    confidence = None
    for line in raw_output.splitlines():
        if line.strip().startswith("Confidence:"):
            confidence = line.strip().split(":", 1)[1].strip()

    evidence = _evidence_from_text_output(raw_output, max_items=6)
    status = "ok" if evidence else "empty"
    return SkillEnvelope(
        tool="compare_topics",
        status=status,
        request=request,
        data={"raw_output": raw_output, "confidence": confidence},
        evidence=evidence,
        diagnostics={
            "topic_a": request["topic_a"],
            "topic_b": request["topic_b"],
            "confidence": confidence,
        },
    )


def build_timeline_skill(payload: BuildTimelineSkillInput) -> SkillEnvelope:
    """Structured skill adapter for build_timeline."""

    request = payload.model_dump(mode="python")
    try:
        raw_output = build_timeline(
            topic=request["topic"],
            days=int(request.get("days", 30)),
            limit=int(request.get("limit", 12)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="build_timeline",
            request=request,
            error="build_timeline_execution_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    if raw_output.startswith("No timeline data") or raw_output.startswith("build_timeline"):
        is_error = "failed" in raw_output or "requires" in raw_output
        return SkillEnvelope(
            tool="build_timeline",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"topic": request["topic"]},
        )

    evidence = _evidence_from_text_output(raw_output, max_items=12)
    return SkillEnvelope(
        tool="build_timeline",
        status="ok",
        request=request,
        data={"raw_output": raw_output, "event_count": len(evidence)},
        evidence=evidence,
        diagnostics={"topic": request["topic"]},
    )


def analyze_landscape_skill(payload: AnalyzeLandscapeSkillInput) -> SkillEnvelope:
    """Structured skill adapter for analyze_landscape."""

    request = payload.model_dump(mode="python")
    try:
        raw_output = analyze_landscape(
            topic=request.get("topic", ""),
            days=int(request.get("days", 30)),
            entities=request.get("entities", ""),
            limit_per_entity=int(request.get("limit_per_entity", 3)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="analyze_landscape",
            request=request,
            error="analyze_landscape_execution_failed",
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    if raw_output.startswith("No landscape data") or raw_output.startswith("analyze_landscape failed"):
        is_error = "failed" in raw_output
        return SkillEnvelope(
            tool="analyze_landscape",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"topic": request.get("topic", "")},
        )

    # Extract confidence from output
    confidence = None
    for line in raw_output.splitlines():
        if line.strip().startswith("Confidence:"):
            confidence = line.strip().split(":", 1)[1].strip()

    evidence = _evidence_from_text_output(raw_output, max_items=12)
    return SkillEnvelope(
        tool="analyze_landscape",
        status="ok",
        request=request,
        data={"raw_output": raw_output, "confidence": confidence},
        evidence=evidence,
        diagnostics={
            "topic": request.get("topic", ""),
            "confidence": confidence,
        },
    )


def fulltext_batch_skill(payload: FulltextBatchSkillInput) -> SkillEnvelope:
    """Structured skill adapter for fulltext_batch."""

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
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
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
        return SkillEnvelope(
            tool="fulltext_batch",
            status="empty",
            request=request,
            data=parsed,
            evidence=[],
            error=parsed.get("error"),
            diagnostics={},
        )

    selected = parsed.get("selected", [])
    articles = parsed.get("articles", [])
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in selected:
        url = str(item.get("url") or "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            evidence.append({
                "url": url,
                "title": item.get("headline"),
                "source": item.get("source_type"),
                "created_at": item.get("created_at"),
                "score": _safe_float(item.get("score")),
            })

    return SkillEnvelope(
        tool="fulltext_batch",
        status="ok",
        request=request,
        data={
            "article_count": len(articles),
            "articles": articles,
        },
        evidence=evidence,
        diagnostics={
            "selection_count": len(selected),
            "article_count": len(articles),
        },
    )


def compare_sources(topic: str, days: int = 14) -> str:
    """Compare HackerNews vs TechCrunch coverage and sentiment for a topic.

    Use this when the user wants to understand how different media sources
    cover the same topic differently.

    Args:
        topic: Entity or keyword to compare across sources.
        days: Lookback window. Default 14, max 90.

    Returns:
        Per-source stats (count, avg_points, sentiment breakdown) and
        top 3 evidence articles per source with URLs.
    """
    print(f"\n[Tool] compare_sources: topic={topic}, days={days}")
    if not topic or not topic.strip():
        return "compare_sources requires topic."

    days = _clamp_int(days, 1, 90)
    topic_clause, topic_params = _build_topic_where_clause(topic)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                source_type,
                COUNT(*) AS cnt,
                ROUND(AVG(points)::numeric, 1) AS avg_points,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neu_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
              AND {topic_clause}
            GROUP BY source_type
            ORDER BY source_type
            """,
            tuple([f"{days} days"] + topic_params),
        )
        stats_rows = cur.fetchall()

        cur.execute(
            f"""
            WITH ranked AS (
                SELECT
                    source_type,
                    COALESCE(title_cn, title) AS headline,
                    url,
                    points,
                    created_at,
                    ROW_NUMBER() OVER (PARTITION BY source_type ORDER BY points DESC NULLS LAST, created_at DESC) AS rn
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND {topic_clause}
            )
            SELECT source_type, headline, url, points, created_at, rn
            FROM ranked
            WHERE rn <= 3
            ORDER BY source_type, rn
            """,
            tuple([f"{days} days"] + topic_params),
        )
        top_rows = cur.fetchall()
        cur.close()

        if not stats_rows:
            return f"No comparison data for '{topic}' in {days} days."

        lines = [f"Source comparison: {topic} (last {days} days)", "Stats:"]
        for src, cnt, avg_points, pos, neu, neg in stats_rows:
            lines.append(f"  {src}: count={cnt}, avg_points={avg_points}, sentiment(P/N/Ng)={pos}/{neu}/{neg}")

        lines.append("Top evidence:")
        for src, headline, url, points, created_at, rn in top_rows:
            lines.append(
                f"  [{src}] #{rn} {headline} | points={points} | {created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error] compare_sources failed: {e}")
        return f"compare_sources failed: {e}"
    finally:
        put_conn(conn)


def compare_topics(topic_a: str, topic_b: str, days: int = 14) -> str:
    """Compare two entities or topics side-by-side with DB evidence.

    Use this for A-vs-B comparisons (e.g. 'OpenAI vs Anthropic',
    'GPU vs TPU'). Provides metrics, momentum, source mix, and evidence.

    Args:
        topic_a: First entity/topic.
        topic_b: Second entity/topic.
        days: Lookback window. Default 14, max 90.

    Returns:
        Side-by-side stats, momentum delta, source mix breakdown,
        top evidence URLs, and a confidence tag.
    """
    print(f"\n[Tool] compare_topics: A={topic_a}, B={topic_b}, days={days}")
    if not topic_a or not topic_a.strip() or not topic_b or not topic_b.strip():
        return "compare_topics requires topic_a and topic_b."

    topic_a = topic_a.strip()
    topic_b = topic_b.strip()
    days = _clamp_int(days, 1, 90)
    split_days = max(3, days // 2)
    prev_days = max(1, days - split_days)
    qa = f"%{topic_a}%"
    qb = f"%{topic_b}%"

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    points,
                    sentiment
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT
                grp,
                COUNT(*) AS cnt,
                ROUND(AVG(points)::numeric, 1) AS avg_points,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neu_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp
            ORDER BY grp
            """,
            (qa, qa, qa, qb, qb, qb, f"{days} days"),
        )
        metric_rows = cur.fetchall()

        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    source_type
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT grp, source_type, COUNT(*) AS cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp, source_type
            ORDER BY grp, cnt DESC, source_type ASC
            """,
            (qa, qa, qa, qb, qb, qb, f"{days} days"),
        )
        source_rows = cur.fetchall()

        cur.execute(
            """
            WITH labeled AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    created_at
                FROM tech_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT
                grp,
                COUNT(*) FILTER (WHERE created_at >= NOW() - %s::interval) AS recent_cnt,
                COUNT(*) FILTER (WHERE created_at < NOW() - %s::interval) AS prev_cnt
            FROM labeled
            WHERE grp IS NOT NULL
            GROUP BY grp
            ORDER BY grp
            """,
            (qa, qa, qa, qb, qb, qb, f"{days} days", f"{split_days} days", f"{split_days} days"),
        )
        momentum_rows = cur.fetchall()

        cur.execute(
            """
            WITH candidates AS (
                SELECT
                    CASE
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                        WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                        ELSE NULL
                    END AS grp,
                    source_type,
                    COALESCE(title_cn, title) AS headline,
                    url,
                    points,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            CASE
                                WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'A'
                                WHEN (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s) THEN 'B'
                                ELSE NULL
                            END
                        ORDER BY points DESC NULLS LAST, created_at DESC
                    ) AS rn
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
            )
            SELECT grp, source_type, headline, url, points, created_at, rn
            FROM candidates
            WHERE grp IS NOT NULL AND rn <= 3
            ORDER BY grp, rn
            """,
            (
                qa, qa, qa, qb, qb, qb,
                qa, qa, qa, qb, qb, qb,
                f"{days} days",
            ),
        )
        top_rows = cur.fetchall()
        cur.close()

        metric_map: dict[str, tuple | None] = {"A": None, "B": None}
        for row in metric_rows:
            metric_map[row[0]] = row

        source_map: dict[str, dict[str, int]] = {"A": {}, "B": {}}
        for grp, source_type, cnt in source_rows:
            bucket = source_map.setdefault(grp, {})
            bucket[source_type] = int(cnt)

        momentum_map: dict[str, tuple[int, int]] = {"A": (0, 0), "B": (0, 0)}
        for grp, recent_cnt, prev_cnt in momentum_rows:
            momentum_map[grp] = (int(recent_cnt or 0), int(prev_cnt or 0))

        def fmt(name: str, row: tuple | None) -> str:
            if not row:
                return f"{name}: count=0, avg_points=0, sentiment(P/N/Ng)=0/0/0"
            _, cnt, avg_points, pos, neu, neg = row
            return f"{name}: count={cnt}, avg_points={avg_points}, sentiment(P/N/Ng)={pos}/{neu}/{neg}"

        lines = [f"Topic comparison: {topic_a} vs {topic_b} (last {days} days)", "Stats:"]
        lines.append("  " + fmt(topic_a, metric_map["A"]))
        lines.append("  " + fmt(topic_b, metric_map["B"]))
        lines.append(f"Time split: recent={split_days}d, previous={prev_days}d")
        lines.append("Momentum:")
        for grp, label in (("A", topic_a), ("B", topic_b)):
            recent_cnt, prev_cnt = momentum_map.get(grp, (0, 0))
            if prev_cnt == 0:
                delta_text = "+new" if recent_cnt > 0 else "0.0%"
            else:
                delta_text = f"{((float(recent_cnt) - float(prev_cnt)) / float(prev_cnt)) * 100:+.1f}%"
            lines.append(
                f"  {label}: recent={recent_cnt}, previous={prev_cnt}, delta={delta_text}"
            )
        lines.append("Source mix:")
        for grp, label in (("A", topic_a), ("B", topic_b)):
            source_bucket = source_map.get(grp, {})
            total_cnt = metric_map[grp][1] if metric_map.get(grp) else 0
            if not source_bucket:
                lines.append(f"  {label}: no source records")
                continue
            mix_parts: list[str] = []
            for source_type, source_cnt in sorted(source_bucket.items(), key=lambda x: (-x[1], x[0])):
                source_share = (float(source_cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
                mix_parts.append(f"{source_type}={source_cnt}({source_share:.1f}%)")
            lines.append(f"  {label}: " + ", ".join(mix_parts))
        lines.append("Evidence URLs:")
        for grp, source_type, headline, url, points, created_at, rn in top_rows:
            label = topic_a if grp == "A" else topic_b
            lines.append(
                f"  [{label}] #{rn} [{source_type}] {headline} | points={points} | "
                f"{created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )
        count_a = metric_map["A"][1] if metric_map["A"] else 0
        count_b = metric_map["B"][1] if metric_map["B"] else 0
        url_count = len(top_rows)
        if count_a > 0 and count_b > 0 and url_count >= 2:
            confidence = "High"
        elif (count_a > 0 or count_b > 0) and url_count >= 1:
            confidence = "Medium"
        else:
            confidence = "Low"
        lines.append(f"Confidence: {confidence}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error] compare_topics failed: {e}")
        return f"compare_topics failed: {e}"
    finally:
        if conn is not None:
            put_conn(conn)


def analyze_landscape(topic: str = "", days: int = 30, entities: str = "", limit_per_entity: int = 3) -> str:
    """Analyze the competitive landscape for a domain with entity-level stats.

    Use this for broad, structural questions like 'What does the AI landscape
    look like?', 'Who are the key players in cybersecurity?', or company-role
    analysis. Provides per-entity article count, sentiment, momentum, source
    mix, signal classification, and evidence URLs.

    Args:
        topic: Domain filter (e.g. 'AI', 'security', 'business').
               Leave empty for all-domain landscape.
        days: Lookback window. Default 30, range 7-180.
        entities: Comma-separated entity names to track.
                  Leave empty for default set (OpenAI, Anthropic, Google,
                  Microsoft, Meta, Amazon, Apple, NVIDIA, etc.).
        limit_per_entity: Max evidence URLs per entity. Default 3, max 5.

    Returns:
        Structured landscape report with coverage stats, entity metrics,
        variable signal counts (Compute/Cost, Algorithm/Efficiency, etc.),
        evidence URLs, and a confidence tag.

    Retry guidance:
        - If 'No landscape data', try increasing days or removing the
          topic filter.
        - If confidence is 'Low', try adding more entities or widening
          the time window.
        - For AI-specific landscape, use topic='AI'.
    """
    topic = (topic or "").strip()
    topic_label = topic or "all"
    print(
        f"\n[Tool] analyze_landscape: topic={topic_label}, days={days}, "
        f"entities={entities or 'default'}, limit_per_entity={limit_per_entity}"
    )
    days = _clamp_int(days, 7, 180)
    limit_per_entity = _clamp_int(limit_per_entity, 1, 5)
    entity_list = _normalize_landscape_entities(entities)

    values_sql = ", ".join(["(%s, %s)"] * len(entity_list))
    params_entities: list[Any] = []
    for name in entity_list:
        params_entities.extend([name, f"%{name}%"])

    topic_where_sql = ""
    topic_params: list[Any] = []
    if topic:
        topic_clause_sql, topic_params = _build_topic_where_clause(topic, table_alias="v")
        topic_where_sql = f"AND {topic_clause_sql}"

    cte = f"""
        WITH entities(canonical, pattern) AS (
            VALUES {values_sql}
        ),
        matched AS (
            SELECT
                e.canonical AS entity,
                v.source_type,
                COALESCE(v.title_cn, v.title) AS headline,
                COALESCE(v.summary, '') AS summary,
                v.url,
                v.points,
                v.sentiment,
                v.created_at
            FROM view_dashboard_news v
            JOIN entities e
              ON (
                  v.title ILIKE e.pattern
                  OR COALESCE(v.title_cn, '') ILIKE e.pattern
                  OR COALESCE(v.summary, '') ILIKE e.pattern
              )
            WHERE v.created_at >= NOW() - %s::interval
              {topic_where_sql}
        )
    """

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT NOW()")
        db_now = _to_utc_naive_datetime(cur.fetchone()[0])
        if db_now is None:
            db_now = datetime.now(timezone.utc).replace(tzinfo=None)

        cur.execute(
            cte
            + """
            SELECT
                entity,
                COUNT(*) AS cnt,
                ROUND(AVG(points)::numeric, 1) AS avg_points,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neu_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM matched
            GROUP BY entity
            ORDER BY cnt DESC, entity ASC
            """,
            tuple(params_entities + [f"{days} days"] + topic_params),
        )
        stats_rows = cur.fetchall()

        cur.execute(
            cte
            + """
            SELECT COUNT(*) AS total_cnt, COUNT(DISTINCT entity) AS active_entities
            FROM matched
            """,
            tuple(params_entities + [f"{days} days"] + topic_params),
        )
        total_cnt, active_entities = cur.fetchone()

        topic_scope_sql = """
            SELECT COUNT(*) AS topic_articles
            FROM view_dashboard_news v
            WHERE v.created_at >= NOW() - %s::interval
        """
        topic_scope_params: list[Any] = [f"{days} days"]
        if topic:
            topic_clause_sql, topic_scope_topic_params = _build_topic_where_clause(topic, table_alias="v")
            topic_scope_sql += f" AND {topic_clause_sql}"
            topic_scope_params.extend(topic_scope_topic_params)
        cur.execute(topic_scope_sql, tuple(topic_scope_params))
        topic_articles = int(cur.fetchone()[0] or 0)

        cur.execute(
            cte
            + """
            , ranked AS (
                SELECT
                    entity,
                    source_type,
                    headline,
                    url,
                    points,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY entity
                        ORDER BY points DESC NULLS LAST, created_at DESC
                    ) AS rn
                FROM matched
            )
            SELECT entity, source_type, headline, url, points, created_at, rn
            FROM ranked
            WHERE rn <= %s
            ORDER BY entity, rn
            """,
            tuple(params_entities + [f"{days} days"] + topic_params + [limit_per_entity]),
        )
        top_rows = cur.fetchall()

        cur.execute(
            cte
            + """
            SELECT entity, source_type, headline, summary, created_at
            FROM matched
            """,
            tuple(params_entities + [f"{days} days"] + topic_params),
        )
        sample_rows = cur.fetchall()
        cur.close()

        if not total_cnt:
            if topic_articles > 0:
                return (
                    f"Topic '{topic_label}' has {topic_articles} articles in the last {days} days, "
                    f"but no tracked entities matched. Try passing explicit entities."
                )
            return (
                f"No landscape data in the last {days} days for entities: "
                f"{', '.join(entity_list)}."
            )

        stat_map: dict[str, tuple[int, Any, int, int, int]] = {}
        for entity, cnt, avg_points, pos_cnt, neu_cnt, neg_cnt in stats_rows:
            stat_map[entity] = (cnt, avg_points, pos_cnt, neu_cnt, neg_cnt)

        split_days = max(3, days // 2)
        prev_days = max(1, days - split_days)
        cutoff_ts = db_now - timedelta(days=split_days)

        source_counts: dict[str, int] = {}
        entity_source_counts: dict[str, dict[str, int]] = {}
        momentum_map: dict[str, dict[str, int]] = {name: {"recent": 0, "previous": 0} for name in entity_list}
        for entity, source_type, headline, summary, created_at in sample_rows:
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            per_entity = entity_source_counts.setdefault(entity, {})
            per_entity[source_type] = per_entity.get(source_type, 0) + 1
            if entity not in momentum_map:
                momentum_map[entity] = {"recent": 0, "previous": 0}
            if _is_recent_timestamp(created_at, cutoff_ts):
                momentum_map[entity]["recent"] += 1
            else:
                momentum_map[entity]["previous"] += 1

        signal_counts = _landscape_signal_counts(sample_rows)

        lines = [
            f"Landscape snapshot: topic={topic_label} (last {days} days)",
            f"Entities requested: {', '.join(entity_list)}",
            (
                f"Coverage: topic_articles={topic_articles}, matched_entity_articles={total_cnt}, "
                f"active_entities={active_entities}/{len(entity_list)}"
            ),
            f"Time split: recent={split_days}d, previous={prev_days}d",
            "Source mix:",
        ]

        if source_counts:
            for source_type, source_cnt in sorted(source_counts.items(), key=lambda x: (-x[1], x[0])):
                source_share = (float(source_cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
                lines.append(f"  {source_type}: count={source_cnt}, share={source_share:.1f}%")
        else:
            lines.append("  no source records")

        lines.extend(
            [
            "Entity stats:",
            ]
        )

        for name in entity_list:
            recent = momentum_map.get(name, {}).get("recent", 0)
            previous = momentum_map.get(name, {}).get("previous", 0)
            if previous == 0:
                delta_text = "+new" if recent > 0 else "0.0%"
            else:
                delta_text = f"{((float(recent) - float(previous)) / float(previous)) * 100:+.1f}%"

            if name not in stat_map:
                lines.append(
                    f"  {name}: count=0, share=0.0%, avg_points=0, "
                    f"sentiment(P/N/Ng)=0/0/0, momentum_recent_vs_prev={recent}/{previous} ({delta_text})"
                )
                continue
            cnt, avg_points, pos_cnt, neu_cnt, neg_cnt = stat_map[name]
            share = (float(cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
            avg_points_value = float(avg_points) if avg_points is not None else 0.0
            top_source_note = ""
            per_entity_source = entity_source_counts.get(name, {})
            if per_entity_source:
                top_source, top_source_cnt = max(per_entity_source.items(), key=lambda x: (x[1], x[0]))
                top_source_share = (float(top_source_cnt) / float(cnt)) * 100 if cnt else 0.0
                top_source_note = f", top_source={top_source}({top_source_share:.1f}%)"
            lines.append(
                f"  {name}: count={cnt}, share={share:.1f}%, avg_points={avg_points_value:.1f}, "
                f"sentiment(P/N/Ng)={pos_cnt}/{neu_cnt}/{neg_cnt}, "
                f"momentum_recent_vs_prev={recent}/{previous} ({delta_text}){top_source_note}"
            )

        lines.append("Variable signals (keyword proxy, headline+summary):")
        for key, label in LANDSCAPE_SIGNAL_LABELS.items():
            signal_cnt = int(signal_counts.get(key, 0))
            signal_share = (float(signal_cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
            lines.append(f"  {label}: count={signal_cnt}, share={signal_share:.1f}%")
        lines.append("Signal note: keyword proxy for triage; verify with evidence URLs.")

        lines.append("Evidence URLs:")
        for entity, source_type, headline, url, points, created_at, rn in top_rows:
            lines.append(
                f"  [{entity}] #{rn} [{source_type}] {headline} | points={points} | "
                f"{created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )

        url_count = len(top_rows)
        coverage_ratio = (float(total_cnt) / float(topic_articles)) if topic_articles > 0 else 1.0
        if (
            active_entities >= min(4, len(entity_list))
            and total_cnt >= 15
            and url_count >= 8
            and coverage_ratio >= 0.4
        ):
            confidence = "High"
        elif active_entities >= 2 and total_cnt >= 4 and url_count >= 2 and coverage_ratio >= 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"
        lines.append(f"Confidence: {confidence}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error] analyze_landscape failed: {e}")
        return f"analyze_landscape failed: {e}"
    finally:
        if conn is not None:
            put_conn(conn)


def analyze_ai_landscape(days: int = 30, entities: str = "", limit_per_entity: int = 3) -> str:
    """Analyze the AI industry landscape. Alias for analyze_landscape(topic='AI').

    Deprecated: prefer using analyze_landscape(topic='AI') directly.
    """
    return analyze_landscape(topic="AI", days=days, entities=entities, limit_per_entity=limit_per_entity)


def build_timeline(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build a chronological event timeline for a topic, company, or product.

    Returns events sorted by time (oldest first) with timestamps, sources,
    sentiment, engagement points, headlines, and URLs.

    Args:
        topic: Entity or keyword to track (e.g. 'OpenAI', 'GPU shortage').
        days: Lookback window in days. Default 30, max 180.
        limit: Max events to return. Default 12, max 40.

    Returns:
        Ranked timeline of events with metadata and URLs.
        Returns 'No timeline data...' if no events found.

    Retry guidance:
        - If no data returned, the tool will automatically retry with a
          wider window (up to 2x). If still empty, try broadening the topic.
        - If fewer than 5 events, consider widening the time window.
    """
    print(f"\n[Tool] build_timeline: topic={topic}, days={days}, limit={limit}")
    if not topic or not topic.strip():
        return "build_timeline requires topic."

    days = _clamp_int(days, 1, 180)
    limit = _clamp_int(limit, 3, 40)
    topic_clause, topic_params = _build_topic_where_clause(topic)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                created_at,
                source_type,
                COALESCE(title_cn, title) AS headline,
                sentiment,
                points,
                url
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
              AND {topic_clause}
            ORDER BY created_at ASC
            LIMIT %s
            """,
            tuple([f"{days} days"] + topic_params + [limit]),
        )
        rows = cur.fetchall()
        cur.close()

        # Auto-retry with wider window if empty and room to expand
        if not rows and days < 90:
            retry_days = min(180, max(60, days * 2))
            print(f"[Tool] build_timeline: empty for {days}d, auto-retrying with {retry_days}d")
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT
                    created_at, source_type,
                    COALESCE(title_cn, title) AS headline,
                    sentiment, points, url
                FROM view_dashboard_news
                WHERE created_at >= NOW() - %s::interval
                  AND {topic_clause}
                ORDER BY created_at ASC
                LIMIT %s
                """,
                tuple([f"{retry_days} days"] + topic_params + [limit]),
            )
            rows = cur.fetchall()
            cur.close()
            if rows:
                days = retry_days

        if not rows:
            return f"No timeline data for '{topic}' in {days} days."

        lines = [f"Timeline: {topic} (last {days} days, max {limit})"]
        for idx, (created_at, src, headline, senti, points, url) in enumerate(rows, 1):
            lines.append(
                f"{idx}. {created_at.strftime('%Y-%m-%d %H:%M')} | {src} | {senti} | points={points}\n"
                f"   {headline}\n"
                f"   {url}"
            )
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error] build_timeline failed: {e}")
        return f"build_timeline failed: {e}"
    finally:
        put_conn(conn)


def fulltext_batch(urls: str, max_chars_per_article: int = 4000, response_format: str = "text") -> str:
    """Batch-read full article text by URL list or keyword auto-selection.

    Can accept either direct URLs or a keyword query. When given keywords
    instead of URLs, automatically selects the most relevant articles.

    Args:
        urls: Comma/newline separated URLs, or a keyword query string
              for auto-selection.
        max_chars_per_article: Max characters per article. Default 4000,
                               range 800-12000.
        response_format: 'text' (default) or 'json'.

    Returns:
        Full text content of each article. When auto-selecting, includes
        a ranked candidate list before the full text.
    """
    print("\n[Tool] fulltext_batch")
    max_chars_per_article = _clamp_int(max_chars_per_article, 800, 12000)
    as_json = response_format.strip().lower() == "json"

    raw_items = _split_urls(urls)
    direct_urls = [x for x in raw_items if _is_probable_url(x)]

    selected: list[tuple[str, str, dict[str, Any]]] = []
    prefix_lines: list[str] = []
    selected_meta: list[dict[str, Any]] = []

    if direct_urls:
        for u in direct_urls[:12]:
            meta = {"selection_mode": "direct", "url": u}
            selected.append(("direct", u, meta))
            selected_meta.append(meta)
    else:
        query = (urls or "").strip()
        if not query:
            return "fulltext_batch requires URLs or a keyword query."

        days = _extract_time_window_days(query, default=14, maximum=120)
        candidates = _lookup_urls_by_query(query=query, days=days, limit=6)
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
                        },
                        "selected": [],
                        "articles": [],
                        "error": f"No candidate articles found for query '{query}'.",
                    }
                )
            return f"No candidate articles found for query '{query}'."

        prefix_lines.append(
            f"No URLs provided. Auto-selected Top {len(candidates)} articles for query '{query}' (window={days}d):"
        )
        for i, row in enumerate(candidates, 1):
            headline, url, source_type, created_at, points, score = row
            prefix_lines.append(
                f"{i}. [{source_type}] {headline} | points={points} | "
                f"score={float(score):.3f} | {created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )
            meta = {
                "selection_mode": "query",
                "query": query,
                "window_days": days,
                "rank": i,
                "headline": headline,
                "source_type": source_type,
                "points": int(points or 0),
                "score": float(score or 0.0),
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
                },
                "selected": selected_meta,
                "articles": article_rows,
            }
        )

    if prefix_lines:
        return "\n".join(prefix_lines) + "\n\n" + "\n\n".join(chunks)
    return "\n\n".join(chunks)
