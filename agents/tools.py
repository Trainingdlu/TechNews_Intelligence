"""Database and retrieval tools for the agent runtime."""

from __future__ import annotations

import json
import os
import re
from datetime import timedelta
from typing import Any

import requests

try:
    from db import get_conn, put_conn
except ImportError:  # package-style import fallback
    from .db import get_conn, put_conn


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
    m = re.search(r"(?:最近|过去|last|recent|past)?\s*(\d{1,3})\s*(?:天|day|days)", text, flags=re.IGNORECASE)
    if not m:
        return _clamp_int(default, 1, maximum)
    return _clamp_int(int(m.group(1)), 1, maximum)


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


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


def get_db_stats() -> str:
    """Get DB freshness and volume stats."""
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
    """Get article counts for recent 21 days."""
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
    """Hybrid search (vector + keyword) for news retrieval."""
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
    """Read full article content by URL from DB raw logs."""
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
    """Filterable retrieval: source/days/category/sentiment/sort/limit."""
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


def trend_analysis(topic: str, window: int = 7) -> str:
    """Analyze momentum in recent window vs previous window."""
    print(f"\n[Tool] trend_analysis: topic={topic}, window={window}")
    if not topic or not topic.strip():
        return "trend_analysis requires topic."

    topic = topic.strip()
    window = _clamp_int(window, 3, 60)
    q = f"%{topic}%"

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            WITH matched AS (
                SELECT created_at, points
                FROM tech_news
                WHERE created_at >= NOW() - (%s::int * INTERVAL '2 day')
                  AND (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)
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
            (window, q, q, q, window, window),
        )
        recent_cnt, recent_avg, prev_cnt, prev_avg = cur.fetchone()

        cur.execute(
            """
            SELECT DATE(created_at) AS day, COUNT(*) AS cnt, ROUND(AVG(points)::numeric, 1) AS avg_points
            FROM tech_news
            WHERE created_at >= NOW() - %s::interval
              AND (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """,
            (f"{window} days", q, q, q),
        )
        daily_rows = cur.fetchall()
        cur.close()

        change_cnt = recent_cnt - prev_cnt
        if prev_cnt > 0:
            pct = (change_cnt / prev_cnt) * 100
            delta = f"{change_cnt:+d} ({pct:+.1f}%)"
        else:
            delta = f"{change_cnt:+d} (no previous baseline)"

        lines = [
            f"Trend for topic: {topic}",
            f"Window: recent {window} days vs previous {window} days",
            f"Article count: {recent_cnt} vs {prev_cnt} -> {delta}",
            f"Avg points: {recent_avg:.1f} vs {prev_avg:.1f}",
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
        return f"trend_analysis failed: {e}"
    finally:
        put_conn(conn)


def compare_sources(topic: str, days: int = 14) -> str:
    """Compare HN vs TechCrunch for one topic."""
    print(f"\n[Tool] compare_sources: topic={topic}, days={days}")
    if not topic or not topic.strip():
        return "compare_sources requires topic."

    days = _clamp_int(days, 1, 90)
    q = f"%{topic.strip()}%"
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                source_type,
                COUNT(*) AS cnt,
                ROUND(AVG(points)::numeric, 1) AS avg_points,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neu_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
              AND (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)
            GROUP BY source_type
            ORDER BY source_type
            """,
            (f"{days} days", q, q, q),
        )
        stats_rows = cur.fetchall()

        cur.execute(
            """
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
                  AND (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)
            )
            SELECT source_type, headline, url, points, created_at, rn
            FROM ranked
            WHERE rn <= 3
            ORDER BY source_type, rn
            """,
            (f"{days} days", q, q, q),
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
    """Compare two entities/topics with DB-backed evidence only."""
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
    """Cross-domain landscape snapshot with entity stats and evidence URLs."""
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
        topic_like = f"%{topic}%"
        topic_where_sql = (
            "AND ("
            "v.title ILIKE %s "
            "OR COALESCE(v.title_cn, '') ILIKE %s "
            "OR COALESCE(v.summary, '') ILIKE %s"
            ")"
        )
        topic_params = [topic_like, topic_like, topic_like]

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
        db_now = cur.fetchone()[0]

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
            topic_like = f"%{topic}%"
            topic_scope_sql += (
                " AND ("
                "v.title ILIKE %s "
                "OR COALESCE(v.title_cn, '') ILIKE %s "
                "OR COALESCE(v.summary, '') ILIKE %s"
                ")"
            )
            topic_scope_params.extend([topic_like, topic_like, topic_like])
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
            if created_at and created_at >= cutoff_ts:
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
    """Compatibility alias for AI landscape."""
    return analyze_landscape(topic="AI", days=days, entities=entities, limit_per_entity=limit_per_entity)


def build_timeline(topic: str, days: int = 30, limit: int = 12) -> str:
    """Build chronological timeline for a topic."""
    print(f"\n[Tool] build_timeline: topic={topic}, days={days}, limit={limit}")
    if not topic or not topic.strip():
        return "build_timeline requires topic."

    days = _clamp_int(days, 1, 180)
    limit = _clamp_int(limit, 3, 40)
    q = f"%{topic.strip()}%"
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                created_at,
                source_type,
                COALESCE(title_cn, title) AS headline,
                sentiment,
                points,
                url
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
              AND (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (f"{days} days", q, q, q, limit),
        )
        rows = cur.fetchall()
        cur.close()
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
    """Batch-read full text by URL list or keyword fallback."""
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
