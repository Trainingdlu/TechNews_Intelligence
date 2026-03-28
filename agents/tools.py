"""Database and retrieval tools for the agent runtime."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

try:
    from db import get_conn, put_conn
except ImportError:  # package-style import fallback
    from .db import get_conn, put_conn


JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"


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
    return None


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
    """Return candidate URLs by keyword query."""
    q = f"%{query.strip()}%"
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COALESCE(title_cn, title) AS headline,
                url,
                source_type,
                created_at,
                points
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
              AND (title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)
            ORDER BY points DESC NULLS LAST, created_at DESC
            LIMIT %s
            """,
            (f"{days} days", q, q, q, limit),
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
        where_parts.append("source_type = %s")
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
        if not rows:
            return "No matching records."

        lines = []
        for idx, row in enumerate(rows, 1):
            title, title_cn, url, summary, senti, points, src, created_at = row
            lines.append(
                f"{idx}. [{src}] {title_cn or title}\n"
                f"   time={created_at.strftime('%Y-%m-%d %H:%M')}, sentiment={senti}, points={points}\n"
                f"   url={url}\n"
                f"   summary={(summary or '')[:220]}"
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

        def fmt(name: str, row: tuple | None) -> str:
            if not row:
                return f"{name}: count=0, avg_points=0, sentiment(P/N/Ng)=0/0/0"
            _, cnt, avg_points, pos, neu, neg = row
            return f"{name}: count={cnt}, avg_points={avg_points}, sentiment(P/N/Ng)={pos}/{neu}/{neg}"

        lines = [f"Topic comparison: {topic_a} vs {topic_b} (last {days} days)", "Stats:"]
        lines.append("  " + fmt(topic_a, metric_map["A"]))
        lines.append("  " + fmt(topic_b, metric_map["B"]))
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


def fulltext_batch(urls: str, max_chars_per_article: int = 4000) -> str:
    """Batch-read full text by URL list or keyword fallback."""
    print("\n[Tool] fulltext_batch")
    max_chars_per_article = _clamp_int(max_chars_per_article, 800, 12000)

    raw_items = _split_urls(urls)
    direct_urls = [x for x in raw_items if _is_probable_url(x)]

    selected: list[tuple[str, str]] = []
    prefix_lines: list[str] = []

    if direct_urls:
        for u in direct_urls[:12]:
            selected.append(("direct", u))
    else:
        query = (urls or "").strip()
        if not query:
            return "fulltext_batch requires URLs or a keyword query."

        candidates = _lookup_urls_by_query(query=query, days=14, limit=5)
        if not candidates:
            return f"No candidate articles found for query '{query}'."

        prefix_lines.append(
            f"No URLs provided. Auto-selected Top {len(candidates)} articles for query '{query}':"
        )
        for i, (headline, url, source_type, created_at, points) in enumerate(candidates, 1):
            prefix_lines.append(
                f"{i}. [{source_type}] {headline} | points={points} | "
                f"{created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )
            selected.append(("query", url))

    chunks: list[str] = []
    for idx, (_, url) in enumerate(selected, 1):
        content = read_news_content(url)
        if len(content) > max_chars_per_article:
            content = content[:max_chars_per_article] + "\n...[truncated]"
        chunks.append(f"=== [{idx}] {url} ===\n{content}")

    if prefix_lines:
        return "\n".join(prefix_lines) + "\n\n" + "\n\n".join(chunks)
    return "\n\n".join(chunks)
