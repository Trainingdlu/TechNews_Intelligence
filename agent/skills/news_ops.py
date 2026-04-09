"""Shared DB-backed primitives used by the runtime outside structured skills."""

from __future__ import annotations

from services.db import get_conn, put_conn


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
    except Exception as exc:
        print(f"[Warn] lookup_url_titles failed: {exc}")
        return {}
    finally:
        if conn is not None:
            put_conn(conn)


def get_db_stats() -> str:
    """Get database freshness statistics and total article count."""
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
    except Exception as exc:
        print(f"[Error] get_db_stats failed: {exc}")
        return f"get_db_stats failed: {exc}"
    finally:
        put_conn(conn)


def list_topics() -> str:
    """Get daily article volume distribution for the most recent 21 days."""
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
        for day, count in rows:
            lines.append(f"  {day.strftime('%Y-%m-%d')}: {count}")
        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] list_topics failed: {exc}")
        return f"list_topics failed: {exc}"
    finally:
        put_conn(conn)


def read_news_content(url: str) -> str:
    """Read full-text content of a single article by URL."""
    print(f"\n[Tool] read_news_content: {url}")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM tech_news WHERE url = %s LIMIT 1", (url,))
        if not cur.fetchone():
            cur.close()
            return f"[Error] URL '{url}' not found in DB. Use URLs returned by search tools."

        cur.execute("SELECT raw_content FROM jina_raw_logs WHERE url = %s LIMIT 1", (url,))
        row = cur.fetchone()
        cur.close()

        if row and row[0]:
            return f"Full content:\n{row[0]}"
        return "URL exists but full content is not available yet."
    except Exception as exc:
        print(f"[Error] read_news_content failed: {exc}")
        return f"read_news_content failed: {exc}"
    finally:
        put_conn(conn)
