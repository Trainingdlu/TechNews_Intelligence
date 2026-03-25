"""Agent 工具函数：供 Gemini 自动调用的数据库查询与全文检索工具"""

import os
import requests

from db import get_conn, put_conn

# ---------------------------------------------------------------------------
# Jina Embeddings
# ---------------------------------------------------------------------------
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"


def _get_query_embedding(query: str) -> list | None:
    """调用 Jina Embeddings API 将查询文本向量化，失败时返回 None"""
    jina_key = os.getenv("JINA_API_KEY", "")
    if not jina_key:
        print("[错误] JINA_API_KEY 未设置，跳过向量化")
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
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"[错误] 向量化失败，将仅使用关键词搜索: {e}")
        return None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def get_db_stats() -> str:
    """
    获取数据库的数据时效性信息，包括最新文章入库时间和文章总数。
    用于在回答用户之前了解自身数据的新鲜程度。
    """
    print("\n[工具执行] 获取数据库统计信息")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT MAX(created_at), COUNT(*) FROM tech_news"
        )
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            return (
                f"数据库统计：共 {row[1]} 篇文章，"
                f"最新文章入库时间为 {row[0].strftime('%Y-%m-%d %H:%M')}。"
            )
        return "数据库为空，暂无文章数据。"
    except Exception as e:
        print(f"[错误] 获取数据库统计失败: {e}")
        return f"获取数据库统计出错：{str(e)}"
    finally:
        put_conn(conn)


def list_topics() -> str:
    """
    获取最近21天每日新闻入库数量概览，帮助判断数据覆盖和分布情况。
    适用于用户提出宽泛问题（如"最近有什么大事"）时先了解数据全貌。
    """
    print("\n[工具执行] 获取近21天文章分布")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                DATE(created_at) AS date,
                COUNT(*)         AS count
            FROM tech_news
            WHERE created_at > NOW() - INTERVAL '21 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC;
        """)
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return "最近21天内无文章入库。"

        lines = ["近21天每日入库文章数："]
        for date, count in rows:
            lines.append(f"  {date.strftime('%Y-%m-%d')}: {count} 篇")
        return "\n".join(lines)
    except Exception as e:
        print(f"[错误] 获取文章分布失败: {e}")
        return f"获取文章分布出错：{str(e)}"
    finally:
        put_conn(conn)


def search_news(query: str, days: int = 21) -> str:
    """
    在新闻数据库中搜索与查询相关的文章。
    使用混合搜索：语义相似度 + 关键词精确匹配，自动合并去重后返回最相关的结果。
    默认搜索最近21天的文章，可通过 days 参数调整时间范围。

    Args:
        query: 用户的搜索查询，例如"英伟达最新GPU发布"、"OpenAI安全问题"
        days: 搜索的时间范围（天数），默认21天。如需搜索更早的历史文章可设为更大值如90或365
    """
    print(f"\n[工具执行] 搜索: {query}")
    limit = 5
    time_filter = f"{days} days"
    conn = get_conn()
    try:
        cur = conn.cursor()
        query_vec = _get_query_embedding(query)

        if query_vec:
            vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
            cur.execute("""
                WITH semantic AS (
                    SELECT t.title, t.url, t.summary, t.sentiment, t.created_at,
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
                    SELECT title, url, summary, sentiment, created_at,
                           1.0 AS score
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
            """, (vec_str, time_filter, vec_str, limit,
                  f"%{query}%", f"%{query}%", time_filter, limit))

            rows = cur.fetchall()
            rows.sort(key=lambda r: r[5], reverse=True)
            rows = rows[:limit]
        else:
            cur.execute("""
                SELECT title, url, summary, sentiment, created_at, 1.0 AS score
                FROM tech_news
                WHERE (title ILIKE %s OR summary ILIKE %s)
                  AND created_at > NOW() - %s::interval
                ORDER BY created_at DESC
                LIMIT %s
            """, (f"%{query}%", f"%{query}%", time_filter, limit))
            rows = cur.fetchall()

        cur.close()

        if not rows:
            return f"[未找到关于'{query}'的最近{days}天相关新闻。可尝试增大days参数扩大搜索范围，或换一个查询词重试]"

        max_score = max(r[5] for r in rows)
        low_relevance_note = ""
        if max_score < 0.5:
            low_relevance_note = "[注意] 相关性较低，库中可能无直接相关内容。以下为最接近的结果：\n\n"

        results = []
        for title, url, summary, sentiment, pub_time, score in rows:
            results.append(
                f"标题：{title}\n"
                f"URL：{url}\n"
                f"摘要：{summary}\n"
                f"情感：{sentiment}\n"
                f"时间：{pub_time.strftime('%Y-%m-%d %H:%M')}\n"
                f"相似度：{score:.3f}"
            )
        return low_relevance_note + "\n---\n".join(results)

    except Exception as e:
        print(f"[错误] 数据库查询出错: {e}")
        return f"[数据库查询出错：{str(e)}，请勿推断或补充，直接告知用户查询出错]"
    finally:
        put_conn(conn)


def read_news_content(url: str) -> str:
    """
    根据新闻URL，从 jina_raw_logs 表中读取新闻全文，以便深入分析。
    url 必须来自 search_news 返回结果，不可自行编造或猜测。

    Args:
        url: 必须是 search_news 返回结果中的URL，不可自行编造
    """
    print(f"\n[工具执行] 读取全文: {url}")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM tech_news WHERE url = %s LIMIT 1", (url,))
        if not cur.fetchone():
            cur.close()
            return (
                f"[错误] URL '{url}' 在数据库中不存在。"
                "你只能使用 search_news 返回的URL，严禁自行编造或猜测URL。"
                "请回到 search_news 的结果中选取URL。"
            )

        cur.execute("SELECT raw_content FROM jina_raw_logs WHERE url = %s LIMIT 1", (url,))
        row = cur.fetchone()
        cur.close()

        if row and row[0]:
            return f"新闻全文内容：\n{row[0]}"
        return "该URL存在于数据库但全文内容暂未抓取，请使用 search_news 返回的摘要进行分析。"
    except Exception as e:
        print(f"[错误] 读取全文出错: {e}")
        return f"读取全文出错：{str(e)}"
    finally:
        put_conn(conn)
