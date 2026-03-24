"""
TechNews Intelligence - Agent Core
===================================
核心引擎：工具函数、连接池、Agent Chat
供 cli.py 或 api.py 调用
"""

import os
import json
import requests
import psycopg2
import psycopg2.pool
from google import genai
from google.genai import types


_pool: psycopg2.pool.SimpleConnectionPool | None = None


def init_db_pool():
    """根据环境变量初始化数据库连接池"""
    global _pool
    if _pool is not None:
        return
    _pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5555")),
        dbname=os.getenv("DB_NAME", "DB"),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
    )


def _get_conn():
    """从池中取连接"""
    if _pool is None:
        init_db_pool()
    return _pool.getconn()


def _put_conn(conn):
    """归还连接到池中"""
    if _pool is not None:
        _pool.putconn(conn)


def close_db_pool():
    """关闭连接池"""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


# ---------------------------------------------------------------------------
# Jina Embeddings
# ---------------------------------------------------------------------------
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"


def _get_query_embedding(query: str) -> list | None:
    """调用 Jina Embeddings API 获取查询向量"""
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
        print(f"[错误] 向量化查询失败，将仅使用关键词搜索: {e}")
        return None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def get_db_stats() -> str:
    """
    获取数据库的数据时效性信息，包括最新文章入库时间和文章总数。
    用于在回答用户之前了解自身数据的新鲜程度。
    """
    print("\n[工具执行] 正在获取数据库统计信息")
    conn = _get_conn()
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
        _put_conn(conn)


def list_topics() -> str:
    """
    获取最近21天每日新闻入库数量概览，帮助判断数据覆盖和分布情况。
    适用于用户提出宽泛问题（如"最近有什么大事"）时先了解数据全貌。
    """
    print("\n[工具执行] 正在获取近21天文章分布")
    conn = _get_conn()
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
        _put_conn(conn)


def search_news(query: str, days: int = 21) -> str:
    """
    在新闻数据库中搜索与查询相关的文章。
    使用混合搜索：语义相似度 + 关键词精确匹配，自动合并去重后返回最相关的结果。
    默认搜索最近21天的文章，可通过 days 参数调整时间范围。

    Args:
        query: 用户的搜索查询，例如"英伟达最新GPU发布"、"OpenAI安全问题"
        days: 搜索的时间范围（天数），默认21天。如需搜索更早的历史文章可设为更大值如90或365
    """
    print(f"\n[工具执行] 正在搜索: {query}")
    limit = 5
    time_filter = f"{days} days"
    conn = _get_conn()
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
        print(f"[错误] 数据库查询发生内部错误: {e}")
        return f"[数据库查询出错：{str(e)}，请勿推断或补充，直接告知用户查询出错]"
    finally:
        _put_conn(conn)


def read_news_content(url: str) -> str:
    """
    根据新闻URL，从 jina_raw_logs 数据库中读取新闻全文内容，以便进行深入分析。
    注意：url 必须来自 search_news 返回的结果，不要自行编造或猜测URL。

    Args:
        url: 必须是 search_news 返回结果中的URL，不可自行编造
    """
    print(f"\n[工具执行] 正在读取新闻全文: {url}")
    conn = _get_conn()
    try:
        cur = conn.cursor()
        # 先验证该 URL 是否真实存在于 tech_news 表中
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
        return f"该URL存在于数据库但全文内容暂未抓取，请使用 search_news 返回的摘要进行分析。"
    except Exception as e:
        print(f"[错误] 读取全文时发生错误: {e}")
        return f"读取全文出错：{str(e)}"
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "# 角色\n"
    "你是一位顶尖的科技行业战略分析师，拥有深厚的技术背景和敏锐的商业直觉。\n"
    "你的用户是科技从业者和决策者，他们需要你回答一个核心问题：**\"So What?\"**\n\n"

    "# 数据库说明\n"
    "- search_news 使用混合搜索（语义相似度 + 关键词匹配），每次最多返回5条，附带相似度分数\n"
    "- 相似度分数含义：1.0 = 关键词精确命中，0~1 = 语义相似度，低于 0.5 表示相关性较弱\n"
    "- 全文内容由网页自动提取，可能含导航栏、页脚等非正文噪声，分析时聚焦正文\n"
    "- 数据每小时更新一次\n\n"

    "# 工作流程（严格执行）\n"
    "0. **时效感知（每轮对话首次交互时执行一次）**：调用 get_db_stats 获取数据库最新文章时间，"
    "后续回答中主动标注数据截止时间。"
    "若用户问题涉及\"最近\"、宽泛话题或无明确实体指向，先调用 list_topics 了解近21天数据分布全貌。\n"
    "1. **搜索**：调用 search_news，传入最能概括用户意图的查询语句。"
    "**若查询是一个笼统概念或行业（如'科技巨头'、'AI发展'），必须分别针对至少3家以上具有代表性的不同公司（如微软、Meta、亚马逊、苹果等）独立进行搜索**，确保视野广度，不能仅局限于一家。"
    "**若查询涉及某家明确的公司，必须至少从两个不同业务维度各搜索一次** "
    "（例如查询'谷歌'时，分别搜索'谷歌 Gemini产品动态'和'谷歌 Android硬件'，"
    "而非两次都围绕同一维度）。"
    "**构造查询词时避免使用会命中历史积累性事件的词汇**（如'诉讼''反垄断''监管史'等），"
    "优先使用指向近期动态的词汇（如'发布''更新''推出''战略'）。"
    "**不要假设具体产品名称**（如GPT-4o、Claude 3等），使用公司名+通用业务方向（如'OpenAI 模型发布'、'Anthropic 产品动态'）来构造查询词。"
    "若返回结果相关性低或数量不足，换角度重试（非公司类查询最多重试1次）。"
    "通读所有结果，优先选择："
    "①相似度分数高、②涉及重大战略动作（融资/收购/发布/监管）、③有第一手数据或独家信息的文章。\n"
    "2. **深挖全文**：从 search_news 返回的结果中，按①相似度分数、②事件重要性综合排序，"
    "选取最有价值的2~4条，**使用其URL字段原样传入** read_news_content 获取全文。"
    "**严禁自行编造、猜测或拼接URL，必须完整复制 search_news 返回的URL。**"
    "相似度低于0.5的文章不读全文，直接跳过。"
    "若返回'未找到'，跳至下一候选，不要停止。\n"
    "3. **分析输出**：基于全文内容，按下方格式进行深度分析。\n\n"

    "# 输出格式\n"
    "## 核心事件\n"
    "2~3句话概括发生了什么（基于全文，非摘要复读）。注明信息时间（最新文章的 created_at）。\n\n"
    "## 深度解读\n"
    "根据核心事件的多少，选择最相关的1~4个维度进行剖析（若事件少则削减维度，宁缺毋滥）。每个维度具体展开1~3个论点，必须引用全文细节支撑。**若全文对应信息不足，直接跳过该维度，"
    "严禁为了填补篇幅而将同一单一事件强加穿插到多个不同维度中（禁止单一话题劫持）。**：\n"
    "- **技术趋势**：技术/产品处于什么发展阶段？解决了什么真实痛点？\n"
    "- **竞争格局**：对主要玩家市场地位有何影响？\n"
    "- **商业影响**：对开发者、企业客户或终端用户意味着什么？\n"
    "- **监管与风险**：是否涉及安全、隐私或政策合规风险？\n"
    "- **生态系统**：对上下游产业链有何连锁反应？\n"
    "- **市场情绪**：多篇文章的 sentiment 字段呈现何种整体倾向？与事件本身是否吻合？\n\n"
    "## 关键洞察\n"
    "2~3条精炼判断：这件事的本质是什么，或接下来最值得关注的具体信号是什么。\n\n"

    "# 铁律（违反即失败）\n"
    "1. **禁止幻觉**：每个论点必须能在全文中找到依据，全文没有的不要提。\n"
    "2. **禁止复读**：不得逐条罗列标题和摘要，价值在于整合与洞察。\n"
    "3. **禁止单一话题劫持**：不得让某一事件（尤其是诉讼、事故等情绪性事件）占据超过一个分析维度。"
    "若同一事件在多篇全文中反复出现，合并为一个论点处理，不得展开多次。\n"
    "4. **禁止空话**：不使用'值得关注''拭目以待'等无信息量表达，每句话必须传递具体信息或明确判断。\n"
    "5. **全文优先**：分析必须基于全文内容。若所有全文获取均失败，在报告头部注明'本分析基于摘要，深度有限'后继续。\n"
    "6. **冲突标注**：若多篇全文在关键事实上矛盾，须明确指出分歧，不得选择性忽略。\n"
    "7. **禁止编造URL**：read_news_content 的 url 参数只能使用 search_news 返回结果中的URL原文。"
    "绝对不允许根据记忆或推测自行构造URL，违反将导致查询失败。"
)

# 注册给 Gemini 的工具列表
AGENT_TOOLS = [search_news, read_news_content, get_db_stats, list_topics]


# ---------------------------------------------------------------------------
# Agent Chat
# ---------------------------------------------------------------------------
def create_agent_chat():
    """
    创建一个有状态的 Gemini Chat 会话对象。
    适用于 CLI 等长连接场景，Chat 对象内部自动维护多轮上下文。

    Returns:
        chat: google.genai Chat 对象
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 环境变量未设置，请检查 .env 文件")

    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            tools=AGENT_TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            ),
        ),
    )
    return chat


def generate_response(history: list[dict], user_message: str) -> str:
    """
    无状态调用：接收完整对话历史 + 新消息，返回 Agent 回复文本。
    适用于 Web API / Bot 等无状态部署场景。

    使用 Chat 接口而非 generate_content，因为只有 Chat 接口
    才支持 automatic_function_calling 的完整循环（自动执行工具并回传结果）。

    Args:
        history: 历史对话列表，格式 [{"role": "user", "parts": [...]}, ...]
        user_message: 本轮用户输入

    Returns:
        Agent 的回复文本
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 环境变量未设置，请检查 .env 文件")

    client = genai.Client(api_key=api_key)

    # 使用 Chat 接口：传入历史记录，支持自动函数调用循环
    chat = client.chats.create(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            tools=AGENT_TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            ),
        ),
        history=history,
    )

    response = chat.send_message(user_message)

    if getattr(response, "candidates", None) is None or not response.candidates:
        return "[错误] 模型未能生成任何候选内容，可能触发了平台安全性拦截机制。"

    parts_texts = [
        part.text
        for part in response.candidates[0].content.parts
        if hasattr(part, "text") and part.text
    ]
    return "".join(parts_texts) if parts_texts else (
        "[错误] 模型未能返回有效文本，可能触发了安全拦截或遇到了解析异常。"
    )
