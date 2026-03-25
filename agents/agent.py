"""LLM 客户端：Chat 工厂与无状态调用接口"""

import os
from google import genai
from google.genai import types

from tools import search_news, read_news_content, get_db_stats, list_topics
from prompts import SYSTEM_INSTRUCTION

AGENT_TOOLS = [search_news, read_news_content, get_db_stats, list_topics]


def _build_config() -> types.GenerateContentConfig:
    """构建 Gemini Agent 的通用配置"""
    return types.GenerateContentConfig(
        tools=AGENT_TOOLS,
        system_instruction=SYSTEM_INSTRUCTION,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=50,
        ),
    )


def create_agent_chat():
    """创建有状态的 Chat 会话，适用于 CLI 等长连接场景。

    Returns:
        google.genai Chat 对象，内部自动维护多轮上下文
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 环境变量未设置")

    client = genai.Client(api_key=api_key)
    return client.chats.create(
        model="gemini-2.5-pro",
        config=_build_config(),
    )


def generate_response(history: list[dict], user_message: str) -> str:
    """无状态调用：接收对话历史 + 新消息，返回 Agent 回复。

    使用 Chat 接口传入 history，确保 automatic_function_calling 正常循环。
    适用于 Bot / Web API 等无状态部署场景。
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 环境变量未设置")

    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model="gemini-2.5-pro",
        config=_build_config(),
        history=history,
    )

    response = chat.send_message(user_message)

    if getattr(response, "candidates", None) is None or not response.candidates:
        return "[错误] 模型未能生成任何候选内容，可能触发了安全拦截。"

    parts_texts = [
        part.text
        for part in response.candidates[0].content.parts
        if hasattr(part, "text") and part.text
    ]
    if not parts_texts:
        return "[错误] 模型未能返回有效文本，可能触发了安全拦截或解析异常。"

    # automatic_function_calling 会在 parts 中累积每轮的中间文本，
    # 最后一段才是工具调用全部完成后的最终结构化分析
    return parts_texts[-1]

