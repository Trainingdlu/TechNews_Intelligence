"""
TechNews Intelligence - Telegram Bot
=====================================
使用 generate_response() 无状态调用，按 chat_id 隔离对话历史
"""

import os
import re
import asyncio
import logging

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from core import generate_response, init_db_pool, close_db_pool

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 按 chat_id 存储对话历史
# 格式: { chat_id: [{"role": "user"/"model", "parts": [{"text": "..."}]}, ...] }
conversation_histories: dict[int, list[dict]] = {}

MAX_HISTORY_TURNS = 20  # 保留最近20轮，防止 context 太长


def _trim_history(history: list[dict]) -> list[dict]:
    """超过 MAX_HISTORY_TURNS 轮时，保留最新的"""
    # 每轮 = user + model 各一条，共2条
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


# ---------------------------------------------------------------------------
# Telegram MarkdownV2 转义
# ---------------------------------------------------------------------------
_MARKDOWNV2_ESCAPE_RE = re.compile(r"([_\[\]()~`>#+\-=|{}.!\\])")


def _escape_markdownv2(text: str) -> str:
    """转义 MarkdownV2 中的特殊字符，但保留常用格式标记。

    保留的格式：**bold**  → *bold*（Telegram 风格）
    其余所有 MarkdownV2 特殊字符均转义，避免发送失败。
    """
    # 1. 将 Gemini 风格的 **bold** 转为占位符
    parts = re.split(r"\*\*(.+?)\*\*", text, flags=re.DOTALL)

    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # 奇数段 = bold 内容：转义内部特殊字符后用 * 包裹
            result.append("*" + _MARKDOWNV2_ESCAPE_RE.sub(r"\\\1", part) + "*")
        else:
            # 偶数段 = 普通文本：转义所有特殊字符（含 *）
            escaped = _MARKDOWNV2_ESCAPE_RE.sub(r"\\\1", part)
            # 额外转义独立的 * 号（不在 bold 标记中的）
            escaped = escaped.replace("*", "\\*")
            result.append(escaped)
    return "".join(result)


async def _send_reply(message, text: str):
    """尝试以 MarkdownV2 发送，失败则回退到纯文本。"""
    chunks = [text[i:i + 4096] for i in range(0, len(text), 4096)]
    for chunk in chunks:
        try:
            await message.reply_text(
                _escape_markdownv2(chunk),
                parse_mode="MarkdownV2",
            )
        except Exception:
            # MarkdownV2 解析失败时回退纯文本
            await message.reply_text(chunk)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_histories[chat_id] = []  # 清空历史
    await update.message.reply_text(
        "你好！我是 TechNews 智能分析助手。\n"
        "发送任意问题开始对话，发送 /clear 可以清空对话历史。"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_histories[chat_id] = []
    await update.message.reply_text("对话历史已清空，可以重新开始。")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    if not user_text:
        return

    # 初始化该用户的历史
    if chat_id not in conversation_histories:
        conversation_histories[chat_id] = []

    # 发送"正在思考"提示
    thinking_msg = await update.message.reply_text("正在分析，请稍候……")

    try:
        history = conversation_histories[chat_id]
        # 使用 asyncio.to_thread 避免阻塞事件循环
        reply = await asyncio.to_thread(generate_response, history, user_text)

        # 更新历史
        history.append({"role": "user", "parts": [{"text": user_text}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        conversation_histories[chat_id] = _trim_history(history)

        # 删除"正在思考"，发送回复
        await thinking_msg.delete()
        await _send_reply(update.message, reply)

    except Exception as e:
        logger.error(f"[chat_id={chat_id}] 处理消息出错: {e}")
        await thinking_msg.edit_text(f"出错：{str(e)}")


async def _post_shutdown(app) -> None:
    """ApplicationBuilder post_shutdown 回调，优雅关闭数据库连接池。"""
    close_db_pool()
    logger.info("数据库连接池已关闭")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN 环境变量未设置")

    init_db_pool()
    logger.info("数据库连接池已初始化")

    app = (
        ApplicationBuilder()
        .token(token)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("启动中")
    app.run_polling()


if __name__ == "__main__":
    main()