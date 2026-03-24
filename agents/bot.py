"""
TechNews Intelligence - Telegram Bot
=====================================
使用 generate_response() 无状态调用，按 chat_id 隔离对话历史
"""

import os
import sys
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
    thinking_msg = await update.message.reply_text("正在分析，请稍候")

    try:
        history = conversation_histories[chat_id]
        reply = generate_response(history, user_text)

        # 更新历史
        history.append({"role": "user", "parts": [{"text": user_text}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        conversation_histories[chat_id] = _trim_history(history)

        # 删除"正在思考"，发送回复
        await thinking_msg.delete()
        # Telegram 单条消息上限 4096 字符，超出则分段发送
        for i in range(0, len(reply), 4096):
            await update.message.reply_text(reply[i:i+4096])

    except Exception as e:
        logger.error(f"[chat_id={chat_id}] 处理消息出错: {e}")
        await thinking_msg.edit_text(f"出错：{str(e)}")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN 环境变量未设置")

    init_db_pool()
    logger.info("数据库连接池已初始化")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("B启动中")
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_db_pool()