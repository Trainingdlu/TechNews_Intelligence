"""Telegram Bot 入口：按 chat_id 隔离对话历史，调用 Agent 无状态接口"""

import os
import re
import asyncio
import logging

from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from agent import generate_response
from db import init_db_pool, close_db_pool, get_conn, put_conn

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 按 chat_id 存储对话历史
# 格式: { chat_id: [{"role": "user"/"model", "parts": [{"text": "..."}]}, ...] }
conversation_histories: dict[int, list[dict]] = {}
chat_history_limits: dict[int, int] = {}

MAX_HISTORY_TURNS = 20  # 保留最近20轮，防止 context 太长


def _trim_history(history: list[dict], max_turns: int | None = None) -> list[dict]:
    """Trim history by max turns (default: MAX_HISTORY_TURNS)."""
    # 每轮 = user + model 各一条，共2条
    turns = max_turns if max_turns is not None else MAX_HISTORY_TURNS
    max_messages = max(1, turns) * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def _parse_admin_ids() -> set[int]:
    raw = os.getenv("TELEGRAM_ADMIN_IDS", "").strip()
    if not raw:
        return set()
    ids: set[int] = set()
    for part in re.split(r"[,\s]+", raw):
        p = part.strip()
        if not p:
            continue
        try:
            ids.add(int(p))
        except Exception:
            continue
    return ids


def _is_admin(update: Update) -> bool:
    user = update.effective_user
    if user is None:
        return False
    admins = _parse_admin_ids()
    if not admins:
        return False
    return int(user.id) in admins


async def _require_admin(update: Update, command_name: str) -> bool:
    if _is_admin(update):
        return True
    await update.message.reply_text(
        f"{command_name} 仅管理员可用。\n"
        "请在环境变量 TELEGRAM_ADMIN_IDS 中配置允许的 Telegram user_id。"
    )
    return False


import html as html_mod


def _render_limited_bold_line(line: str) -> str:
    """Keep bold only for short segment-summary labels, plain text otherwise.

    Supported bold patterns:
    - **Summary**
    - **Summary**: detail
    - **Summary**\uFF1Adetail
    """
    # Case 1: full-line summary label: **Summary**
    m = re.match(r"^(\s*)\*\*(.+?)\*\*\s*$", line)
    if m:
        indent, label = m.group(1), m.group(2)
        return f"{html_mod.escape(indent)}<b>{html_mod.escape(label)}</b>"

    # Case 2: prefix summary label: **Summary**: detail
    m = re.match(r"^(\s*)\*\*(.+?)\*\*\s*([:\uFF1A])\s*(.*)$", line)
    if m:
        indent, label, colon, rest = m.groups()
        prefix = f"{html_mod.escape(indent)}<b>{html_mod.escape(label)}</b>{html_mod.escape(colon)}"
        if rest:
            return f"{prefix} {html_mod.escape(rest)}"
        return prefix

    # Other lines: strip markdown bold markers and keep plain text.
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    return html_mod.escape(plain)


def _chunk_by_lines(text: str, max_len: int = 4096) -> list[str]:
    """Split text into Telegram-safe chunks without breaking HTML tags mid-line."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    cur_len = 0

    for line in lines:
        if len(line) > max_len:
            if current:
                chunks.append("\n".join(current))
                current = []
                cur_len = 0
            for i in range(0, len(line), max_len):
                chunks.append(line[i:i + max_len])
            continue

        add_len = len(line) + (1 if current else 0)
        if current and (cur_len + add_len > max_len):
            chunks.append("\n".join(current))
            current = [line]
            cur_len = len(line)
            continue

        current.append(line)
        cur_len += add_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def _format_for_telegram(text: str) -> str:
    """将 Gemini 输出的 Markdown 转换为 Telegram HTML 格式。

    策略：
    - # / ## / ### 标题 → <b>粗体</b>
    - **正文加粗** → 去除星号，保留纯文本
    - 其余内容 HTML 转义，防止解析错误
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # 处理 Markdown 标题行
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)
            title = html_mod.escape(title)
            if title:
                result.append(f"<b>{title}</b>")
            continue

        # 将 markdown 无序列表统一为 "-"，避免 TG 中星号列表不稳定。
        line = re.sub(r"^(\s*)[\*\u2022]\s+", r"\1- ", line)

        # 正文：只允许分段综述标签加粗；其余正文全部普通文本。
        result.append(_render_limited_bold_line(line))
    return "\n".join(result)


async def _send_reply(message, text: str):
    """尝试以 HTML 模式发送，失败则回退到纯文本。"""
    formatted = _format_for_telegram(text)
    chunks = _chunk_by_lines(formatted, 4096)
    for chunk in chunks:
        try:
            await message.reply_text(chunk, parse_mode="HTML")
        except Exception:
            # HTML 解析失败时回退纯文本（去除标签）
            plain = re.sub(r"<[^>]+>", "", chunk)
            plain = html_mod.unescape(plain)
            await message.reply_text(plain)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_histories[chat_id] = []  # 清空历史
    await update.message.reply_text(
        "你好！我是 TechNews 智能分析助手。\n"
        "发送任意问题开始对话。\n"
        "输入 /menu 查看可用命令。"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_histories[chat_id] = []
    await update.message.reply_text("对话历史已清空，可以重新开始。")


def _extract_command_arg(text: str) -> str:
    if not text:
        return ""
    parts = text.strip().split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "可用命令：\n"
        "/start - 开始并重置当前会话\n"
        "/menu - 查看命令菜单\n"
        "/settings - 查看或修改当前设置（管理员）\n"
        "/quota <token> - 查询 API Token 配额（管理员）\n"
        "/clear - 清空对话历史\n"
        "/help - 查看帮助"
    )


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _require_admin(update, "/settings"):
        return
    chat_id = update.effective_chat.id

    def _usage() -> str:
        return (
            "用法：\n"
            "/settings                    查看当前设置\n"
            "/settings strict on|off      设置严格模式\n"
            "/settings runtime langchain|legacy  设置运行时\n"
            "/settings history <1-100>    设置历史轮数上限"
        )

    arg = _extract_command_arg(update.message.text or "")
    if not arg:
        runtime = os.getenv("AGENT_RUNTIME", "langchain").strip().lower()
        strict_mode = os.getenv("AGENT_RUNTIME_STRICT", "false").strip().lower()
        chat_limit = chat_history_limits.get(chat_id, MAX_HISTORY_TURNS)
        await update.message.reply_text(
            "当前设置：\n"
            f"- 运行时: {runtime}\n"
            f"- 严格模式: {strict_mode}\n"
            f"- 当前 chat 历史轮数上限: {chat_limit}\n"
            f"- 默认历史轮数上限: {MAX_HISTORY_TURNS}\n"
            "- 消息格式: HTML（仅标题/分段综述加粗）\n\n"
            + _usage()
        )
        return

    parts = arg.split()
    key = parts[0].lower()
    if key == "strict":
        if len(parts) < 2 or parts[1].lower() not in {"on", "off"}:
            await update.message.reply_text("参数错误。\n" + _usage())
            return
        val = "true" if parts[1].lower() == "on" else "false"
        os.environ["AGENT_RUNTIME_STRICT"] = val
        await update.message.reply_text(f"已更新：strict = {parts[1].lower()}（AGENT_RUNTIME_STRICT={val}）")
        return

    if key == "runtime":
        if len(parts) < 2 or parts[1].lower() not in {"langchain", "legacy"}:
            await update.message.reply_text("参数错误。\n" + _usage())
            return
        val = parts[1].lower()
        os.environ["AGENT_RUNTIME"] = val
        await update.message.reply_text(f"已更新：runtime = {val}")
        return

    if key == "history":
        if len(parts) < 2:
            await update.message.reply_text("参数错误。\n" + _usage())
            return
        try:
            n = int(parts[1])
        except Exception:
            await update.message.reply_text("history 需为整数。\n" + _usage())
            return
        n = max(1, min(100, n))
        chat_history_limits[chat_id] = n
        # 仅裁剪当前 chat 的历史，避免影响其他用户
        hist = conversation_histories.get(chat_id, [])
        conversation_histories[chat_id] = _trim_history(hist, max_turns=n)
        await update.message.reply_text(f"已更新：当前 chat history = {n}")
        return

    await update.message.reply_text("不支持的设置项。\n" + _usage())


async def quota(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _require_admin(update, "/quota"):
        return
    token = _extract_command_arg(update.message.text or "")
    if not token:
        await update.message.reply_text(
            "用法：/quota <token>\n"
            "示例：/quota abcdef123456"
        )
        return

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT quota, used, status
            FROM access_tokens
            WHERE token = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (token,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            await update.message.reply_text("未找到该 token，请检查后重试。")
            return

        total, used, status = int(row[0]), int(row[1]), str(row[2] or "")
        remaining = max(0, total - used)
        await update.message.reply_text(
            "配额信息：\n"
            f"- 总配额: {total}\n"
            f"- 已使用: {used}\n"
            f"- 剩余: {remaining}\n"
            f"- 状态: {status}"
        )
    except Exception as e:
        logger.error(f"查询配额失败: {e}")
        await update.message.reply_text("查询配额失败，请稍后重试。")
    finally:
        put_conn(conn)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await menu(update, context)


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
        # 使用 asyncio.to_thread 避免阻塞事件循环
        reply = await asyncio.to_thread(generate_response, history, user_text)

        # 更新历史
        history.append({"role": "user", "parts": [{"text": user_text}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        history_limit = chat_history_limits.get(chat_id, MAX_HISTORY_TURNS)
        conversation_histories[chat_id] = _trim_history(history, max_turns=history_limit)

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


async def _post_init(app) -> None:
    """Initialize Telegram slash-command menu."""
    commands = [
        BotCommand("start", "开始并重置会话"),
        BotCommand("menu", "查看命令菜单"),
        BotCommand("settings", "查看或修改当前设置"),
        BotCommand("quota", "查询 token 配额(管理员)"),
        BotCommand("clear", "清空对话历史"),
        BotCommand("help", "帮助"),
    ]
    try:
        await app.bot.set_my_commands(commands)
    except Exception as e:
        logger.warning(f"设置 Telegram 命令菜单失败: {e}")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN 环境变量未设置")

    init_db_pool()
    logger.info("数据库连接池已初始化")

    app = (
        ApplicationBuilder()
        .token(token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CommandHandler("settings", settings))
    app.add_handler(CommandHandler("quota", quota))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("启动中")
    app.run_polling()


if __name__ == "__main__":
    main()
