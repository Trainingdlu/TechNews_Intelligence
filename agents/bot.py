"""Telegram Bot 入口：按 chat_id 隔离对话历史，调用 Agent 无状态接口"""

import os
import re
import asyncio
import logging
import time
from urllib.parse import urlparse, unquote

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
chat_request_log: dict[int, list[float]] = {}
url_title_cache: dict[str, str] = {}

MAX_HISTORY_TURNS = 20  # 保留最近20轮，防止 context 太长


def _trim_history(history: list[dict], max_turns: int | None = None) -> list[dict]:
    """Trim history by max turns (default: MAX_HISTORY_TURNS)."""
    # 每轮 = user + model 各一条，共2条
    turns = max_turns if max_turns is not None else MAX_HISTORY_TURNS
    max_messages = max(1, turns) * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def _rate_limit_window_sec() -> int:
    try:
        return max(1, int(os.getenv("BOT_RATE_WINDOW_SEC", "10")))
    except Exception:
        return 10


def _rate_limit_max_requests() -> int:
    try:
        return max(1, int(os.getenv("BOT_RATE_LIMIT", "3")))
    except Exception:
        return 3


def _auto_delete_seconds() -> int:
    try:
        return max(0, int(os.getenv("BOT_AUTO_DELETE_SEC", "30")))
    except Exception:
        return 30


def _send_retry_attempts() -> int:
    try:
        return max(0, min(5, int(os.getenv("BOT_SEND_RETRY_ATTEMPTS", "2"))))
    except Exception:
        return 2


def _send_retry_base_delay_sec() -> float:
    try:
        return max(0.1, float(os.getenv("BOT_SEND_RETRY_BASE_SEC", "0.8")))
    except Exception:
        return 0.8


def _max_citation_urls() -> int:
    try:
        return max(1, min(30, int(os.getenv("BOT_MAX_CITATION_URLS", "12"))))
    except Exception:
        return 12


def _consume_chat_rate_token(chat_id: int) -> tuple[bool, int]:
    """Return (allowed, retry_after_seconds)."""
    now = time.time()
    window = _rate_limit_window_sec()
    maximum = _rate_limit_max_requests()
    logs = [t for t in chat_request_log.get(chat_id, []) if now - t < window]
    if len(logs) >= maximum:
        retry_after = max(1, int(window - (now - logs[0])) + 1)
        chat_request_log[chat_id] = logs
        return False, retry_after
    logs.append(now)
    chat_request_log[chat_id] = logs
    return True, 0


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


def _is_private_chat(update: Update) -> bool:
    chat = update.effective_chat
    return bool(chat and chat.type == "private")


async def _require_admin(update: Update, command_name: str, private_only: bool = False) -> bool:
    if private_only and not _is_private_chat(update):
        await update.message.reply_text(f"{command_name} 仅支持私聊机器人使用。")
        return False
    if _is_admin(update):
        return True
    await update.message.reply_text(
        f"{command_name} 仅管理员可用。\n"
        "请在环境变量 TELEGRAM_ADMIN_IDS 中配置允许的 Telegram user_id。"
    )
    return False


def _schedule_delete_messages(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_ids: list[int], delay_sec: int):
    if delay_sec <= 0 or not message_ids:
        return

    async def _job():
        await asyncio.sleep(delay_sec)
        for mid in message_ids:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=mid)
            except Exception:
                pass

    asyncio.create_task(_job())


import html as html_mod


_URL_PATTERN = re.compile(r"https?://[^\s<>\]\)}`]+")


def _extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for m in _URL_PATTERN.finditer(text or ""):
        u = (m.group(0) or "").strip().rstrip(".,;:!?")
        if not u or u in seen:
            continue
        seen.add(u)
        urls.append(u)
    return urls


def _fallback_title_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = (p.path or "").strip("/")
        if not path:
            return host or url
        tail = unquote(path.split("/")[-1]).strip()
        tail = re.sub(r"\.(html?|php)$", "", tail, flags=re.IGNORECASE)
        tail = re.sub(r"[-_]+", " ", tail).strip()
        if not tail:
            return host or url
        if len(tail) > 80:
            tail = tail[:77] + "..."
        return tail
    except Exception:
        return url


def _lookup_url_titles(urls: list[str]) -> dict[str, str]:
    if not urls:
        return {}
    unique_urls = list(dict.fromkeys(u for u in urls if u))

    result = {u: url_title_cache[u] for u in unique_urls if u in url_title_cache}
    missing = [u for u in unique_urls if u not in result]
    if not missing:
        return result

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT url, COALESCE(NULLIF(title_cn, ''), '')
            FROM tech_news
            WHERE url = ANY(%s)
            """,
            (missing,),
        )
        rows = cur.fetchall()
        cur.close()
        for row in rows:
            u = str(row[0] or "")
            title = str(row[1] or "").strip()
            if u and title:
                result[u] = title
                url_title_cache[u] = title
    except Exception as e:
        logger.warning(f"URL title lookup failed: {e}")
    finally:
        put_conn(conn)

    for u in unique_urls:
        if u not in result:
            result[u] = "暂无中文标题"
    return result


def _escape_and_linkify(text: str, url_title_map: dict[str, str]) -> str:
    if not text:
        return ""
    out: list[str] = []
    cursor = 0
    for m in _URL_PATTERN.finditer(text):
        matched = (m.group(0) or "").strip()
        raw = matched.rstrip(".,;:!?")
        if not raw:
            continue
        suffix = matched[len(raw):]
        start, end = m.span()
        out.append(html_mod.escape(text[cursor:start]))
        title = (url_title_map.get(raw) or "").strip() or "暂无中文标题"
        safe_url = html_mod.escape(raw, quote=True)
        safe_title = html_mod.escape(title)
        out.append(f'<a href="{safe_url}">{safe_title}</a>')
        if suffix:
            out.append(html_mod.escape(suffix))
        cursor = end
    out.append(html_mod.escape(text[cursor:]))
    return "".join(out)


def _render_limited_bold_line(line: str, url_title_map: dict[str, str]) -> str:
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
            return f"{prefix} {_escape_and_linkify(rest, url_title_map)}"
        return prefix

    # Other lines: strip markdown bold markers and keep plain text.
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    plain = re.sub(r"`(https?://[^`]+)`", r"\1", plain)
    return _escape_and_linkify(plain, url_title_map)


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


def _format_for_telegram(text: str, url_title_map: dict[str, str] | None = None) -> str:
    """将 Gemini 输出的 Markdown 转换为 Telegram HTML 格式。

    策略：
    - # / ## / ### 标题 → <b>粗体</b>
    - **正文加粗** → 去除星号，保留纯文本
    - 其余内容 HTML 转义，防止解析错误
    """
    lines = text.split("\n")
    title_map = url_title_map or {}
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
        result.append(_render_limited_bold_line(line, title_map))
    return "\n".join(result)


_EVIDENCE_HEADER_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:来源|证据来源|source(?:s)?|evidence(?:\s+sources?|\s+urls?)?)\s*:?\s*$",
    re.IGNORECASE,
)


def _strip_existing_evidence_section(text: str) -> str:
    lines = (text or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if _EVIDENCE_HEADER_RE.match(line.strip()):
            start = i
            break
    if start is None:
        return (text or "").rstrip()
    return "\n".join(lines[:start]).rstrip()


def _apply_inline_citations(text: str, ordered_urls: list[str]) -> str:
    out = text or ""
    for idx, url in enumerate(ordered_urls, 1):
        cite = f"[{idx}]"
        out = out.replace(f"`{url}`", cite)
        out = out.replace(url, cite)
    return out


def _build_evidence_section(ordered_urls: list[str]) -> str:
    lines = ["## 来源"]
    for idx, url in enumerate(ordered_urls, 1):
        lines.append(f"- [{idx}] {url}")
    return "\n".join(lines)


async def _reply_text_with_retry(message, text: str, parse_mode: str | None = None):
    attempts = _send_retry_attempts()
    base_delay = _send_retry_base_delay_sec()
    last_error = None
    for attempt in range(attempts + 1):
        try:
            kwargs = {"disable_web_page_preview": True}
            if parse_mode:
                kwargs["parse_mode"] = parse_mode
            return await message.reply_text(text, **kwargs)
        except Exception as e:
            last_error = e
            if attempt >= attempts:
                break
            wait_sec = base_delay * (2 ** attempt)
            logger.warning(
                f"Message send failed (attempt {attempt + 1}/{attempts + 1}), retry in {wait_sec:.1f}s: {e}"
            )
            await asyncio.sleep(wait_sec)
    raise last_error if last_error else RuntimeError("send failed")


async def _send_reply(message, text: str):
    """尝试以 HTML 模式发送，失败则回退到纯文本。"""
    base_text = _strip_existing_evidence_section(text)
    scan_text = base_text if base_text else text
    urls = _extract_urls(scan_text)[:_max_citation_urls()]
    title_map: dict[str, str] = {}
    if urls:
        try:
            title_map = await asyncio.to_thread(_lookup_url_titles, urls)
        except Exception as e:
            logger.warning(f"Prepare URL title map failed: {e}")

    render_text = base_text if base_text else text
    if urls:
        body = base_text if base_text else text
        body = _apply_inline_citations(body, urls)
        evidence = _build_evidence_section(urls)
        render_text = f"{body}\n\n{evidence}" if body else evidence

    formatted = _format_for_telegram(render_text, url_title_map=title_map)
    chunks = _chunk_by_lines(formatted, 4096)
    for chunk in chunks:
        try:
            await _reply_text_with_retry(message, chunk, parse_mode="HTML")
        except Exception:
            # HTML 解析失败时回退纯文本（去除标签）
            plain = re.sub(r"<[^>]+>", "", chunk)
            plain = html_mod.unescape(plain)
            await _reply_text_with_retry(message, plain, parse_mode=None)


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
        "/whoami - 查看当前 Telegram 身份\n"
        "/status - 查看机器人状态（管理员）\n"
        "/settings - 查看或修改当前设置（管理员）\n"
        "/quota <token> - 查询 API Token 配额（管理员）\n"
        "/clear - 清空对话历史\n"
        "/help - 查看帮助"
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat = update.effective_chat
    user_id = user.id if user else 0
    username = f"@{user.username}" if (user and user.username) else "(none)"
    chat_type = chat.type if chat else "(unknown)"
    admin_text = "yes" if _is_admin(update) else "no"
    await update.message.reply_text(
        "身份信息：\n"
        f"- user_id: {user_id}\n"
        f"- username: {username}\n"
        f"- chat_type: {chat_type}\n"
        f"- is_admin: {admin_text}"
    )


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _require_admin(update, "/settings", private_only=True):
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
    if not await _require_admin(update, "/quota", private_only=True):
        return
    chat_id = update.effective_chat.id
    token = _extract_command_arg(update.message.text or "")
    if not token:
        msg = await update.message.reply_text(
            "用法：/quota <token>\n"
            "示例：/quota abcdef123456"
        )
        _schedule_delete_messages(
            context,
            chat_id,
            [update.message.message_id, msg.message_id],
            _auto_delete_seconds(),
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
            msg = await update.message.reply_text("未找到该 token，请检查后重试。")
            _schedule_delete_messages(
                context,
                chat_id,
                [update.message.message_id, msg.message_id],
                _auto_delete_seconds(),
            )
            return

        total, used, status = int(row[0]), int(row[1]), str(row[2] or "")
        remaining = max(0, total - used)
        msg = await update.message.reply_text(
            "配额信息：\n"
            f"- 总配额: {total}\n"
            f"- 已使用: {used}\n"
            f"- 剩余: {remaining}\n"
            f"- 状态: {status}"
        )
        _schedule_delete_messages(
            context,
            chat_id,
            [update.message.message_id, msg.message_id],
            _auto_delete_seconds(),
        )
    except Exception as e:
        logger.error(f"查询配额失败: {e}")
        msg = await update.message.reply_text("查询配额失败，请稍后重试。")
        _schedule_delete_messages(
            context,
            chat_id,
            [update.message.message_id, msg.message_id],
            _auto_delete_seconds(),
        )
    finally:
        put_conn(conn)


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _require_admin(update, "/status", private_only=True):
        return

    runtime = os.getenv("AGENT_RUNTIME", "langchain").strip().lower()
    strict_mode = os.getenv("AGENT_RUNTIME_STRICT", "false").strip().lower()
    active_sessions = sum(1 for h in conversation_histories.values() if h)
    total_sessions = len(conversation_histories)
    per_chat_override = len(chat_history_limits)
    window = _rate_limit_window_sec()
    limit = _rate_limit_max_requests()

    db_ok = False
    db_err = ""
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        db_ok = True
    except Exception as e:
        db_err = str(e)
    finally:
        if conn is not None:
            put_conn(conn)

    lines = [
        "系统状态：",
        f"- runtime: {runtime}",
        f"- strict: {strict_mode}",
        f"- db: {'ok' if db_ok else 'error'}",
        f"- sessions(active/total): {active_sessions}/{total_sessions}",
        f"- chat_history_overrides: {per_chat_override}",
        f"- rate_limit: {limit} req / {window}s",
    ]
    if db_err:
        lines.append(f"- db_error: {db_err[:120]}")
    await update.message.reply_text("\n".join(lines))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await menu(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()

    if not user_text:
        return

    allowed, retry_after = _consume_chat_rate_token(chat_id)
    if not allowed:
        await _reply_text_with_retry(update.message, f"请求过于频繁，请在 {retry_after} 秒后重试。")
        return

    # 初始化该用户的历史
    if chat_id not in conversation_histories:
        conversation_histories[chat_id] = []

    # 发送"正在思考"提示
    thinking_msg = await _reply_text_with_retry(update.message, "正在分析，请稍候")

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
        try:
            await thinking_msg.delete()
        except Exception:
            pass
        await _send_reply(update.message, reply)

    except Exception as e:
        logger.error(f"[chat_id={chat_id}] message handling error: {e}")
        try:
            await thinking_msg.edit_text(f"出错：{str(e)}")
        except Exception:
            await _reply_text_with_retry(update.message, f"出错：{str(e)}")


async def _post_shutdown(app) -> None:
    """ApplicationBuilder post_shutdown 回调，优雅关闭数据库连接池。"""
    close_db_pool()
    logger.info("数据库连接池已关闭")


async def _post_init(app) -> None:
    """Initialize Telegram slash-command menu."""
    commands = [
        BotCommand("start", "开始并重置会话"),
        BotCommand("menu", "查看命令菜单"),
        BotCommand("whoami", "查看当前身份"),
        BotCommand("status", "查看系统状态(管理员)"),
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
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("settings", settings))
    app.add_handler(CommandHandler("quota", quota))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("启动中")
    app.run_polling()


if __name__ == "__main__":
    main()
