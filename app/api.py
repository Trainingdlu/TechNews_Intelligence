"""Web API 入口：FastAPI + Token 自动发放 + 限额管理"""

import os
import json
import secrets
import time
import asyncio
import logging
import html
import hmac
import hashlib
import uuid
from threading import Lock
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, StreamingResponse
from cachetools import TTLCache
from pydantic import BaseModel, EmailStr, Field
from psycopg2.extras import Json

from agent import AgentGenerationError, generate_response_payload
from agent.clarification import (
    resolve_user_message_with_followup_context,
    resolve_user_message_with_history_clarification,
)
from services.db import init_db_pool, close_db_pool, get_conn, put_conn
from services.mail import (
    send_token_email,
    send_quota_exhausted_to_admin,
    send_quota_exhausted_to_user,
    send_quota_upgraded,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "agent" / ".env", override=False)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://agentapi.trainingcqy.com")
APPROVE_LINK_SECRET = os.getenv("APPROVE_LINK_SECRET", os.getenv("SECRET_KEY", ""))
APPROVE_LINK_TTL_SEC = max(60, int(os.getenv("APPROVE_LINK_TTL_SEC", "86400")))
DEFAULT_QUOTA = 10
UPGRADED_QUOTA = 50
LEGACY_TRIAL_QUOTA = 15
DEFAULT_SUBSCRIPTION_SOURCES = [
    "HackerNews",
    "TechCrunch",
]
SUBSCRIPTION_FREQUENCIES = ["daily"]

# ---------------------------------------------------------------------------
# Rate Limiter（内存级，按 IP 限流）
# ---------------------------------------------------------------------------
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "5"))
RATE_WINDOW = 60
RATE_TRACK_MAX_IPS = max(100, int(os.getenv("API_RATE_TRACK_MAX_IPS", "10000")))

def _new_rate_log_cache(*, maxsize: int | None = None, ttl: int | None = None) -> TTLCache[str, list[float]]:
    cache_maxsize = maxsize if maxsize is not None else RATE_TRACK_MAX_IPS
    cache_ttl = ttl if ttl is not None else RATE_WINDOW
    return TTLCache(maxsize=max(1, cache_maxsize), ttl=max(1, cache_ttl), timer=time.time)


_request_log: TTLCache[str, list[float]] = _new_rate_log_cache()
_rate_lock = Lock()


def _check_rate_limit(client_ip: str):
    now = time.time()
    with _rate_lock:
        timestamps = [t for t in _request_log.get(client_ip, []) if now - t < RATE_WINDOW]
        if len(timestamps) >= RATE_LIMIT:
            _request_log[client_ip] = timestamps
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
        timestamps.append(now)
        _request_log[client_ip] = timestamps


def _build_approve_signature(record_id: int, exp: int) -> str:
    secret = APPROVE_LINK_SECRET.strip().encode("utf-8")
    if not secret:
        raise RuntimeError("APPROVE_LINK_SECRET is empty")
    payload = f"{record_id}:{exp}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _build_signed_approve_url(record_id: int) -> str | None:
    if not APPROVE_LINK_SECRET.strip():
        return None
    exp = int(time.time()) + APPROVE_LINK_TTL_SEC
    sig = _build_approve_signature(record_id, exp)
    return f"{API_BASE_URL}/approve/{record_id}?exp={exp}&sig={sig}"


def _is_valid_approve_signature(record_id: int, exp: int | None, sig: str | None) -> bool:
    if exp is None or not sig:
        return False
    if not APPROVE_LINK_SECRET.strip():
        return False
    if exp < int(time.time()):
        return False
    try:
        expected = _build_approve_signature(record_id, exp)
    except Exception:
        return False
    return hmac.compare_digest(expected, sig)


def _render_approve_confirmation_page(record_id: int, exp: int, sig: str) -> str:
    safe_sig = html.escape(sig, quote=True)
    action = f"/approve/{record_id}?exp={exp}&sig={safe_sig}"
    return (
        "<html><body style=\"font-family:sans-serif;max-width:560px;margin:32px auto;\">"
        "<h2>审批确认</h2>"
        "<p>请确认是否将该用户额度提升至升级配额。</p>"
        f"<form method=\"post\" action=\"{action}\">"
        "<button type=\"submit\" style=\"padding:8px 16px;cursor:pointer;\">确认批准</button>"
        "</form>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Token 验证（从数据库校验）
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer()


def _repair_legacy_upgraded_usage(
    cur,
    record_id: int,
    quota: int,
    used: int,
    status: str,
    notified: bool,
) -> tuple[int, bool, bool]:
    """
    兼容旧逻辑：
    历史版本升级只把 quota 改为 50，未重置 used，导致升级后额度被试用阶段占用。
    命中后仅修复一次：used 减去旧试用额度，并清除 notified 标记。
    """
    if (
        status == "upgraded"
        and quota == UPGRADED_QUOTA
        and bool(notified)
        and used >= LEGACY_TRIAL_QUOTA
    ):
        repaired_used = max(used - LEGACY_TRIAL_QUOTA, 0)
        cur.execute(
            "UPDATE access_tokens SET used = %s, notified = FALSE WHERE id = %s",
            (repaired_used, record_id),
        )
        logger.info(
            "quota compatibility repaired for record_id=%s: used %s -> %s",
            record_id,
            used,
            repaired_used,
        )
        return repaired_used, False, True
    return used, bool(notified), False


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> dict:
    """校验 Bearer Token 并返回 token 记录"""
    token = credentials.credentials
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, quota, used, status, notified FROM access_tokens WHERE token = %s",
            (token,),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            raise HTTPException(status_code=401, detail="无效的 Token")

        token_id, email, quota, used, status, notified = row
        used, _, repaired = _repair_legacy_upgraded_usage(
            cur=cur,
            record_id=int(token_id),
            quota=int(quota),
            used=int(used),
            status=str(status or ""),
            notified=bool(notified),
        )
        if repaired:
            conn.commit()
        cur.close()
        return {
            "id": int(token_id),
            "email": email,
            "token": token,
            "quota": int(quota),
            "used": int(used),
            "status": str(status or ""),
        }
    finally:
        put_conn(conn)


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db_pool()
    logger.info("数据库连接池已初始化")
    try:
        yield
    finally:
        close_db_pool()
        logger.info("数据库连接池已关闭")


app = FastAPI(title="TechNews Agent API", docs_url=None, redoc_url=None, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AccessRequest(BaseModel):
    email: EmailStr


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ClarificationResponsePayload(BaseModel):
    kind: str = "clarification_required"
    reason: str
    question: str
    hints: list[str] = []
    original_question: str = ""


class ChatResponse(BaseModel):
    reply: str
    kind: str = "answer"
    clarification: ClarificationResponsePayload | None = None
    citation_urls: list[str] = Field(default_factory=list)
    remaining: int
    quota: int


class QuotaResponse(BaseModel):
    quota: int
    used: int
    remaining: int
    status: str


class SubscriptionRequest(BaseModel):
    email: EmailStr
    name: str | None = None
    sources: list[str] | None = None
    frequency: str = "daily"
    timezone: str = "Asia/Shanghai"


class UnsubscribeRequest(BaseModel):
    email: EmailStr


class SubscriptionResponse(BaseModel):
    email: EmailStr
    name: str | None = None
    is_active: bool
    sources: list[str]
    frequency: str
    timezone: str


def _sse_event(event: str, data: dict) -> str:
    """Encode one SSE event payload."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _status_text_from_progress(payload: dict) -> str | None:
    """Map runtime progress payload to UI-facing status text."""
    stage = str(payload.get("stage", "")).strip().lower()
    if stage == "understanding":
        return "正在理解问题"
    if stage == "retrieving":
        return "正在检索相关新闻"
    if stage == "analyzing":
        return "正在整理关键信息"
    if stage == "finalizing":
        return "正在生成回复"
    return None


def _remaining_after_refund(reservation: dict) -> int:
    quota = int(reservation.get("quota", 0) or 0)
    remaining = int(reservation.get("remaining", 0) or 0) + 1
    if quota > 0:
        remaining = min(remaining, quota)
    return max(remaining, 0)


def _normalize_sources(sources: list[str] | None) -> list[str]:
    allowed_sources = _fetch_active_source_names()

    if not sources:
        return allowed_sources.copy()

    normalized: list[str] = []
    allowed_map = {s.lower(): s for s in allowed_sources}
    for source in sources:
        key = source.strip().lower()
        if key in allowed_map and allowed_map[key] not in normalized:
            normalized.append(allowed_map[key])

    if not normalized:
        raise HTTPException(status_code=400, detail="invalid_sources")

    return normalized


def _row_to_subscription(row) -> SubscriptionResponse:
    sources = row[3] if isinstance(row[3], list) else []
    return SubscriptionResponse(
        email=row[0],
        name=row[1],
        is_active=row[2],
        sources=sources,
        frequency=row[4] or "daily",
        timezone=row[5] or "Asia/Shanghai",
    )


def _fetch_active_source_names() -> list[str]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_name
            FROM public.source_registry
            WHERE is_active = TRUE
            ORDER BY priority ASC, source_name ASC
            """
        )
        rows = cur.fetchall()
        cur.close()
        names = [r[0] for r in rows if r and r[0]]
        if names:
            return names
        return DEFAULT_SUBSCRIPTION_SOURCES.copy()
    except Exception:
        return DEFAULT_SUBSCRIPTION_SOURCES.copy()
    finally:
        put_conn(conn)


def _reserve_quota_or_403(token_info: dict) -> dict:
    """Atomically reserve one quota unit. Returns reservation snapshot."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE access_tokens
            SET used = used + 1
            WHERE id = %s AND used < quota
            RETURNING used, quota, notified
            """,
            (token_info["id"],),
        )
        row = cur.fetchone()
        if not row:
            conn.rollback()
            cur.close()
            raise HTTPException(status_code=403, detail="quota_exhausted")

        used, quota, notified = row
        conn.commit()
        cur.close()
        return {
            "used": int(used),
            "quota": int(quota),
            "remaining": int(quota - used),
            "notified": bool(notified),
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"[{token_info['email']}] 额度预扣失败: {e}")
        raise HTTPException(status_code=500, detail="服务暂时不可用，请稍后重试")
    finally:
        put_conn(conn)


def _refund_reserved_quota(token_info: dict):
    """Best-effort quota refund used when generation fails after reservation."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE access_tokens
            SET used = GREATEST(used - 1, 0)
            WHERE id = %s
            RETURNING used, quota
            """,
            (token_info["id"],),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        if row:
            logger.info(
                "[%s] quota refund applied: used=%s quota=%s",
                token_info["email"],
                int(row[0]),
                int(row[1]),
            )
    except Exception as e:
        conn.rollback()
        logger.error(f"[{token_info['email']}] 额度退款失败: {e}")
    finally:
        put_conn(conn)


def _maybe_send_quota_exhausted_notifications(token_info: dict, reservation: dict):
    """When remaining is 0, mark exhausted+notified once and send mail notifications."""
    if int(reservation.get("remaining", 0)) > 0:
        return

    conn = get_conn()
    row = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE access_tokens
            SET status = 'exhausted', notified = TRUE
            WHERE id = %s AND notified = FALSE AND used >= quota
            RETURNING email
            """,
            (token_info["id"],),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        logger.error(f"[{token_info['email']}] 标记额度耗尽失败: {e}")
        return
    finally:
        put_conn(conn)

    if not row:
        return

    email = str(row[0] or token_info["email"])
    approve_url = _build_signed_approve_url(token_info["id"])
    try:
        if approve_url:
            send_quota_exhausted_to_admin(ADMIN_EMAIL, email, token_info["id"], approve_url)
        else:
            logger.error("APPROVE_LINK_SECRET 未配置，跳过管理员审批邮件发送")
        send_quota_exhausted_to_user(email)
        logger.info(f"额度耗尽通知已发送: {email}")
    except Exception as e:
        logger.error(f"[{email}] 额度耗尽邮件发送失败: {e}")


# ---------------------------------------------------------------------------
# Routes — Token 管理
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/subscription-options")
def get_subscription_options():
    return {
        "sources": _fetch_active_source_names(),
        "frequencies": SUBSCRIPTION_FREQUENCIES,
        "default_timezone": "Asia/Shanghai",
    }


@app.get("/subscriptions", response_model=SubscriptionResponse)
def get_subscription(email: EmailStr):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT email, name, is_active, source_preferences, frequency, timezone
            FROM subscribers
            WHERE email = %s
            LIMIT 1
            """,
            (email,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            raise HTTPException(status_code=404, detail="subscription_not_found")
        return _row_to_subscription(row)
    finally:
        put_conn(conn)


@app.post("/subscriptions", response_model=SubscriptionResponse)
def subscribe_daily_brief(body: SubscriptionRequest):
    sources = _normalize_sources(body.sources)
    frequency = body.frequency.lower()
    if frequency not in SUBSCRIPTION_FREQUENCIES:
        raise HTTPException(status_code=400, detail="invalid_frequency")

    timezone = body.timezone.strip() if body.timezone else "Asia/Shanghai"
    if len(timezone) > 50:
        raise HTTPException(status_code=400, detail="invalid_timezone")

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO subscribers (
                email, name, is_active, source_preferences, frequency, timezone, updated_at
            )
            VALUES (%s, %s, TRUE, %s, %s, %s, NOW())
            ON CONFLICT (email) DO UPDATE
            SET
                name = COALESCE(EXCLUDED.name, subscribers.name),
                is_active = TRUE,
                source_preferences = EXCLUDED.source_preferences,
                frequency = EXCLUDED.frequency,
                timezone = EXCLUDED.timezone,
                updated_at = NOW()
            RETURNING email, name, is_active, source_preferences, frequency, timezone
            """,
            (
                body.email,
                body.name.strip()[:50] if body.name else None,
                Json(sources),
                frequency,
                timezone,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        return _row_to_subscription(row)
    except Exception as e:
        conn.rollback()
        logger.error(f"订阅保存失败: {e}")
        raise HTTPException(status_code=500, detail="subscription_save_failed")
    finally:
        put_conn(conn)


@app.post("/subscriptions/unsubscribe")
def unsubscribe_daily_brief(body: UnsubscribeRequest):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE subscribers
            SET is_active = FALSE, updated_at = NOW()
            WHERE email = %s
            RETURNING id
            """,
            (body.email,),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        if not row:
            raise HTTPException(status_code=404, detail="subscription_not_found")
        return {"message": "unsubscribed"}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"退订失败: {e}")
        raise HTTPException(status_code=500, detail="subscription_unsubscribe_failed")
    finally:
        put_conn(conn)


@app.post("/request-access")
def request_access(body: AccessRequest):
    """访客提交邮箱，自动生成 Token 发送邮件"""
    token = secrets.token_urlsafe(32)
    conn = get_conn()
    try:
        cur = conn.cursor()
        # 检查该邮箱是否已有活跃 Token
        cur.execute(
            """
            SELECT id, token, quota, used, status, notified
            FROM access_tokens
            WHERE email = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (body.email,),
        )
        existing = cur.fetchone()
        if existing:
            ex_id, ex_token, ex_quota, ex_used, ex_status, ex_notified = existing
            ex_used, _, repaired = _repair_legacy_upgraded_usage(
                cur=cur,
                record_id=int(ex_id),
                quota=int(ex_quota),
                used=int(ex_used),
                status=str(ex_status or ""),
                notified=bool(ex_notified),
            )
            if repaired:
                conn.commit()
            if ex_status in ("active", "upgraded") and ex_used < ex_quota:
                cur.close()
                # 重新发送已有 Token
                send_token_email(body.email, ex_token, ex_quota - ex_used)
                return {"message": "Token 已重新发送至邮箱", "request_id": None}

        # 创建新 Token
        cur.execute(
            "INSERT INTO access_tokens (email, token, quota) VALUES (%s, %s, %s) RETURNING id",
            (body.email, token, DEFAULT_QUOTA),
        )
        record_id = cur.fetchone()[0]
        conn.commit()
        cur.close()

        send_token_email(body.email, token, DEFAULT_QUOTA)
        logger.info(f"新 Token 已发放: {body.email} (id={record_id})")
        return {"message": "Token 已发送至邮箱", "request_id": record_id}
    except Exception as e:
        conn.rollback()
        logger.error(f"创建 Token 失败: {e}")
        raise HTTPException(status_code=500, detail="服务异常，请稍后重试")
    finally:
        put_conn(conn)


@app.get("/quota/{token}", response_model=QuotaResponse)
def get_quota(token: str):
    """查询 Token 剩余额度和状态"""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT quota, used, status FROM access_tokens WHERE token = %s",
            (token,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            raise HTTPException(status_code=404, detail="Token 不存在")
        return QuotaResponse(
            quota=row[0], used=row[1], remaining=row[0] - row[1], status=row[2],
        )
    finally:
        put_conn(conn)


@app.get("/approve/{record_id}", response_class=HTMLResponse)
def approve_page(
    record_id: int,
    exp: int | None = Query(default=None),
    sig: str | None = Query(default=None),
):
    """管理员打开审批页：仅展示确认按钮，不执行状态变更。"""
    if not APPROVE_LINK_SECRET.strip():
        logger.error("APPROVE_LINK_SECRET 未配置，拒绝审批请求")
        return HTMLResponse("<h2>服务配置错误</h2><p>审批密钥未配置。</p>", status_code=503)
    if not _is_valid_approve_signature(record_id, exp, sig):
        return HTMLResponse("<h2>审批链接无效或已过期</h2>", status_code=403)
    return HTMLResponse(_render_approve_confirmation_page(record_id, int(exp), str(sig)))


@app.post("/approve/{record_id}", response_class=HTMLResponse)
def approve(
    record_id: int,
    exp: int | None = Query(default=None),
    sig: str | None = Query(default=None),
):
    """管理员确认后执行审批：提升用户额度。"""
    if not APPROVE_LINK_SECRET.strip():
        logger.error("APPROVE_LINK_SECRET 未配置，拒绝审批请求")
        return HTMLResponse("<h2>服务配置错误</h2><p>审批密钥未配置。</p>", status_code=503)
    if not _is_valid_approve_signature(record_id, exp, sig):
        return HTMLResponse("<h2>审批链接无效或已过期</h2>", status_code=403)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT email, status FROM access_tokens WHERE id = %s", (record_id,),
        )
        row = cur.fetchone()
        if not row:
            return HTMLResponse("<h2>记录不存在</h2>", status_code=404)
        email, status = row
        if status == "upgraded":
            return HTMLResponse(f"<h2>已批准过</h2><p>{email} 的额度此前已提升。</p>")

        cur.execute(
            "UPDATE access_tokens SET quota = %s, status = 'upgraded', upgraded_at = NOW() WHERE id = %s",
            (UPGRADED_QUOTA, record_id),
        )
        conn.commit()
        cur.close()

        send_quota_upgraded(email, UPGRADED_QUOTA)
        logger.info(f"{email} 额度已提升至")
        return HTMLResponse(
            f"<h2>{email}</h2><p>额度已提升至 {UPGRADED_QUOTA} 次，通知邮件已发送。</p>"
        )
    except Exception as e:
        conn.rollback()
        logger.error(f"审批处理失败: {e}")
        return HTMLResponse(f"<h2>处理失败</h2><p>{e}</p>", status_code=500)
    finally:
        put_conn(conn)


# ---------------------------------------------------------------------------
# Routes — 对话
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    token_info: dict = Depends(_verify_token),
):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    reservation = _reserve_quota_or_403(token_info)
    request_id = uuid.uuid4().hex

    logger.info(
        "[%s] 对话: request_id=%s message=%s...",
        token_info["email"],
        request_id,
        body.message[:50],
    )
    effective_message, pending = resolve_user_message_with_history_clarification(body.history, body.message)
    if pending is not None:
        logger.info(
            "[%s] clarification follow-up detected: request_id=%s reason=%s",
            token_info["email"],
            request_id,
            pending.reason,
        )
    else:
        effective_message, followup_profile = resolve_user_message_with_followup_context(
            body.history,
            effective_message,
        )
        if bool(followup_profile.get("augmented")):
            logger.info(
                "[%s] follow-up context augmented: request_id=%s score=%.3f decision=%s",
                token_info["email"],
                request_id,
                float(followup_profile.get("score", 0.0)),
                str(followup_profile.get("decision", "fresh")),
            )
    try:
        payload = await asyncio.to_thread(
            generate_response_payload,
            body.history,
            effective_message,
            request_id=request_id,
        )
    except HTTPException:
        _refund_reserved_quota(token_info)
        raise
    except asyncio.CancelledError:
        _refund_reserved_quota(token_info)
        raise
    except AgentGenerationError as e:
        _refund_reserved_quota(token_info)
        logger.warning(
            "[%s] generation blocked: request_id=%s error=%s",
            token_info["email"],
            request_id,
            e,
        )
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _refund_reserved_quota(token_info)
        logger.error(
            "[%s] 处理失败: request_id=%s error=%s",
            token_info["email"],
            request_id,
            e,
        )
        raise HTTPException(status_code=500, detail="服务暂时不可用，请稍后重试")

    kind = str(payload.get("kind", "answer")).strip().lower()
    if kind == "clarification_required":
        _refund_reserved_quota(token_info)
        clarification_payload = payload.get("clarification", {})
        question = str(payload.get("text", "")).strip()
        return ChatResponse(
            reply=question,
            kind="clarification_required",
            clarification=clarification_payload if isinstance(clarification_payload, dict) else None,
            citation_urls=[],
            remaining=_remaining_after_refund(reservation),
            quota=int(reservation["quota"]),
        )

    reply = str(payload.get("text", "")).strip()
    citation_urls = payload.get("citation_urls", [])
    if not isinstance(citation_urls, list):
        citation_urls = []
    _maybe_send_quota_exhausted_notifications(token_info, reservation)
    return ChatResponse(
        reply=reply,
        kind="answer",
        citation_urls=[str(url).strip() for url in citation_urls if str(url).strip()],
        remaining=max(int(reservation["remaining"]), 0),
        quota=int(reservation["quota"]),
    )


@app.post("/chat-stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    token_info: dict = Depends(_verify_token),
):
    """SSE chat endpoint: emits status updates then final payload."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    reservation = _reserve_quota_or_403(token_info)
    request_id = uuid.uuid4().hex

    logger.info(
        "[%s] 对话(流式): request_id=%s message=%s...",
        token_info["email"],
        request_id,
        body.message[:50],
    )
    effective_message, pending = resolve_user_message_with_history_clarification(body.history, body.message)
    if pending is not None:
        logger.info(
            "[%s] clarification follow-up detected(stream): request_id=%s reason=%s",
            token_info["email"],
            request_id,
            pending.reason,
        )
    else:
        effective_message, followup_profile = resolve_user_message_with_followup_context(
            body.history,
            effective_message,
        )
        if bool(followup_profile.get("augmented")):
            logger.info(
                "[%s] follow-up context augmented(stream): request_id=%s score=%.3f decision=%s",
                token_info["email"],
                request_id,
                float(followup_profile.get("score", 0.0)),
                str(followup_profile.get("decision", "fresh")),
            )

    async def event_generator():
        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue[dict] = asyncio.Queue()
        done_event = asyncio.Event()
        result_holder: dict[str, dict] = {}
        error_holder: dict[str, Exception] = {}
        refunded = False
        completed = False
        worker_task: asyncio.Task | None = None

        def refund_once():
            nonlocal refunded
            if refunded:
                return
            refunded = True
            _refund_reserved_quota(token_info)

        def progress_callback(payload: dict[str, str]):
            stage = str(payload.get("stage", "")).strip()
            if not stage:
                return
            loop.call_soon_threadsafe(progress_queue.put_nowait, payload)

        def worker():
            try:
                result_holder["payload"] = generate_response_payload(
                    body.history,
                    effective_message,
                    progress_callback=progress_callback,
                    request_id=request_id,
                )
            except Exception as exc:
                error_holder["exc"] = exc
            finally:
                loop.call_soon_threadsafe(done_event.set)

        try:
            worker_task = asyncio.create_task(asyncio.to_thread(worker))
            last_status = ""

            while True:
                if done_event.is_set() and progress_queue.empty():
                    break
                try:
                    payload = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue

                status_text = _status_text_from_progress(payload)
                if status_text and status_text != last_status:
                    last_status = status_text
                    yield _sse_event("status", {"text": status_text})

            await worker_task

            if "exc" in error_holder:
                exc = error_holder["exc"]
                refund_once()
                if isinstance(exc, AgentGenerationError):
                    logger.warning(
                        "[%s] generation blocked: request_id=%s error=%s",
                        token_info["email"],
                        request_id,
                        exc,
                    )
                    yield _sse_event("error", {"detail": str(exc), "status_code": 503})
                    return
                logger.error(
                    "[%s] 处理失败: request_id=%s error=%s",
                    token_info["email"],
                    request_id,
                    exc,
                )
                yield _sse_event("error", {"detail": "服务暂时不可用，请稍后重试", "status_code": 500})
                return

            payload = result_holder.get("payload", {})
            kind = str(payload.get("kind", "answer")).strip().lower()
            if kind == "clarification_required":
                clarification_payload = payload.get("clarification", {})
                question = str(payload.get("text", "")).strip()
                refund_once()
                completed = True
                yield _sse_event("final", {
                    "kind": "clarification_required",
                    "reply": question,
                    "clarification": clarification_payload if isinstance(clarification_payload, dict) else None,
                    "citation_urls": [],
                    "remaining": _remaining_after_refund(reservation),
                    "quota": int(reservation["quota"]),
                })
                return

            reply = str(payload.get("text", "")).strip()
            citation_urls = payload.get("citation_urls", [])
            if not isinstance(citation_urls, list):
                citation_urls = []
            _maybe_send_quota_exhausted_notifications(token_info, reservation)
            completed = True
            yield _sse_event("final", {
                "kind": "answer",
                "reply": reply,
                "citation_urls": [str(url).strip() for url in citation_urls if str(url).strip()],
                "remaining": max(int(reservation["remaining"]), 0),
                "quota": int(reservation["quota"]),
            })
        finally:
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
                try:
                    await worker_task
                except Exception:
                    pass
            if not completed:
                refund_once()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

