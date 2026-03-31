"""Web API 入口：FastAPI + Token 自动发放 + 限额管理"""

import os
import secrets
import time
import asyncio
import logging
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from psycopg2.extras import Json

from agent import AgentGenerationError, generate_response
from db import init_db_pool, close_db_pool, get_conn, put_conn
from mail import (
    send_token_email,
    send_quota_exhausted_to_admin,
    send_quota_exhausted_to_user,
    send_quota_upgraded,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://agentapi.trainingcqy.com")
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

_request_log: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str):
    now = time.time()
    timestamps = _request_log[client_ip]
    _request_log[client_ip] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
    _request_log[client_ip].append(now)


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
app = FastAPI(title="TechNews Agent API", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db_pool()
    logger.info("数据库连接池已初始化")


@app.on_event("shutdown")
def shutdown():
    close_db_pool()
    logger.info("数据库连接池已关闭")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AccessRequest(BaseModel):
    email: EmailStr


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
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
def approve(record_id: int):
    """管理员点击邮件链接，提升用户额度"""
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

    # 检查额度
    if token_info["used"] >= token_info["quota"]:
        raise HTTPException(
            status_code=403,
            detail="quota_exhausted",
        )

    logger.info(f"[{token_info['email']}] 对话: {body.message[:50]}...")
    try:
        reply = await asyncio.to_thread(
            generate_response, body.history, body.message,
        )

        # 扣减额度
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE access_tokens SET used = used + 1 WHERE id = %s RETURNING used, quota, notified",
                (token_info["id"],),
            )
            row = cur.fetchone()
            new_used, quota, notified = row
            remaining = quota - new_used

            # 额度耗尽时触发通知
            if remaining <= 0 and not notified:
                cur.execute(
                    "UPDATE access_tokens SET status = 'exhausted', notified = TRUE WHERE id = %s",
                    (token_info["id"],),
                )
                approve_url = f"{API_BASE_URL}/approve/{token_info['id']}"
                send_quota_exhausted_to_admin(
                    ADMIN_EMAIL, token_info["email"], token_info["id"], approve_url,
                )
                send_quota_exhausted_to_user(token_info["email"])
                logger.info(f"额度耗尽通知已发送: {token_info['email']}")

            conn.commit()
            cur.close()
        finally:
            put_conn(conn)

        return ChatResponse(reply=reply, remaining=max(remaining, 0))

    except HTTPException:
        raise
    except AgentGenerationError as e:
        logger.warning(f"[{token_info['email']}] generation blocked: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[{token_info['email']}] 处理失败: {e}")
        raise HTTPException(status_code=500, detail="服务暂时不可用，请稍后重试")

