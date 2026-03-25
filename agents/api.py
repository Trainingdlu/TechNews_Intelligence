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

from agent import generate_response
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
DEFAULT_QUOTA = 15
UPGRADED_QUOTA = 50

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


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> dict:
    """校验 Bearer Token 并返回 token 记录"""
    token = credentials.credentials
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, quota, used, status FROM access_tokens WHERE token = %s",
            (token,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            raise HTTPException(status_code=401, detail="无效的 Token")
        return {
            "id": row[0], "email": row[1], "token": token,
            "quota": row[2], "used": row[3], "status": row[4],
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


class QuotaResponse(BaseModel):
    quota: int
    used: int
    remaining: int
    status: str


# ---------------------------------------------------------------------------
# Routes — Token 管理
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/request-access")
def request_access(body: AccessRequest):
    """访客提交邮箱，自动生成 Token 发送邮件"""
    token = secrets.token_urlsafe(32)
    conn = get_conn()
    try:
        cur = conn.cursor()
        # 检查该邮箱是否已有活跃 Token
        cur.execute(
            "SELECT token, quota, used, status FROM access_tokens WHERE email = %s ORDER BY created_at DESC LIMIT 1",
            (body.email,),
        )
        existing = cur.fetchone()
        if existing:
            ex_token, ex_quota, ex_used, ex_status = existing
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
        logger.info(f"审批通过: {email} 额度提升至 {UPGRADED_QUOTA}")
        return HTMLResponse(
            f"<h2>已批准 ✓</h2><p>{email} 的额度已提升至 {UPGRADED_QUOTA} 次，通知邮件已发送。</p>"
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
    except Exception as e:
        logger.error(f"[{token_info['email']}] 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
