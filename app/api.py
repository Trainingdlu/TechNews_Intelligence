"""Web API entrypoint: FastAPI routes and response adapters."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "agent" / ".env", override=True)

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import EmailStr

from app import rate_limit, security
from app.schemas import (
    AccessRequest,
    ChatRequest,
    ChatResponse,
    QuotaResponse,
    SubscriptionRequest,
    SubscriptionResponse,
    UnsubscribeRequest,
)
from app.services import access_token_service, chat_service, subscription_service
from services.db import close_db_pool, init_db_pool

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_bearer_scheme = HTTPBearer()


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> dict:
    return access_token_service.verify_token(credentials.credentials)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/subscription-options")
def get_subscription_options():
    return subscription_service.subscription_options()


@app.get("/subscriptions", response_model=SubscriptionResponse)
def get_subscription(email: EmailStr):
    return subscription_service.get_subscription_by_email(str(email))


@app.post("/subscriptions", response_model=SubscriptionResponse)
def subscribe_daily_brief(body: SubscriptionRequest):
    return subscription_service.subscribe_daily_brief(body)


@app.post("/subscriptions/unsubscribe")
def unsubscribe_daily_brief(body: UnsubscribeRequest):
    return subscription_service.unsubscribe_daily_brief(body)


@app.post("/request-access")
def request_access(body: AccessRequest):
    return access_token_service.request_access_token(str(body.email))


@app.get("/quota/{token}", response_model=QuotaResponse)
def get_quota(token: str):
    return access_token_service.get_quota(token)


@app.get("/approve/{record_id}", response_class=HTMLResponse)
def approve_page(
    record_id: int,
    tier: int | None = Query(default=None),
    exp: int | None = Query(default=None),
    sig: str | None = Query(default=None),
):
    """管理员打开审批页：仅展示确认按钮，不执行状态变更。"""
    if not security.APPROVE_LINK_SECRET.strip():
        logger.error("APPROVE_LINK_SECRET 未配置，拒绝审批请求")
        return HTMLResponse("<h2>服务配置错误</h2><p>审批密钥未配置。</p>", status_code=503)
    if not security.is_valid_approve_signature(record_id, tier, exp, sig):
        return HTMLResponse("<h2>审批链接无效或已过期</h2>", status_code=403)
    return HTMLResponse(security.render_approve_confirmation_page(record_id, int(tier), int(exp), str(sig)))


@app.post("/approve/{record_id}", response_class=HTMLResponse)
def approve(
    record_id: int,
    tier: int | None = Query(default=None),
    exp: int | None = Query(default=None),
    sig: str | None = Query(default=None),
):
    """管理员确认后执行审批：提升用户额度。"""
    if not security.APPROVE_LINK_SECRET.strip():
        logger.error("APPROVE_LINK_SECRET 未配置，拒绝审批请求")
        return HTMLResponse("<h2>服务配置错误</h2><p>审批密钥未配置。</p>", status_code=503)
    if not security.is_valid_approve_signature(record_id, tier, exp, sig):
        return HTMLResponse("<h2>审批链接无效或已过期</h2>", status_code=403)

    html, status_code = access_token_service.approve_access_request(record_id, int(tier))
    return HTMLResponse(html, status_code=status_code)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    token_info: dict = Depends(_verify_token),
):
    client_ip = request.client.host if request.client else "unknown"
    rate_limit.check_rate_limit(client_ip)
    request_id = uuid.uuid4().hex
    return await chat_service.handle_chat_request(body, token_info, request_id)


@app.post("/chat-stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    token_info: dict = Depends(_verify_token),
):
    """SSE chat endpoint: emits status updates then final payload."""
    client_ip = request.client.host if request.client else "unknown"
    rate_limit.check_rate_limit(client_ip)
    request_id = uuid.uuid4().hex
    turn = chat_service.prepare_chat_turn(body, token_info, request_id, stream=True)
    return StreamingResponse(
        chat_service.stream_chat_events(turn),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{thread_id}")
def get_history(thread_id: str, token_info: dict = Depends(_verify_token)):
    messages = chat_service.load_thread_history_or_404(
        thread_id,
        token_id=int(token_info["id"]),
        limit=chat_service.WEB_HISTORY_DISPLAY_LIMIT,
    )
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/history/{thread_id}")
def delete_history(thread_id: str, token_info: dict = Depends(_verify_token)):
    chat_service.delete_thread_or_404(thread_id, token_id=int(token_info["id"]))
    return {"status": "deleted", "thread_id": thread_id}
