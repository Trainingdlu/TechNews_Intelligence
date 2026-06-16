"""Access token request, verification, quota lookup, and approval workflows."""

from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException

from app.schemas import QuotaResponse
from app.services import quota_service
from services.db import get_conn, put_conn
from services.mail import send_quota_upgraded, send_token_email

logger = logging.getLogger(__name__)


def verify_token(token: str) -> dict:
    """Validate a bearer token and return its access-token record."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, email, quota, used, status, notified, tier, unlimited
            FROM access_tokens
            WHERE token = %s
            """,
            (token,),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            raise HTTPException(status_code=401, detail="无效的 Token")

        token_id, email, quota, used, status, notified, tier, unlimited = row
        cur.close()
        return {
            "id": int(token_id),
            "email": email,
            "token": token,
            "quota": int(quota),
            "used": int(used),
            "status": str(status or ""),
            "notified": bool(notified),
            "tier": int(tier),
            "unlimited": bool(unlimited),
        }
    finally:
        put_conn(conn)


def request_access_token(email: str) -> dict:
    """Create a new access token or resend a still-usable existing token."""
    token = secrets.token_urlsafe(32)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, token, quota, used, status, notified, tier, unlimited
            FROM access_tokens
            WHERE email = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email,),
        )
        existing = cur.fetchone()
        if existing:
            ex_id, ex_token, ex_quota, ex_used, ex_status, ex_notified, ex_tier, ex_unlimited = existing
            del ex_id, ex_notified, ex_tier
            existing_is_usable = bool(ex_unlimited) or (
                str(ex_status or "") == "active" and int(ex_used) < int(ex_quota)
            )
            if existing_is_usable:
                cur.close()
                remaining = int(ex_quota) if bool(ex_unlimited) else int(ex_quota) - int(ex_used)
                send_token_email(email, ex_token, remaining)
                return {"message": "Token 已重新发送至邮箱", "request_id": None}

        cur.execute(
            "INSERT INTO access_tokens (email, token, quota) VALUES (%s, %s, %s) RETURNING id",
            (email, token, quota_service.DEFAULT_QUOTA),
        )
        record_id = cur.fetchone()[0]
        conn.commit()
        cur.close()

        send_token_email(email, token, quota_service.DEFAULT_QUOTA)
        logger.info("new access token issued: %s (id=%s)", email, record_id)
        return {"message": "Token 已发送至邮箱", "request_id": record_id}
    except Exception as e:
        conn.rollback()
        logger.error("creating access token failed: %s", e)
        raise HTTPException(status_code=500, detail="服务异常，请稍后重试")
    finally:
        put_conn(conn)


def get_quota(token: str) -> QuotaResponse:
    """Return token quota usage and status."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT quota, used, status, unlimited FROM access_tokens WHERE token = %s",
            (token,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            raise HTTPException(status_code=404, detail="Token 不存在")
        quota, used, status, unlimited = row
        return QuotaResponse(
            quota=int(quota),
            used=int(used),
            remaining=max(int(quota - used), 0),
            status=str(status or ""),
            unlimited=bool(unlimited),
        )
    finally:
        put_conn(conn)


def approve_access_request(record_id: int, tier: int) -> tuple[str, int]:
    """Approve a pending quota request for the tier embedded in the signed link."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT email, tier FROM access_tokens WHERE id = %s",
            (record_id,),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            return "<h2>记录不存在</h2>", 404

        email, current_tier = row
        current_tier = int(current_tier)
        requested_tier = int(tier)
        if current_tier >= quota_service.TOP_TIER:
            cur.close()
            return f"<h2>{email}</h2><p>已达最高额度档位。</p>", 200
        if current_tier != requested_tier:
            cur.close()
            return f"<h2>{email}</h2><p>审批链接已失效或已批准。</p>", 409

        next_tier = current_tier + 1
        new_quota = quota_service.QUOTA_TIERS[next_tier]
        cur.execute(
            """
            UPDATE access_tokens
            SET tier = %s, quota = %s, status = 'active', notified = FALSE, upgraded_at = NOW()
            WHERE id = %s
            """,
            (next_tier, new_quota, record_id),
        )
        conn.commit()
        cur.close()

        send_quota_upgraded(email, new_quota)
        logger.info("%s quota upgraded to tier=%s quota=%s", email, next_tier, new_quota)
        return f"<h2>{email}</h2><p>额度已提升至 {new_quota} 次，通知邮件已发送。</p>", 200
    except Exception as e:
        conn.rollback()
        logger.error("approval handling failed: %s", e)
        return f"<h2>处理失败</h2><p>{e}</p>", 500
    finally:
        put_conn(conn)
