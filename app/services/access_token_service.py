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
    """校验 Bearer Token 并返回 token 记录"""
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
        used, _, repaired = quota_service.repair_legacy_upgraded_usage(
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


def request_access_token(email: str) -> dict:
    """访客提交邮箱，自动生成 Token 发送邮件"""
    token = secrets.token_urlsafe(32)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, token, quota, used, status, notified
            FROM access_tokens
            WHERE email = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email,),
        )
        existing = cur.fetchone()
        if existing:
            ex_id, ex_token, ex_quota, ex_used, ex_status, ex_notified = existing
            ex_used, _, repaired = quota_service.repair_legacy_upgraded_usage(
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
                send_token_email(email, ex_token, ex_quota - ex_used)
                return {"message": "Token 已重新发送至邮箱", "request_id": None}

        cur.execute(
            "INSERT INTO access_tokens (email, token, quota) VALUES (%s, %s, %s) RETURNING id",
            (email, token, quota_service.DEFAULT_QUOTA),
        )
        record_id = cur.fetchone()[0]
        conn.commit()
        cur.close()

        send_token_email(email, token, quota_service.DEFAULT_QUOTA)
        logger.info(f"新 Token 已发放: {email} (id={record_id})")
        return {"message": "Token 已发送至邮箱", "request_id": record_id}
    except Exception as e:
        conn.rollback()
        logger.error(f"创建 Token 失败: {e}")
        raise HTTPException(status_code=500, detail="服务异常，请稍后重试")
    finally:
        put_conn(conn)


def get_quota(token: str) -> QuotaResponse:
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


def approve_access_request(record_id: int) -> tuple[str, int]:
    """管理员确认后执行审批：提升用户额度。"""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT email, status FROM access_tokens WHERE id = %s", (record_id,),
        )
        row = cur.fetchone()
        if not row:
            return "<h2>记录不存在</h2>", 404
        email, status = row
        if status == "upgraded":
            return f"<h2>已批准过</h2><p>{email} 的额度此前已提升。</p>", 200

        cur.execute(
            "UPDATE access_tokens SET quota = %s, status = 'upgraded', upgraded_at = NOW() WHERE id = %s",
            (quota_service.UPGRADED_QUOTA, record_id),
        )
        conn.commit()
        cur.close()

        send_quota_upgraded(email, quota_service.UPGRADED_QUOTA)
        logger.info(f"{email} 额度已提升至")
        return f"<h2>{email}</h2><p>额度已提升至 {quota_service.UPGRADED_QUOTA} 次，通知邮件已发送。</p>", 200
    except Exception as e:
        conn.rollback()
        logger.error(f"审批处理失败: {e}")
        return f"<h2>处理失败</h2><p>{e}</p>", 500
    finally:
        put_conn(conn)
