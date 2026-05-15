"""Quota reservation, refund, and exhaustion notification workflows."""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException

from app import security
from services.db import get_conn, put_conn
from services.mail import (
    send_quota_exhausted_to_admin,
    send_quota_exhausted_to_user,
)

logger = logging.getLogger(__name__)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
DEFAULT_QUOTA = 10
UPGRADED_QUOTA = 50
LEGACY_TRIAL_QUOTA = 15


def repair_legacy_upgraded_usage(
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


def reserve_quota_or_403(token_info: dict) -> dict:
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


def reserve_quota_or_notify_403(token_info: dict) -> dict:
    try:
        return reserve_quota_or_403(token_info)
    except HTTPException as exc:
        if exc.status_code == 403 and exc.detail == "quota_exhausted":
            maybe_send_quota_exhausted_notifications(token_info, {"remaining": 0})
        raise


def refund_reserved_quota(token_info: dict) -> None:
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


def maybe_send_quota_exhausted_notifications(token_info: dict, reservation: dict) -> None:
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
            SET status = 'exhausted'
            WHERE id = %s AND used >= quota
            RETURNING email, notified
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
    already_notified = bool(row[1])
    if already_notified:
        return

    admin_email = ADMIN_EMAIL.strip()
    if not admin_email:
        logger.error("ADMIN_EMAIL 未配置，无法发送管理员审批邮件；notified 保持 false: %s", email)
        return

    approve_url = security.build_signed_approve_url(token_info["id"])
    if not approve_url:
        logger.error("APPROVE_LINK_SECRET 未配置，无法发送管理员审批邮件；notified 保持 false: %s", email)
        return

    admin_sent = send_quota_exhausted_to_admin(admin_email, email, token_info["id"], approve_url)
    if not admin_sent:
        logger.error("管理员审批邮件发送失败；notified 保持 false: admin=%s user=%s", admin_email, email)
        return

    user_sent = send_quota_exhausted_to_user(email)
    if not user_sent:
        logger.error("用户等待审批通知邮件发送失败: %s", email)
    mark_quota_exhausted_notified(token_info["id"])
    logger.info("额度耗尽审批邮件已发送: admin=%s user=%s", admin_email, email)


def mark_quota_exhausted_notified(record_id: int) -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE access_tokens
            SET notified = TRUE
            WHERE id = %s AND used >= quota
            """,
            (record_id,),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        logger.error("标记额度耗尽通知成功状态失败: %s", e)
    finally:
        put_conn(conn)


def remaining_after_refund(reservation: dict) -> int:
    quota = int(reservation.get("quota", 0) or 0)
    remaining = int(reservation.get("remaining", 0) or 0) + 1
    if quota > 0:
        remaining = min(remaining, quota)
    return max(remaining, 0)
