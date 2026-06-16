"""Quota reservation, refund, and lifecycle notification workflows."""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException

from app import security
from services.db import get_conn, put_conn
from services.mail import (
    send_quota_capped_to_admin,
    send_quota_exhausted_to_admin,
    send_quota_exhausted_to_user,
)

logger = logging.getLogger(__name__)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
DEFAULT_QUOTA = 10
QUOTA_TIERS = [10, 50, 100, 200]
TOP_TIER = len(QUOTA_TIERS) - 1


def reserve_quota_or_403(token_info: dict) -> dict:
    """Atomically reserve one quota unit. Returns the reservation snapshot."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE access_tokens
            SET used = used + 1
            WHERE id = %s AND (unlimited OR used < quota)
            RETURNING used, quota, tier, notified, unlimited
            """,
            (token_info["id"],),
        )
        row = cur.fetchone()
        if not row:
            conn.rollback()
            cur.close()
            raise HTTPException(status_code=403, detail="quota_exhausted")

        used, quota, tier, notified, unlimited = row
        conn.commit()
        cur.close()
        return {
            "used": int(used),
            "quota": int(quota),
            "remaining": max(int(quota - used), 0),
            "tier": int(tier),
            "notified": bool(notified),
            "unlimited": bool(unlimited),
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[%s] quota reservation failed: %s", token_info["email"], e)
        raise HTTPException(status_code=500, detail="service_unavailable")
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
        logger.error("[%s] quota refund failed: %s", token_info["email"], e)
    finally:
        put_conn(conn)


def maybe_send_quota_exhausted_notifications(token_info: dict, reservation: dict) -> None:
    """Handle one-time pending or capped notifications when a quota tier is exhausted."""
    if bool(reservation.get("unlimited") or token_info.get("unlimited")):
        return
    if int(reservation.get("remaining", 0)) > 0:
        return

    conn = get_conn()
    row = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE access_tokens
            SET status = CASE WHEN tier >= %s THEN 'capped' ELSE 'pending' END
            WHERE id = %s AND used >= quota
            RETURNING email, tier, quota, notified
            """,
            (TOP_TIER, token_info["id"]),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        logger.error("[%s] quota exhaustion status update failed: %s", token_info["email"], e)
        return
    finally:
        put_conn(conn)

    if not row:
        return

    email = str(row[0] or token_info["email"])
    tier = int(row[1])
    quota = int(row[2])
    already_notified = bool(row[3])
    if already_notified:
        return

    admin_email = ADMIN_EMAIL.strip()
    if not admin_email:
        logger.error("ADMIN_EMAIL is empty; quota notification remains pending: %s", email)
        return

    if tier >= TOP_TIER:
        capped_sent = send_quota_capped_to_admin(admin_email, email, quota)
        if not capped_sent:
            logger.error(
                "quota capped admin notice failed; notified remains false: admin=%s user=%s",
                admin_email,
                email,
            )
            return
        mark_quota_exhausted_notified(token_info["id"])
        logger.info("quota capped notice sent: admin=%s user=%s", admin_email, email)
        return

    approve_url = security.build_signed_approve_url(token_info["id"], tier)
    if not approve_url:
        logger.error("approval link secret is empty; quota notification remains pending: %s", email)
        return

    admin_sent = send_quota_exhausted_to_admin(admin_email, email, token_info["id"], approve_url)
    if not admin_sent:
        logger.error(
            "quota approval admin mail failed; notified remains false: admin=%s user=%s",
            admin_email,
            email,
        )
        return

    user_sent = send_quota_exhausted_to_user(email)
    if not user_sent:
        logger.error("quota pending user mail failed: %s", email)
    mark_quota_exhausted_notified(token_info["id"])
    logger.info("quota approval request sent: admin=%s user=%s", admin_email, email)


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
        logger.error("marking quota notification sent failed: %s", e)
    finally:
        put_conn(conn)


def remaining_after_refund(reservation: dict) -> int:
    quota = int(reservation.get("quota", 0) or 0)
    remaining = int(reservation.get("remaining", 0) or 0) + 1
    if quota > 0:
        remaining = min(remaining, quota)
    return max(remaining, 0)


def status_from_reservation(reservation: dict) -> str:
    """Derive the user-facing quota status from a reservation snapshot."""
    if bool(reservation.get("unlimited")):
        return "active"
    if int(reservation.get("remaining", 0)) > 0:
        return "active"
    if int(reservation.get("tier", 0)) >= TOP_TIER:
        return "capped"
    return "pending"
