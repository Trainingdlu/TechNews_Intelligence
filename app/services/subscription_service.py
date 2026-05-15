"""Daily brief subscription use cases."""

from __future__ import annotations

import logging

from fastapi import HTTPException
from psycopg2.extras import Json

from app.schemas import SubscriptionRequest, SubscriptionResponse, UnsubscribeRequest
from services.db import get_conn, put_conn

logger = logging.getLogger(__name__)

DEFAULT_SUBSCRIPTION_SOURCES = [
    "HackerNews",
    "TechCrunch",
]
SUBSCRIPTION_FREQUENCIES = ["daily"]
DEFAULT_TIMEZONE = "Asia/Shanghai"


def fetch_active_source_names() -> list[str]:
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


def normalize_sources(sources: list[str] | None) -> list[str]:
    allowed_sources = fetch_active_source_names()

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


def row_to_subscription(row) -> SubscriptionResponse:
    sources = row[3] if isinstance(row[3], list) else []
    return SubscriptionResponse(
        email=row[0],
        name=row[1],
        is_active=row[2],
        sources=sources,
        frequency=row[4] or "daily",
        timezone=row[5] or DEFAULT_TIMEZONE,
    )


def subscription_options() -> dict:
    return {
        "sources": fetch_active_source_names(),
        "frequencies": SUBSCRIPTION_FREQUENCIES,
        "default_timezone": DEFAULT_TIMEZONE,
    }


def get_subscription_by_email(email: str) -> SubscriptionResponse:
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
        return row_to_subscription(row)
    finally:
        put_conn(conn)


def subscribe_daily_brief(body: SubscriptionRequest) -> SubscriptionResponse:
    sources = normalize_sources(body.sources)
    frequency = body.frequency.lower()
    if frequency not in SUBSCRIPTION_FREQUENCIES:
        raise HTTPException(status_code=400, detail="invalid_frequency")

    timezone = body.timezone.strip() if body.timezone else DEFAULT_TIMEZONE
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
        return row_to_subscription(row)
    except Exception as e:
        conn.rollback()
        logger.error(f"订阅保存失败: {e}")
        raise HTTPException(status_code=500, detail="subscription_save_failed")
    finally:
        put_conn(conn)


def unsubscribe_daily_brief(body: UnsubscribeRequest) -> dict:
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
