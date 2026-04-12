"""Persistent conversation thread service.

Transport-agnostic DB primitives for:
- creating threads
- appending conversation messages
- loading history in current `history` shape
- listing recent threads
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from psycopg2.extras import Json

from services.db import db_cursor, db_transaction

DEFAULT_THREAD_CHANNEL = "generic"
DEFAULT_RECENT_THREADS_LIMIT = 20
MAX_RECENT_THREADS_LIMIT = 100
MAX_HISTORY_LOAD_LIMIT = 500


class ConversationThreadNotFoundError(LookupError):
    """Raised when a thread_id cannot be found."""


def create_thread(
    *,
    thread_id: str | None = None,
    channel: str = DEFAULT_THREAD_CHANNEL,
    subject: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a thread and return thread metadata."""
    thread_key = _normalize_thread_id(thread_id, generate_if_missing=True)
    channel_value = _normalize_channel(channel)
    metadata_value = _normalize_metadata(metadata)
    subject_value = _normalize_subject(subject)

    with db_transaction() as (_, cur):
        cur.execute(
            """
            INSERT INTO public.conversation_threads (
                thread_id, channel, subject, metadata
            )
            VALUES (%s, %s, %s, %s)
            RETURNING thread_id, channel, subject, metadata, created_at, updated_at, last_message_at
            """,
            (thread_key, channel_value, subject_value, Json(metadata_value)),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError("failed to create conversation thread")
    return _thread_row_to_dict(row, default_message_count=0)


def append_message(
    thread_id: str,
    message: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one message to an existing thread and update thread timestamp."""
    thread_key = _normalize_thread_id(thread_id, generate_if_missing=False)
    role, parts, payload = _normalize_message(message)
    metadata_value = _normalize_metadata(metadata)

    with db_transaction() as (_, cur):
        cur.execute(
            "SELECT 1 FROM public.conversation_threads WHERE thread_id = %s LIMIT 1",
            (thread_key,),
        )
        if cur.fetchone() is None:
            raise ConversationThreadNotFoundError(f"thread not found: {thread_key}")

        cur.execute(
            """
            INSERT INTO public.conversation_messages (
                thread_id, role, parts, payload, metadata
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, thread_id, role, parts, payload, metadata, created_at
            """,
            (
                thread_key,
                role,
                Json(parts),
                Json(payload),
                Json(metadata_value),
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("failed to append conversation message")

        cur.execute(
            """
            UPDATE public.conversation_threads
            SET last_message_at = %s, updated_at = NOW()
            WHERE thread_id = %s
            """,
            (row[6], thread_key),
        )

    return _message_row_to_dict(row)


def load_history(thread_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load persisted history and map to current `history` structure."""
    thread_key = _normalize_thread_id(thread_id, generate_if_missing=False)
    limit_value = _normalize_optional_limit(limit, max_limit=MAX_HISTORY_LOAD_LIMIT)

    with db_cursor() as (_, cur):
        if limit_value is None:
            cur.execute(
                """
                SELECT payload, role, parts
                FROM public.conversation_messages
                WHERE thread_id = %s
                ORDER BY id ASC
                """,
                (thread_key,),
            )
            rows = cur.fetchall()
        else:
            cur.execute(
                """
                SELECT payload, role, parts
                FROM public.conversation_messages
                WHERE thread_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (thread_key, limit_value),
            )
            rows = cur.fetchall()
            rows.reverse()

    return [_history_item_from_row(row) for row in rows]


def list_recent_threads(
    *,
    limit: int = DEFAULT_RECENT_THREADS_LIMIT,
    channel: str | None = None,
) -> list[dict[str, Any]]:
    """List recent threads ordered by latest activity time."""
    limit_value = _normalize_limit(limit, max_limit=MAX_RECENT_THREADS_LIMIT)
    channel_value = _normalize_channel(channel) if channel is not None else None

    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT
                t.thread_id,
                t.channel,
                t.subject,
                t.metadata,
                t.created_at,
                t.updated_at,
                t.last_message_at,
                COALESCE(stats.message_count, 0) AS message_count
            FROM public.conversation_threads t
            LEFT JOIN (
                SELECT thread_id, COUNT(*)::INTEGER AS message_count
                FROM public.conversation_messages
                GROUP BY thread_id
            ) stats
                ON stats.thread_id = t.thread_id
            WHERE (%s IS NULL OR t.channel = %s)
            ORDER BY COALESCE(t.last_message_at, t.created_at) DESC, t.created_at DESC
            LIMIT %s
            """,
            (channel_value, channel_value, limit_value),
        )
        rows = cur.fetchall()

    return [_thread_row_to_dict(row) for row in rows]


def _history_item_from_row(row: tuple[Any, Any, Any]) -> dict[str, Any]:
    payload, role, parts = row
    if isinstance(payload, dict):
        return copy.deepcopy(payload)
    return {
        "role": str(role or ""),
        "parts": copy.deepcopy(parts if isinstance(parts, list) else []),
    }


def _thread_row_to_dict(
    row: tuple[Any, ...], *, default_message_count: int | None = None,
) -> dict[str, Any]:
    thread_id, channel, subject, metadata, created_at, updated_at, last_message_at = row[:7]
    if len(row) >= 8:
        message_count = int(row[7] or 0)
    else:
        message_count = int(default_message_count or 0)

    return {
        "thread_id": str(thread_id),
        "channel": str(channel),
        "subject": subject,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "created_at": created_at,
        "updated_at": updated_at,
        "last_message_at": last_message_at,
        "message_count": message_count,
    }


def _message_row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    message_id, thread_id, role, parts, payload, metadata, created_at = row
    return {
        "id": int(message_id),
        "thread_id": str(thread_id),
        "role": str(role),
        "parts": parts if isinstance(parts, list) else [],
        "payload": payload if isinstance(payload, dict) else {},
        "metadata": metadata if isinstance(metadata, dict) else {},
        "created_at": created_at,
    }


def _normalize_thread_id(thread_id: str | None, *, generate_if_missing: bool) -> str:
    if thread_id is None:
        if not generate_if_missing:
            raise ValueError("thread_id cannot be empty")
        value = uuid.uuid4().hex
    else:
        value = str(thread_id).strip()
    if not value:
        raise ValueError("thread_id cannot be empty")
    if len(value) > 64:
        raise ValueError("thread_id is too long (max 64)")
    return value


def _normalize_channel(channel: str) -> str:
    value = str(channel).strip()
    if not value:
        raise ValueError("channel cannot be empty")
    if len(value) > 32:
        raise ValueError("channel is too long (max 32)")
    return value


def _normalize_subject(subject: str | None) -> str | None:
    if subject is None:
        return None
    value = str(subject).strip()
    if not value:
        return None
    return value[:256]


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    return copy.deepcopy(metadata)


def _normalize_message(message: dict[str, Any]) -> tuple[str, list[Any], dict[str, Any]]:
    if not isinstance(message, dict):
        raise TypeError("message must be a dict")

    payload = copy.deepcopy(message)
    role = str(payload.get("role", "")).strip()
    if not role:
        raise ValueError("message.role is required")

    parts = payload.get("parts")
    if not isinstance(parts, list):
        parts = []
    else:
        parts = copy.deepcopy(parts)

    return role, parts, payload


def _normalize_optional_limit(value: int | None, *, max_limit: int) -> int | None:
    if value is None:
        return None
    return _normalize_limit(value, max_limit=max_limit)


def _normalize_limit(value: int, *, max_limit: int) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError("limit must be an integer") from exc
    if parsed <= 0:
        raise ValueError("limit must be greater than 0")
    return min(parsed, max_limit)
