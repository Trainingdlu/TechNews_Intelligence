"""Unit tests for persistent conversation service primitives."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from services import conversations as conv_mod


class _FakeCursor:
    def __init__(
        self,
        *,
        fetchone_values: list[tuple | None] | None = None,
        fetchall_value: list[tuple] | None = None,
    ):
        self.calls: list[tuple[str, tuple]] = []
        self._fetchone_values = list(fetchone_values or [])
        self._fetchall_value = list(fetchall_value or [])

    def execute(self, sql: str, params: tuple = ()):
        self.calls.append((sql, params))

    def fetchone(self):
        if not self._fetchone_values:
            return None
        return self._fetchone_values.pop(0)

    def fetchall(self):
        return list(self._fetchall_value)


def _transaction_context(cursor: _FakeCursor):
    @contextmanager
    def _ctx():
        yield None, cursor

    return _ctx


def _cursor_context(cursor: _FakeCursor):
    @contextmanager
    def _ctx():
        yield None, cursor

    return _ctx


def test_create_thread_returns_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
    cursor = _FakeCursor(
        fetchone_values=[
            ("thread-001", "web", "weekly summary", {"tenant": "acme"}, now, now, None),
        ],
    )
    monkeypatch.setattr(conv_mod, "Json", lambda value: value)
    monkeypatch.setattr(conv_mod, "db_transaction", _transaction_context(cursor))

    out = conv_mod.create_thread(
        thread_id="thread-001",
        channel="web",
        subject="weekly summary",
        metadata={"tenant": "acme"},
    )

    assert out["thread_id"] == "thread-001"
    assert out["channel"] == "web"
    assert out["metadata"] == {"tenant": "acme"}
    assert out["message_count"] == 0
    assert "INSERT INTO public.conversation_threads" in cursor.calls[0][0]
    assert cursor.calls[0][1] == ("thread-001", "web", "weekly summary", {"tenant": "acme"})


def test_append_message_updates_thread_last_message_at(monkeypatch: pytest.MonkeyPatch) -> None:
    created_at = datetime(2026, 4, 12, 13, 0, 0, tzinfo=timezone.utc)
    payload = {"role": "user", "parts": [{"text": "hello"}], "trace_id": "abc"}
    cursor = _FakeCursor(
        fetchone_values=[
            (1,),
            (
                8,
                "thread-001",
                "user",
                [{"text": "hello"}],
                payload,
                {"source": "web"},
                created_at,
            ),
        ],
    )
    monkeypatch.setattr(conv_mod, "Json", lambda value: value)
    monkeypatch.setattr(conv_mod, "db_transaction", _transaction_context(cursor))

    out = conv_mod.append_message(
        "thread-001",
        payload,
        metadata={"source": "web"},
    )

    assert out["id"] == 8
    assert out["thread_id"] == "thread-001"
    assert out["payload"] == payload
    assert "SELECT 1 FROM public.conversation_threads" in cursor.calls[0][0]
    assert "INSERT INTO public.conversation_messages" in cursor.calls[1][0]
    assert cursor.calls[1][1] == (
        "thread-001",
        "user",
        [{"text": "hello"}],
        payload,
        {"source": "web"},
    )
    assert "UPDATE public.conversation_threads" in cursor.calls[2][0]
    assert cursor.calls[2][1] == (created_at, "thread-001")


def test_append_message_raises_when_thread_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(fetchone_values=[None])
    monkeypatch.setattr(conv_mod, "Json", lambda value: value)
    monkeypatch.setattr(conv_mod, "db_transaction", _transaction_context(cursor))

    with pytest.raises(conv_mod.ConversationThreadNotFoundError):
        conv_mod.append_message(
            "missing-thread",
            {"role": "user", "parts": [{"text": "hello"}]},
        )


def test_load_history_returns_lossless_payload_and_role_parts_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_payload = {
        "role": "user",
        "parts": [{"text": "how are you"}],
        "custom": {"a": 1},
    }
    cursor = _FakeCursor(
        fetchall_value=[
            (stored_payload, "user", [{"text": "how are you"}]),
            (None, "model", [{"text": "fine"}]),
        ],
    )
    monkeypatch.setattr(conv_mod, "db_cursor", _cursor_context(cursor))

    history = conv_mod.load_history("thread-001")

    assert history == [
        stored_payload,
        {"role": "model", "parts": [{"text": "fine"}]},
    ]
    assert "ORDER BY id ASC" in cursor.calls[0][0]
    assert cursor.calls[0][1] == ("thread-001",)


def test_load_history_with_limit_preserves_chronological_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Query returns DESC rows when limit is set; service should reverse back to ASC order.
    cursor = _FakeCursor(
        fetchall_value=[
            ({"role": "model", "parts": [{"text": "msg-2"}]}, "model", [{"text": "msg-2"}]),
            ({"role": "user", "parts": [{"text": "msg-1"}]}, "user", [{"text": "msg-1"}]),
        ],
    )
    monkeypatch.setattr(conv_mod, "db_cursor", _cursor_context(cursor))

    history = conv_mod.load_history("thread-001", limit=2)

    assert history == [
        {"role": "user", "parts": [{"text": "msg-1"}]},
        {"role": "model", "parts": [{"text": "msg-2"}]},
    ]
    assert "ORDER BY id DESC" in cursor.calls[0][0]
    assert cursor.calls[0][1] == ("thread-001", 2)


def test_list_recent_threads_maps_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 12, 14, 0, 0, tzinfo=timezone.utc)
    cursor = _FakeCursor(
        fetchall_value=[
            ("thread-a", "web", "A", {"tenant": "x"}, now, now, now, 3),
            ("thread-b", "web", "B", {"tenant": "x"}, now, now, None, 0),
        ],
    )
    monkeypatch.setattr(conv_mod, "db_cursor", _cursor_context(cursor))

    rows = conv_mod.list_recent_threads(limit=2, channel="web")

    assert [r["thread_id"] for r in rows] == ["thread-a", "thread-b"]
    assert rows[0]["message_count"] == 3
    assert rows[1]["message_count"] == 0
    assert cursor.calls[0][1] == ("web", "web", 2)
