"""Focused tests for API service modules split out of app.api."""

from __future__ import annotations

import pytest

from app.schemas import SubscriptionRequest, UnsubscribeRequest
from app.services import subscription_service


class _Cursor:
    def __init__(self, conn):  # noqa: ANN001
        self.conn = conn

    def execute(self, sql, params=None):  # noqa: ANN001
        self.conn.executed.append((str(sql), tuple(params or ())))

    def fetchone(self):  # noqa: ANN001
        return self.conn.fetchone_row

    def fetchall(self):  # noqa: ANN001
        return self.conn.fetchall_rows

    def close(self) -> None:
        self.conn.closed_cursors += 1


class _Conn:
    def __init__(self, *, fetchone_row=None, fetchall_rows=None):  # noqa: ANN001
        self.fetchone_row = fetchone_row
        self.fetchall_rows = list(fetchall_rows or [])
        self.executed: list[tuple[str, tuple]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed_cursors = 0

    def cursor(self):  # noqa: ANN001
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _install_conn(monkeypatch: pytest.MonkeyPatch, conn: _Conn) -> _Conn:
    monkeypatch.setattr(subscription_service, "get_conn", lambda: conn)
    monkeypatch.setattr(subscription_service, "put_conn", lambda _conn: None)
    return conn


def test_normalize_sources_uses_active_sources_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subscription_service, "fetch_active_source_names", lambda: ["HackerNews", "TechCrunch"])

    assert subscription_service.normalize_sources(["hackernews", "TECHCRUNCH", "hackernews"]) == [
        "HackerNews",
        "TechCrunch",
    ]


def test_get_subscription_response_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_conn(
        monkeypatch,
        _Conn(fetchone_row=("user@example.com", "Alice", True, ["HackerNews"], "daily", "Asia/Shanghai")),
    )

    result = subscription_service.get_subscription_by_email("user@example.com")

    assert result.email == "user@example.com"
    assert result.name == "Alice"
    assert result.is_active is True
    assert result.sources == ["HackerNews"]
    assert result.frequency == "daily"
    assert result.timezone == "Asia/Shanghai"


def test_subscribe_daily_brief_response_compatibility(
    email_validator_stub,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _install_conn(
        monkeypatch,
        _Conn(fetchone_row=("user@example.com", "Alice", True, ["HackerNews"], "daily", "Asia/Shanghai")),
    )
    monkeypatch.setattr(subscription_service, "normalize_sources", lambda _sources: ["HackerNews"])

    result = subscription_service.subscribe_daily_brief(
        SubscriptionRequest(
            email="user@example.com",
            name=" Alice ",
            sources=["hackernews"],
            frequency="daily",
            timezone="Asia/Shanghai",
        )
    )

    assert result.email == "user@example.com"
    assert result.name == "Alice"
    assert result.sources == ["HackerNews"]
    assert conn.commits == 1
    assert conn.executed[0][1][1] == "Alice"


def test_unsubscribe_daily_brief_response_compatibility(
    email_validator_stub,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _install_conn(monkeypatch, _Conn(fetchone_row=(1,)))

    result = subscription_service.unsubscribe_daily_brief(UnsubscribeRequest(email="user@example.com"))

    assert result == {"message": "unsubscribed"}
    assert conn.commits == 1
