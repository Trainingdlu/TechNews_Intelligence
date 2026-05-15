"""Unit tests for API security helpers and rate limiter cache controls."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture()
def api_mod(
    email_validator_stub,  # noqa: ANN001
    agent_dependency_stubs,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(sys.modules, "app.api", raising=False)
    return importlib.import_module("app.api")


@pytest.fixture()
def api_state(api_mod):  # noqa: ANN001
    security_mod = api_mod.security
    rate_mod = api_mod.rate_limit
    original = {
        "approve_link_secret": security_mod.APPROVE_LINK_SECRET,
        "approve_link_ttl_sec": security_mod.APPROVE_LINK_TTL_SEC,
        "rate_window": rate_mod.RATE_WINDOW,
        "rate_limit": rate_mod.RATE_LIMIT,
        "request_log": rate_mod.request_log,
    }
    yield security_mod, rate_mod
    security_mod.APPROVE_LINK_SECRET = original["approve_link_secret"]
    security_mod.APPROVE_LINK_TTL_SEC = original["approve_link_ttl_sec"]
    rate_mod.RATE_WINDOW = original["rate_window"]
    rate_mod.RATE_LIMIT = original["rate_limit"]
    rate_mod.request_log = original["request_log"]


def test_approve_signature_validation(api_state) -> None:  # noqa: ANN001
    security_mod, _ = api_state
    security_mod.APPROVE_LINK_SECRET = "unit-test-secret"
    with patch.object(security_mod.time, "time", return_value=1000.0):
        sig = security_mod.build_approve_signature(7, 1060)
        assert security_mod.is_valid_approve_signature(7, 1060, sig)
        assert not security_mod.is_valid_approve_signature(7, 999, sig)
        assert not security_mod.is_valid_approve_signature(7, 1060, sig + "00")


def test_signed_approve_url_contains_exp_and_valid_signature(api_state) -> None:  # noqa: ANN001
    security_mod, _ = api_state
    security_mod.APPROVE_LINK_SECRET = "another-secret"
    security_mod.APPROVE_LINK_TTL_SEC = 120

    with patch.object(security_mod.time, "time", return_value=2000.0):
        url = security_mod.build_signed_approve_url(12)

    assert url is not None
    assert "/approve/12" in str(url)
    assert "exp=2120" in str(url)

    query = str(url).split("?", maxsplit=1)[1]
    parts = dict(item.split("=", maxsplit=1) for item in query.split("&"))
    with patch.object(security_mod.time, "time", return_value=2000.0):
        assert security_mod.is_valid_approve_signature(
            12,
            int(parts["exp"]),
            parts["sig"],
        )


def test_confirmation_page_uses_post_form(api_state) -> None:  # noqa: ANN001
    security_mod, _ = api_state
    html_doc = security_mod.render_approve_confirmation_page(9, 123456, "abc123")
    assert '<form method="post"' in html_doc
    assert '/approve/9?exp=123456&sig=abc123' in html_doc


class _QuotaNotifyCursor:
    def __init__(self, conn):  # noqa: ANN001
        self.conn = conn
        self.row = None

    def execute(self, sql, params):  # noqa: ANN001
        self.conn.executed.append((str(sql), tuple(params or ())))
        if "RETURNING email, notified" in str(sql):
            self.row = self.conn.first_row
        else:
            self.row = None

    def fetchone(self):  # noqa: ANN001
        return self.row

    def close(self) -> None:
        self.conn.closed_cursors += 1


class _QuotaNotifyConn:
    def __init__(self, first_row):  # noqa: ANN001
        self.first_row = first_row
        self.executed: list[tuple[str, tuple]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed_cursors = 0

    def cursor(self):  # noqa: ANN001
        return _QuotaNotifyCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _install_quota_notify_conn(quota_mod, monkeypatch: pytest.MonkeyPatch, first_row):  # noqa: ANN001
    conn = _QuotaNotifyConn(first_row)
    monkeypatch.setattr(quota_mod, "get_conn", lambda: conn)
    monkeypatch.setattr(quota_mod, "put_conn", lambda _conn: None)
    return conn


def test_quota_exhausted_sends_admin_then_marks_notified(api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    quota_mod = api_mod.chat_service.quota_service
    conn = _install_quota_notify_conn(quota_mod, monkeypatch, ("user@example.com", False))
    calls: list[tuple[str, str]] = []
    quota_mod.ADMIN_EMAIL = "admin@example.com"
    monkeypatch.setattr(quota_mod.security, "build_signed_approve_url", lambda _record_id: "https://api.example.com/approve/7")
    monkeypatch.setattr(
        quota_mod,
        "send_quota_exhausted_to_admin",
        lambda admin, user, _record_id, _url: calls.append(("admin", f"{admin}|{user}")) or True,
    )
    monkeypatch.setattr(
        quota_mod,
        "send_quota_exhausted_to_user",
        lambda user: calls.append(("user", user)) or True,
    )

    quota_mod.maybe_send_quota_exhausted_notifications(
        {"id": 7, "email": "fallback@example.com"},
        {"remaining": 0},
    )

    assert calls == [
        ("admin", "admin@example.com|user@example.com"),
        ("user", "user@example.com"),
    ]
    assert any("SET notified = TRUE" in sql for sql, _params in conn.executed)


def test_quota_exhausted_missing_admin_does_not_mark_notified(api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    quota_mod = api_mod.chat_service.quota_service
    conn = _install_quota_notify_conn(quota_mod, monkeypatch, ("user@example.com", False))
    quota_mod.ADMIN_EMAIL = ""
    monkeypatch.setattr(quota_mod.security, "build_signed_approve_url", lambda _record_id: "https://api.example.com/approve/7")
    monkeypatch.setattr(
        quota_mod,
        "send_quota_exhausted_to_admin",
        lambda *_args, **_kwargs: pytest.fail("admin email should not be attempted"),
    )
    monkeypatch.setattr(
        quota_mod,
        "send_quota_exhausted_to_user",
        lambda *_args, **_kwargs: pytest.fail("user waiting email should not be sent without admin email"),
    )

    quota_mod.maybe_send_quota_exhausted_notifications(
        {"id": 7, "email": "fallback@example.com"},
        {"remaining": 0},
    )

    assert not any("SET notified = TRUE" in sql for sql, _params in conn.executed)


def test_quota_exhausted_admin_send_failure_does_not_notify_user_or_mark(api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    quota_mod = api_mod.chat_service.quota_service
    conn = _install_quota_notify_conn(quota_mod, monkeypatch, ("user@example.com", False))
    quota_mod.ADMIN_EMAIL = "admin@example.com"
    monkeypatch.setattr(quota_mod.security, "build_signed_approve_url", lambda _record_id: "https://api.example.com/approve/7")
    monkeypatch.setattr(quota_mod, "send_quota_exhausted_to_admin", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        quota_mod,
        "send_quota_exhausted_to_user",
        lambda *_args, **_kwargs: pytest.fail("user waiting email should not be sent when admin mail fails"),
    )

    quota_mod.maybe_send_quota_exhausted_notifications(
        {"id": 7, "email": "fallback@example.com"},
        {"remaining": 0},
    )

    assert not any("SET notified = TRUE" in sql for sql, _params in conn.executed)


def test_reserve_quota_403_triggers_exhausted_notification_retry(api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    quota_mod = api_mod.chat_service.quota_service
    calls: list[tuple[dict, dict]] = []

    def _raise_quota_exhausted(_token_info):  # noqa: ANN001
        raise HTTPException(status_code=403, detail="quota_exhausted")

    monkeypatch.setattr(quota_mod, "reserve_quota_or_403", _raise_quota_exhausted)
    monkeypatch.setattr(
        quota_mod,
        "maybe_send_quota_exhausted_notifications",
        lambda token_info, reservation: calls.append((token_info, reservation)),
    )

    with pytest.raises(HTTPException):
        quota_mod.reserve_quota_or_notify_403({"id": 7, "email": "user@example.com"})

    assert calls == [({"id": 7, "email": "user@example.com"}, {"remaining": 0})]


def test_rate_limiter_cache_evicts_oldest_key(api_state) -> None:  # noqa: ANN001
    _, rate_mod = api_state
    rate_mod.RATE_LIMIT = 10
    rate_mod.RATE_WINDOW = 60
    rate_mod.request_log = rate_mod.new_rate_log_cache(maxsize=2, ttl=60)

    with patch.object(rate_mod.time, "time", return_value=1000.0):
        rate_mod.check_rate_limit("1.1.1.1")
        rate_mod.check_rate_limit("2.2.2.2")
        rate_mod.check_rate_limit("3.3.3.3")

    assert len(rate_mod.request_log) == 2
    assert "1.1.1.1" not in rate_mod.request_log


def test_rate_limiter_filters_stale_timestamps(api_state) -> None:  # noqa: ANN001
    _, rate_mod = api_state
    rate_mod.RATE_LIMIT = 10
    rate_mod.RATE_WINDOW = 10
    rate_mod.request_log = rate_mod.new_rate_log_cache(maxsize=10, ttl=60)

    with patch.object(rate_mod.time, "time", return_value=1000.0):
        rate_mod.check_rate_limit("9.9.9.9")

    with patch.object(rate_mod.time, "time", return_value=1001.0):
        rate_mod.check_rate_limit("9.9.9.9")

    with patch.object(rate_mod.time, "time", return_value=1012.0):
        rate_mod.check_rate_limit("9.9.9.9")

    assert rate_mod.request_log["9.9.9.9"] == [1012.0]


def _install_token_thread_memory(chat_mod, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001
    threads: dict[str, dict] = {
        "thread-owned-by-a": {
            "owner_token_id": 1,
            "history": [],
        }
    }
    append_calls: list[tuple[str, int, dict]] = []

    def _load_history_for_token(thread_id, token_id, limit=None):  # noqa: ANN001
        del limit
        row = threads.get(str(thread_id))
        if row is None or int(row["owner_token_id"]) != int(token_id):
            raise chat_mod.ConversationThreadNotFoundError(thread_id)
        return list(row["history"])

    def _append_message_for_token(thread_id, token_id, message, metadata=None):  # noqa: ANN001
        del metadata
        row = threads.get(str(thread_id))
        if row is None or int(row["owner_token_id"]) != int(token_id):
            raise chat_mod.ConversationThreadNotFoundError(thread_id)
        row["history"].append(message)
        append_calls.append((str(thread_id), int(token_id), message))
        return {"thread_id": thread_id}

    monkeypatch.setattr(chat_mod, "load_history_for_token", _load_history_for_token)
    monkeypatch.setattr(chat_mod, "append_message_for_token", _append_message_for_token)
    monkeypatch.setattr(chat_mod, "schedule_thread_memory_update", lambda **_kwargs: None)
    return threads, append_calls


def test_chat_returns_404_for_cross_token_thread(api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    token_state = {
        "id": 2,
        "email": "token-b@example.com",
        "quota": 10,
        "used": 0,
        "status": "active",
    }
    api_mod.app.dependency_overrides[api_mod._verify_token] = lambda: dict(token_state)  # pylint: disable=protected-access
    monkeypatch.setattr(api_mod.rate_limit, "check_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod.chat_service.quota_service, "reserve_quota_or_notify_403", lambda *_a, **_k: {"remaining": 4, "quota": 10, "used": 6})
    monkeypatch.setattr(api_mod.chat_service.quota_service, "maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod.chat_service.quota_service, "refund_reserved_quota", lambda *_a, **_k: None)
    _install_token_thread_memory(api_mod.chat_service, monkeypatch)

    client = TestClient(api_mod.app)
    try:
        response = client.post(
            "/chat",
            json={"message": "hello", "thread_id": "thread-owned-by-a"},
            headers={"Authorization": "Bearer test"},
        )
    finally:
        api_mod.app.dependency_overrides.clear()
        client.close()

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation_thread_not_found"}


def test_chat_stream_returns_404_for_cross_token_thread(api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    token_state = {
        "id": 2,
        "email": "token-b@example.com",
        "quota": 10,
        "used": 0,
        "status": "active",
    }
    api_mod.app.dependency_overrides[api_mod._verify_token] = lambda: dict(token_state)  # pylint: disable=protected-access
    monkeypatch.setattr(api_mod.rate_limit, "check_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod.chat_service.quota_service, "reserve_quota_or_notify_403", lambda *_a, **_k: {"remaining": 4, "quota": 10, "used": 6})
    monkeypatch.setattr(api_mod.chat_service.quota_service, "maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod.chat_service.quota_service, "refund_reserved_quota", lambda *_a, **_k: None)
    _install_token_thread_memory(api_mod.chat_service, monkeypatch)

    client = TestClient(api_mod.app)
    try:
        response = client.post(
            "/chat-stream",
            json={"message": "hello", "thread_id": "thread-owned-by-a"},
            headers={"Authorization": "Bearer test"},
        )
    finally:
        api_mod.app.dependency_overrides.clear()
        client.close()

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation_thread_not_found"}


def test_chat_allows_owner_thread_and_persists_turn(api_mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    token_state = {
        "id": 1,
        "email": "token-a@example.com",
        "quota": 10,
        "used": 0,
        "status": "active",
    }
    api_mod.app.dependency_overrides[api_mod._verify_token] = lambda: dict(token_state)  # pylint: disable=protected-access
    monkeypatch.setattr(api_mod.rate_limit, "check_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod.chat_service.quota_service, "reserve_quota_or_notify_403", lambda *_a, **_k: {"remaining": 4, "quota": 10, "used": 6})
    monkeypatch.setattr(api_mod.chat_service.quota_service, "maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod.chat_service.quota_service, "refund_reserved_quota", lambda *_a, **_k: None)
    _, append_calls = _install_token_thread_memory(api_mod.chat_service, monkeypatch)
    monkeypatch.setattr(
        api_mod.chat_service,
        "generate_response_payload",
        lambda *_a, **_k: {"kind": "answer", "text": "ok", "citation_urls": []},
    )

    client = TestClient(api_mod.app)
    try:
        response = client.post(
            "/chat",
            json={"message": "owner hello", "thread_id": "thread-owned-by-a"},
            headers={"Authorization": "Bearer test"},
        )
    finally:
        api_mod.app.dependency_overrides.clear()
        client.close()

    assert response.status_code == 200
    assert response.json()["thread_id"] == "thread-owned-by-a"
    assert len(append_calls) == 2
    assert append_calls[0][1] == 1
    assert append_calls[1][1] == 1
