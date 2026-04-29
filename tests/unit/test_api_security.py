"""Unit tests for API security helpers and rate limiter cache controls."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest
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
    original = {
        "approve_link_secret": api_mod.APPROVE_LINK_SECRET,
        "approve_link_ttl_sec": api_mod.APPROVE_LINK_TTL_SEC,
        "rate_window": api_mod.RATE_WINDOW,
        "rate_limit": api_mod.RATE_LIMIT,
        "request_log": api_mod._request_log,  # pylint: disable=protected-access
    }
    yield api_mod
    api_mod.APPROVE_LINK_SECRET = original["approve_link_secret"]
    api_mod.APPROVE_LINK_TTL_SEC = original["approve_link_ttl_sec"]
    api_mod.RATE_WINDOW = original["rate_window"]
    api_mod.RATE_LIMIT = original["rate_limit"]
    api_mod._request_log = original["request_log"]  # pylint: disable=protected-access


def test_approve_signature_validation(api_state) -> None:  # noqa: ANN001
    api_state.APPROVE_LINK_SECRET = "unit-test-secret"
    with patch.object(api_state.time, "time", return_value=1000.0):
        sig = api_state._build_approve_signature(7, 1060)  # pylint: disable=protected-access
        assert api_state._is_valid_approve_signature(7, 1060, sig)  # pylint: disable=protected-access
        assert not api_state._is_valid_approve_signature(7, 999, sig)  # pylint: disable=protected-access
        assert not api_state._is_valid_approve_signature(7, 1060, sig + "00")  # pylint: disable=protected-access


def test_signed_approve_url_contains_exp_and_valid_signature(api_state) -> None:  # noqa: ANN001
    api_state.APPROVE_LINK_SECRET = "another-secret"
    api_state.APPROVE_LINK_TTL_SEC = 120

    with patch.object(api_state.time, "time", return_value=2000.0):
        url = api_state._build_signed_approve_url(12)  # pylint: disable=protected-access

    assert url is not None
    assert "/approve/12" in str(url)
    assert "exp=2120" in str(url)

    query = str(url).split("?", maxsplit=1)[1]
    parts = dict(item.split("=", maxsplit=1) for item in query.split("&"))
    with patch.object(api_state.time, "time", return_value=2000.0):
        assert api_state._is_valid_approve_signature(  # pylint: disable=protected-access
            12,
            int(parts["exp"]),
            parts["sig"],
        )


def test_confirmation_page_uses_post_form(api_state) -> None:  # noqa: ANN001
    html_doc = api_state._render_approve_confirmation_page(9, 123456, "abc123")  # pylint: disable=protected-access
    assert '<form method="post"' in html_doc
    assert '/approve/9?exp=123456&sig=abc123' in html_doc


def test_rate_limiter_cache_evicts_oldest_key(api_state) -> None:  # noqa: ANN001
    api_state.RATE_LIMIT = 10
    api_state.RATE_WINDOW = 60
    api_state._request_log = api_state._new_rate_log_cache(maxsize=2, ttl=60)  # pylint: disable=protected-access

    with patch.object(api_state.time, "time", return_value=1000.0):
        api_state._check_rate_limit("1.1.1.1")  # pylint: disable=protected-access
        api_state._check_rate_limit("2.2.2.2")  # pylint: disable=protected-access
        api_state._check_rate_limit("3.3.3.3")  # pylint: disable=protected-access

    assert len(api_state._request_log) == 2  # pylint: disable=protected-access
    assert "1.1.1.1" not in api_state._request_log  # pylint: disable=protected-access


def test_rate_limiter_filters_stale_timestamps(api_state) -> None:  # noqa: ANN001
    api_state.RATE_LIMIT = 10
    api_state.RATE_WINDOW = 10
    api_state._request_log = api_state._new_rate_log_cache(maxsize=10, ttl=60)  # pylint: disable=protected-access

    with patch.object(api_state.time, "time", return_value=1000.0):
        api_state._check_rate_limit("9.9.9.9")  # pylint: disable=protected-access

    with patch.object(api_state.time, "time", return_value=1001.0):
        api_state._check_rate_limit("9.9.9.9")  # pylint: disable=protected-access

    with patch.object(api_state.time, "time", return_value=1012.0):
        api_state._check_rate_limit("9.9.9.9")  # pylint: disable=protected-access

    assert api_state._request_log["9.9.9.9"] == [1012.0]  # pylint: disable=protected-access


def _install_token_thread_memory(api_mod, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001
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
            raise api_mod.ConversationThreadNotFoundError(thread_id)
        return list(row["history"])

    def _append_message_for_token(thread_id, token_id, message, metadata=None):  # noqa: ANN001
        del metadata
        row = threads.get(str(thread_id))
        if row is None or int(row["owner_token_id"]) != int(token_id):
            raise api_mod.ConversationThreadNotFoundError(thread_id)
        row["history"].append(message)
        append_calls.append((str(thread_id), int(token_id), message))
        return {"thread_id": thread_id}

    monkeypatch.setattr(api_mod, "load_history_for_token", _load_history_for_token)
    monkeypatch.setattr(api_mod, "append_message_for_token", _append_message_for_token)
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
    monkeypatch.setattr(api_mod, "_check_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod, "_reserve_quota_or_403", lambda *_a, **_k: {"remaining": 4, "quota": 10, "used": 6})
    monkeypatch.setattr(api_mod, "_maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod, "_refund_reserved_quota", lambda *_a, **_k: None)
    _install_token_thread_memory(api_mod, monkeypatch)

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
    monkeypatch.setattr(api_mod, "_check_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod, "_reserve_quota_or_403", lambda *_a, **_k: {"remaining": 4, "quota": 10, "used": 6})
    monkeypatch.setattr(api_mod, "_maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod, "_refund_reserved_quota", lambda *_a, **_k: None)
    _install_token_thread_memory(api_mod, monkeypatch)

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
    monkeypatch.setattr(api_mod, "_check_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod, "_reserve_quota_or_403", lambda *_a, **_k: {"remaining": 4, "quota": 10, "used": 6})
    monkeypatch.setattr(api_mod, "_maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod, "_refund_reserved_quota", lambda *_a, **_k: None)
    _, append_calls = _install_token_thread_memory(api_mod, monkeypatch)
    monkeypatch.setattr(
        api_mod,
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
