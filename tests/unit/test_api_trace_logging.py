"""API trace request_id propagation tests."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from unittest.mock import patch

import pytest


@pytest.fixture()
def api_mod(
    email_validator_stub,  # noqa: ANN001
    agent_dependency_stubs,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(sys.modules, "app.api", raising=False)
    return importlib.import_module("app.api")


def _fake_request(ip: str = "127.0.0.1"):
    return types.SimpleNamespace(client=types.SimpleNamespace(host=ip))


def test_chat_logs_and_forwards_trace_request_id(
    api_mod,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def _fake_generate_response_payload(_history, _message, progress_callback=None, request_id=None):
        del progress_callback
        captured["request_id"] = str(request_id)
        return {"kind": "answer", "text": "reply-ok"}

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(api_mod, "generate_response_payload", _fake_generate_response_payload)
    monkeypatch.setattr(api_mod.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(api_mod, "_check_rate_limit", lambda _ip: None)
    monkeypatch.setattr(
        api_mod,
        "_reserve_quota_or_403",
        lambda _token: {"remaining": 9, "quota": 10, "notified": False},
    )
    monkeypatch.setattr(api_mod, "_maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)

    body = api_mod.ChatRequest(message="hello", history=[])
    token_info = {"id": 1, "email": "user@example.com"}

    with patch.object(api_mod.logger, "info") as info_log:
        response = asyncio.run(api_mod.chat(body, _fake_request(), token_info))

    assert response.reply == "reply-ok"
    assert info_log.call_count >= 1
    # logger.info(fmt, email, request_id, message_preview)
    logged_request_id = str(info_log.call_args.args[2])
    assert logged_request_id
    assert captured["request_id"] == logged_request_id


def test_chat_stream_logs_and_forwards_trace_request_id(
    api_mod,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def _fake_generate_response_payload(_history, _message, progress_callback=None, request_id=None):
        if callable(progress_callback):
            progress_callback({"stage": "understanding"})
            progress_callback({"stage": "finalizing"})
        captured["request_id"] = str(request_id)
        return {"kind": "answer", "text": "reply-stream"}

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def _collect_stream(response) -> str:
        chunks: list[str] = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode("utf-8"))
            else:
                chunks.append(str(chunk))
        return "".join(chunks)

    monkeypatch.setattr(api_mod, "generate_response_payload", _fake_generate_response_payload)
    monkeypatch.setattr(api_mod.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(api_mod, "_check_rate_limit", lambda _ip: None)
    monkeypatch.setattr(
        api_mod,
        "_reserve_quota_or_403",
        lambda _token: {"remaining": 8, "quota": 10, "notified": False},
    )
    monkeypatch.setattr(api_mod, "_maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    monkeypatch.setattr(api_mod, "_refund_reserved_quota", lambda *_a, **_k: None)

    body = api_mod.ChatRequest(message="stream hello", history=[])
    token_info = {"id": 1, "email": "user@example.com"}

    with patch.object(api_mod.logger, "info") as info_log:
        response = asyncio.run(api_mod.chat_stream(body, _fake_request(), token_info))

    payload = asyncio.run(_collect_stream(response))

    assert "event: final" in payload
    assert info_log.call_count >= 1
    logged_request_id = str(info_log.call_args.args[2])
    assert logged_request_id
    assert captured["request_id"] == logged_request_id
