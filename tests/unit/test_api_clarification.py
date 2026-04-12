"""Unit tests for /chat and /chat-stream clarification behavior."""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import MagicMock

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
def api_client(
    api_mod,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
):
    api_mod.app.dependency_overrides[api_mod._verify_token] = lambda: {  # pylint: disable=protected-access
        "id": 1,
        "email": "unit@test.local",
        "quota": 10,
        "used": 0,
        "status": "active",
    }
    monkeypatch.setattr(api_mod, "_check_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(
        api_mod,
        "_reserve_quota_or_403",
        lambda *_a, **_k: {"remaining": 4, "quota": 10, "used": 6},
    )
    monkeypatch.setattr(api_mod, "_maybe_send_quota_exhausted_notifications", lambda *_a, **_k: None)
    refund_spy = MagicMock()
    monkeypatch.setattr(api_mod, "_refund_reserved_quota", refund_spy)

    client = TestClient(api_mod.app)
    try:
        yield client, api_mod, refund_spy
    finally:
        api_mod.app.dependency_overrides.clear()
        client.close()


def _parse_sse_events(raw_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in str(raw_text or "").strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        payload_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", maxsplit=1)[1].strip()
            elif line.startswith("data:"):
                payload_lines.append(line.split(":", maxsplit=1)[1].strip())
        if not payload_lines:
            continue
        events.append((event_name, json.loads("\n".join(payload_lines))))
    return events


def _clarification_payload() -> dict:
    return {
        "kind": "clarification_required",
        "text": "你更想看最近 7 天还是最近 30 天？",
        "url_title_map": {},
        "clarification": {
            "kind": "clarification_required",
            "reason": "insufficient_evidence",
            "question": "你更想看最近 7 天还是最近 30 天？",
            "hints": [
                "时间范围：最近 7 天 / 最近 30 天",
                "来源范围：HackerNews / TechCrunch / 全部来源",
            ],
            "original_question": "帮我分析 AI 行业",
        },
    }


def _source_conflict_payload() -> dict:
    return {
        "kind": "clarification_required",
        "text": "当前多来源结论存在冲突，请确认范围。",
        "url_title_map": {},
        "clarification": {
            "kind": "clarification_required",
            "reason": "source_conflict",
            "question": "当前多来源结论存在冲突，请确认范围。",
            "hints": [
                "来源范围：仅 HackerNews / 仅 TechCrunch / 双来源对比",
                "时间范围：最近 7 天 / 14 天 / 30 天",
                "分析维度：趋势 / 对比 / 时间线 / 格局",
            ],
            "original_question": "OpenAI 现在前景怎么样？",
        },
    }


def test_chat_returns_clarification_payload_not_503(api_client) -> None:  # noqa: ANN001
    client, api_mod, refund_spy = api_client
    api_mod.generate_response_payload = lambda *_a, **_k: _clarification_payload()

    response = client.post(
        "/chat",
        json={"message": "帮我分析 AI 行业", "history": []},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "clarification_required"
    assert data["clarification"]["reason"] == "insufficient_evidence"
    assert data["reply"] == "你更想看最近 7 天还是最近 30 天？"
    assert refund_spy.call_count == 1


def test_chat_stream_carries_source_conflict_reason(api_client) -> None:  # noqa: ANN001
    client, api_mod, _ = api_client
    api_mod.generate_response_payload = lambda *_a, **_k: _source_conflict_payload()

    response = client.post(
        "/chat-stream",
        json={"message": "OpenAI 现在前景怎么样？", "history": []},
        headers={"Authorization": "Bearer test"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    final_events = [payload for name, payload in events if name == "final"]
    assert final_events
    assert final_events[-1]["kind"] == "clarification_required"
    assert final_events[-1]["clarification"]["reason"] == "source_conflict"


def test_chat_stream_emits_final_clarification_event(api_client) -> None:  # noqa: ANN001
    client, api_mod, refund_spy = api_client

    def _fake_generate(_history, _message, progress_callback=None, request_id=None):  # noqa: ANN001
        if callable(progress_callback):
            progress_callback({"stage": "understanding"})
        return _clarification_payload()

    api_mod.generate_response_payload = _fake_generate

    response = client.post(
        "/chat-stream",
        json={"message": "帮我分析 AI 行业", "history": []},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_names = [name for name, _ in events]
    assert "error" not in event_names
    final_events = [payload for name, payload in events if name == "final"]
    assert final_events
    assert final_events[-1]["kind"] == "clarification_required"
    assert refund_spy.call_count == 1


def test_clarification_followup_retries_with_merged_message(api_client) -> None:  # noqa: ANN001
    client, api_mod, refund_spy = api_client
    seen_messages: list[str] = []

    def _fake_generate(_history, message, progress_callback=None, request_id=None):  # noqa: ANN001
        seen_messages.append(str(message))
        if len(seen_messages) == 1:
            return _clarification_payload()
        return {
            "kind": "answer",
            "text": "分析结论 [1]\n\n## 来源\n- [1] https://example.com/a",
            "url_title_map": {"https://example.com/a": "Example A"},
        }

    api_mod.generate_response_payload = _fake_generate

    first = client.post(
        "/chat",
        json={"message": "帮我分析 AI 行业", "history": []},
        headers={"Authorization": "Bearer test"},
    )
    assert first.status_code == 200
    clarification = first.json()["clarification"]

    history = [
        {"role": "user", "parts": [{"text": "帮我分析 AI 行业"}]},
        {
            "role": "model",
            "kind": "clarification_required",
            "clarification": clarification,
            "parts": [{"text": clarification["question"]}],
        },
    ]
    followup = "最近 30 天，只看 TechCrunch，聚焦 OpenAI，做趋势对比"
    second = client.post(
        "/chat",
        json={"message": followup, "history": history},
        headers={"Authorization": "Bearer test"},
    )

    assert second.status_code == 200
    second_data = second.json()
    assert second_data["kind"] == "answer"
    assert second_data["reply"].startswith("分析结论")
    assert len(seen_messages) == 2
    assert "原问题：帮我分析 AI 行业" in seen_messages[1]
    assert f"用户补充澄清：{followup}" in seen_messages[1]
    assert refund_spy.call_count == 1


def test_chat_normal_answer_path_unchanged(api_client) -> None:  # noqa: ANN001
    client, api_mod, refund_spy = api_client
    api_mod.generate_response_payload = lambda *_a, **_k: {
        "kind": "answer",
        "text": "正常回答 [1]\n\n## 来源\n- [1] https://example.com/n",
        "url_title_map": {"https://example.com/n": "Example N"},
    }

    response = client.post(
        "/chat",
        json={"message": "最近 AI 动态", "history": []},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "answer"
    assert data["reply"].startswith("正常回答")
    assert data["clarification"] is None
    assert refund_spy.call_count == 0
