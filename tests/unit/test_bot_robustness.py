"""Unit tests for Telegram bot transport-layer robustness."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest
from cachetools import TTLCache


@pytest.fixture()
def bot_mod(
    agent_dependency_stubs,  # noqa: ANN001
    telegram_import_stubs,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(sys.modules, "app.bot", raising=False)
    return importlib.import_module("app.bot")


@pytest.fixture(autouse=True)
def reset_bot_runtime_state(bot_mod):  # noqa: ANN001
    env = os.environ.copy()
    original = {
        "conversation_histories": bot_mod.conversation_histories,
        "chat_history_limits": bot_mod.chat_history_limits,
        "chat_request_log": bot_mod.chat_request_log,
    }
    bot_mod.chat_request_log.clear()
    yield
    os.environ.clear()
    os.environ.update(env)
    bot_mod.conversation_histories = original["conversation_histories"]
    bot_mod.chat_history_limits = original["chat_history_limits"]
    bot_mod.chat_request_log = original["chat_request_log"]
    bot_mod.chat_request_log.clear()
    bot_mod.conversation_histories.clear()
    bot_mod.chat_history_limits.clear()


class _FakeMessage:
    def __init__(self, fail_html: bool = False, fail_count: int = 0):
        self.fail_html = fail_html
        self.fail_count = fail_count
        self.calls: list[tuple[str, dict]] = []

    async def reply_text(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("transient")
        if self.fail_html and kwargs.get("parse_mode") == "HTML":
            raise ValueError("bad html")
        return types.SimpleNamespace(message_id=len(self.calls))


class _ThinkingMessage:
    def __init__(self):
        self.edits: list[str] = []

    async def delete(self):
        return None

    async def edit_text(self, text: str):
        self.edits.append(text)
        return None


def test_extract_urls_dedup_and_strip_punctuation(bot_mod) -> None:  # noqa: ANN001
    text = "A https://a.com/x, B https://b.com/y. A2 https://a.com/x"
    assert bot_mod._extract_urls(text) == ["https://a.com/x", "https://b.com/y"]  # pylint: disable=protected-access


def test_format_for_telegram_linkify_uses_title(bot_mod) -> None:  # noqa: ANN001
    text = "- https://a.com/news"
    html = bot_mod._format_for_telegram(  # pylint: disable=protected-access
        text,
        {"https://a.com/news": "中文标题A"},
    )
    assert '<a href="https://a.com/news">中文标题A</a>' in html


def test_format_for_telegram_linkify_fallback_to_url_slug(bot_mod) -> None:  # noqa: ANN001
    text = "- https://a.com/path/to/my-post-name"
    html = bot_mod._format_for_telegram(text, {})  # pylint: disable=protected-access
    assert "my post name" in html


def test_format_for_telegram_source_bullet_with_bracketed_title(bot_mod) -> None:  # noqa: ANN001
    text = "## 来源\n- [1] [[安全] 测试标题](https://a.com/news-1)"
    html = bot_mod._format_for_telegram(text, {})  # pylint: disable=protected-access
    assert '<a href="https://a.com/news-1">[安全] 测试标题</a>' in html
    assert "](<a href=" not in html


def test_format_for_telegram_inline_citation_links_to_sources(bot_mod) -> None:  # noqa: ANN001
    text = (
        "OpenAI update [1], Google update [2]\n\n"
        "## Sources\n"
        "- [1] https://a.com/openai\n"
        "- [2] [Google News](https://b.com/google)"
    )
    html = bot_mod._format_for_telegram(text, {})  # pylint: disable=protected-access
    assert '<a href="https://a.com/openai">[1]</a>' in html
    assert '<a href="https://b.com/google">[2]</a>' in html
    assert 'href="https://agent.trainingcqy.com/[1]"' not in html
    assert 'href="https://agent.trainingcqy.com/[2]"' not in html


def test_render_source_bullet_markdown_link_returns_none_for_non_source(bot_mod) -> None:  # noqa: ANN001
    rendered = bot_mod._render_source_bullet_markdown_link(  # pylint: disable=protected-access
        "普通文本 [标题](https://a.com/x)"
    )
    assert rendered is None


def test_reply_text_with_retry_succeeds_after_transient_failures(bot_mod) -> None:  # noqa: ANN001
    os.environ["BOT_SEND_RETRY_ATTEMPTS"] = "3"
    os.environ["BOT_SEND_RETRY_BASE_SEC"] = "0.1"
    msg = _FakeMessage(fail_count=2)
    with patch.object(bot_mod.asyncio, "sleep", new=AsyncMock()):
        asyncio.run(bot_mod._reply_text_with_retry(msg, "hello"))  # pylint: disable=protected-access
    assert len(msg.calls) == 3


def test_reply_text_with_retry_raises_after_max_attempts(bot_mod) -> None:  # noqa: ANN001
    os.environ["BOT_SEND_RETRY_ATTEMPTS"] = "1"
    os.environ["BOT_SEND_RETRY_BASE_SEC"] = "0.1"
    msg = _FakeMessage(fail_count=5)
    with patch.object(bot_mod.asyncio, "sleep", new=AsyncMock()):
        with pytest.raises(RuntimeError):
            asyncio.run(bot_mod._reply_text_with_retry(msg, "hello"))  # pylint: disable=protected-access
    assert len(msg.calls) == 2


def test_send_reply_uses_agent_payload_and_html_fallback(bot_mod) -> None:  # noqa: ANN001
    os.environ["BOT_SEND_RETRY_ATTEMPTS"] = "0"
    msg = _FakeMessage(fail_html=True)
    text = "## 来源\n- [1] https://a.com/n1"
    title_map = {"https://a.com/n1": "中文A"}

    with patch.object(bot_mod, "get_conn", side_effect=AssertionError("bot should not query DB when sending")):
        asyncio.run(bot_mod._send_reply(msg, text, url_title_map=title_map))  # pylint: disable=protected-access

    assert len(msg.calls) >= 2
    first_text, first_kwargs = msg.calls[0]
    assert first_kwargs.get("parse_mode") == "HTML"
    assert "中文A" in first_text

    second_text, second_kwargs = msg.calls[-1]
    assert second_kwargs.get("parse_mode") is None
    assert "中文A" in second_text


def test_rate_limit_blocks_with_retry_after(bot_mod) -> None:  # noqa: ANN001
    os.environ["BOT_RATE_LIMIT"] = "2"
    os.environ["BOT_RATE_WINDOW_SEC"] = "10"
    with patch.object(bot_mod.time, "time", return_value=1000.0):
        assert bot_mod._consume_chat_rate_token(1) == (True, 0)  # pylint: disable=protected-access
        assert bot_mod._consume_chat_rate_token(1) == (True, 0)  # pylint: disable=protected-access
        allowed, retry_after = bot_mod._consume_chat_rate_token(1)  # pylint: disable=protected-access

    assert not allowed
    assert retry_after >= 1


def test_session_cache_evicts_oldest_when_over_capacity(bot_mod) -> None:  # noqa: ANN001
    bot_mod.conversation_histories = TTLCache(maxsize=2, ttl=3600)
    bot_mod.chat_history_limits = TTLCache(maxsize=2, ttl=3600)

    bot_mod.conversation_histories[1] = []
    bot_mod.conversation_histories[2] = []
    bot_mod.conversation_histories[3] = []

    bot_mod.chat_history_limits[1] = 10
    bot_mod.chat_history_limits[2] = 10
    bot_mod.chat_history_limits[3] = 10

    assert len(bot_mod.conversation_histories) == 2
    assert 1 not in bot_mod.conversation_histories
    assert 1 not in bot_mod.chat_history_limits


def test_rate_log_cache_evicts_oldest_chat(bot_mod) -> None:  # noqa: ANN001
    os.environ["BOT_RATE_LIMIT"] = "10"
    os.environ["BOT_RATE_WINDOW_SEC"] = "60"
    bot_mod.chat_request_log = TTLCache(maxsize=2, ttl=3600)

    with patch.object(bot_mod.time, "time", return_value=1000.0):
        assert bot_mod._consume_chat_rate_token(1) == (True, 0)  # pylint: disable=protected-access
        assert bot_mod._consume_chat_rate_token(2) == (True, 0)  # pylint: disable=protected-access
        assert bot_mod._consume_chat_rate_token(3) == (True, 0)  # pylint: disable=protected-access

    assert len(bot_mod.chat_request_log) == 2
    assert 1 not in bot_mod.chat_request_log


def test_handle_message_generation_error_not_pollute_history(bot_mod) -> None:  # noqa: ANN001
    chat_id = 999
    thinking = _ThinkingMessage()
    update = types.SimpleNamespace(
        effective_chat=types.SimpleNamespace(id=chat_id),
        message=types.SimpleNamespace(text="analyze AI trend in recent 10 days"),
    )
    context = types.SimpleNamespace(bot=None)

    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.object(bot_mod, "_consume_chat_rate_token", return_value=(True, 0)),
        patch.object(bot_mod, "_reply_text_with_retry", new=AsyncMock(return_value=thinking)),
        patch.object(bot_mod.asyncio, "to_thread", new=AsyncMock(side_effect=_to_thread)),
        patch.object(
            bot_mod,
            "generate_response_payload",
            side_effect=bot_mod.AgentGenerationError("抱歉，当前模型服务暂时不可用。"),
        ),
    ):
        asyncio.run(bot_mod.handle_message(update, context))

    assert bot_mod.conversation_histories.get(chat_id, []) == []
    assert thinking.edits == ["抱歉，当前模型服务暂时不可用。"]


def test_handle_message_clarification_roundtrip(bot_mod) -> None:  # noqa: ANN001
    chat_id = 1001
    update_first = types.SimpleNamespace(
        effective_chat=types.SimpleNamespace(id=chat_id),
        message=types.SimpleNamespace(text="帮我分析 AI 行业"),
    )
    update_second = types.SimpleNamespace(
        effective_chat=types.SimpleNamespace(id=chat_id),
        message=types.SimpleNamespace(text="最近30天，只看 TechCrunch，聚焦 OpenAI，做趋势"),
    )
    context = types.SimpleNamespace(bot=None)
    seen_messages: list[str] = []

    clarification_payload = {
        "kind": "clarification_required",
        "reason": "insufficient_evidence",
        "question": "你更想看最近 7 天还是最近 30 天？",
        "hints": [
            "时间范围：最近 7 天 / 最近 30 天",
            "来源范围：HackerNews / TechCrunch / 全部来源",
        ],
        "original_question": "帮我分析 AI 行业",
    }

    def _fake_generate(_history, message):  # noqa: ANN001
        seen_messages.append(str(message))
        if len(seen_messages) == 1:
            return {
                "kind": "clarification_required",
                "text": clarification_payload["question"],
                "clarification": clarification_payload,
                "url_title_map": {},
            }
        return {
            "kind": "answer",
            "text": "最终回答 [1]\n\n## 来源\n- [1] https://example.com/a",
            "url_title_map": {"https://example.com/a": "Example A"},
        }

    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    thinking_1 = _ThinkingMessage()
    thinking_2 = _ThinkingMessage()
    send_reply_mock = AsyncMock()

    with (
        patch.object(bot_mod, "_consume_chat_rate_token", return_value=(True, 0)),
        patch.object(bot_mod, "_reply_text_with_retry", new=AsyncMock(side_effect=[thinking_1, thinking_2])),
        patch.object(bot_mod, "_send_reply", new=send_reply_mock),
        patch.object(bot_mod.asyncio, "to_thread", new=AsyncMock(side_effect=_to_thread)),
        patch.object(bot_mod, "generate_response_payload", side_effect=_fake_generate),
    ):
        asyncio.run(bot_mod.handle_message(update_first, context))
        asyncio.run(bot_mod.handle_message(update_second, context))

    history = bot_mod.conversation_histories.get(chat_id, [])
    assert len(history) == 4
    assert history[1]["kind"] == "clarification_required"
    assert history[1]["clarification"]["reason"] == "insufficient_evidence"
    assert len(seen_messages) == 2
    assert "原问题：帮我分析 AI 行业" in seen_messages[1]
    assert "用户补充澄清：最近30天，只看 TechCrunch，聚焦 OpenAI，做趋势" in seen_messages[1]
    sent_texts = [call.args[1] for call in send_reply_mock.await_args_list]
    assert any("你更想看最近 7 天还是最近 30 天" in text for text in sent_texts)
    assert any("最终回答" in text for text in sent_texts)


def test_handle_message_source_conflict_reason_kept_in_history(bot_mod) -> None:  # noqa: ANN001
    chat_id = 1002
    thinking = _ThinkingMessage()
    update = types.SimpleNamespace(
        effective_chat=types.SimpleNamespace(id=chat_id),
        message=types.SimpleNamespace(text="OpenAI 现在前景怎么样？"),
    )
    context = types.SimpleNamespace(bot=None)

    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.object(bot_mod, "_consume_chat_rate_token", return_value=(True, 0)),
        patch.object(bot_mod, "_reply_text_with_retry", new=AsyncMock(return_value=thinking)),
        patch.object(bot_mod, "_send_reply", new=AsyncMock()),
        patch.object(bot_mod.asyncio, "to_thread", new=AsyncMock(side_effect=_to_thread)),
        patch.object(
            bot_mod,
            "generate_response_payload",
            return_value={
                "kind": "clarification_required",
                "text": "当前多来源结论存在冲突，请确认范围。",
                "clarification": {
                    "kind": "clarification_required",
                    "reason": "source_conflict",
                    "question": "当前多来源结论存在冲突，请确认范围。",
                    "hints": ["来源范围", "时间范围", "分析维度"],
                    "original_question": "OpenAI 现在前景怎么样？",
                },
                "url_title_map": {},
            },
        ),
    ):
        asyncio.run(bot_mod.handle_message(update, context))

    history = bot_mod.conversation_histories.get(chat_id, [])
    assert len(history) == 2
    assert history[1]["kind"] == "clarification_required"
    assert history[1]["clarification"]["reason"] == "source_conflict"
