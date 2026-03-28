"""Robustness tests for Telegram bot formatting/retry pipeline."""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

AGENTS_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))


def _install_bot_import_stubs() -> None:
    """Install minimal stubs so importing bot.py does not require external deps."""
    if "telegram" not in sys.modules:
        telegram_mod = types.ModuleType("telegram")

        class Update:  # pragma: no cover - structure-only stub
            pass

        class BotCommand:  # pragma: no cover - structure-only stub
            def __init__(self, command: str, description: str):
                self.command = command
                self.description = description

        telegram_mod.Update = Update
        telegram_mod.BotCommand = BotCommand
        sys.modules["telegram"] = telegram_mod

    if "telegram.ext" not in sys.modules:
        ext_mod = types.ModuleType("telegram.ext")

        class _Builder:  # pragma: no cover - structure-only stub
            def token(self, *_args, **_kwargs):
                return self

            def post_init(self, *_args, **_kwargs):
                return self

            def post_shutdown(self, *_args, **_kwargs):
                return self

            def build(self):
                return types.SimpleNamespace(
                    add_handler=lambda *_a, **_k: None,
                    run_polling=lambda *_a, **_k: None,
                    bot=types.SimpleNamespace(set_my_commands=AsyncMock()),
                )

        class _ContextTypes:  # pragma: no cover - structure-only stub
            DEFAULT_TYPE = object

        ext_mod.ApplicationBuilder = _Builder
        ext_mod.CommandHandler = lambda *_a, **_k: None
        ext_mod.MessageHandler = lambda *_a, **_k: None
        ext_mod.ContextTypes = _ContextTypes
        ext_mod.filters = types.SimpleNamespace(TEXT=1, COMMAND=2)
        sys.modules["telegram.ext"] = ext_mod

    if "agent" not in sys.modules:
        agent_mod = types.ModuleType("agent")
        agent_mod.generate_response = lambda _h, _m: "ok"
        sys.modules["agent"] = agent_mod

    if "db" not in sys.modules:
        db_mod = types.ModuleType("db")
        db_mod.init_db_pool = lambda: None
        db_mod.close_db_pool = lambda: None
        db_mod.get_conn = lambda: None
        db_mod.put_conn = lambda _conn: None
        sys.modules["db"] = db_mod


_install_bot_import_stubs()
import bot as bot_mod  # noqa: E402  pylint: disable=wrong-import-position


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


class BotRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.copy()
        bot_mod.chat_request_log.clear()
        bot_mod.url_title_cache.clear()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env)
        bot_mod.chat_request_log.clear()
        bot_mod.url_title_cache.clear()

    def test_extract_urls_dedup_and_strip_punctuation(self) -> None:
        text = "A https://a.com/x, B https://b.com/y. A2 https://a.com/x"
        self.assertEqual(
            bot_mod._extract_urls(text),  # pylint: disable=protected-access
            ["https://a.com/x", "https://b.com/y"],
        )

    def test_strip_existing_source_section(self) -> None:
        text = "## Summary\nBody\n\n## 来源\n- [1] https://a.com\n- [2] https://b.com"
        out = bot_mod._strip_existing_evidence_section(text)  # pylint: disable=protected-access
        self.assertEqual(out, "## Summary\nBody")

    def test_apply_inline_citations_replaces_plain_and_code_url(self) -> None:
        text = "A https://a.com/x\nB `https://b.com/y`"
        out = bot_mod._apply_inline_citations(  # pylint: disable=protected-access
            text,
            ["https://a.com/x", "https://b.com/y"],
        )
        self.assertIn("[1]", out)
        self.assertIn("[2]", out)
        self.assertNotIn("https://a.com/x", out)
        self.assertNotIn("https://b.com/y", out)

    def test_build_source_section_contains_indices(self) -> None:
        out = bot_mod._build_evidence_section(["https://a.com", "https://b.com"])  # pylint: disable=protected-access
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("## "))
        self.assertIn("[1]", out)
        self.assertIn("[2]", out)

    def test_format_for_telegram_linkify_uses_title(self) -> None:
        text = "- https://a.com/news"
        html = bot_mod._format_for_telegram(  # pylint: disable=protected-access
            text,
            {"https://a.com/news": "中文标题A"},
        )
        self.assertIn('<a href="https://a.com/news">中文标题A</a>', html)

    def test_reply_text_with_retry_succeeds_after_transient_failures(self) -> None:
        os.environ["BOT_SEND_RETRY_ATTEMPTS"] = "3"
        os.environ["BOT_SEND_RETRY_BASE_SEC"] = "0.1"
        msg = _FakeMessage(fail_count=2)
        with patch.object(bot_mod.asyncio, "sleep", new=AsyncMock()):
            asyncio.run(bot_mod._reply_text_with_retry(msg, "hello"))  # pylint: disable=protected-access
        self.assertEqual(len(msg.calls), 3)

    def test_reply_text_with_retry_raises_after_max_attempts(self) -> None:
        os.environ["BOT_SEND_RETRY_ATTEMPTS"] = "1"
        os.environ["BOT_SEND_RETRY_BASE_SEC"] = "0.1"
        msg = _FakeMessage(fail_count=5)
        with patch.object(bot_mod.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(RuntimeError):
                asyncio.run(bot_mod._reply_text_with_retry(msg, "hello"))  # pylint: disable=protected-access
        self.assertEqual(len(msg.calls), 2)

    def test_send_reply_rebuilds_sources_and_html_fallback(self) -> None:
        os.environ["BOT_SEND_RETRY_ATTEMPTS"] = "0"
        os.environ["BOT_MAX_CITATION_URLS"] = "8"
        msg = _FakeMessage(fail_html=True)
        text = (
            "## 分析\n"
            "观点见 https://a.com/n1 和 `https://b.com/n2`。\n\n"
            "## 证据来源\n"
            "- https://old.com/x"
        )
        with patch.object(
            bot_mod, "_lookup_url_titles", return_value={"https://a.com/n1": "中文A", "https://b.com/n2": "中文B"}
        ):
            asyncio.run(bot_mod._send_reply(msg, text))  # pylint: disable=protected-access

        # First call attempts HTML, second call is plain fallback.
        self.assertGreaterEqual(len(msg.calls), 2)
        first_text, first_kwargs = msg.calls[0]
        self.assertEqual(first_kwargs.get("parse_mode"), "HTML")
        self.assertIn("[1]", first_text)
        self.assertIn("[2]", first_text)
        self.assertNotIn("old.com", first_text)

        second_text, second_kwargs = msg.calls[-1]
        self.assertTrue("parse_mode" not in second_kwargs)
        self.assertIn("中文A", second_text)
        self.assertIn("中文B", second_text)

    def test_rate_limit_blocks_with_retry_after(self) -> None:
        os.environ["BOT_RATE_LIMIT"] = "2"
        os.environ["BOT_RATE_WINDOW_SEC"] = "10"
        with patch.object(bot_mod.time, "time", return_value=1000.0):
            self.assertEqual(bot_mod._consume_chat_rate_token(1), (True, 0))  # pylint: disable=protected-access
            self.assertEqual(bot_mod._consume_chat_rate_token(1), (True, 0))  # pylint: disable=protected-access
            allowed, retry_after = bot_mod._consume_chat_rate_token(1)  # pylint: disable=protected-access
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)


if __name__ == "__main__":
    unittest.main()
