"""Unit tests for structured tool outputs in tools.py."""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from unittest.mock import patch

try:
    from agents.tests.utils.bootstrap import ensure_agents_on_path
except ModuleNotFoundError:
    from utils.bootstrap import ensure_agents_on_path

ensure_agents_on_path()

import tools as tools_mod  # noqa: E402  pylint: disable=wrong-import-position


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _sql, _params=()):
        return None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        return None


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


class ToolStructuredOutputTests(unittest.TestCase):
    def test_query_news_json_mode_returns_structured_payload(self) -> None:
        rows = [
            (
                "OpenAI launches X",
                "[AI] OpenAI 发布 X",
                "https://a.com",
                "summary A",
                "Positive",
                88,
                "TechCrunch",
                datetime(2026, 3, 28, 10, 30, 0),
            )
        ]
        fake_conn = _FakeConn(rows)
        with (
            patch.object(tools_mod, "get_conn", lambda: fake_conn),
            patch.object(tools_mod, "put_conn", lambda _conn: None),
        ):
            out = tools_mod.query_news(query="OpenAI", source="TechCrunch", days=7, response_format="json")

        payload = json.loads(out)
        self.assertEqual(payload["tool"], "query_news")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["records"][0]["url"], "https://a.com")

    def test_fulltext_batch_json_mode_empty_candidates(self) -> None:
        with patch.object(tools_mod, "_lookup_urls_by_query", lambda **_kwargs: []):
            out = tools_mod.fulltext_batch("OpenAI Voice Engine", response_format="json")
        payload = json.loads(out)
        self.assertEqual(payload["tool"], "fulltext_batch")
        self.assertEqual(payload["status"], "empty")
        self.assertEqual(payload["articles"], [])

    def test_fulltext_batch_json_mode_contains_selected_and_articles(self) -> None:
        candidates = [
            ("A title", "https://a.com", "TechCrunch", datetime(2026, 3, 28, 10, 0, 0), 10, 2.345),
            ("B title", "https://b.com", "HackerNews", datetime(2026, 3, 28, 9, 0, 0), 8, 1.876),
        ]
        with (
            patch.object(tools_mod, "_lookup_urls_by_query", lambda **_kwargs: candidates),
            patch.object(tools_mod, "read_news_content", lambda _url: "Full content:\nHello world"),
        ):
            out = tools_mod.fulltext_batch("OpenAI recent 14 days", response_format="json")

        payload = json.loads(out)
        self.assertEqual(payload["tool"], "fulltext_batch")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["selected"]), 2)
        self.assertEqual(len(payload["articles"]), 2)
        self.assertEqual(payload["articles"][0]["url"], "https://a.com")

