"""Unit tests for structured outputs in skill modules."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agent.skills import fulltext_batch as fulltext_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.skills import helpers as helpers_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.skills import query_news as query_news_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.skills import sql_builders as sql_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.skills import trend_analysis as trend_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.skills.schemas import QueryNewsSkillInput, TrendAnalysisSkillInput  # noqa: E402  pylint: disable=wrong-import-position


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
    def test_is_recent_timestamp_handles_naive_aware(self) -> None:
        cutoff = datetime(2026, 3, 29, 0, 0, 0, tzinfo=timezone.utc)
        naive_recent = datetime(2026, 3, 29, 1, 0, 0)
        naive_old = datetime(2026, 3, 28, 1, 0, 0)
        self.assertTrue(helpers_mod._is_recent_timestamp(naive_recent, cutoff))
        self.assertFalse(helpers_mod._is_recent_timestamp(naive_old, cutoff - timedelta(hours=1)))

    def test_is_recent_timestamp_normalizes_non_utc_offsets(self) -> None:
        cutoff = datetime(2026, 3, 29, 0, 0, 0)
        shanghai_tz = timezone(timedelta(hours=8))
        aware_local = datetime(2026, 3, 29, 7, 30, 0, tzinfo=shanghai_tz)
        self.assertFalse(helpers_mod._is_recent_timestamp(aware_local, cutoff))

    def test_is_recent_timestamp_accepts_iso8601_string(self) -> None:
        cutoff = datetime(2026, 3, 29, 0, 30, 0)
        iso_value = "2026-03-29T01:00:00Z"
        self.assertTrue(helpers_mod._is_recent_timestamp(iso_value, cutoff))

    def test_extract_time_window_days_supports_week_and_month(self) -> None:
        self.assertEqual(helpers_mod._extract_time_window_days("最近2周相关新闻", default=14), 14)
        self.assertEqual(helpers_mod._extract_time_window_days("past 1 month coverage", default=14), 30)

    def test_expand_topic_terms_for_ai_domain(self) -> None:
        terms = sql_mod._expand_topic_terms("AI")
        lowered = {t.lower() for t in terms}
        self.assertIn("ai", lowered)
        self.assertIn("gpt", lowered)
        self.assertIn("gemini", lowered)

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
            patch.object(query_news_mod, "get_conn", lambda: fake_conn),
            patch.object(query_news_mod, "put_conn", lambda _conn: None),
        ):
            out = query_news_mod.query_news(query="OpenAI", source="TechCrunch", days=7, response_format="json")

        payload = json.loads(out)
        self.assertEqual(payload["tool"], "query_news")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["records"][0]["url"], "https://a.com")

    def test_fulltext_batch_json_mode_empty_candidates(self) -> None:
        with patch.object(fulltext_mod, "lookup_candidates_by_query", lambda **_kwargs: ([], {})):
            out = fulltext_mod.fulltext_batch("OpenAI Voice Engine", response_format="json")
        payload = json.loads(out)
        self.assertEqual(payload["tool"], "fulltext_batch")
        self.assertEqual(payload["status"], "empty")
        self.assertEqual(payload["articles"], [])

    def test_fulltext_batch_json_mode_contains_selected_and_articles(self) -> None:
        candidates = [
            {
                "title": "A title",
                "url": "https://a.com",
                "summary": "A summary",
                "sentiment": "Neutral",
                "source_type": "TechCrunch",
                "created_at": datetime(2026, 3, 28, 10, 0, 0),
                "points": 10,
                "score": 2.345,
            },
            {
                "title": "B title",
                "url": "https://b.com",
                "summary": "B summary",
                "sentiment": "Positive",
                "source_type": "HackerNews",
                "created_at": datetime(2026, 3, 28, 9, 0, 0),
                "points": 8,
                "score": 1.876,
            },
        ]
        with (
            patch.object(
                fulltext_mod,
                "lookup_candidates_by_query",
                lambda **_kwargs: (
                    candidates,
                    {"rerank_mode": "none", "candidate_count": 2, "top_k": 2, "fallback": False},
                ),
            ),
            patch.object(fulltext_mod, "read_news_content", lambda _url: "Full content:\nHello world"),
        ):
            out = fulltext_mod.fulltext_batch("OpenAI recent 14 days", response_format="json")

        payload = json.loads(out)
        self.assertEqual(payload["tool"], "fulltext_batch")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["selected"]), 2)
        self.assertEqual(len(payload["articles"]), 2)
        self.assertEqual(payload["articles"][0]["url"], "https://a.com")

    def test_query_news_skill_returns_envelope_with_evidence(self) -> None:
        query_payload = {
            "tool": "query_news",
            "status": "ok",
            "request": {
                "query": "OpenAI",
                "source": "all",
                "days": 7,
                "category": "",
                "sentiment": "",
                "sort": "time_desc",
                "limit": 3,
            },
            "count": 1,
            "records": [
                {
                    "rank": 1,
                    "source": "TechCrunch",
                    "title": "OpenAI launches model",
                    "title_cn": "[AI] OpenAI 发新模型",
                    "url": "https://example.com/a",
                    "summary": "summary",
                    "sentiment": "Positive",
                    "score": None,
                    "points": 99,
                    "created_at": "2026-03-28T10:30:00",
                }
            ],
        }
        with patch.object(query_news_mod, "query_news", lambda **_kwargs: json.dumps(query_payload)):
            envelope = query_news_mod.query_news_skill(QueryNewsSkillInput(query="OpenAI", days=7, limit=3))

        self.assertEqual(envelope.tool, "query_news")
        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["count"], 1)
        self.assertEqual(len(envelope.evidence), 1)
        self.assertEqual(envelope.evidence[0].url, "https://example.com/a")
        self.assertEqual(envelope.evidence[0].score, 99.0)

    def test_trend_analysis_skill_backfills_evidence(self) -> None:
        trend_payload = {
            "tool": "trend_analysis",
            "status": "ok",
            "request": {"topic": "OpenAI", "window": 7},
            "data": {
                "topic": "OpenAI",
                "window": 7,
                "recent_count": 8,
                "previous_count": 5,
                "count_delta": 3,
                "count_delta_pct": 60.0,
                "avg_points_recent": 50.0,
                "avg_points_previous": 41.0,
                "daily": [],
            },
        }
        evidence_payload = {
            "tool": "query_news",
            "status": "ok",
            "request": {"query": "OpenAI"},
            "count": 1,
            "records": [
                {
                    "rank": 1,
                    "source": "HackerNews",
                    "title": "OpenAI topic article",
                    "title_cn": "",
                    "url": "https://example.com/evidence",
                    "summary": "evidence summary",
                    "sentiment": "Neutral",
                    "points": 42,
                    "created_at": "2026-03-28T08:00:00",
                }
            ],
        }
        with (
            patch.object(trend_mod, "trend_analysis", lambda **_kwargs: json.dumps(trend_payload)),
            patch.object(trend_mod, "query_news", lambda **_kwargs: json.dumps(evidence_payload)),
        ):
            envelope = trend_mod.trend_analysis_skill(TrendAnalysisSkillInput(topic="OpenAI", window=7))

        self.assertEqual(envelope.tool, "trend_analysis")
        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["recent_count"], 8)
        self.assertEqual(len(envelope.evidence), 1)
        self.assertEqual(envelope.evidence[0].url, "https://example.com/evidence")
