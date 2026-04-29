"""Unit tests for structured outputs in tool modules."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agent.core.tool_catalog import iter_tool_definitions
from agent.core.tool_contracts import ToolEnvelope, build_tool_empty_envelope, build_tool_error_envelope
from agent.tools import analyze_landscape as landscape_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.tools import build_timeline as timeline_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.tools import compare_sources as compare_sources_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.tools import compare_topics as compare_topics_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.tools import fulltext_batch as fulltext_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.tools import helpers as helpers_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.tools import query_news as query_news_mod  # noqa: E402  pylint: disable=wrong-import-position

from agent.tools import trend_analysis as trend_mod  # noqa: E402  pylint: disable=wrong-import-position
from agent.tools.schemas import (  # noqa: E402  pylint: disable=wrong-import-position
    AnalyzeLandscapeToolInput,
    BuildTimelineToolInput,
    CompareSourcesToolInput,
    CompareTopicsToolInput,
    QueryNewsToolInput,
    TrendAnalysisToolInput,
)


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


class _SequenceCursor:
    def __init__(self, rowsets):
        self._rowsets = list(rowsets)
        self._current = []

    def execute(self, _sql, _params=()):
        self._current = self._rowsets.pop(0) if self._rowsets else []

    def fetchall(self):
        return list(self._current)

    def fetchone(self):
        return self._current[0] if self._current else None

    def close(self):
        return None


class _SequenceConn:
    def __init__(self, rowsets):
        self._cursor = _SequenceCursor(rowsets)

    def cursor(self):
        return self._cursor


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
        self.assertEqual(helpers_mod._extract_time_window_days("最近 1 周相关新闻", default=14), 7)
        self.assertEqual(helpers_mod._extract_time_window_days("past 1 month coverage", default=14), 30)



    def test_query_news_json_mode_returns_structured_payload(self) -> None:
        rows = [
            (
                "OpenAI launches X",
                "[AI] OpenAI 鍙戝竷 X",
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

    def test_query_news_tool_returns_envelope_with_evidence(self) -> None:
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
                    "title_cn": "[AI] OpenAI 鍙戞柊妯″瀷",
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
            envelope = query_news_mod.query_news_tool(QueryNewsToolInput(query="OpenAI", days=7, limit=3))

        self.assertEqual(envelope.tool, "query_news")
        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["count"], 1)
        self.assertEqual(len(envelope.evidence), 1)
        self.assertEqual(envelope.evidence[0].url, "https://example.com/a")
        self.assertEqual(envelope.evidence[0].score, 99.0)

    def test_trend_analysis_tool_backfills_evidence(self) -> None:
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
            envelope = trend_mod.trend_analysis_tool(TrendAnalysisToolInput(topic="OpenAI", window=7))

        self.assertEqual(envelope.tool, "trend_analysis")
        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["recent_count"], 8)
        self.assertEqual(len(envelope.evidence), 1)
        self.assertEqual(envelope.evidence[0].url, "https://example.com/evidence")

    def test_envelope_helpers_add_stable_empty_and_error_contracts(self) -> None:
        empty = build_tool_empty_envelope("search_news", {"query": "x"}, "no_related_news")
        self.assertEqual(empty.status, "empty")
        self.assertEqual(empty.diagnostics["empty_reason"], "no_related_news")

        error = build_tool_error_envelope("search_news", {"query": "x"}, "search_news_failed")
        self.assertEqual(error.status, "error")
        self.assertEqual(error.error_code, "search_news_failed")
        self.assertEqual(error.diagnostics["error_code"], "search_news_failed")

        envelope = ToolEnvelope(
            tool="search_news",
            status="ok",
            request={},
            evidence=[
                {
                    "url": "https://example.com/a",
                    "rank": 1,
                    "match_score": 0.91,
                    "score_components": {"semantic_score": 0.91},
                    "metadata": {"points": 10},
                }
            ],
        )
        self.assertEqual(envelope.evidence[0].rank, 1)
        self.assertEqual(envelope.evidence[0].match_score, 0.91)

    def test_catalog_descriptions_are_specific_for_routing(self) -> None:
        descriptions = {definition.name: definition.description for definition in iter_tool_definitions()}
        self.assertIn("evidence", descriptions["search_news"].lower())
        self.assertIn("match-score", descriptions["compare_topics"].lower())
        self.assertIn("url", descriptions["read_news_content"].lower())

    def test_compare_sources_tool_uses_structured_rows_for_evidence(self) -> None:
        now = datetime(2026, 3, 28, 10, 0, 0)
        fake_conn = _SequenceConn(
            [
                [("HackerNews", 1, 12.0, 1, 0, 0), ("TechCrunch", 1, 8.0, 0, 1, 0)],
                [
                    ("HackerNews", "HN title", "https://example.com/hn", 12, now, 1),
                    ("TechCrunch", "TC title", "https://example.com/tc", 8, now, 1),
                ],
            ]
        )
        with (
            patch.object(
                compare_sources_mod,
                "fetch_semantic_url_pool",
                lambda *_args, **_kwargs: [("https://example.com/hn", 0.91), ("https://example.com/tc", 0.83)],
            ),
            patch.object(compare_sources_mod, "get_conn", lambda: fake_conn),
            patch.object(compare_sources_mod, "put_conn", lambda _conn: None),
            patch.object(compare_sources_mod, "retrieve_and_rerank", lambda *_args, **_kwargs: ([], [], {})),
        ):
            envelope = compare_sources_mod.compare_sources_tool(
                CompareSourcesToolInput(topic="OpenAI", days=14)
            )

        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["metrics_by_source"]["HackerNews"]["count"], 1)
        self.assertEqual(envelope.evidence[0].url, "https://example.com/hn")
        self.assertEqual(envelope.evidence[0].match_score, 0.91)
        self.assertEqual(envelope.diagnostics["evidence_count"], 2)

    def test_build_timeline_tool_uses_structured_events_for_evidence(self) -> None:
        now = datetime(2026, 3, 28, 10, 0, 0)
        fake_conn = _SequenceConn(
            [
                [(now, "TechCrunch", "Launch event", "Positive", 99, "https://example.com/timeline")],
            ]
        )
        with (
            patch.object(
                timeline_mod,
                "fetch_semantic_url_pool",
                lambda *_args, **_kwargs: [("https://example.com/timeline", 0.77)],
            ),
            patch.object(timeline_mod, "get_conn", lambda: fake_conn),
            patch.object(timeline_mod, "put_conn", lambda _conn: None),
            patch.object(timeline_mod, "retrieve_and_rerank", lambda *_args, **_kwargs: ([], [], {})),
        ):
            envelope = timeline_mod.build_timeline_tool(
                BuildTimelineToolInput(topic="OpenAI", days=30, limit=3)
            )

        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["event_count"], 1)
        self.assertEqual(envelope.data["events"][0]["title"], "Launch event")
        self.assertEqual(envelope.evidence[0].url, "https://example.com/timeline")
        self.assertEqual(envelope.evidence[0].match_score, 0.77)

    def test_compare_topics_match_score_arbitrates_intersection_urls(self) -> None:
        def fake_pool(query, **_kwargs):
            if query == "OpenAI":
                return [("https://example.com/shared", 0.40), ("https://example.com/a", 0.90)]
            return [("https://example.com/shared", 0.85), ("https://example.com/b", 0.70)]

        with patch.object(compare_topics_mod, "fetch_semantic_url_pool", fake_pool):
            pool_a, pool_b = compare_topics_mod._resolve_topic_pool_scores("OpenAI", "Anthropic", days=14)

        self.assertEqual([url for url, _score in pool_a], ["https://example.com/a"])
        self.assertEqual(
            [url for url, _score in pool_b],
            ["https://example.com/shared", "https://example.com/b"],
        )

    def test_compare_topics_tool_uses_structured_rows_for_evidence(self) -> None:
        now = datetime(2026, 3, 28, 10, 0, 0)
        fake_conn = _SequenceConn(
            [
                [("A", 1, 20.0, 1, 0, 0), ("B", 1, 10.0, 0, 1, 0)],
                [("A", "TechCrunch", 1), ("B", "HackerNews", 1)],
                [("A", 1, 0), ("B", 0, 1)],
                [
                    ("A", "TechCrunch", "OpenAI title", "https://example.com/a", 20, now, 1),
                    ("B", "HackerNews", "Anthropic title", "https://example.com/b", 10, now, 1),
                ],
            ]
        )

        def fake_pool(query, **_kwargs):
            if query == "OpenAI":
                return [("https://example.com/a", 0.92)]
            return [("https://example.com/b", 0.81)]

        with (
            patch.object(compare_topics_mod, "fetch_semantic_url_pool", fake_pool),
            patch.object(compare_topics_mod, "get_conn", lambda: fake_conn),
            patch.object(compare_topics_mod, "put_conn", lambda _conn: None),
        ):
            envelope = compare_topics_mod.compare_topics_tool(
                CompareTopicsToolInput(topic_a="OpenAI", topic_b="Anthropic", days=14)
            )

        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["metrics_by_topic"]["OpenAI"]["count"], 1)
        self.assertEqual(envelope.evidence[0].url, "https://example.com/a")
        self.assertEqual(envelope.evidence[0].match_score, 0.92)

    def test_analyze_landscape_tool_uses_structured_rows_for_evidence(self) -> None:
        now = datetime(2026, 3, 28, 10, 0, 0)
        fake_conn = _SequenceConn(
            [
                [
                    (
                        "OpenAI evidence",
                        "https://example.com/openai",
                        "summary",
                        "TechCrunch",
                        42,
                        "Positive",
                        now,
                    )
                ],
                [(12,)],
            ]
        )
        with (
            patch.object(
                landscape_mod,
                "_fetch_entity_url_pools",
                lambda *_args, **_kwargs: ({"OpenAI": ["https://example.com/openai"]}, {"https://example.com/openai": "OpenAI"}),
            ),
            patch.object(landscape_mod, "_get_signal_anchors", lambda: {}),
            patch.object(landscape_mod, "get_conn", lambda: fake_conn),
            patch.object(landscape_mod, "put_conn", lambda _conn: None),
            patch.object(landscape_mod, "retrieve_and_rerank", lambda *_args, **_kwargs: ([], [], {})),
        ):
            envelope = landscape_mod.analyze_landscape_tool(
                AnalyzeLandscapeToolInput(topic="", days=30, entities="OpenAI", limit_per_entity=1)
            )

        self.assertEqual(envelope.status, "ok")
        self.assertEqual(envelope.data["coverage"]["matched_entity_articles"], 1)
        self.assertEqual(envelope.data["entity_stats"]["OpenAI"]["count"], 1)
        self.assertEqual(envelope.evidence[0].url, "https://example.com/openai")



