"""Unit tests for rerank integration on search/retrieval paths."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import json
import requests

from agent.skills import fulltext_batch as fulltext_mod
from agent.skills import rerank as rerank_mod
from agent.skills import retrieval as retrieval_mod
from agent.skills import search_news as search_news_mod


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


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return dict(self._payload)


def test_resolve_rerank_mode_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.setenv("NEWS_RERANK_MODE", "cross")
    assert rerank_mod.resolve_rerank_mode() == rerank_mod.RERANK_MODE_CROSS_ENCODER
    assert rerank_mod.resolve_rerank_mode("none") == rerank_mod.RERANK_MODE_NONE

    monkeypatch.setenv("NEWS_RERANK_MODE", "llm")
    assert rerank_mod.resolve_rerank_mode() == rerank_mod.RERANK_MODE_LLM
    assert rerank_mod.resolve_rerank_mode("unknown-mode") == rerank_mod.RERANK_MODE_NONE


def test_search_news_rerank_off_keeps_recall_order() -> None:
    rows = [
        ("Title A", "https://a.com", "Summary A", "Neutral", datetime(2026, 4, 10, 10, 0, 0), 1.0),
        ("Title B", "https://b.com", "Summary B", "Positive", datetime(2026, 4, 9, 10, 0, 0), 1.0),
    ]
    fake_conn = _FakeConn(rows)
    with (
        patch.object(search_news_mod, "get_conn", lambda: fake_conn),
        patch.object(search_news_mod, "put_conn", lambda _conn: None),
        patch.object(search_news_mod, "_get_query_embedding", lambda _q: None),
    ):
        output = search_news_mod.search_news("OpenAI", days=7, rerank_mode="none")
        raw_json = search_news_mod.search_news(
            "OpenAI",
            days=7,
            rerank_mode="none",
            response_format="json",
        )

    assert "[Rerank]" not in output
    assert output.index("https://a.com") < output.index("https://b.com")
    payload = json.loads(raw_json)
    assert payload["rerank"]["rerank_mode"] == "none"
    assert payload["rerank"]["candidate_count"] == 2
    assert payload["rerank"]["top_k"] == 2
    assert payload["rerank"]["fallback"] is False


def test_search_news_rerank_on_uses_reranker_order(monkeypatch) -> None:
    rows = [
        ("Title A", "https://a.com", "Summary A", "Neutral", datetime(2026, 4, 10, 10, 0, 0), 1.0),
        ("Title B", "https://b.com", "Summary B", "Positive", datetime(2026, 4, 9, 10, 0, 0), 1.0),
    ]
    fake_conn = _FakeConn(rows)

    def _fake_post(_url, *, headers=None, json=None, timeout=None):
        assert headers is not None
        assert json is not None
        assert "query" in json
        assert "documents" in json
        assert len(json["documents"]) == 2
        assert timeout is not None
        return _FakeResponse(
            {
                "model": json["model"],
                "results": [
                    {"index": 1, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.32},
                ],
            }
        )

    monkeypatch.setenv("JINA_API_KEY", "fake-key")
    monkeypatch.setenv("SEARCH_NEWS_RERANK_MODE", "cross_encoder")
    with (
        patch.object(search_news_mod, "get_conn", lambda: fake_conn),
        patch.object(search_news_mod, "put_conn", lambda _conn: None),
        patch.object(search_news_mod, "_get_query_embedding", lambda _q: None),
        patch.object(rerank_mod.requests, "post", _fake_post),
    ):
        output = search_news_mod.search_news("OpenAI", days=7, rerank_mode="cross_encoder")
        raw_json = search_news_mod.search_news(
            "OpenAI",
            days=7,
            rerank_mode="cross_encoder",
            response_format="json",
        )

    assert "[Rerank]" not in output
    assert output.index("https://b.com") < output.index("https://a.com")
    payload = json.loads(raw_json)
    assert payload["rerank"]["rerank_mode"] == "cross_encoder"
    assert payload["rerank"]["candidate_count"] == 2
    assert payload["rerank"]["top_k"] == 2
    assert payload["rerank"]["fallback"] is False
    assert payload["rerank"]["model"] == "jina-reranker-v2-base-multilingual"


def test_search_news_rerank_failure_fallbacks_to_recall_order(monkeypatch) -> None:
    rows = [
        ("Title A", "https://a.com", "Summary A", "Neutral", datetime(2026, 4, 10, 10, 0, 0), 1.0),
        ("Title B", "https://b.com", "Summary B", "Positive", datetime(2026, 4, 9, 10, 0, 0), 1.0),
    ]
    fake_conn = _FakeConn(rows)

    def _raise_timeout(*_args, **_kwargs):
        raise requests.Timeout("rerank timeout")

    monkeypatch.setenv("JINA_API_KEY", "fake-key")
    with (
        patch.object(search_news_mod, "get_conn", lambda: fake_conn),
        patch.object(search_news_mod, "put_conn", lambda _conn: None),
        patch.object(search_news_mod, "_get_query_embedding", lambda _q: None),
        patch.object(rerank_mod.requests, "post", _raise_timeout),
    ):
        output = search_news_mod.search_news("OpenAI", days=7, rerank_mode="cross_encoder")
        raw_json = search_news_mod.search_news(
            "OpenAI",
            days=7,
            rerank_mode="cross_encoder",
            response_format="json",
        )

    assert "[Rerank]" not in output
    assert output.index("https://a.com") < output.index("https://b.com")
    payload = json.loads(raw_json)
    assert payload["rerank"]["rerank_mode"] == "cross_encoder"
    assert payload["rerank"]["fallback"] is True


def test_search_news_skill_keeps_rerank_meta_in_diagnostics(monkeypatch) -> None:
    rows = [
        ("Title A", "https://a.com", "Summary A", "Neutral", datetime(2026, 4, 10, 10, 0, 0), 1.0),
        ("Title B", "https://b.com", "Summary B", "Positive", datetime(2026, 4, 9, 10, 0, 0), 1.0),
    ]
    fake_conn = _FakeConn(rows)

    def _fake_post(_url, *, headers=None, json=None, timeout=None):
        del headers, timeout
        return _FakeResponse(
            {
                "model": json["model"],
                "results": [
                    {"index": 1, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.32},
                ],
            }
        )

    monkeypatch.setenv("JINA_API_KEY", "fake-key")
    monkeypatch.setenv("SEARCH_NEWS_RERANK_MODE", "cross_encoder")
    with (
        patch.object(search_news_mod, "get_conn", lambda: fake_conn),
        patch.object(search_news_mod, "put_conn", lambda _conn: None),
        patch.object(search_news_mod, "_get_query_embedding", lambda _q: None),
        patch.object(rerank_mod.requests, "post", _fake_post),
    ):
        envelope = search_news_mod.search_news_skill(search_news_mod.SearchNewsSkillInput(query="OpenAI", days=7))

    assert envelope.status == "ok"
    assert envelope.diagnostics["rerank"]["rerank_mode"] == "cross_encoder"
    assert envelope.diagnostics["rerank"]["candidate_count"] == 2
    raw_output = str((envelope.data or {}).get("raw_output") or "")
    assert "[Rerank]" not in raw_output


def test_lookup_urls_by_query_rerank_off_keeps_recall_order() -> None:
    rows = [
        ("Headline A", "https://a.com", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
        ("Headline B", "https://b.com", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
    ]
    fake_conn = _FakeConn(rows)

    with (
        patch.object(retrieval_mod, "get_conn", lambda: fake_conn),
        patch.object(retrieval_mod, "put_conn", lambda _conn: None),
        patch.object(retrieval_mod, "_get_query_embedding", lambda _q: None),
    ):
        ranked_rows, meta = retrieval_mod._lookup_urls_by_query(
            query="OpenAI",
            days=14,
            limit=2,
            rerank_mode="none",
            include_rerank_meta=True,
        )

    assert ranked_rows == rows
    assert meta["rerank_mode"] == "none"
    assert meta["candidate_count"] == 2
    assert meta["top_k"] == 2
    assert meta["fallback"] is False


def test_lookup_urls_by_query_empty_query_returns_structured_meta() -> None:
    rows, meta = retrieval_mod._lookup_urls_by_query(
        query="   ",
        rerank_mode="cross_encoder",
        include_rerank_meta=True,
    )
    assert rows == []
    assert meta["candidate_count"] == 0
    assert meta["top_k"] == 0
    assert meta["fallback"] is False


def test_lookup_urls_by_query_rerank_on_uses_reranker_order(monkeypatch) -> None:
    rows = [
        ("Headline A", "https://a.com", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
        ("Headline B", "https://b.com", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
    ]
    fake_conn = _FakeConn(rows)

    def _fake_post(_url, *, headers=None, json=None, timeout=None):
        assert headers is not None
        assert json is not None
        assert json["query"] == "OpenAI"
        assert len(json["documents"]) == 2
        assert timeout is not None
        return _FakeResponse(
            {
                "model": json["model"],
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.12},
                ],
            }
        )

    monkeypatch.setenv("JINA_API_KEY", "fake-key")
    with (
        patch.object(retrieval_mod, "get_conn", lambda: fake_conn),
        patch.object(retrieval_mod, "put_conn", lambda _conn: None),
        patch.object(retrieval_mod, "_get_query_embedding", lambda _q: None),
        patch.object(rerank_mod.requests, "post", _fake_post),
    ):
        ranked_rows, meta = retrieval_mod._lookup_urls_by_query(
            query="OpenAI",
            days=14,
            limit=2,
            rerank_mode="cross_encoder",
            include_rerank_meta=True,
        )

    assert ranked_rows[0][1] == "https://b.com"
    assert ranked_rows[1][1] == "https://a.com"
    assert meta["rerank_mode"] == "cross_encoder"
    assert meta["candidate_count"] == 2
    assert meta["top_k"] == 2
    assert meta["fallback"] is False


def test_lookup_urls_by_query_rerank_failure_fallbacks_to_recall_order(monkeypatch) -> None:
    rows = [
        ("Headline A", "https://a.com", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
        ("Headline B", "https://b.com", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
    ]
    fake_conn = _FakeConn(rows)

    def _raise_timeout(*_args, **_kwargs):
        raise requests.Timeout("rerank timeout")

    monkeypatch.setenv("JINA_API_KEY", "fake-key")
    with (
        patch.object(retrieval_mod, "get_conn", lambda: fake_conn),
        patch.object(retrieval_mod, "put_conn", lambda _conn: None),
        patch.object(retrieval_mod, "_get_query_embedding", lambda _q: None),
        patch.object(rerank_mod.requests, "post", _raise_timeout),
    ):
        ranked_rows, meta = retrieval_mod._lookup_urls_by_query(
            query="OpenAI",
            days=14,
            limit=2,
            rerank_mode="cross_encoder",
            include_rerank_meta=True,
        )

    assert ranked_rows == rows
    assert meta["rerank_mode"] == "cross_encoder"
    assert meta["candidate_count"] == 2
    assert meta["top_k"] == 2
    assert meta["fallback"] is True


def test_fulltext_batch_auto_select_uses_reranked_order_and_meta() -> None:
    reranked_rows = [
        ("Headline B", "https://b.com", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
        ("Headline A", "https://a.com", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
    ]
    rerank_meta = {
        "rerank_mode": "cross_encoder",
        "candidate_count": 12,
        "top_k": 2,
        "fallback": False,
    }

    with (
        patch.object(
            fulltext_mod,
            "_lookup_urls_by_query",
            lambda **_kwargs: (reranked_rows, rerank_meta),
        ),
        patch.object(fulltext_mod, "read_news_content", lambda _url: "Full content body"),
    ):
        raw = fulltext_mod.fulltext_batch("OpenAI recent 14 days", response_format="json", rerank_mode="cross_encoder")

    payload = json.loads(raw)
    assert payload["status"] == "ok"
    assert payload["rerank"]["rerank_mode"] == "cross_encoder"
    assert payload["rerank"]["candidate_count"] == 12
    assert payload["rerank"]["top_k"] == 2
    assert payload["rerank"]["fallback"] is False
    assert payload["selected"][0]["url"] == "https://b.com"
    assert payload["selected"][1]["url"] == "https://a.com"


def test_fulltext_batch_text_output_hides_rerank_debug_line() -> None:
    reranked_rows = [
        ("Headline B", "https://b.com", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
        ("Headline A", "https://a.com", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
    ]
    rerank_meta = {
        "rerank_mode": "cross_encoder",
        "candidate_count": 12,
        "top_k": 2,
        "fallback": False,
    }

    with (
        patch.object(
            fulltext_mod,
            "_lookup_urls_by_query",
            lambda **_kwargs: (reranked_rows, rerank_meta),
        ),
        patch.object(fulltext_mod, "read_news_content", lambda _url: "Full content body"),
    ):
        text = fulltext_mod.fulltext_batch("OpenAI recent 14 days", response_format="text", rerank_mode="cross_encoder")

    assert "[Rerank]" not in text


def test_fulltext_batch_skill_exposes_rerank_meta_in_diagnostics() -> None:
    payload = {
        "tool": "fulltext_batch",
        "status": "ok",
        "request": {"urls_or_query": "OpenAI"},
        "rerank": {
            "rerank_mode": "cross_encoder",
            "candidate_count": 6,
            "top_k": 2,
            "fallback": False,
            "model": "jina-reranker-v2-base-multilingual",
        },
        "selected": [
            {"url": "https://a.com", "headline": "A", "source_type": "TechCrunch", "created_at": "2026-04-10T10:00:00", "score": 2.1},
            {"url": "https://b.com", "headline": "B", "source_type": "HackerNews", "created_at": "2026-04-09T10:00:00", "score": 1.9},
        ],
        "articles": [
            {"url": "https://a.com", "content": "A", "meta": {}},
            {"url": "https://b.com", "content": "B", "meta": {}},
        ],
    }
    with patch.object(fulltext_mod, "fulltext_batch", lambda **_kwargs: json.dumps(payload, ensure_ascii=False)):
        envelope = fulltext_mod.fulltext_batch_skill(fulltext_mod.FulltextBatchSkillInput(urls="OpenAI", max_chars_per_article=2000))

    assert envelope.status == "ok"
    assert envelope.diagnostics["rerank"]["rerank_mode"] == "cross_encoder"
    assert envelope.diagnostics["rerank"]["candidate_count"] == 6
