"""Unit tests for rerank integration on search/retrieval paths."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

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

    def rollback(self):
        return None


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return dict(self._payload)


def _candidate(
    *,
    title: str,
    url: str,
    summary: str,
    sentiment: str,
    source_type: str,
    created_at: datetime,
    points: int,
    score: float,
) -> dict:
    return {
        "title": title,
        "url": url,
        "summary": summary,
        "sentiment": sentiment,
        "source_type": source_type,
        "created_at": created_at,
        "points": points,
        "score": score,
    }


def test_resolve_rerank_mode_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.setenv("NEWS_RERANK_MODE", "llm_rerank")
    assert rerank_mod.resolve_rerank_mode() == rerank_mod.RERANK_MODE_LLM
    assert rerank_mod.resolve_rerank_mode("none") == rerank_mod.RERANK_MODE_NONE

    monkeypatch.setenv("NEWS_RERANK_MODE", "unknown_mode")
    assert rerank_mod.resolve_rerank_mode() == rerank_mod.RERANK_MODE_NONE

    monkeypatch.setenv("NEWS_RERANK_MODE", "llm")
    assert rerank_mod.resolve_rerank_mode() == rerank_mod.RERANK_MODE_LLM
    assert rerank_mod.resolve_rerank_mode("unknown-mode") == rerank_mod.RERANK_MODE_NONE


def test_search_news_rerank_off_keeps_recall_order() -> None:
    candidates = [
        _candidate(
            title="Title A",
            url="https://a.com",
            summary="Summary A",
            sentiment="Neutral",
            source_type="TechCrunch",
            created_at=datetime(2026, 4, 10, 10, 0, 0),
            points=10,
            score=1.0,
        ),
        _candidate(
            title="Title B",
            url="https://b.com",
            summary="Summary B",
            sentiment="Positive",
            source_type="HackerNews",
            created_at=datetime(2026, 4, 9, 10, 0, 0),
            points=8,
            score=0.9,
        ),
    ]
    rerank_meta = {"rerank_mode": "none", "candidate_count": 2, "top_k": 2, "fallback": False}

    with patch.object(search_news_mod, "lookup_candidates_by_query", lambda **_kwargs: (candidates, rerank_meta)):
        output = search_news_mod.search_news("OpenAI", days=7, rerank_mode="none")
        raw_json = search_news_mod.search_news(
            "OpenAI",
            days=7,
            rerank_mode="none",
            response_format="json",
        )

    assert output.index("https://a.com") < output.index("https://b.com")
    payload = json.loads(raw_json)
    assert payload["rerank"]["rerank_mode"] == "none"
    assert payload["rerank"]["candidate_count"] == 2
    assert payload["rerank"]["top_k"] == 2
    assert payload["rerank"]["fallback"] is False


def test_search_news_rerank_on_uses_reranker_order() -> None:
    candidates = [
        _candidate(
            title="Title B",
            url="https://b.com",
            summary="Summary B",
            sentiment="Positive",
            source_type="HackerNews",
            created_at=datetime(2026, 4, 9, 10, 0, 0),
            points=8,
            score=1.2,
        ),
        _candidate(
            title="Title A",
            url="https://a.com",
            summary="Summary A",
            sentiment="Neutral",
            source_type="TechCrunch",
            created_at=datetime(2026, 4, 10, 10, 0, 0),
            points=10,
            score=1.0,
        ),
    ]
    rerank_meta = {
        "rerank_mode": "llm_rerank",
        "candidate_count": 2,
        "top_k": 2,
        "fallback": False,
        "model": "jina-reranker-v3",
    }

    with patch.object(search_news_mod, "lookup_candidates_by_query", lambda **_kwargs: (candidates, rerank_meta)):
        output = search_news_mod.search_news("OpenAI", days=7, rerank_mode="llm_rerank")
        raw_json = search_news_mod.search_news(
            "OpenAI",
            days=7,
            rerank_mode="llm_rerank",
            response_format="json",
        )

    assert output.index("https://b.com") < output.index("https://a.com")
    payload = json.loads(raw_json)
    assert payload["rerank"]["rerank_mode"] == "llm_rerank"
    assert payload["rerank"]["candidate_count"] == 2
    assert payload["rerank"]["top_k"] == 2
    assert payload["rerank"]["fallback"] is False
    assert payload["rerank"]["model"] == "jina-reranker-v3"


def test_search_news_rerank_failure_fallbacks_to_recall_order() -> None:
    candidates = [
        _candidate(
            title="Title A",
            url="https://a.com",
            summary="Summary A",
            sentiment="Neutral",
            source_type="TechCrunch",
            created_at=datetime(2026, 4, 10, 10, 0, 0),
            points=10,
            score=1.0,
        ),
        _candidate(
            title="Title B",
            url="https://b.com",
            summary="Summary B",
            sentiment="Positive",
            source_type="HackerNews",
            created_at=datetime(2026, 4, 9, 10, 0, 0),
            points=8,
            score=0.9,
        ),
    ]
    rerank_meta = {"rerank_mode": "llm_rerank", "candidate_count": 2, "top_k": 2, "fallback": True}

    with patch.object(search_news_mod, "lookup_candidates_by_query", lambda **_kwargs: (candidates, rerank_meta)):
        output = search_news_mod.search_news("OpenAI", days=7, rerank_mode="llm_rerank")
        raw_json = search_news_mod.search_news(
            "OpenAI",
            days=7,
            rerank_mode="llm_rerank",
            response_format="json",
        )

    assert output.index("https://a.com") < output.index("https://b.com")
    payload = json.loads(raw_json)
    assert payload["rerank"]["rerank_mode"] == "llm_rerank"
    assert payload["rerank"]["fallback"] is True


def test_search_news_skill_keeps_rerank_meta_in_diagnostics() -> None:
    candidates = [
        _candidate(
            title="Title B",
            url="https://b.com",
            summary="Summary B",
            sentiment="Positive",
            source_type="HackerNews",
            created_at=datetime(2026, 4, 9, 10, 0, 0),
            points=8,
            score=1.2,
        ),
        _candidate(
            title="Title A",
            url="https://a.com",
            summary="Summary A",
            sentiment="Neutral",
            source_type="TechCrunch",
            created_at=datetime(2026, 4, 10, 10, 0, 0),
            points=10,
            score=1.0,
        ),
    ]
    rerank_meta = {
        "rerank_mode": "llm_rerank",
        "candidate_count": 2,
        "top_k": 2,
        "fallback": False,
        "model": "jina-reranker-v3",
    }

    with patch.object(search_news_mod, "lookup_candidates_by_query", lambda **_kwargs: (candidates, rerank_meta)):
        envelope = search_news_mod.search_news_skill(search_news_mod.SearchNewsSkillInput(query="OpenAI", days=7))

    assert envelope.status == "ok"
    assert envelope.diagnostics["rerank"]["rerank_mode"] == "llm_rerank"
    assert envelope.diagnostics["rerank"]["candidate_count"] == 2


def test_lookup_candidates_by_query_rerank_off_keeps_recall_order() -> None:
    rows = [
        ("Headline A", "https://a.com", "Summary A", "Neutral", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
        ("Headline B", "https://b.com", "Summary B", "Positive", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
    ]
    fake_conn = _FakeConn(rows)

    with (
        patch.object(retrieval_mod, "get_conn", lambda: fake_conn),
        patch.object(retrieval_mod, "put_conn", lambda _conn: None),
        patch.object(retrieval_mod, "_get_query_embedding", lambda _q: None),
    ):
        candidates, meta = retrieval_mod.lookup_candidates_by_query(
            query="OpenAI",
            days=14,
            limit=2,
            rerank_mode="none",
        )

    assert [item["url"] for item in candidates] == ["https://a.com", "https://b.com"]
    assert meta["rerank_mode"] == "none"
    assert meta["candidate_count"] == 2
    assert meta["top_k"] == 2
    assert meta["fallback"] is False


def test_lookup_candidates_by_query_empty_query_returns_structured_meta() -> None:
    candidates, meta = retrieval_mod.lookup_candidates_by_query(
        query="   ",
        rerank_mode="llm_rerank",
    )
    assert candidates == []
    assert meta["candidate_count"] == 0
    assert meta["top_k"] == 0
    assert meta["fallback"] is False


def test_lookup_candidates_by_query_rerank_on_uses_reranker_order(monkeypatch) -> None:
    rows = [
        ("Headline A", "https://a.com", "Summary A", "Neutral", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
        ("Headline B", "https://b.com", "Summary B", "Positive", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
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
        candidates, meta = retrieval_mod.lookup_candidates_by_query(
            query="OpenAI",
            days=14,
            limit=2,
            rerank_mode="llm_rerank",
        )

    assert candidates[0]["url"] == "https://b.com"
    assert candidates[1]["url"] == "https://a.com"
    assert meta["rerank_mode"] == "llm_rerank"
    assert meta["candidate_count"] == 2
    assert meta["top_k"] == 2
    assert meta["fallback"] is False


def test_lookup_candidates_by_query_rerank_failure_fallbacks_to_recall_order(monkeypatch) -> None:
    rows = [
        ("Headline A", "https://a.com", "Summary A", "Neutral", "TechCrunch", datetime(2026, 4, 10, 10, 0, 0), 10, 1.11),
        ("Headline B", "https://b.com", "Summary B", "Positive", "HackerNews", datetime(2026, 4, 9, 10, 0, 0), 8, 1.01),
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
        candidates, meta = retrieval_mod.lookup_candidates_by_query(
            query="OpenAI",
            days=14,
            limit=2,
            rerank_mode="llm_rerank",
        )

    assert [item["url"] for item in candidates] == ["https://a.com", "https://b.com"]
    assert meta["rerank_mode"] == "llm_rerank"
    assert meta["candidate_count"] == 2
    assert meta["top_k"] == 2
    assert meta["fallback"] is True


def test_fulltext_batch_auto_select_uses_reranked_order_and_meta() -> None:
    reranked_candidates = [
        _candidate(
            title="Headline B",
            url="https://b.com",
            summary="Summary B",
            sentiment="Positive",
            source_type="HackerNews",
            created_at=datetime(2026, 4, 9, 10, 0, 0),
            points=8,
            score=1.01,
        ),
        _candidate(
            title="Headline A",
            url="https://a.com",
            summary="Summary A",
            sentiment="Neutral",
            source_type="TechCrunch",
            created_at=datetime(2026, 4, 10, 10, 0, 0),
            points=10,
            score=1.11,
        ),
    ]
    rerank_meta = {
        "rerank_mode": "llm_rerank",
        "candidate_count": 12,
        "top_k": 2,
        "fallback": False,
    }

    with (
        patch.object(fulltext_mod, "lookup_candidates_by_query", lambda **_kwargs: (reranked_candidates, rerank_meta)),
        patch.object(fulltext_mod, "read_news_content", lambda _url: "Full content body"),
    ):
        raw = fulltext_mod.fulltext_batch("OpenAI recent 14 days", response_format="json", rerank_mode="llm_rerank")

    payload = json.loads(raw)
    assert payload["status"] == "ok"
    assert payload["rerank"]["rerank_mode"] == "llm_rerank"
    assert payload["rerank"]["candidate_count"] == 12
    assert payload["rerank"]["top_k"] == 2
    assert payload["rerank"]["fallback"] is False
    assert payload["selected"][0]["url"] == "https://b.com"
    assert payload["selected"][1]["url"] == "https://a.com"


def test_fulltext_batch_text_output_hides_rerank_debug_line() -> None:
    reranked_candidates = [
        _candidate(
            title="Headline B",
            url="https://b.com",
            summary="Summary B",
            sentiment="Positive",
            source_type="HackerNews",
            created_at=datetime(2026, 4, 9, 10, 0, 0),
            points=8,
            score=1.01,
        ),
        _candidate(
            title="Headline A",
            url="https://a.com",
            summary="Summary A",
            sentiment="Neutral",
            source_type="TechCrunch",
            created_at=datetime(2026, 4, 10, 10, 0, 0),
            points=10,
            score=1.11,
        ),
    ]
    rerank_meta = {
        "rerank_mode": "llm_rerank",
        "candidate_count": 12,
        "top_k": 2,
        "fallback": False,
    }

    with (
        patch.object(fulltext_mod, "lookup_candidates_by_query", lambda **_kwargs: (reranked_candidates, rerank_meta)),
        patch.object(fulltext_mod, "read_news_content", lambda _url: "Full content body"),
    ):
        text = fulltext_mod.fulltext_batch("OpenAI recent 14 days", response_format="text", rerank_mode="llm_rerank")

    assert "[Rerank]" not in text


def test_fulltext_batch_skill_exposes_rerank_meta_in_diagnostics() -> None:
    payload = {
        "tool": "fulltext_batch",
        "status": "ok",
        "request": {"urls_or_query": "OpenAI"},
        "rerank": {
            "rerank_mode": "llm_rerank",
            "candidate_count": 6,
            "top_k": 2,
            "fallback": False,
            "model": "jina-reranker-v3",
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
    assert envelope.diagnostics["rerank"]["rerank_mode"] == "llm_rerank"
    assert envelope.diagnostics["rerank"]["candidate_count"] == 6
