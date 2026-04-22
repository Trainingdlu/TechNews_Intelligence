"""Tests for semantic pool — replacement for the old topic-expansion config tests.

The original test_topic_config.py validated hardcoded dictionary loading from
``sql_builders.py``.  That module has been deprecated in favour of
``semantic_pool.fetch_semantic_url_pool``.  These tests verify the new
semantic infrastructure at a unit level (mocked embedding API).
"""

from __future__ import annotations

from unittest.mock import patch

from agent.skills.semantic_pool import (
    _cache_get,
    _cache_set,
    _embedding_cache,
    _evict_stale_cache,
    _get_embedding_with_retry,
)


def test_cache_set_and_get_happy_path() -> None:
    """Stored embeddings are retrievable within TTL."""
    _embedding_cache.clear()
    vec = [0.1, 0.2, 0.3]
    _cache_set("test_query", vec)
    result = _cache_get("test_query")
    assert result == vec
    _embedding_cache.clear()


def test_cache_get_returns_none_for_missing_key() -> None:
    _embedding_cache.clear()
    assert _cache_get("nonexistent") is None


def test_evict_stale_cache_removes_expired_entries() -> None:
    """Entries older than TTL are evicted."""
    _embedding_cache.clear()
    # Insert an entry with a timestamp far in the past
    _embedding_cache["old_query"] = ([0.1], 0.0)
    _evict_stale_cache()
    assert "old_query" not in _embedding_cache
    _embedding_cache.clear()


def test_get_embedding_with_retry_returns_on_first_success() -> None:
    """If the first call succeeds, no retry occurs."""
    _embedding_cache.clear()
    fake_vec = [0.5] * 10
    with patch(
        "agent.skills.semantic_pool._get_query_embedding",
        return_value=fake_vec,
    ) as mock_embed:
        result = _get_embedding_with_retry("hello")
    assert result == fake_vec
    mock_embed.assert_called_once()
    _embedding_cache.clear()


def test_get_embedding_with_retry_retries_on_failure(monkeypatch) -> None:
    """On first failure, the function retries and succeeds on the second call."""
    _embedding_cache.clear()
    monkeypatch.setenv("SEMANTIC_POOL_RETRY_COUNT", "1")
    fake_vec = [0.9] * 10
    call_count = {"n": 0}

    def _mock_embed(query: str):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None  # first call fails
        return fake_vec

    with patch(
        "agent.skills.semantic_pool._get_query_embedding",
        side_effect=_mock_embed,
    ):
        # Reduce delay to speed up test
        with patch("agent.skills.semantic_pool._DEFAULT_RETRY_DELAY", 0.01):
            result = _get_embedding_with_retry("hello")
    assert result == fake_vec
    assert call_count["n"] == 2
    _embedding_cache.clear()
