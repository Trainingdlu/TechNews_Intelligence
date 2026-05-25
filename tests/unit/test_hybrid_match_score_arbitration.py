from __future__ import annotations

from agent.tools import analyze_landscape as landscape_mod
from agent.tools import compare_topics as compare_mod


def test_compare_topics_intersection_arbitration_uses_higher_match_score(monkeypatch) -> None:
    def _fake_fetch(query: str, *, days: int, limit: int):  # noqa: ARG001
        if query == "topic-a":
            return [
                ("https://example.com/shared", 0.42),
                ("https://example.com/a-only", 0.70),
            ]
        if query == "topic-b":
            return [
                ("https://example.com/shared", 0.88),
                ("https://example.com/b-only", 0.66),
            ]
        return []

    monkeypatch.setattr(compare_mod, "fetch_hybrid_url_pool", _fake_fetch)
    urls_a, urls_b = compare_mod._resolve_topic_pools("topic-a", "topic-b", days=14)

    assert "https://example.com/shared" not in urls_a
    assert "https://example.com/shared" in urls_b
    assert "https://example.com/a-only" in urls_a
    assert "https://example.com/b-only" in urls_b


def test_analyze_landscape_entity_arbitration_uses_higher_match_score(monkeypatch) -> None:
    def _fake_fetch(query: str, *, days: int, limit: int):  # noqa: ARG001
        if query.startswith("EntityA"):
            return [
                ("https://example.com/shared", 0.31),
                ("https://example.com/a-only", 0.62),
            ]
        if query.startswith("EntityB"):
            return [
                ("https://example.com/shared", 0.91),
                ("https://example.com/b-only", 0.58),
            ]
        return []

    monkeypatch.setattr(landscape_mod, "fetch_hybrid_url_pool", _fake_fetch)
    entity_url_map, url_to_entity = landscape_mod._fetch_entity_url_pools(
        ["EntityA", "EntityB"],
        topic="AI",
        days=30,
        limit_per_entity=20,
    )

    assert url_to_entity["https://example.com/shared"] == "EntityB"
    assert "https://example.com/shared" not in entity_url_map["EntityA"]
    assert "https://example.com/shared" in entity_url_map["EntityB"]
