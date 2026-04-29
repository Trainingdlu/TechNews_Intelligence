from __future__ import annotations

import random

from eval.corpus_sampler import build_pools, pack_cluster


def _doc(idx: int, *, source: str, day: int, sim: float, topic_group: str = "") -> dict:
    row = {
        "doc_id": f"doc_{idx}",
        "url": f"https://example.com/{idx}",
        "title": f"title {idx}",
        "summary": f"summary {idx}",
        "evidence_text": f"summary {idx}",
        "published_at": f"2026-04-{day:02d}T00:00:00",
        "source": source,
        "seed_similarity": sim,
        "channels": ["semantic"],
        "embedding_available": True,
    }
    if topic_group:
        row["topic_group"] = topic_group
    return row


def test_pack_compare_sources_preserves_source_coverage() -> None:
    task = {
        "skill": "compare_sources",
        "scenario": "normal",
        "sampling": {"pool_size": 8, "sources": ["HackerNews", "TechCrunch"]},
    }
    docs = [
        *[_doc(i, source="HackerNews", day=i, sim=0.9 - i * 0.01) for i in range(1, 5)],
        *[_doc(i, source="TechCrunch", day=i, sim=0.8 - i * 0.01) for i in range(5, 9)],
    ]

    selected, meta = pack_cluster(task, docs, pool_size=8)

    assert meta["packing_constraints_passed"] is True
    assert {doc["source"] for doc in selected} == {"HackerNews", "TechCrunch"}


def test_pack_timeline_preserves_time_spread() -> None:
    task = {"skill": "build_timeline", "scenario": "normal", "sampling": {"pool_size": 8}}
    docs = [_doc(i, source="S", day=i, sim=0.7 + i * 0.01) for i in range(1, 13)]

    selected, meta = pack_cluster(task, docs, pool_size=8)

    assert meta["packing_constraints_passed"] is True
    selected_days = sorted(int(doc["published_at"][8:10]) for doc in selected)
    assert selected_days[0] <= 2
    assert selected_days[-1] >= 10


def test_pack_compare_topics_balances_topic_groups() -> None:
    task = {"skill": "compare_topics", "scenario": "normal", "sampling": {"pool_size": 10}}
    docs = [
        *[_doc(i, source="S1", day=i, sim=0.9, topic_group="A") for i in range(1, 7)],
        *[_doc(i, source="S2", day=i, sim=0.8, topic_group="B") for i in range(7, 13)],
    ]

    selected, meta = pack_cluster(task, docs, pool_size=10)

    assert meta["packing_constraints_passed"] is True
    assert len([doc for doc in selected if doc["topic_group"] == "A"]) == 5
    assert len([doc for doc in selected if doc["topic_group"] == "B"]) == 5


def test_empty_build_pools_uses_negative_candidates_without_positive_floor(monkeypatch) -> None:
    import eval.corpus_sampler as sampler

    monkeypatch.setattr(sampler, "_recall_sim_floor", lambda: 0.3)
    task = {
        "task_id": "search_news.empty",
        "skill": "search_news",
        "scenario": "empty",
        "sampling": {"pool_size": 8},
    }
    candidates = [_doc(i, source="S", day=i, sim=0.05) for i in range(1, 10)]

    pools, meta = build_pools(task, candidates, pools_per_task=1, rng=random.Random(1))

    assert meta["cluster_mode"] == "empty_negative"
    assert len(pools) == 1
    assert all(doc["seed_similarity"] <= 0.2 for doc in pools[0].docs)
