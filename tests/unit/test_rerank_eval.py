"""Tests for offline rerank benchmark utilities."""

from __future__ import annotations

import json
from pathlib import Path

from eval import rerank_eval as rerank_eval_mod


def test_rerank_eval_none_mode_keeps_baseline_equal() -> None:
    dataset_path = Path("eval/datasets/rerank_mini.json")
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = rerank_eval_mod.evaluate_dataset(cases, mode="none", top_k=3)
    summary = report["summary"]
    assert summary["case_count"] >= 1
    assert abs(summary["delta_avg_ndcg"]) < 1e-12
    assert abs(summary["delta_avg_mrr"]) < 1e-12


def test_rerank_eval_supports_custom_rerank_order(monkeypatch) -> None:
    cases = [
        {
            "id": "toy",
            "query": "openai voice",
            "candidates": [
                {"title": "irrelevant", "url": "https://a.com", "label": 0},
                {"title": "high relevance", "url": "https://b.com", "label": 3},
                {"title": "medium relevance", "url": "https://c.com", "label": 2},
            ],
        }
    ]

    def _fake_rerank_candidates(_query, candidates, *, mode=None, top_k=None, env_keys=None):
        del mode, env_keys
        ordered = [dict(candidates[1]), dict(candidates[2]), dict(candidates[0])]
        return ordered[: int(top_k or len(ordered))], {
            "rerank_mode": "cross_encoder",
            "candidate_count": len(candidates),
            "top_k": int(top_k or len(candidates)),
            "fallback": False,
            "model": "fake",
        }

    monkeypatch.setattr(rerank_eval_mod, "rerank_candidates", _fake_rerank_candidates)
    report = rerank_eval_mod.evaluate_dataset(cases, mode="cross_encoder", top_k=3)
    summary = report["summary"]
    assert summary["delta_avg_ndcg"] > 0
    assert summary["delta_avg_mrr"] > 0
