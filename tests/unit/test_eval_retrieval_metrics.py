"""Unit tests for retrieval-level eval metrics."""

from __future__ import annotations

import math

from eval.eval_core import mrr_at_k, ndcg_at_k, recall_at_k, summarize_case_results


def test_retrieval_metrics_with_url_normalization() -> None:
    pred_urls = [
        "HTTPS://WWW.Example.com/A/",
        "https://irrelevant.com/doc",
        "https://OTHER.com/",
    ]
    gold_urls = [
        "https://example.com/a",
        "https://other.com",
    ]

    assert recall_at_k(pred_urls, gold_urls, 5) == 1.0
    assert mrr_at_k(pred_urls, gold_urls, 10) == 1.0

    ndcg = ndcg_at_k(pred_urls, gold_urls, 10)
    assert ndcg is not None
    expected = (1.0 + (1.0 / math.log2(4))) / (1.0 + (1.0 / math.log2(3)))
    assert abs(ndcg - expected) < 1e-6


def test_retrieval_metrics_gold_empty_returns_none() -> None:
    pred_urls = ["https://example.com/a"]
    assert recall_at_k(pred_urls, [], 5) is None
    assert mrr_at_k(pred_urls, None, 10) is None
    assert ndcg_at_k(pred_urls, [], 10) is None


def test_retrieval_metrics_support_domain_level_matching() -> None:
    pred_urls = ["https://example.com/news/a"]
    gold_urls = ["https://example.com/reports/quarterly"]
    assert recall_at_k(pred_urls, gold_urls, 5) == 0.0
    assert recall_at_k(pred_urls, gold_urls, 5, domain_only=True) == 1.0


def test_summarize_case_results_ignores_cases_without_retrieval_gold() -> None:
    case_with_gold = {
        "metrics": {
            "run_count": 1,
            "avg_pairwise_similarity": 1.0,
            "unique_response_ratio": 0.0,
            "min_url_hit_rate": 1.0,
            "phrase_hit_rate": 1.0,
            "fact_hit_rate": 1.0,
            "fact_group_hit_rate": 1.0,
            "tool_path_hit_rate": 1.0,
            "tool_path_accept_hit_rate": 1.0,
            "source_domain_hit_rate": 1.0,
            "forbidden_claim_rate": 0.0,
            "error_rate": 0.0,
            "retrieval_has_gold": True,
            "recall_at_5": 0.8,
            "recall_at_10": 1.0,
            "mrr_at_10": 0.5,
            "ndcg_at_10": 0.7,
        }
    }
    case_without_gold = {
        "metrics": {
            "run_count": 1,
            "avg_pairwise_similarity": 1.0,
            "unique_response_ratio": 0.0,
            "min_url_hit_rate": 1.0,
            "phrase_hit_rate": 1.0,
            "fact_hit_rate": 1.0,
            "fact_group_hit_rate": 1.0,
            "tool_path_hit_rate": 1.0,
            "tool_path_accept_hit_rate": 1.0,
            "source_domain_hit_rate": 1.0,
            "forbidden_claim_rate": 0.0,
            "error_rate": 0.0,
            "retrieval_has_gold": False,
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr_at_10": None,
            "ndcg_at_10": None,
        }
    }

    summary = summarize_case_results([case_with_gold, case_without_gold])
    assert summary["retrieval_case_count"] == 1
    assert summary["avg_recall_at_5"] == 0.8
    assert summary["avg_recall_at_10"] == 1.0
    assert summary["avg_mrr_at_10"] == 0.5
    assert summary["avg_ndcg_at_10"] == 0.7
