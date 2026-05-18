from __future__ import annotations

from eval.build_resume_metrics import build_resume_metrics, render_markdown


def _retrieval_report(summary: dict) -> dict:
    return {
        "generated_at": "2026-05-18T00:00:00+00:00",
        "dataset": "eval/datasets/retrieval_cases.jsonl",
        "events": "eval/datasets/event_cards.jsonl",
        "summary": summary,
        "env": {"EVAL_RECALL_PROFILE": "base", "NEWS_RERANK_MODE": "none", "JINA_API_KEY_PRESENT": False},
    }


def test_resume_metrics_builds_resume_ready_cards_when_manual_gate_passes() -> None:
    baseline = _retrieval_report(
        {
            "single_event_case_count": 150,
            "broad_topic_case_count": 50,
            "avg_single_event_hit_at_k": 0.6,
            "avg_single_mrr_at_k": 0.5,
            "avg_event_set_recall_at_k": 0.4,
            "avg_irrelevant_event_ratio_at_k": 0.3,
            "avg_event_diversity_at_k": 2.0,
        }
    )
    candidate = _retrieval_report(
        {
            "single_event_case_count": 150,
            "broad_topic_case_count": 50,
            "avg_single_event_hit_at_k": 0.72,
            "avg_single_mrr_at_k": 0.58,
            "avg_event_set_recall_at_k": 0.62,
            "avg_irrelevant_event_ratio_at_k": 0.18,
            "avg_event_diversity_at_k": 3.1,
        }
    )
    generation = {"summary": {"case_count": 50, "avg_claim_coverage": 0.84, "avg_unsupported_url_rate": 0.02, "avg_forbidden_hit_count": 0.0}}
    report = build_resume_metrics(
        baseline_retrieval=baseline,
        candidate_retrieval=candidate,
        generation=generation,
        manual_review_status="pass",
        manual_fail_rate=0.05,
        manual_card_sample_size=20,
        manual_case_sample_size=30,
    )
    assert report["manual_quality_gate"]["allowed_for_resume"] is True
    assert report["system_under_test"]["candidate_2"]["status"] == "skipped"
    cards = {card["name"]: card for card in report["metric_cards"]}
    assert abs(cards["single_event_hit_at_5"]["delta"] - 0.12) < 1e-9
    assert cards["broad_irrelevant_event_ratio_at_5"]["direction"] == "lower_better"
    markdown = render_markdown(report)
    assert "在 150 条单事件检索样本上" in markdown
    assert "非证据 URL 泄漏率为 2.0%" in markdown


def test_resume_metrics_blocks_resume_use_when_manual_gate_fails() -> None:
    baseline = _retrieval_report({"single_event_case_count": 1, "avg_single_event_hit_at_k": 0.0})
    candidate = _retrieval_report({"single_event_case_count": 1, "avg_single_event_hit_at_k": 1.0})
    report = build_resume_metrics(
        baseline_retrieval=baseline,
        candidate_retrieval=candidate,
        manual_review_status="pass",
        manual_fail_rate=0.2,
        manual_card_sample_size=20,
        manual_case_sample_size=30,
    )
    assert report["manual_quality_gate"]["allowed_for_resume"] is False
    assert "Do not use these numbers" in render_markdown(report)
