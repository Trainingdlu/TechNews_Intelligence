from __future__ import annotations

from datetime import date

from eval.run_retrieval_eval import _days_from_case, _summary_by_query_type


def test_days_from_case_uses_iso_date_not_day_of_month() -> None:
    case = {"time_window": "2026-05-02"}
    assert _days_from_case(case, 30, today=date(2026, 5, 17)) == 17


def test_days_from_case_uses_earliest_date_in_range() -> None:
    case = {"time_window": "2026-04-28 至 2026-05-03"}
    assert _days_from_case(case, 30, today=date(2026, 5, 17)) == 21


def test_days_from_case_supports_explicit_relative_window() -> None:
    assert _days_from_case({"time_window": "14d"}, 30, today=date(2026, 5, 17)) == 14
    assert _days_from_case({"time_window": "最近7天"}, 30, today=date(2026, 5, 17)) == 7


def test_summary_by_query_type() -> None:
    summary = _summary_by_query_type(
        [
            {
                "query_type": "single_event",
                "scores": {"exact_hit_at_k": 1.0, "event_hit_at_k": 1.0, "mrr_at_k": 1.0},
            },
            {
                "query_type": "single_event",
                "scores": {"exact_hit_at_k": 0.0, "event_hit_at_k": 0.0, "mrr_at_k": 0.0},
            },
            {
                "query_type": "latest_update",
                "scores": {"exact_hit_at_k": 1.0, "event_hit_at_k": 1.0, "mrr_at_k": 0.5},
            },
        ]
    )
    assert summary["single_event"]["case_count"] == 2
    assert summary["single_event"]["exact_hit_rate"] == 0.5
    assert summary["latest_update"]["avg_mrr_at_k"] == 0.5


def test_summary_by_query_type_includes_broad_topic_metrics() -> None:
    summary = _summary_by_query_type(
        [
            {
                "query_type": "topic_overview",
                "case_kind": "broad_topic",
                "scores": {
                    "event_set_recall_at_k": 0.5,
                    "event_diversity_at_k": 2,
                    "irrelevant_event_ratio_at_k": 0.25,
                    "source_diversity_at_k": 2,
                },
            },
            {
                "query_type": "topic_overview",
                "case_kind": "broad_topic",
                "scores": {
                    "event_set_recall_at_k": 1.0,
                    "event_diversity_at_k": 4,
                    "irrelevant_event_ratio_at_k": 0.0,
                    "source_diversity_at_k": 3,
                },
            },
        ]
    )
    topic = summary["topic_overview"]
    assert topic["broad_topic_case_count"] == 2
    assert topic["avg_event_set_recall_at_k"] == 0.75
    assert topic["avg_event_diversity_at_k"] == 3.0
    assert topic["avg_irrelevant_event_ratio_at_k"] == 0.125
