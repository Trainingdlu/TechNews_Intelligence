from __future__ import annotations

from eval.audit_event_eval_quality import build_quality_report


def test_quality_report_blocks_when_manual_review_is_missing() -> None:
    report = build_quality_report(
        event_cards=[],
        retrieval_cases=[],
        generation_cases=[],
        e2e_cases=[],
        manual_review_rows=[],
    )
    assert report["summary"]["allowed_for_resume"] is False
    assert report["acceptance"]["manual_event_card_sample"]["ok"] is False
    assert report["acceptance"]["manual_fail_rate"]["ok"] is False


def test_quality_report_counts_manual_samples_and_fail_rate() -> None:
    manual_rows = (
        [{"item_type": "event_card", "item_id": f"card_{idx}", "passed": True} for idx in range(20)]
        + [{"item_type": "case", "item_id": f"case_{idx}", "passed": True} for idx in range(29)]
        + [{"item_type": "case", "item_id": "case_bad", "passed": False}]
    )
    report = build_quality_report(
        event_cards=[],
        retrieval_cases=[],
        generation_cases=[],
        e2e_cases=[],
        manual_review_rows=manual_rows,
    )
    assert report["summary"]["manual_card_sample_size"] == 20
    assert report["summary"]["manual_case_sample_size"] == 30
    assert report["summary"]["manual_fail_rate"] == 1 / 50
    assert report["acceptance"]["manual_fail_rate"]["ok"] is True
