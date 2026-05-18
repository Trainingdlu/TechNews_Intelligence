"""Quality gate for event-card driven evaluation datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from news_eval_schema import (
        load_e2e_cases,
        load_event_cards,
        load_generation_cases,
        load_retrieval_cases,
        read_jsonl,
    )
except ImportError:  # pragma: no cover
    from .news_eval_schema import (
        load_e2e_cases,
        load_event_cards,
        load_generation_cases,
        load_retrieval_cases,
        read_jsonl,
    )


MIN_EVENT_CARDS = 100
MIN_SINGLE_RETRIEVAL_CASES = 150
MIN_BROAD_RETRIEVAL_CASES = 50
MIN_GENERATION_CASES = 50
MIN_E2E_CASES = 50
MIN_MANUAL_EVENT_CARD_SAMPLE = 20
MIN_MANUAL_CASE_SAMPLE = 30
MAX_MANUAL_FAIL_RATE = 0.10


def _issue(code: str, item_type: str, item_id: str, detail: str) -> dict[str, str]:
    return {"code": code, "item_type": item_type, "item_id": item_id, "detail": detail}


def _audit_event_cards(cards: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for card in cards:
        event_id = str(card.get("event_id", "")).strip()
        if len(card.get("core_urls", []) or []) < 1:
            issues.append(_issue("missing_core_url", "event_card", event_id, "event card needs at least one core URL"))
        if not card.get("facts"):
            issues.append(_issue("missing_facts", "event_card", event_id, "event card needs verifiable facts"))
        if not card.get("entities"):
            issues.append(_issue("missing_entities", "event_card", event_id, "event card needs entities for broad-topic diversity"))
        if len(card.get("facts", []) or []) > 8:
            issues.append(_issue("too_many_facts", "event_card", event_id, "large fact sets often indicate multiple events merged into one card"))
    return issues


def _audit_retrieval_cases(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        case_kind = str(case.get("case_kind", "single_event")).strip()
        if case_kind == "single_event":
            if not case.get("gold_event_id"):
                issues.append(_issue("missing_gold_event_id", "retrieval_case", case_id, "single-event case needs gold_event_id"))
            if not case.get("gold_urls"):
                issues.append(_issue("missing_gold_urls", "retrieval_case", case_id, "single-event case needs gold_urls"))
        elif case_kind == "broad_topic":
            if len(case.get("gold_event_ids", []) or []) < 2:
                issues.append(_issue("broad_topic_too_small", "retrieval_case", case_id, "broad-topic case needs at least two gold_event_ids"))
            if not case.get("topic"):
                issues.append(_issue("missing_topic", "retrieval_case", case_id, "broad-topic case needs topic"))
            if not case.get("expected_entities") and not case.get("expected_event_types"):
                issues.append(
                    _issue(
                        "missing_broad_facets",
                        "retrieval_case",
                        case_id,
                        "broad-topic case should expose expected_entities or expected_event_types",
                    )
                )
        else:
            issues.append(_issue("unknown_case_kind", "retrieval_case", case_id, case_kind))
    return issues


def _audit_generation_cases(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        claims = case.get("required_claims", []) or []
        sources = case.get("required_claim_sources", []) or []
        if len(sources) != len(claims):
            issues.append(_issue("claim_source_mismatch", "generation_case", case_id, "each required claim needs one evidence URL mapping"))
        for claim in claims:
            text = str(claim or "").strip()
            if len(text) > 120:
                issues.append(_issue("claim_not_atomic", "generation_case", case_id, "required claim is too long to be reliably checked"))
                break
    return issues


def _manual_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [row for row in payload["items"] if isinstance(row, dict)]
    return []


def _manual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    card_rows = [row for row in rows if str(row.get("item_type", "")).lower() in {"event_card", "card"}]
    case_rows = [row for row in rows if str(row.get("item_type", "")).lower() in {"case", "retrieval_case", "generation_case", "e2e_case"}]
    failed = [row for row in rows if row.get("passed") is False or str(row.get("status", "")).lower() in {"fail", "failed"}]
    total = len(rows)
    fail_rate = (len(failed) / total) if total else None
    if total == 0:
        status = "unknown"
    elif fail_rate is not None and fail_rate <= MAX_MANUAL_FAIL_RATE:
        status = "pass"
    else:
        status = "fail"
    return {
        "manual_review_status": status,
        "manual_card_sample_size": len(card_rows),
        "manual_case_sample_size": len(case_rows),
        "manual_fail_count": len(failed),
        "manual_fail_rate": fail_rate,
    }


def build_quality_report(
    *,
    event_cards: list[dict[str, Any]],
    retrieval_cases: list[dict[str, Any]],
    generation_cases: list[dict[str, Any]],
    e2e_cases: list[dict[str, Any]],
    manual_review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    single_cases = [case for case in retrieval_cases if case.get("case_kind") == "single_event"]
    broad_cases = [case for case in retrieval_cases if case.get("case_kind") == "broad_topic"]
    issues = (
        _audit_event_cards(event_cards)
        + _audit_retrieval_cases(retrieval_cases)
        + _audit_generation_cases(generation_cases)
    )
    manual = _manual_summary(manual_review_rows)
    acceptance = {
        "event_cards": {"actual": len(event_cards), "required": MIN_EVENT_CARDS, "ok": len(event_cards) >= MIN_EVENT_CARDS},
        "single_event_retrieval_cases": {
            "actual": len(single_cases),
            "required": MIN_SINGLE_RETRIEVAL_CASES,
            "ok": len(single_cases) >= MIN_SINGLE_RETRIEVAL_CASES,
        },
        "broad_topic_retrieval_cases": {
            "actual": len(broad_cases),
            "required": MIN_BROAD_RETRIEVAL_CASES,
            "ok": len(broad_cases) >= MIN_BROAD_RETRIEVAL_CASES,
        },
        "generation_cases": {
            "actual": len(generation_cases),
            "required": MIN_GENERATION_CASES,
            "ok": len(generation_cases) >= MIN_GENERATION_CASES,
        },
        "e2e_cases": {"actual": len(e2e_cases), "required": MIN_E2E_CASES, "ok": len(e2e_cases) >= MIN_E2E_CASES},
        "manual_event_card_sample": {
            "actual": manual["manual_card_sample_size"],
            "required": MIN_MANUAL_EVENT_CARD_SAMPLE,
            "ok": manual["manual_card_sample_size"] >= MIN_MANUAL_EVENT_CARD_SAMPLE,
        },
        "manual_case_sample": {
            "actual": manual["manual_case_sample_size"],
            "required": MIN_MANUAL_CASE_SAMPLE,
            "ok": manual["manual_case_sample_size"] >= MIN_MANUAL_CASE_SAMPLE,
        },
        "manual_fail_rate": {
            "actual": manual["manual_fail_rate"],
            "required_max": MAX_MANUAL_FAIL_RATE,
            "ok": manual["manual_fail_rate"] is not None and manual["manual_fail_rate"] <= MAX_MANUAL_FAIL_RATE,
        },
    }
    deterministic_ok = not issues
    acceptance_ok = all(item.get("ok") is True for item in acceptance.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "event_card_count": len(event_cards),
            "retrieval_case_count": len(retrieval_cases),
            "single_event_retrieval_case_count": len(single_cases),
            "broad_topic_retrieval_case_count": len(broad_cases),
            "generation_case_count": len(generation_cases),
            "e2e_case_count": len(e2e_cases),
            "deterministic_issue_count": len(issues),
            **manual,
            "allowed_for_resume": deterministic_ok and acceptance_ok,
        },
        "acceptance": acceptance,
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit event-eval dataset quality.")
    parser.add_argument("--events", type=Path, default=Path("eval/datasets/event_cards.jsonl"))
    parser.add_argument("--retrieval", type=Path, default=Path("eval/datasets/retrieval_cases.jsonl"))
    parser.add_argument("--generation", type=Path, default=Path("eval/datasets/generation_cases.jsonl"))
    parser.add_argument("--e2e", type=Path, default=Path("eval/datasets/e2e_cases.jsonl"))
    parser.add_argument("--manual-review", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval/reports/event_eval_quality_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_quality_report(
        event_cards=load_event_cards(args.events),
        retrieval_cases=load_retrieval_cases(args.retrieval),
        generation_cases=load_generation_cases(args.generation),
        e2e_cases=load_e2e_cases(args.e2e),
        manual_review_rows=_manual_rows(args.manual_review),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[EventEvalQuality] output={args.output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
