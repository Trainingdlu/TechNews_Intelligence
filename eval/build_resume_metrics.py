"""Build resume-ready metric artifacts from event-eval reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_CONFIG = {
    "name": "Baseline",
    "EVAL_RECALL_PROFILE": "base",
    "NEWS_RERANK_MODE": "none",
}

CANDIDATE_WIDE_CONFIG = {
    "name": "Candidate 1",
    "EVAL_RECALL_PROFILE": "wide",
    "NEWS_RERANK_MODE": "none",
}

CANDIDATE_RERANK_CONFIG = {
    "name": "Candidate 2",
    "EVAL_RECALL_PROFILE": "wide",
    "NEWS_RERANK_MODE": "llm_rerank",
}

METRIC_DEFINITIONS = {
    "single_event_hit_at_5": "Single-event retrieval: fraction of cases where top-k results contain any URL mapped to the gold_event_id.",
    "single_event_mrr_at_5": "Single-event retrieval: mean reciprocal rank of the first exact gold URL in top-k results.",
    "broad_event_set_recall_at_5": "Broad-topic retrieval: matched gold_event_ids divided by total gold_event_ids.",
    "broad_irrelevant_event_ratio_at_5": "Broad-topic retrieval: top-k result ratio not mapped to gold_event_ids or acceptable_event_ids.",
    "broad_event_diversity_at_5": "Broad-topic retrieval: average distinct gold events covered by top-k results.",
    "generation_claim_coverage": "Generation with fixed evidence: required atomic claims covered by the answer divided by required claims.",
    "generation_unsupported_url_rate": "Generation with fixed evidence: cited URLs not present in evidence divided by cited URLs.",
    "generation_forbidden_claim_hit_count": "Generation with fixed evidence: average forbidden-claim hits per answer.",
    "e2e_failure_attribution": "End-to-end cases grouped by trace-based attribution: retrieval, generation, guard/citation, evidence use, system, or OK.",
}


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get(data: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_metric_value(metric_name: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if any(token in metric_name for token in ("hit", "recall", "ratio", "rate", "coverage")):
        return _pct(value)
    return _num(value)


def _fingerprint_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    any_content = False
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.exists() or not path.is_file():
            continue
        any_content = True
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16] if any_content else ""


def _dataset_fingerprint(reports: list[dict[str, Any] | None]) -> str:
    paths: list[Path] = []
    digest = hashlib.sha256()
    for report in reports:
        if not isinstance(report, dict):
            continue
        for key in ("dataset", "events"):
            raw = str(report.get(key) or "").strip()
            if not raw:
                continue
            path = Path(raw)
            if path.exists():
                paths.append(path)
            else:
                digest.update(f"{key}={raw}".encode("utf-8"))
                digest.update(b"\0")
    file_fingerprint = _fingerprint_paths(paths)
    if file_fingerprint:
        digest.update(file_fingerprint.encode("utf-8"))
    return digest.hexdigest()[:16]


def _config_observed(report: dict[str, Any] | None, fallback: dict[str, str]) -> dict[str, Any]:
    env = _get(report, "env", {}) or {}
    return {
        "expected": fallback,
        "observed": {
            "EVAL_RECALL_PROFILE": env.get("EVAL_RECALL_PROFILE", ""),
            "NEWS_RERANK_MODE": env.get("NEWS_RERANK_MODE", ""),
            "JINA_API_KEY_PRESENT": env.get("JINA_API_KEY_PRESENT", False),
        },
    }


def _rerank_status(rerank_report: dict[str, Any] | None) -> dict[str, Any]:
    if not rerank_report:
        return {
            "included_in_resume_conclusion": False,
            "status": "skipped",
            "reason": "rerank report not provided",
        }
    env_has_key = bool(_get(rerank_report, "env.JINA_API_KEY_PRESENT", False))
    local_has_key = bool(os.getenv("JINA_API_KEY"))
    if not env_has_key and not local_has_key:
        return {
            "included_in_resume_conclusion": False,
            "status": "skipped",
            "reason": "JINA_API_KEY not available",
        }
    return {
        "included_in_resume_conclusion": True,
        "status": "included",
        "reason": "",
    }


def _audit_gate(
    *,
    audit_report: dict[str, Any] | None,
    manual_review_status: str,
    manual_fail_rate: float | None,
    manual_card_sample_size: int,
    manual_case_sample_size: int,
) -> dict[str, Any]:
    audit_fail_rate = _to_float(_get(audit_report, "summary.manual_fail_rate"))
    fail_rate = manual_fail_rate if manual_fail_rate is not None else audit_fail_rate
    status = str(manual_review_status or "unknown").strip().lower()
    if status not in {"unknown", "pass", "fail"}:
        status = "unknown"
    if audit_report and status == "unknown":
        status = str(_get(audit_report, "summary.manual_review_status", "unknown")).strip().lower()
    audit_allowed = _get(audit_report, "summary.allowed_for_resume")
    card_sample_size = manual_card_sample_size or _to_int(_get(audit_report, "summary.manual_card_sample_size"))
    case_sample_size = manual_case_sample_size or _to_int(_get(audit_report, "summary.manual_case_sample_size"))
    manual_ready = (
        status == "pass"
        and fail_rate is not None
        and fail_rate <= 0.10
        and card_sample_size >= 20
        and case_sample_size >= 30
    )
    blocked = (not manual_ready) or audit_allowed is False
    return {
        "manual_review_status": status,
        "manual_fail_rate": fail_rate,
        "manual_card_sample_size": card_sample_size,
        "manual_case_sample_size": case_sample_size,
        "max_allowed_manual_fail_rate": 0.10,
        "audit_allowed_for_resume": audit_allowed,
        "allowed_for_resume": not blocked,
    }


def _metric_card(
    *,
    name: str,
    n: int,
    baseline_value: float | None = None,
    candidate_value: float | None = None,
    generation_value: float | None = None,
    direction: str = "higher_better",
    dataset_fingerprint: str,
    run_date: str,
    includes_rerank: bool,
    manual_gate: dict[str, Any],
) -> dict[str, Any]:
    delta = None
    if baseline_value is not None and candidate_value is not None:
        delta = candidate_value - baseline_value
    return {
        "name": name,
        "definition": METRIC_DEFINITIONS[name],
        "n": n,
        "dataset_fingerprint": dataset_fingerprint,
        "baseline_config": BASELINE_CONFIG,
        "candidate_config": CANDIDATE_WIDE_CONFIG,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "generation_value": generation_value,
        "delta": delta,
        "direction": direction,
        "run_date": run_date,
        "includes_rerank": includes_rerank,
        "manual_review_status": manual_gate["manual_review_status"],
        "manual_fail_rate": manual_gate["manual_fail_rate"],
        "allowed_for_resume": bool(manual_gate["allowed_for_resume"]),
    }


def build_resume_metrics(
    *,
    baseline_retrieval: dict[str, Any],
    candidate_retrieval: dict[str, Any],
    rerank_retrieval: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
    e2e_baseline: dict[str, Any] | None = None,
    e2e_candidate: dict[str, Any] | None = None,
    audit_report: dict[str, Any] | None = None,
    manual_review_status: str = "unknown",
    manual_fail_rate: float | None = None,
    manual_card_sample_size: int = 0,
    manual_case_sample_size: int = 0,
) -> dict[str, Any]:
    run_date = datetime.now(timezone.utc).date().isoformat()
    fingerprint = _dataset_fingerprint(
        [baseline_retrieval, candidate_retrieval, rerank_retrieval, generation, e2e_baseline, e2e_candidate]
    )
    rerank = _rerank_status(rerank_retrieval)
    manual_gate = _audit_gate(
        audit_report=audit_report,
        manual_review_status=manual_review_status,
        manual_fail_rate=manual_fail_rate,
        manual_card_sample_size=manual_card_sample_size,
        manual_case_sample_size=manual_case_sample_size,
    )

    b_summary = baseline_retrieval.get("summary", {})
    c_summary = candidate_retrieval.get("summary", {})
    g_summary = (generation or {}).get("summary", {})

    metric_cards = [
        _metric_card(
            name="single_event_hit_at_5",
            n=min(_to_int(b_summary.get("single_event_case_count")), _to_int(c_summary.get("single_event_case_count"))),
            baseline_value=_to_float(b_summary.get("avg_single_event_hit_at_k", b_summary.get("avg_event_hit_at_k"))),
            candidate_value=_to_float(c_summary.get("avg_single_event_hit_at_k", c_summary.get("avg_event_hit_at_k"))),
            direction="higher_better",
            dataset_fingerprint=fingerprint,
            run_date=run_date,
            includes_rerank=False,
            manual_gate=manual_gate,
        ),
        _metric_card(
            name="single_event_mrr_at_5",
            n=min(_to_int(b_summary.get("single_event_case_count")), _to_int(c_summary.get("single_event_case_count"))),
            baseline_value=_to_float(b_summary.get("avg_single_mrr_at_k", b_summary.get("avg_mrr_at_k"))),
            candidate_value=_to_float(c_summary.get("avg_single_mrr_at_k", c_summary.get("avg_mrr_at_k"))),
            direction="higher_better",
            dataset_fingerprint=fingerprint,
            run_date=run_date,
            includes_rerank=False,
            manual_gate=manual_gate,
        ),
        _metric_card(
            name="broad_event_set_recall_at_5",
            n=min(_to_int(b_summary.get("broad_topic_case_count")), _to_int(c_summary.get("broad_topic_case_count"))),
            baseline_value=_to_float(b_summary.get("avg_event_set_recall_at_k")),
            candidate_value=_to_float(c_summary.get("avg_event_set_recall_at_k")),
            direction="higher_better",
            dataset_fingerprint=fingerprint,
            run_date=run_date,
            includes_rerank=False,
            manual_gate=manual_gate,
        ),
        _metric_card(
            name="broad_irrelevant_event_ratio_at_5",
            n=min(_to_int(b_summary.get("broad_topic_case_count")), _to_int(c_summary.get("broad_topic_case_count"))),
            baseline_value=_to_float(b_summary.get("avg_irrelevant_event_ratio_at_k")),
            candidate_value=_to_float(c_summary.get("avg_irrelevant_event_ratio_at_k")),
            direction="lower_better",
            dataset_fingerprint=fingerprint,
            run_date=run_date,
            includes_rerank=False,
            manual_gate=manual_gate,
        ),
        _metric_card(
            name="broad_event_diversity_at_5",
            n=min(_to_int(b_summary.get("broad_topic_case_count")), _to_int(c_summary.get("broad_topic_case_count"))),
            baseline_value=_to_float(b_summary.get("avg_event_diversity_at_k")),
            candidate_value=_to_float(c_summary.get("avg_event_diversity_at_k")),
            direction="higher_better",
            dataset_fingerprint=fingerprint,
            run_date=run_date,
            includes_rerank=False,
            manual_gate=manual_gate,
        ),
    ]

    if generation:
        metric_cards.extend(
            [
                _metric_card(
                    name="generation_claim_coverage",
                    n=_to_int(g_summary.get("case_count")),
                    generation_value=_to_float(g_summary.get("avg_claim_coverage")),
                    direction="higher_better",
                    dataset_fingerprint=fingerprint,
                    run_date=run_date,
                    includes_rerank=False,
                    manual_gate=manual_gate,
                ),
                _metric_card(
                    name="generation_unsupported_url_rate",
                    n=_to_int(g_summary.get("case_count")),
                    generation_value=_to_float(g_summary.get("avg_unsupported_url_rate")),
                    direction="lower_better",
                    dataset_fingerprint=fingerprint,
                    run_date=run_date,
                    includes_rerank=False,
                    manual_gate=manual_gate,
                ),
                _metric_card(
                    name="generation_forbidden_claim_hit_count",
                    n=_to_int(g_summary.get("case_count")),
                    generation_value=_to_float(g_summary.get("avg_forbidden_hit_count")),
                    direction="lower_better",
                    dataset_fingerprint=fingerprint,
                    run_date=run_date,
                    includes_rerank=False,
                    manual_gate=manual_gate,
                ),
            ]
        )

    e2e_summary = {
        "baseline_case_count": _to_int(_get(e2e_baseline, "summary.case_count")),
        "candidate_case_count": _to_int(_get(e2e_candidate, "summary.case_count")),
        "baseline_attribution_counts": _get(e2e_baseline, "summary.attribution_counts", {}) or {},
        "candidate_attribution_counts": _get(e2e_candidate, "summary.attribution_counts", {}) or {},
    }
    if e2e_baseline or e2e_candidate:
        metric_cards.append(
            _metric_card(
                name="e2e_failure_attribution",
                n=min(e2e_summary["baseline_case_count"], e2e_summary["candidate_case_count"])
                or max(e2e_summary["baseline_case_count"], e2e_summary["candidate_case_count"]),
                generation_value=None,
                direction="diagnostic",
                dataset_fingerprint=fingerprint,
                run_date=run_date,
                includes_rerank=False,
                manual_gate=manual_gate,
            )
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "manual_quality_gate": manual_gate,
        "system_under_test": {
            "baseline": _config_observed(baseline_retrieval, BASELINE_CONFIG),
            "candidate_1": _config_observed(candidate_retrieval, CANDIDATE_WIDE_CONFIG),
            "candidate_2": {
                **_config_observed(rerank_retrieval, CANDIDATE_RERANK_CONFIG),
                **rerank,
            },
        },
        "metric_cards": metric_cards,
        "e2e_summary": e2e_summary,
    }


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["manual_quality_gate"]
    cards = report["metric_cards"]
    allowed = bool(gate["allowed_for_resume"])
    lines = [
        "# Resume Metrics",
        "",
        f"- Dataset fingerprint: `{report['dataset_fingerprint']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Manual review: `{gate['manual_review_status']}`; fail rate: `{_pct(gate['manual_fail_rate'])}`",
        f"- Resume use allowed: `{allowed}`",
        "",
        "## System Under Test",
        "",
        "- Baseline: `EVAL_RECALL_PROFILE=base`, `NEWS_RERANK_MODE=none`",
        "- Candidate 1: `EVAL_RECALL_PROFILE=wide`, `NEWS_RERANK_MODE=none`",
        "- Candidate 2: `EVAL_RECALL_PROFILE=wide`, `NEWS_RERANK_MODE=llm_rerank`",
    ]
    candidate_2 = report["system_under_test"]["candidate_2"]
    if candidate_2.get("status") == "skipped":
        lines.append(f"- Candidate 2 status: skipped ({candidate_2.get('reason')})")
    lines.extend(["", "## Metric Cards", ""])
    for card in cards:
        if card.get("generation_value") is not None:
            value = _format_metric_value(card["name"], card["generation_value"])
            lines.append(f"- `{card['name']}`: N={card['n']}, value={value}. {card['definition']}")
            continue
        base = _format_metric_value(card["name"], card.get("baseline_value"))
        cand = _format_metric_value(card["name"], card.get("candidate_value"))
        delta = _format_metric_value(card["name"], card.get("delta"))
        lines.append(f"- `{card['name']}`: N={card['n']}, baseline={base}, candidate={cand}, delta={delta}. {card['definition']}")

    lines.extend(["", "## Resume-Ready Statements", ""])
    if not allowed:
        lines.append("Manual review gate failed or is marked failed. Do not use these numbers in the resume yet.")
        return "\n".join(lines) + "\n"

    by_name = {card["name"]: card for card in cards}
    single = by_name.get("single_event_hit_at_5")
    if single and single.get("baseline_value") is not None and single.get("candidate_value") is not None:
        lines.append(
            f"- 在 {single['n']} 条单事件检索样本上，event hit@5 从 {_pct(single['baseline_value'])} 提升到 {_pct(single['candidate_value'])}。"
        )
    broad = by_name.get("broad_event_set_recall_at_5")
    irrelevant = by_name.get("broad_irrelevant_event_ratio_at_5")
    if broad and irrelevant and broad.get("baseline_value") is not None and broad.get("candidate_value") is not None:
        lines.append(
            f"- 在 {broad['n']} 条主题概览样本上，事件集合覆盖率从 {_pct(broad['baseline_value'])} 提升到 {_pct(broad['candidate_value'])}，无关事件比例从 {_pct(irrelevant.get('baseline_value'))} 降到 {_pct(irrelevant.get('candidate_value'))}。"
        )
    gen = by_name.get("generation_unsupported_url_rate")
    if gen and gen.get("generation_value") is not None:
        lines.append(f"- 在 {gen['n']} 条固定证据生成样本上，非证据 URL 泄漏率为 {_pct(gen['generation_value'])}。")
    e2e = by_name.get("e2e_failure_attribution")
    if e2e:
        lines.append(f"- 在 {e2e['n']} 条端到端样本中，通过 Trace 将失败归因到检索、生成、引用守卫、证据使用和系统错误等环节。")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resume-ready metric artifacts.")
    parser.add_argument("--baseline-retrieval", type=Path, required=True)
    parser.add_argument("--candidate-retrieval", type=Path, required=True)
    parser.add_argument("--rerank-retrieval", type=Path, default=None)
    parser.add_argument("--generation", type=Path, default=None)
    parser.add_argument("--e2e-baseline", type=Path, default=None)
    parser.add_argument("--e2e-candidate", type=Path, default=None)
    parser.add_argument("--audit-report", type=Path, default=None)
    parser.add_argument("--manual-review-status", choices=["unknown", "pass", "fail"], default="unknown")
    parser.add_argument("--manual-fail-rate", type=float, default=None)
    parser.add_argument("--manual-card-sample-size", type=int, default=0)
    parser.add_argument("--manual-case-sample-size", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=Path("eval/reports/resume_metrics.json"))
    parser.add_argument("--output-md", type=Path, default=Path("eval/reports/resume_metrics.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _read_json(args.baseline_retrieval)
    candidate = _read_json(args.candidate_retrieval)
    if baseline is None:
        raise FileNotFoundError(f"baseline retrieval report not found: {args.baseline_retrieval}")
    if candidate is None:
        raise FileNotFoundError(f"candidate retrieval report not found: {args.candidate_retrieval}")

    report = build_resume_metrics(
        baseline_retrieval=baseline,
        candidate_retrieval=candidate,
        rerank_retrieval=_read_json(args.rerank_retrieval),
        generation=_read_json(args.generation),
        e2e_baseline=_read_json(args.e2e_baseline),
        e2e_candidate=_read_json(args.e2e_candidate),
        audit_report=_read_json(args.audit_report),
        manual_review_status=args.manual_review_status,
        manual_fail_rate=args.manual_fail_rate,
        manual_card_sample_size=max(0, int(args.manual_card_sample_size)),
        manual_case_sample_size=max(0, int(args.manual_case_sample_size)),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[ResumeMetrics] json={args.output_json}")
    print(f"[ResumeMetrics] markdown={args.output_md}")


if __name__ == "__main__":
    main()
