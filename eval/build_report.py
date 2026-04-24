"""Build final markdown report for recall/analysis matrix evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON root object: {path}")
    return payload


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _metric_value(group: dict[str, Any], metric: str, field: str = "current") -> Any:
    metrics = group.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    row = metrics.get(metric, {})
    if not isinstance(row, dict):
        return None
    return row.get(field)


def _find_baseline_group(report: dict[str, Any]) -> str:
    return str(report.get("baseline_group", "")).strip()


def _find_group(report: dict[str, Any], group_id: str) -> dict[str, Any] | None:
    for group in report.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        if str(group.get("group_id", "")).strip() == group_id:
            return group
    return None


def _select_best_group(main_report: dict[str, Any]) -> str:
    baseline = _find_baseline_group(main_report)
    best_group = baseline
    best_score = float("-inf")
    for group in main_report.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        gid = str(group.get("group_id", "")).strip()
        if not gid or gid == baseline:
            continue
        score = _metric_value(group, "retrieval_rcs")
        if score is None:
            score = _metric_value(group, "avg_recall_at_10")
        try:
            value = float(score)
        except Exception:
            continue
        if value > best_score:
            best_score = value
            best_group = gid
    return best_group


def _render_retrieval_section(main_report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Retrieval Leaderboard")
    lines.append("")
    lines.append("| Group | RCS | Recall@10 | MRR@10 | nDCG@10 | Hit@5 | Hit@10 | DepthGain | ErrorRate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in main_report.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        gid = str(group.get("group_id", "")).strip() or "-"
        lines.append(
            "| "
            f"{gid} | "
            f"{_fmt_float(_metric_value(group, 'retrieval_rcs'))} | "
            f"{_fmt_float(_metric_value(group, 'avg_recall_at_10'))} | "
            f"{_fmt_float(_metric_value(group, 'avg_mrr_at_10'))} | "
            f"{_fmt_float(_metric_value(group, 'avg_ndcg_at_10'))} | "
            f"{_fmt_float(_metric_value(group, 'avg_hit_rate_at_5'))} | "
            f"{_fmt_float(_metric_value(group, 'avg_hit_rate_at_10'))} | "
            f"{_fmt_float(_metric_value(group, 'retrieval_depth_gain'))} | "
            f"{_fmt_float(_metric_value(group, 'avg_error_rate'))} |"
        )
    lines.append("")
    return lines


def _render_empty_risk_section(main_report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Empty Risk Board")
    lines.append("")
    lines.append("| Group | Empty Response Acc | Empty Hallucination Rate | Recovery Suggestion Rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for group in main_report.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        gid = str(group.get("group_id", "")).strip() or "-"
        lines.append(
            "| "
            f"{gid} | "
            f"{_fmt_float(_metric_value(group, 'empty_response_accuracy'))} | "
            f"{_fmt_float(_metric_value(group, 'empty_hallucination_rate'))} | "
            f"{_fmt_float(_metric_value(group, 'empty_recovery_suggestion_rate'))} |"
        )
    lines.append("")
    return lines


def _render_analysis_section(main_report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Analysis Board")
    lines.append("")
    lines.append("| Group | Claim Support | Unsupported Claim Rate | Contradiction Rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for group in main_report.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        gid = str(group.get("group_id", "")).strip() or "-"
        lines.append(
            "| "
            f"{gid} | "
            f"{_fmt_float(_metric_value(group, 'analysis_claim_support_rate'))} | "
            f"{_fmt_float(_metric_value(group, 'analysis_unsupported_claim_rate'))} | "
            f"{_fmt_float(_metric_value(group, 'analysis_contradiction_rate'))} |"
        )
    lines.append("")
    return lines


def _render_baseline_vs_best(main_report: dict[str, Any], best_group_id: str) -> list[str]:
    baseline_id = _find_baseline_group(main_report)
    baseline = _find_group(main_report, baseline_id)
    best = _find_group(main_report, best_group_id)
    lines: list[str] = []
    lines.append("## Baseline vs Best")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_id}`")
    lines.append(f"- Best Group: `{best_group_id}`")
    lines.append("")
    lines.append("| Metric | Baseline | Best | Delta(best-baseline) |")
    lines.append("| --- | ---: | ---: | ---: |")
    metrics = [
        ("retrieval_rcs", "RCS"),
        ("avg_recall_at_10", "Recall@10"),
        ("avg_mrr_at_10", "MRR@10"),
        ("avg_ndcg_at_10", "nDCG@10"),
        ("avg_hit_rate_at_10", "Hit@10"),
        ("empty_hallucination_rate", "EmptyHallucinationRate"),
        ("avg_error_rate", "ErrorRate"),
    ]
    for key, label in metrics:
        base_val = _metric_value(baseline or {}, key)
        best_val = _metric_value(best or {}, key)
        delta = None
        if base_val is not None and best_val is not None:
            try:
                delta = float(best_val) - float(base_val)
            except Exception:
                delta = None
        lines.append(f"| {label} | {_fmt_float(base_val)} | {_fmt_float(best_val)} | {_fmt_float(delta)} |")
    lines.append("")
    return lines


def _render_judge_section(judge_report: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    lines.append("## Judge Audit (Baseline + Best)")
    lines.append("")
    if judge_report is None:
        lines.append("- Judge leaderboard not provided; skipped.")
        lines.append("")
        return lines

    lines.append("| Group | Faithfulness | Relevancy |")
    lines.append("| --- | ---: | ---: |")
    for group in judge_report.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        gid = str(group.get("group_id", "")).strip() or "-"
        lines.append(
            "| "
            f"{gid} | "
            f"{_fmt_float(_metric_value(group, 'generation_faithfulness'))} | "
            f"{_fmt_float(_metric_value(group, 'generation_relevancy'))} |"
        )
    lines.append("")
    return lines


def _render_recommendations(main_report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Recommendations")
    lines.append("")
    items = main_report.get("recommendations", []) or []
    if not isinstance(items, list) or not items:
        lines.append("- No auto recommendation.")
    else:
        for item in items:
            lines.append(f"- {str(item)}")
    lines.append("")
    return lines


def _parse_args() -> argparse.Namespace:
    eval_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build final evaluation markdown report.")
    parser.add_argument("--main-leaderboard-json", type=Path, required=True, help="Main matrix leaderboard JSON path.")
    parser.add_argument("--main-leaderboard-md", type=Path, default=None, help="Main matrix leaderboard markdown path.")
    parser.add_argument("--judge-leaderboard-json", type=Path, default=None, help="Judge matrix leaderboard JSON path.")
    parser.add_argument("--run-id", type=str, required=True, help="Pipeline run id.")
    parser.add_argument("--dataset-version", type=str, required=True, help="Frozen dataset version.")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=eval_dir / "reports" / "final" / "latest.md",
        help="Output markdown path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    main_json = args.main_leaderboard_json.resolve()
    if not main_json.exists():
        raise FileNotFoundError(f"Main leaderboard JSON not found: {main_json}")
    main_report = _load_json(main_json)

    judge_report = None
    if args.judge_leaderboard_json:
        judge_json = args.judge_leaderboard_json.resolve()
        if judge_json.exists():
            judge_report = _load_json(judge_json)

    best_group_id = _select_best_group(main_report)
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"- Generated At (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Run ID: `{str(args.run_id).strip()}`")
    lines.append(f"- Dataset Version: `{str(args.dataset_version).strip()}`")
    lines.append(f"- Main Leaderboard JSON: `{main_json}`")
    if args.main_leaderboard_md:
        lines.append(f"- Main Leaderboard MD: `{str(args.main_leaderboard_md.resolve())}`")
    if args.judge_leaderboard_json:
        lines.append(f"- Judge Leaderboard JSON: `{str(args.judge_leaderboard_json.resolve())}`")
    lines.append("")

    lines.extend(_render_retrieval_section(main_report))
    lines.extend(_render_empty_risk_section(main_report))
    lines.extend(_render_analysis_section(main_report))
    lines.extend(_render_baseline_vs_best(main_report, best_group_id))
    lines.extend(_render_judge_section(judge_report))
    lines.extend(_render_recommendations(main_report))

    output = args.output_md.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    print(f"[EvalReport] main_leaderboard={main_json}")
    if args.judge_leaderboard_json:
        print(f"[EvalReport] judge_leaderboard={args.judge_leaderboard_json.resolve()}")
    print(f"[EvalReport] output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

