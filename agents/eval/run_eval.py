"""Batch evaluation runner for TechNews agent stability."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from eval_core import (
        build_baseline_comparison,
        evaluate_case_outputs,
        evaluate_quality_gates,
        extract_urls,
        summarize_case_results,
    )
except ImportError:  # package-style import fallback
    from .eval_core import (
        build_baseline_comparison,
        evaluate_case_outputs,
        evaluate_quality_gates,
        extract_urls,
        summarize_case_results,
    )


def _bootstrap_imports() -> tuple[Any, Any, Any]:
    agents_dir = Path(__file__).resolve().parents[1]
    if str(agents_dir) not in sys.path:
        sys.path.insert(0, str(agents_dir))

    from agent import (  # pylint: disable=import-outside-toplevel
        generate_response,
        get_route_metrics_snapshot,
        reset_route_metrics,
    )

    return generate_response, get_route_metrics_snapshot, reset_route_metrics


def _load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc

            question = str(item.get("question", "")).strip()
            if not question:
                raise ValueError(f"Missing question at line {i}")

            case_id = str(item.get("id", f"case_{len(cases)+1}")).strip()
            case = {
                "id": case_id,
                "category": str(item.get("category", "general")),
                "question": question,
                "min_urls": int(item.get("min_urls", 0) or 0),
                "must_contain": list(item.get("must_contain", [])),
            }
            cases.append(case)
    return cases


def _run_case(
    case: dict[str, Any],
    runs_per_question: int,
    sleep_seconds: float,
    generate_response: Any,
    include_outputs: bool,
) -> dict[str, Any]:
    outputs: list[str] = []
    output_meta: list[dict[str, Any]] = []
    for idx in range(1, runs_per_question + 1):
        try:
            text = generate_response([], case["question"])
        except Exception as exc:  # noqa: BLE001 - eval should continue on failures
            text = f"[EVAL_ERROR] {type(exc).__name__}: {exc}"

        outputs.append(text)
        output_meta.append(
            {
                "run_index": idx,
                "url_count": len(extract_urls(text)),
            }
        )

        if sleep_seconds > 0 and idx < runs_per_question:
            time.sleep(sleep_seconds)

    metrics = evaluate_case_outputs(
        outputs=outputs,
        min_urls=case.get("min_urls", 0),
        must_contain=case.get("must_contain", []),
    )

    item: dict[str, Any] = {
        "id": case["id"],
        "category": case.get("category", "general"),
        "question": case["question"],
        "constraints": {
            "min_urls": case.get("min_urls", 0),
            "must_contain": case.get("must_contain", []),
        },
        "metrics": metrics,
        "runs": output_meta,
    }
    if include_outputs:
        item["outputs"] = outputs
    return item


def _build_arg_parser(default_dataset: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stability eval for TechNews agent.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--runs-per-question",
        type=int,
        default=3,
        help="How many repeated runs per question.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between repeated runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "latest.json",
        help="Where to write the report JSON.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, only run the first N cases.",
    )
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include full model outputs in report JSON.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline report JSON for regression comparison.",
    )
    parser.add_argument(
        "--fail-on-avg-error-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_error_rate is greater than this threshold.",
    )
    parser.add_argument(
        "--fail-on-fallback-rate-total",
        type=float,
        default=None,
        help="Fail if route_metrics.fallback_rate_total is greater than this threshold.",
    )
    parser.add_argument(
        "--fail-on-fallback-rate-langchain",
        type=float,
        default=None,
        help="Fail if route_metrics.fallback_rate_langchain is greater than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-min-url-hit-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_min_url_hit_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-phrase-hit-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_phrase_hit_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-pairwise-similarity",
        type=float,
        default=None,
        help="Fail if summary.avg_pairwise_similarity is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-langchain-success-rate",
        type=float,
        default=None,
        help="Fail if route_metrics.langchain_success_rate is lower than this threshold.",
    )
    return parser


def _build_gate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def _add(name: str, metric_path: str, op: str, value: float | None) -> None:
        if value is None:
            return
        specs.append(
            {
                "name": name,
                "metric_path": metric_path,
                "op": op,
                "threshold": float(value),
            }
        )

    _add(
        "avg_error_rate_max",
        "summary.avg_error_rate",
        "max",
        args.fail_on_avg_error_rate,
    )
    _add(
        "fallback_rate_total_max",
        "route_metrics.fallback_rate_total",
        "max",
        args.fail_on_fallback_rate_total,
    )
    _add(
        "fallback_rate_langchain_max",
        "route_metrics.fallback_rate_langchain",
        "max",
        args.fail_on_fallback_rate_langchain,
    )
    _add(
        "avg_min_url_hit_rate_min",
        "summary.avg_min_url_hit_rate",
        "min",
        args.fail_on_avg_min_url_hit_rate,
    )
    _add(
        "avg_phrase_hit_rate_min",
        "summary.avg_phrase_hit_rate",
        "min",
        args.fail_on_avg_phrase_hit_rate,
    )
    _add(
        "avg_pairwise_similarity_min",
        "summary.avg_pairwise_similarity",
        "min",
        args.fail_on_avg_pairwise_similarity,
    )
    _add(
        "langchain_success_rate_min",
        "route_metrics.langchain_success_rate",
        "min",
        args.fail_on_langchain_success_rate,
    )
    return specs


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_arg_parser(eval_dir / "questions_default.jsonl")
    args = parser.parse_args()

    if args.runs_per_question < 1:
        raise ValueError("--runs-per-question must be >= 1")

    agents_dir = eval_dir.parent
    env_path = agents_dir / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

    generate_response, get_route_metrics_snapshot, reset_route_metrics = _bootstrap_imports()

    dataset_path = args.dataset.resolve()
    cases = _load_cases(dataset_path)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    reset_route_metrics()

    started = time.time()
    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(
            _run_case(
                case=case,
                runs_per_question=args.runs_per_question,
                sleep_seconds=max(0.0, float(args.sleep_seconds)),
                generate_response=generate_response,
                include_outputs=bool(args.include_outputs),
            )
        )

    elapsed = time.time() - started
    summary = summarize_case_results(results)
    route_metrics = get_route_metrics_snapshot()

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "runs_per_question": args.runs_per_question,
        "elapsed_seconds": round(elapsed, 3),
        "summary": summary,
        "route_metrics": route_metrics,
        "cases": results,
    }

    if args.baseline:
        baseline_path = args.baseline.resolve()
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["baseline"] = {
            "path": str(baseline_path),
            "comparison": build_baseline_comparison(
                current_report=report,
                baseline_report=baseline_report,
            ),
        }

    gate_specs = _build_gate_specs(args)
    gate_result = evaluate_quality_gates(report, gate_specs)
    report["quality_gate"] = {
        "rules": gate_specs,
        "result": gate_result,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[Eval] cases={summary['case_count']} "
        f"runs={summary['run_count_total']} "
        f"avg_similarity={summary['avg_pairwise_similarity']:.3f} "
        f"avg_unique_ratio={summary['avg_unique_response_ratio']:.3f} "
        f"avg_url_hit={summary['avg_min_url_hit_rate']:.3f} "
        f"avg_error={summary['avg_error_rate']:.3f}"
    )
    print(
        "[Eval] route: "
        f"fallback_total={route_metrics.get('fallback_rate_total', 0.0):.2%} "
        f"fallback_langchain={route_metrics.get('fallback_rate_langchain', 0.0):.2%} "
        f"forced_route={route_metrics.get('forced_route_rate', 0.0):.2%}"
    )

    if "baseline" in report:
        bc = report["baseline"]["comparison"]
        print(
            "[Eval] baseline: "
            f"improved={bc.get('improved_count', 0)} "
            f"regressed={bc.get('regressed_count', 0)} "
            f"unchanged={bc.get('unchanged_count', 0)} "
            f"missing={bc.get('missing_count', 0)}"
        )

    print(
        "[Eval] gate: "
        f"rules={gate_result.get('total', 0)} "
        f"passed={gate_result.get('passed_count', 0)} "
        f"failed={gate_result.get('failed_count', 0)}"
    )
    for item in gate_result.get("failed", []):
        if item.get("reason"):
            print(
                f"[Eval][GateFail] {item.get('name')} {item.get('metric_path')} "
                f"reason={item.get('reason')}"
            )
        else:
            print(
                f"[Eval][GateFail] {item.get('name')} {item.get('metric_path')} "
                f"value={item.get('value')} op={item.get('op')} threshold={item.get('threshold')}"
            )

    print(f"[Eval] report={output_path}")

    return 0 if gate_result.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
