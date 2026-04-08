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
    from capabilities import CAPABILITY_CATALOG, supported_capabilities
    from dataset_loader import (
        filter_eval_cases,
        load_eval_cases,
        parse_csv_filter_arg,
        summarize_case_matrix,
    )
    from eval_core import (
        build_baseline_comparison,
        evaluate_case_outputs,
        evaluate_quality_gates,
        extract_urls,
        summarize_case_results,
    )
except ImportError:  # package-style import fallback
    from .capabilities import CAPABILITY_CATALOG, supported_capabilities
    from .dataset_loader import (
        filter_eval_cases,
        load_eval_cases,
        parse_csv_filter_arg,
        summarize_case_matrix,
    )
    from .eval_core import (
        build_baseline_comparison,
        evaluate_case_outputs,
        evaluate_quality_gates,
        extract_urls,
        summarize_case_results,
    )


ROUTE_METRICS_SCHEMA_VERSION = 3


def _bootstrap_imports() -> tuple[Any, Any, Any, Any]:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from agent import (  # pylint: disable=import-outside-toplevel
        generate_response_eval_payload,
        get_last_tool_calls_snapshot,
        get_route_metrics_snapshot,
        reset_route_metrics,
    )

    return (
        generate_response_eval_payload,
        get_last_tool_calls_snapshot,
        get_route_metrics_snapshot,
        reset_route_metrics,
    )


def _run_case(
    case: dict[str, Any],
    runs_per_question: int,
    sleep_seconds: float,
    generate_response_eval_payload: Any,
    get_last_tool_calls_snapshot: Any,
    include_outputs: bool,
) -> dict[str, Any]:
    outputs: list[str] = []
    output_meta: list[dict[str, Any]] = []
    tool_calls_by_run: list[list[str]] = []
    for idx in range(1, runs_per_question + 1):
        try:
            payload = generate_response_eval_payload([], case["question"])
            text = str(payload.get("text", ""))
            tools = payload.get("tool_calls", [])
            if not isinstance(tools, list):
                tools = []
        except Exception as exc:  # noqa: BLE001 - eval should continue on failures
            text = f"[EVAL_ERROR] {type(exc).__name__}: {exc}"
            try:
                tools = get_last_tool_calls_snapshot()
            except Exception:
                tools = []

        outputs.append(text)
        tool_calls_by_run.append([str(t).strip() for t in tools if str(t).strip()])
        output_meta.append(
            {
                "run_index": idx,
                "url_count": len(extract_urls(text)),
                "tool_calls": tool_calls_by_run[-1],
            }
        )

        if sleep_seconds > 0 and idx < runs_per_question:
            time.sleep(sleep_seconds)

    metrics = evaluate_case_outputs(
        outputs=outputs,
        min_urls=case.get("min_urls", 0),
        must_contain=case.get("must_contain", []),
        expected_facts=case.get("expected_facts", []),
        expected_fact_groups=case.get("expected_fact_groups", []),
        required_tools=case.get("required_tools", []),
        acceptable_tool_paths=case.get("acceptable_tool_paths", []),
        must_not_contain=case.get("must_not_contain", []),
        expected_source_domains=case.get("expected_source_domains", []),
        run_tool_calls=tool_calls_by_run,
    )

    item: dict[str, Any] = {
        "id": case["id"],
        "category": case.get("category", "general"),
        "capability": case.get("capability", "general_qa"),
        "question": case["question"],
        "constraints": {
            "min_urls": case.get("min_urls", 0),
            "must_contain": case.get("must_contain", []),
            "expected_facts": case.get("expected_facts", []),
            "expected_fact_groups": case.get("expected_fact_groups", []),
            "required_tools": case.get("required_tools", []),
            "acceptable_tool_paths": case.get("acceptable_tool_paths", []),
            "must_not_contain": case.get("must_not_contain", []),
            "expected_source_domains": case.get("expected_source_domains", []),
        },
        "tags": case.get("tags", []),
        "metrics": metrics,
        "runs": output_meta,
    }
    if include_outputs:
        item["outputs"] = outputs
    return item


def _build_arg_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stability eval for TechNews agent.")
    parser.add_argument(
        "--suite",
        type=str,
        default="default",
        help="Dataset suite name under eval/datasets (default|smoke).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional JSONL dataset path. If set, --suite is ignored.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated category filter (e.g. compare,timeline,landscape).",
    )
    parser.add_argument(
        "--capabilities",
        type=str,
        default="",
        help="Comma-separated capability filter (e.g. compare_topics,timeline).",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include cases marked as enabled=false in dataset.",
    )
    parser.add_argument(
        "--strict-capability-check",
        dest="strict_capability_check",
        action="store_true",
        default=True,
        help="Fail fast if dataset references unsupported capabilities.",
    )
    parser.add_argument(
        "--no-strict-capability-check",
        dest="strict_capability_check",
        action="store_false",
        help="Do not fail when dataset has unsupported capabilities (fallback to general_qa).",
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
        default=eval_dir / "reports" / "latest.json",
        help="Where to write the report JSON.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, only run the first N selected cases.",
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
        "--fail-on-react-error-rate",
        type=float,
        default=None,
        help="Fail if route_metrics.react_error_rate is greater than this threshold.",
    )
    parser.add_argument(
        "--fail-on-react-success-rate",
        type=float,
        default=None,
        help="Fail if route_metrics.react_success_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-react-recursion-limit-rate",
        type=float,
        default=None,
        help="Fail if route_metrics.react_recursion_limit_rate is greater than this threshold.",
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
        "--fail-on-avg-fact-hit-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_fact_hit_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-tool-path-hit-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_tool_path_hit_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-fact-group-hit-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_fact_group_hit_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-tool-path-accept-hit-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_tool_path_accept_hit_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-source-domain-hit-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_source_domain_hit_rate is lower than this threshold.",
    )
    parser.add_argument(
        "--fail-on-avg-forbidden-claim-rate",
        type=float,
        default=None,
        help="Fail if summary.avg_forbidden_claim_rate is greater than this threshold.",
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

    _add("avg_error_rate_max", "summary.avg_error_rate", "max", args.fail_on_avg_error_rate)
    _add(
        "react_error_rate_max",
        "route_metrics.react_error_rate",
        "max",
        args.fail_on_react_error_rate,
    )
    _add(
        "react_recursion_limit_rate_max",
        "route_metrics.react_recursion_limit_rate",
        "max",
        args.fail_on_react_recursion_limit_rate,
    )
    _add(
        "avg_min_url_hit_rate_min",
        "summary.avg_min_url_hit_rate",
        "min",
        args.fail_on_avg_min_url_hit_rate,
    )
    _add("avg_phrase_hit_rate_min", "summary.avg_phrase_hit_rate", "min", args.fail_on_avg_phrase_hit_rate)
    _add(
        "avg_pairwise_similarity_min",
        "summary.avg_pairwise_similarity",
        "min",
        args.fail_on_avg_pairwise_similarity,
    )
    _add(
        "avg_fact_hit_rate_min",
        "summary.avg_fact_hit_rate",
        "min",
        args.fail_on_avg_fact_hit_rate,
    )
    _add(
        "avg_tool_path_hit_rate_min",
        "summary.avg_tool_path_hit_rate",
        "min",
        args.fail_on_avg_tool_path_hit_rate,
    )
    _add(
        "avg_fact_group_hit_rate_min",
        "summary.avg_fact_group_hit_rate",
        "min",
        args.fail_on_avg_fact_group_hit_rate,
    )
    _add(
        "avg_tool_path_accept_hit_rate_min",
        "summary.avg_tool_path_accept_hit_rate",
        "min",
        args.fail_on_avg_tool_path_accept_hit_rate,
    )
    _add(
        "avg_source_domain_hit_rate_min",
        "summary.avg_source_domain_hit_rate",
        "min",
        args.fail_on_avg_source_domain_hit_rate,
    )
    _add(
        "avg_forbidden_claim_rate_max",
        "summary.avg_forbidden_claim_rate",
        "max",
        args.fail_on_avg_forbidden_claim_rate,
    )
    _add(
        "react_success_rate_min",
        "route_metrics.react_success_rate",
        "min",
        args.fail_on_react_success_rate,
    )
    return specs


def _resolve_dataset_path(eval_dir: Path, args: argparse.Namespace) -> Path:
    if args.dataset:
        return args.dataset.resolve()
    return (eval_dir / "datasets" / f"{args.suite.strip().lower()}.jsonl").resolve()


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_arg_parser(eval_dir)
    args = parser.parse_args()

    if args.runs_per_question < 1:
        raise ValueError("--runs-per-question must be >= 1")

    project_root = eval_dir.parent
    env_path = project_root / "agent" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

    dataset_path = _resolve_dataset_path(eval_dir, args)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cases = load_eval_cases(
        dataset_path,
        strict_capability_check=bool(args.strict_capability_check),
        include_disabled=bool(args.include_disabled),
    )

    category_filter = parse_csv_filter_arg(args.categories)
    capability_filter = parse_csv_filter_arg(args.capabilities)
    if capability_filter:
        unknown = capability_filter.difference(supported_capabilities())
        if unknown:
            raise ValueError(
                f"Unknown capabilities in --capabilities: {sorted(unknown)}; "
                f"supported={sorted(supported_capabilities())}"
            )

    cases = filter_eval_cases(
        cases,
        categories=category_filter,
        capabilities=capability_filter,
    )
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("No eval cases selected. Check dataset/filter arguments.")

    selection = summarize_case_matrix(cases)
    selection["suite"] = args.suite
    selection["dataset"] = str(dataset_path)
    selection["filters"] = {
        "categories": sorted(category_filter),
        "capabilities": sorted(capability_filter),
        "max_cases": int(args.max_cases or 0),
        "include_disabled": bool(args.include_disabled),
        "strict_capability_check": bool(args.strict_capability_check),
    }

    (
        generate_response_eval_payload,
        get_last_tool_calls_snapshot,
        get_route_metrics_snapshot,
        reset_route_metrics,
    ) = _bootstrap_imports()

    reset_route_metrics()

    started = time.time()
    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(
            _run_case(
                case=case,
                runs_per_question=args.runs_per_question,
                sleep_seconds=max(0.0, float(args.sleep_seconds)),
                generate_response_eval_payload=generate_response_eval_payload,
                get_last_tool_calls_snapshot=get_last_tool_calls_snapshot,
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
        "route_metrics_schema_version": ROUTE_METRICS_SCHEMA_VERSION,
        "selection": selection,
        "capability_catalog": CAPABILITY_CATALOG,
        "summary": summary,
        "route_metrics": route_metrics,
        "cases": results,
    }

    if args.baseline:
        baseline_path = args.baseline.resolve()
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_schema = int(baseline_report.get("route_metrics_schema_version", 1) or 1)
        report["baseline"] = {
            "path": str(baseline_path),
            "route_metrics_schema_version": baseline_schema,
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
        "[Eval] selection: "
        f"cases={selection['case_count']} "
        f"categories={selection.get('categories', {})} "
        f"capabilities={selection.get('capabilities', {})}"
    )
    print(
        f"[Eval] cases={summary['case_count']} "
        f"runs={summary['run_count_total']} "
        f"avg_similarity={summary['avg_pairwise_similarity']:.3f} "
        f"avg_unique_ratio={summary['avg_unique_response_ratio']:.3f} "
        f"avg_url_hit={summary['avg_min_url_hit_rate']:.3f} "
        f"avg_fact_hit={summary['avg_fact_hit_rate']:.3f} "
        f"avg_fact_group_hit={summary['avg_fact_group_hit_rate']:.3f} "
        f"avg_tool_path_hit={summary['avg_tool_path_hit_rate']:.3f} "
        f"avg_tool_path_accept_hit={summary['avg_tool_path_accept_hit_rate']:.3f} "
        f"avg_source_domain_hit={summary['avg_source_domain_hit_rate']:.3f} "
        f"avg_forbidden_claim={summary['avg_forbidden_claim_rate']:.3f} "
        f"avg_error={summary['avg_error_rate']:.3f}"
    )
    print(
        "[Eval] route: "
        f"react_attempts={int(route_metrics.get('react_attempts', 0))} "
        f"react_success={int(route_metrics.get('react_success', 0))} "
        f"react_error={int(route_metrics.get('react_error', 0))} "
        f"react_recursion_limit_hit={int(route_metrics.get('react_recursion_limit_hit', 0))} "
        f"react_success_rate={route_metrics.get('react_success_rate', 0.0):.2%} "
        f"react_error_rate={route_metrics.get('react_error_rate', 0.0):.2%} "
        f"react_recursion_limit_rate={route_metrics.get('react_recursion_limit_rate', 0.0):.2%}"
    )

    if "baseline" in report:
        bc = report["baseline"]["comparison"]
        baseline_schema = int(report["baseline"].get("route_metrics_schema_version", 1))
        if baseline_schema != ROUTE_METRICS_SCHEMA_VERSION:
            print(
                "[Eval][Warn] baseline schema mismatch: "
                f"baseline={baseline_schema} current={ROUTE_METRICS_SCHEMA_VERSION}; "
                "regenerate baseline for accurate route metrics comparison."
            )
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
