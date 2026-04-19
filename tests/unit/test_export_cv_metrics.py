"""Unit tests for eval/export_cv_metrics.py."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

import eval.export_cv_metrics as export_cv_metrics


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_export_cv_metrics")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metric_row(
    *,
    layer: str,
    direction: str,
    current: float,
    baseline: float,
    n: int,
    ci_lower: float,
    ci_upper: float,
    method: str = "bootstrap_delta_mean",
) -> dict:
    delta_abs = current - baseline
    delta_pct = 0.0 if abs(baseline) <= 1e-12 else (delta_abs / abs(baseline)) * 100.0
    return {
        "layer": layer,
        "direction": direction,
        "current": current,
        "baseline": baseline,
        "delta_abs": delta_abs,
        "delta_pct": round(delta_pct, 4),
        "n": n,
        "ci_95": {
            "lower": ci_lower,
            "upper": ci_upper,
            "method": method,
        },
    }


def _build_fixture(case_dir: Path) -> dict[str, Path]:
    reports_dir = case_dir / "reports"
    matrix_dir = reports_dir / "matrix"
    judge_dir = reports_dir / "judge"
    trace_dir = reports_dir / "trace"
    failure_dir = reports_dir / "failures"
    leaderboard_dir = reports_dir / "leaderboard"

    g0_run_eval = matrix_dir / "g0.json"
    g1_run_eval = matrix_dir / "g1.json"
    g0_judge = judge_dir / "g0.json"
    g1_judge = judge_dir / "g1.json"
    matrix_manifest = reports_dir / "pipeline_manifest.json"
    leaderboard_json = leaderboard_dir / "latest.json"
    trace_map_jsonl = trace_dir / "trace_map.jsonl"
    failure_json = failure_dir / "key_failures.json"

    _write_json(
        g0_run_eval,
        {
            "dataset": "eval/datasets/versions/v20260417/default.jsonl",
            "selection": {"suite": "regression"},
            "summary": {
                "case_count": 2,
                "retrieval_case_count": 2,
                "avg_recall_at_5": 0.30,
                "avg_recall_at_10": 0.40,
                "avg_mrr_at_10": 0.30,
                "avg_ndcg_at_10": 0.35,
                "avg_error_rate": 0.20,
                "avg_tool_path_hit_rate": 0.60,
                "avg_tool_path_accept_hit_rate": 0.65,
            },
            "route_metrics": {
                "react_attempts": 10,
                "react_success": 8,
                "react_success_rate": 0.80,
            },
            "cases": [
                {
                    "id": "case_1",
                    "metrics": {
                        "recall_at_5": 0.20,
                        "recall_at_10": 0.30,
                        "mrr_at_10": 0.20,
                        "ndcg_at_10": 0.25,
                        "error_rate": 0.20,
                        "tool_path_hit_rate": 0.50,
                        "tool_path_accept_hit_rate": 0.60,
                    },
                    "runs": [
                        {
                            "request_id": "req-g0-c1-r1",
                            "trace_summary": {"request_id": "req-g0-c1-r1", "latency_ms": 120},
                        }
                    ],
                },
                {
                    "id": "case_2",
                    "metrics": {
                        "recall_at_5": 0.40,
                        "recall_at_10": 0.50,
                        "mrr_at_10": 0.40,
                        "ndcg_at_10": 0.45,
                        "error_rate": 0.20,
                        "tool_path_hit_rate": 0.70,
                        "tool_path_accept_hit_rate": 0.70,
                    },
                    "runs": [
                        {
                            "request_id": "req-g0-c2-r1",
                            "trace_summary": {"request_id": "req-g0-c2-r1", "latency_ms": 220},
                        }
                    ],
                },
            ],
        },
    )

    _write_json(
        g1_run_eval,
        {
            "dataset": "eval/datasets/versions/v20260417/default.jsonl",
            "selection": {"suite": "regression"},
            "summary": {
                "case_count": 2,
                "retrieval_case_count": 2,
                "avg_recall_at_5": 0.45,
                "avg_recall_at_10": 0.60,
                "avg_mrr_at_10": 0.45,
                "avg_ndcg_at_10": 0.50,
                "avg_error_rate": 0.10,
                "avg_tool_path_hit_rate": 0.80,
                "avg_tool_path_accept_hit_rate": 0.85,
            },
            "route_metrics": {
                "react_attempts": 10,
                "react_success": 9,
                "react_success_rate": 0.90,
            },
            "cases": [
                {
                    "id": "case_1",
                    "metrics": {
                        "recall_at_5": 0.35,
                        "recall_at_10": 0.55,
                        "mrr_at_10": 0.40,
                        "ndcg_at_10": 0.45,
                        "error_rate": 0.10,
                        "tool_path_hit_rate": 0.75,
                        "tool_path_accept_hit_rate": 0.80,
                    },
                    "runs": [
                        {
                            "request_id": "req-g1-c1-r1",
                            "trace_summary": {"request_id": "req-g1-c1-r1", "latency_ms": 100},
                        }
                    ],
                },
                {
                    "id": "case_2",
                    "metrics": {
                        "recall_at_5": 0.55,
                        "recall_at_10": 0.65,
                        "mrr_at_10": 0.50,
                        "ndcg_at_10": 0.55,
                        "error_rate": 0.10,
                        "tool_path_hit_rate": 0.85,
                        "tool_path_accept_hit_rate": 0.90,
                    },
                    "runs": [
                        {
                            "request_id": "req-g1-c2-r1",
                            "trace_summary": {"request_id": "req-g1-c2-r1", "latency_ms": 180},
                        }
                    ],
                },
            ],
        },
    )

    _write_json(
        g0_judge,
        {
            "case_count": 2,
            "summary": {"avg_composite": 3.50},
            "rows": [
                {"case_id": "case_1", "scores": {"composite": 3.40}},
                {"case_id": "case_2", "scores": {"composite": 3.60}},
            ],
        },
    )
    _write_json(
        g1_judge,
        {
            "case_count": 2,
            "summary": {"avg_composite": 4.20},
            "rows": [
                {"case_id": "case_1", "scores": {"composite": 4.10}},
                {"case_id": "case_2", "scores": {"composite": 4.30}},
            ],
        },
    )

    _write_json(
        matrix_manifest,
        {
            "generated_at_utc": "2026-04-17T12:00:00+00:00",
            "groups": [
                {
                    "id": "G0_baseline",
                    "description": "baseline",
                    "output": str(g0_run_eval.resolve()),
                    "judge_output": str(g0_judge.resolve()),
                    "env_overrides": {
                        "EVAL_RETRIEVAL_VARIANT": "baseline",
                        "NEWS_RERANK_MODE": "none",
                        "EVAL_AGENT_VARIANT": "baseline",
                    },
                },
                {
                    "id": "G1_candidate",
                    "description": "candidate",
                    "output": str(g1_run_eval.resolve()),
                    "judge_output": str(g1_judge.resolve()),
                    "env_overrides": {
                        "EVAL_RETRIEVAL_VARIANT": "retrieval_full_optimized",
                        "NEWS_RERANK_MODE": "llm_rerank",
                        "EVAL_AGENT_VARIANT": "agent_optimized",
                    },
                },
            ],
        },
    )

    _write_json(
        leaderboard_json,
        {
            "generated_at_utc": "2026-04-17T12:05:00+00:00",
            "manifest_path": str(matrix_manifest.resolve()),
            "baseline_group": "G0_baseline",
            "dataset": {
                "version": "v20260417",
                "path": "eval/datasets/versions/v20260417/default.jsonl",
                "suite": "regression",
            },
            "groups": [
                {
                    "group_id": "G0_baseline",
                    "description": "baseline",
                    "sources": {
                        "run_eval": str(g0_run_eval.resolve()),
                        "judge": str(g0_judge.resolve()),
                    },
                    "metrics": {
                        "avg_recall_at_10": _metric_row(layer="retrieval", direction="higher_better", current=0.40, baseline=0.40, n=2, ci_lower=0.0, ci_upper=0.0),
                        "avg_mrr_at_10": _metric_row(layer="retrieval", direction="higher_better", current=0.30, baseline=0.30, n=2, ci_lower=0.0, ci_upper=0.0),
                        "avg_ndcg_at_10": _metric_row(layer="retrieval", direction="higher_better", current=0.35, baseline=0.35, n=2, ci_lower=0.0, ci_upper=0.0),
                        "avg_error_rate": _metric_row(layer="tool_path", direction="lower_better", current=0.20, baseline=0.20, n=2, ci_lower=0.0, ci_upper=0.0),
                        "react_success_rate": _metric_row(layer="tool_path", direction="higher_better", current=0.80, baseline=0.80, n=10, ci_lower=0.0, ci_upper=0.0),
                        "p95_latency": _metric_row(layer="tool_path", direction="lower_better", current=220.0, baseline=220.0, n=2, ci_lower=0.0, ci_upper=0.0, method="bootstrap_delta_p95"),
                        "avg_composite": _metric_row(layer="judge", direction="higher_better", current=3.50, baseline=3.50, n=2, ci_lower=0.0, ci_upper=0.0),
                    },
                },
                {
                    "group_id": "G1_candidate",
                    "description": "candidate",
                    "sources": {
                        "run_eval": str(g1_run_eval.resolve()),
                        "judge": str(g1_judge.resolve()),
                    },
                    "metrics": {
                        "avg_recall_at_10": _metric_row(layer="retrieval", direction="higher_better", current=0.60, baseline=0.40, n=2, ci_lower=0.08, ci_upper=0.32),
                        "avg_mrr_at_10": _metric_row(layer="retrieval", direction="higher_better", current=0.45, baseline=0.30, n=2, ci_lower=0.03, ci_upper=0.27),
                        "avg_ndcg_at_10": _metric_row(layer="retrieval", direction="higher_better", current=0.50, baseline=0.35, n=2, ci_lower=0.03, ci_upper=0.27),
                        "avg_error_rate": _metric_row(layer="tool_path", direction="lower_better", current=0.10, baseline=0.20, n=2, ci_lower=-0.18, ci_upper=-0.02),
                        "react_success_rate": _metric_row(layer="tool_path", direction="higher_better", current=0.90, baseline=0.80, n=10, ci_lower=0.01, ci_upper=0.19),
                        "p95_latency": _metric_row(layer="tool_path", direction="lower_better", current=180.0, baseline=220.0, n=2, ci_lower=-55.0, ci_upper=-5.0, method="bootstrap_delta_p95"),
                        "avg_composite": _metric_row(layer="judge", direction="higher_better", current=4.20, baseline=3.50, n=2, ci_lower=0.20, ci_upper=1.20),
                    },
                },
            ],
        },
    )

    _write_jsonl(
        trace_map_jsonl,
        [
            {"run_id": "run-g1-c1-r1", "trace_id": "trace-g1-c1-r1", "metadata": {"request_id": "req-g1-c1-r1"}},
            {"run_id": "run-g1-c2-r1", "trace_id": "trace-g1-c2-r1", "metadata": {"request_id": "req-g1-c2-r1"}},
        ],
    )
    _write_json(
        failure_json,
        [
            {
                "group_id": "G1_candidate",
                "layer": "retrieval",
                "metric": "avg_recall_at_10",
                "reason": "historical retrieval miss",
                "request_id": "req-g1-c1-r1",
            },
            {
                "group_id": "G1_candidate",
                "layer": "tool_path",
                "metric": "react_success_rate",
                "reason": "tool timeout",
                "run_id": "run-tool-1",
            },
        ],
    )

    return {
        "leaderboard": leaderboard_json,
        "manifest": matrix_manifest,
        "trace_map": trace_map_jsonl,
        "failure": failure_json,
    }


def test_build_cv_metrics_report_generates_quant_and_evidence() -> None:
    with _case_dir() as case_dir:
        paths = _build_fixture(case_dir)
        report = export_cv_metrics.build_cv_metrics_report(
            leaderboard_path=paths["leaderboard"],
            matrix_manifest_path=paths["manifest"],
            failure_sample_paths=[paths["failure"]],
            trace_artifact_paths=[paths["trace_map"]],
            baseline_group=None,
            candidate_groups=None,
            max_run_refs=10,
        )

        assert report["experiment"]["baseline_group"] == "G0_baseline"
        assert report["experiment"]["candidate_groups"] == ["G1_candidate"]
        assert report["trace_mapping"]["request_id_count"] >= 2

        candidate = next(item for item in report["groups"] if item["group_id"] == "G1_candidate")
        metric_map = {row["metric"]: row for row in candidate["metrics"]}
        assert "avg_recall_at_5" in metric_map
        assert float(metric_map["avg_recall_at_5"]["delta_pct"]) > 0
        assert float(metric_map["avg_composite"]["delta_abs"]) > 0
        assert metric_map["avg_recall_at_10"]["ci_95"]["lower"] != "missing"

        retrieval_attr = next(item for item in candidate["layer_attribution"] if item["layer"] == "retrieval")
        assert retrieval_attr["status"] in {"supported", "observed", "mixed"}

        conclusions = report["conclusions"]
        assert conclusions
        assert any(item["type"] == "relative_improvement" for item in conclusions)
        assert any(item["type"] == "absolute_improvement" for item in conclusions)
        assert any(item["type"] == "stability" for item in conclusions)
        assert any(item["type"] == "reproducibility" for item in conclusions)
        for item in conclusions:
            evidence = item.get("evidence", {})
            assert evidence.get("metric_files")
            assert evidence.get("run_ids")

        summary_md = str(report.get("summary_markdown", ""))
        assert "Reproducibility" in summary_md
        assert "avg_recall_at_5" in summary_md


def test_build_cv_metrics_report_falls_back_to_request_id_without_trace_map() -> None:
    with _case_dir() as case_dir:
        paths = _build_fixture(case_dir)
        report = export_cv_metrics.build_cv_metrics_report(
            leaderboard_path=paths["leaderboard"],
            matrix_manifest_path=paths["manifest"],
            failure_sample_paths=[paths["failure"]],
            trace_artifact_paths=[],
            baseline_group="G0_baseline",
            candidate_groups=["G1_candidate"],
            max_run_refs=10,
        )

        fields = {
            str(item.get("evidence", {}).get("run_id_field"))
            for item in report.get("conclusions", [])
            if isinstance(item, dict)
        }
        assert "request_id" in fields
