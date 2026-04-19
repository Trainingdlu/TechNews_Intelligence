"""Unit tests for eval/build_leaderboard.py."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

from eval import build_leaderboard


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_build_leaderboard")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_run_eval_report(
    *,
    dataset: str,
    suite: str,
    avg_recall_at_10: float,
    avg_mrr_at_10: float,
    avg_ndcg_at_10: float,
    avg_error_rate: float,
    react_attempts: int,
    react_success: int,
    case_rows: list[dict],
) -> dict:
    return {
        "dataset": dataset,
        "selection": {"suite": suite},
        "summary": {
            "case_count": len(case_rows),
            "retrieval_case_count": len(case_rows),
            "avg_recall_at_10": avg_recall_at_10,
            "avg_mrr_at_10": avg_mrr_at_10,
            "avg_ndcg_at_10": avg_ndcg_at_10,
            "avg_error_rate": avg_error_rate,
        },
        "route_metrics": {
            "react_attempts": react_attempts,
            "react_success": react_success,
            "react_success_rate": (react_success / react_attempts) if react_attempts else 0.0,
        },
        "cases": case_rows,
    }


def _make_judge_report(avg_composite: float, rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "summary": {"avg_composite": avg_composite},
        "rows": rows,
    }


def test_build_leaderboard_report_baseline_delta_and_n() -> None:
    with _case_dir() as case_dir:
        reports_dir = case_dir / "reports"
        matrix_dir = reports_dir / "matrix"
        judge_dir = reports_dir / "judge"
        matrix_dir.mkdir(parents=True, exist_ok=True)

        g0_run_eval = matrix_dir / "g0.json"
        g1_run_eval = matrix_dir / "g1.json"
        g0_judge = judge_dir / "g0_judge.json"
        g1_judge = judge_dir / "g1_judge.json"

        _write_json(
            g0_run_eval,
            _make_run_eval_report(
                dataset="eval/datasets/versions/v202604/default.jsonl",
                suite="default",
                avg_recall_at_10=0.40,
                avg_mrr_at_10=0.30,
                avg_ndcg_at_10=0.35,
                avg_error_rate=0.20,
                react_attempts=10,
                react_success=8,
                case_rows=[
                    {
                        "metrics": {
                            "recall_at_10": 0.30,
                            "mrr_at_10": 0.20,
                            "ndcg_at_10": 0.25,
                            "error_rate": 0.0,
                        },
                        "runs": [{"trace_summary": {"latency_ms": 100}}],
                    },
                    {
                        "metrics": {
                            "recall_at_10": 0.50,
                            "mrr_at_10": 0.40,
                            "ndcg_at_10": 0.45,
                            "error_rate": 0.4,
                        },
                        "runs": [{"trace_summary": {"latency_ms": 200}}],
                    },
                ],
            ),
        )
        _write_json(
            g1_run_eval,
            _make_run_eval_report(
                dataset="eval/datasets/versions/v202604/default.jsonl",
                suite="default",
                avg_recall_at_10=0.60,
                avg_mrr_at_10=0.50,
                avg_ndcg_at_10=0.55,
                avg_error_rate=0.10,
                react_attempts=10,
                react_success=9,
                case_rows=[
                    {
                        "metrics": {
                            "recall_at_10": 0.55,
                            "mrr_at_10": 0.45,
                            "ndcg_at_10": 0.50,
                            "error_rate": 0.0,
                        },
                        "runs": [{"trace_summary": {"latency_ms": 90}}],
                    },
                    {
                        "metrics": {
                            "recall_at_10": 0.65,
                            "mrr_at_10": 0.55,
                            "ndcg_at_10": 0.60,
                            "error_rate": 0.2,
                        },
                        "runs": [{"trace_summary": {"latency_ms": 180}}],
                    },
                ],
            ),
        )

        _write_json(g0_judge, _make_judge_report(3.50, [{"composite": 3.2}, {"composite": 3.8}]))
        _write_json(g1_judge, _make_judge_report(4.20, [{"composite": 4.0}, {"composite": 4.4}]))

        manifest = matrix_dir / "matrix_manifest.json"
        _write_json(
            manifest,
            {
                "generated_at_utc": "2026-04-17T00:00:00+00:00",
                "groups": [
                    {
                        "id": "G0_baseline",
                        "description": "baseline",
                        "output": str(g0_run_eval),
                        "judge_output": str(g0_judge),
                    },
                    {
                        "id": "G1_candidate",
                        "description": "candidate",
                        "output": str(g1_run_eval),
                        "judge_output": str(g1_judge),
                    },
                ],
            },
        )

        report = build_leaderboard.build_leaderboard_report(
            manifest_path=manifest,
            eval_dir=Path("eval").resolve(),
            baseline_group="G0_baseline",
            top_k=3,
            bootstrap_samples=300,
            bootstrap_seed=17,
        )

        assert report["baseline_group"] == "G0_baseline"
        assert report["dataset"]["version"] == "v202604"
        assert len(report["groups"]) == 2
        g1 = next(item for item in report["groups"] if item["group_id"] == "G1_candidate")
        recall_metric = g1["metrics"]["avg_recall_at_10"]
        assert recall_metric["delta_abs"] > 0
        assert recall_metric["n"] == 2
        assert "ci_95" in recall_metric
        assert g1["metrics"]["avg_composite"]["delta_abs"] > 0
        assert isinstance(report["top_gains"], list) and report["top_gains"]


def test_build_leaderboard_report_missing_optional_sources_warns() -> None:
    with _case_dir() as case_dir:
        matrix_dir = case_dir / "reports" / "matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        g0_run_eval = matrix_dir / "g0.json"
        g1_run_eval = matrix_dir / "g1.json"

        _write_json(
            g0_run_eval,
            _make_run_eval_report(
                dataset="eval/datasets/smoke.jsonl",
                suite="smoke",
                avg_recall_at_10=0.20,
                avg_mrr_at_10=0.20,
                avg_ndcg_at_10=0.20,
                avg_error_rate=0.10,
                react_attempts=5,
                react_success=4,
                case_rows=[{"metrics": {"recall_at_10": 0.2, "mrr_at_10": 0.2, "ndcg_at_10": 0.2, "error_rate": 0.1}}],
            ),
        )
        _write_json(
            g1_run_eval,
            _make_run_eval_report(
                dataset="eval/datasets/smoke.jsonl",
                suite="smoke",
                avg_recall_at_10=0.30,
                avg_mrr_at_10=0.30,
                avg_ndcg_at_10=0.30,
                avg_error_rate=0.08,
                react_attempts=5,
                react_success=5,
                case_rows=[{"metrics": {"recall_at_10": 0.3, "mrr_at_10": 0.3, "ndcg_at_10": 0.3, "error_rate": 0.08}}],
            ),
        )

        manifest = matrix_dir / "matrix_manifest.json"
        _write_json(
            manifest,
            {
                "generated_at_utc": "2026-04-17T00:00:00+00:00",
                "groups": [
                    {"id": "G0_baseline", "output": str(g0_run_eval)},
                    {"id": "G1_candidate", "output": str(g1_run_eval)},
                ],
            },
        )

        report = build_leaderboard.build_leaderboard_report(
            manifest_path=manifest,
            eval_dir=Path("eval").resolve(),
            baseline_group="G0_baseline",
            top_k=3,
            bootstrap_samples=200,
            bootstrap_seed=9,
        )

        g1 = next(item for item in report["groups"] if item["group_id"] == "G1_candidate")
        assert g1["metrics"]["avg_composite"]["current"] == "missing"
        warnings = report.get("warnings", [])
        assert any("missing judge file" in str(item) for item in warnings)
