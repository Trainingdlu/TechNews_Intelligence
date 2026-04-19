"""Unit tests for full eval pipeline orchestration."""

from __future__ import annotations

import argparse
import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from typing import Any

import pytest

from eval import run_full_eval_pipeline


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_run_full_eval_pipeline")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _arg_value(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return str(command[idx + 1])


def _make_args(
    *,
    dataset_version: str,
    matrix_config: Path,
    run_id: str,
    resume: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_version=dataset_version,
        matrix_config=matrix_config,
        baseline="G0_baseline",
        candidates="G1_candidate",
        run_id=run_id,
        resume=resume,
    )


def _make_matrix_config(path: Path) -> None:
    _write_json(
        path,
        {
            "groups": [
                {"id": "G0_baseline", "description": "baseline", "env": {}},
                {"id": "G1_candidate", "description": "candidate", "env": {}},
            ]
        },
    )


def _prepare_dataset(dataset_version: str) -> Path:
    dataset_dir = Path("eval/datasets/versions") / dataset_version
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "regression.jsonl"
    dataset_path.write_text(
        '{"id":"case_1","category":"general","capability":"general_qa","question":"Q?"}\n',
        encoding="utf-8",
    )
    return dataset_dir


def _fake_subprocess_factory(
    *,
    missing_run_eval_field: bool = False,
):
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def _fake_run_subprocess(
        command: list[str],
        *,
        cwd: Path,  # noqa: ARG001 - keep signature aligned
        env: dict[str, str] | None = None,
    ) -> None:
        calls.append((list(command), dict(env) if env is not None else None))
        script_name = Path(command[1]).name

        if script_name == "run_matrix_eval.py":
            output_dir = Path(_arg_value(command, "--output-dir"))
            dataset_path = _arg_value(command, "--dataset")
            groups = [g for g in _arg_value(command, "--groups").split(",") if g]
            timestamp = "20260417T120000Z"
            manifest_groups: list[dict[str, Any]] = []
            for idx, group_id in enumerate(groups):
                run_eval_path = output_dir / f"{timestamp}_{group_id}.json"
                report = {
                    "dataset": dataset_path,
                    "selection": {"suite": "regression"},
                    "summary": {
                        "case_count": 1,
                        "retrieval_case_count": 1,
                        "avg_recall_at_10": 0.50 + (0.10 * idx),
                        "avg_mrr_at_10": 0.40 + (0.10 * idx),
                        "avg_ndcg_at_10": 0.45 + (0.10 * idx),
                        "avg_error_rate": 0.10 - (0.02 * idx),
                    },
                    "route_metrics": {
                        "react_attempts": 1,
                        "react_success": 1,
                        "react_success_rate": 1.0,
                    },
                    "cases": [
                        {
                            "id": "case_1",
                            "question": "Q?",
                            "outputs": ["Answer text"],
                            "metrics": {
                                "recall_at_10": 0.5 + (0.1 * idx),
                                "mrr_at_10": 0.4 + (0.1 * idx),
                                "ndcg_at_10": 0.45 + (0.1 * idx),
                                "error_rate": 0.0,
                            },
                            "runs": [{"trace_summary": {"latency_ms": 100 + idx}}],
                        }
                    ],
                }
                if missing_run_eval_field:
                    report["summary"].pop("avg_recall_at_10", None)
                _write_json(run_eval_path, report)
                manifest_groups.append(
                    {
                        "id": group_id,
                        "status": "ok",
                        "output": str(run_eval_path.resolve()),
                    }
                )
            manifest = output_dir / f"{timestamp}_manifest.json"
            _write_json(
                manifest,
                {
                    "generated_at_utc": "2026-04-17T12:00:00+00:00",
                    "groups": manifest_groups,
                },
            )
            return

        if script_name == "run_judge_eval.py":
            output = Path(_arg_value(command, "--output"))
            _write_json(
                output,
                {
                    "source_report": _arg_value(command, "--report"),
                    "case_count": 1,
                    "summary": {"avg_composite": 4.1},
                    "rows": [
                        {
                            "case_id": "case_1",
                            "scores": {
                                "accuracy": 4,
                                "groundedness": 4,
                                "coherence": 4,
                                "completeness": 4,
                                "helpfulness": 4,
                                "composite": 4.1,
                            },
                            "evidence": {
                                "accuracy": "ok",
                                "groundedness": "ok",
                                "coherence": "ok",
                                "completeness": "ok",
                                "helpfulness": "ok",
                            },
                            "verdict": "good",
                        }
                    ],
                },
            )
            return

        if script_name == "build_leaderboard.py":
            output_json = Path(_arg_value(command, "--output-json"))
            output_md = Path(_arg_value(command, "--output-md"))
            baseline = _arg_value(command, "--baseline-group")
            manifest_path = Path(_arg_value(command, "--manifest"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            group_ids = [str(item.get("id")) for item in manifest.get("groups", []) if isinstance(item, dict)]
            candidate = next((gid for gid in group_ids if gid != baseline), "G1_candidate")

            def _metric_row(cur: float, base: float) -> dict[str, Any]:
                return {
                    "current": cur,
                    "baseline": base,
                    "delta_abs": cur - base,
                    "n": 1,
                    "ci_95": {"lower": -0.1, "upper": 0.2, "method": "bootstrap_delta_mean"},
                }

            leaderboard = {
                "baseline_group": baseline,
                "dataset": {
                    "version": "vtest",
                    "path": "eval/datasets/versions/vtest/regression.jsonl",
                },
                "metric_order": ["avg_recall_at_10", "avg_composite"],
                "groups": [
                    {
                        "group_id": baseline,
                        "metrics": {
                            "avg_recall_at_10": _metric_row(0.5, 0.5),
                            "avg_composite": _metric_row(4.0, 4.0),
                        },
                    },
                    {
                        "group_id": candidate,
                        "metrics": {
                            "avg_recall_at_10": _metric_row(0.6, 0.5),
                            "avg_composite": _metric_row(4.2, 4.0),
                        },
                    },
                ],
            }
            _write_json(output_json, leaderboard)
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text("# Leaderboard\n", encoding="utf-8")
            return

        raise AssertionError(f"Unexpected command: {command}")

    return _fake_run_subprocess, calls


def test_full_pipeline_main_happy_path(monkeypatch) -> None:  # noqa: ANN001
    with _case_dir() as case_dir:
        run_id = f"pipeline_{uuid.uuid4().hex}"
        dataset_version = f"vtest_{uuid.uuid4().hex[:8]}"
        dataset_dir = _prepare_dataset(dataset_version)
        matrix_config = case_dir / "matrix.json"
        _make_matrix_config(matrix_config)
        run_dir = Path("eval/reports") / run_id

        fake_runner, calls = _fake_subprocess_factory()
        args = _make_args(
            dataset_version=dataset_version,
            matrix_config=matrix_config,
            run_id=run_id,
            resume=False,
        )
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: args)
        monkeypatch.setattr(run_full_eval_pipeline, "_run_subprocess", fake_runner)

        try:
            exit_code = run_full_eval_pipeline.main()
            assert exit_code == 0

            summary_path = run_dir / "pipeline_summary.json"
            assert summary_path.exists()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            leaderboard_json = Path(summary["leaderboard_json"])
            assert leaderboard_json.exists()

            leaderboard = json.loads(leaderboard_json.read_text(encoding="utf-8"))
            candidate_group = next(
                item for item in leaderboard["groups"] if item["group_id"] == "G1_candidate"
            )
            metric = candidate_group["metrics"]["avg_recall_at_10"]
            assert "current" in metric
            assert "baseline" in metric
            assert "delta_abs" in metric
            assert "ci_95" in metric
            assert "lower" in metric["ci_95"]
            assert "upper" in metric["ci_95"]

            matrix_calls = [item for item in calls if Path(item[0][1]).name == "run_matrix_eval.py"]
            assert matrix_calls, "run_matrix_eval command not invoked"
            matrix_cmd = matrix_calls[0][0]
            assert "--dataset" in matrix_cmd
            assert "--include-outputs" in matrix_cmd
            assert "--include-trace-summary" in matrix_cmd
        finally:
            rmtree(dataset_dir, ignore_errors=True)
            rmtree(run_dir, ignore_errors=True)


def test_full_pipeline_resume_skips_subprocess_steps(monkeypatch) -> None:  # noqa: ANN001
    with _case_dir() as case_dir:
        run_id = f"pipeline_{uuid.uuid4().hex}"
        dataset_version = f"vtest_{uuid.uuid4().hex[:8]}"
        dataset_dir = _prepare_dataset(dataset_version)
        matrix_config = case_dir / "matrix.json"
        _make_matrix_config(matrix_config)
        run_dir = Path("eval/reports") / run_id

        fake_runner, _calls = _fake_subprocess_factory()
        first_args = _make_args(
            dataset_version=dataset_version,
            matrix_config=matrix_config,
            run_id=run_id,
            resume=False,
        )
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: first_args)
        monkeypatch.setattr(run_full_eval_pipeline, "_run_subprocess", fake_runner)

        try:
            assert run_full_eval_pipeline.main() == 0

            resume_calls: list[list[str]] = []

            def _resume_runner(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:  # noqa: ANN001
                resume_calls.append(command)

            resume_args = _make_args(
                dataset_version=dataset_version,
                matrix_config=matrix_config,
                run_id=run_id,
                resume=True,
            )
            monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: resume_args)
            monkeypatch.setattr(run_full_eval_pipeline, "_run_subprocess", _resume_runner)

            assert run_full_eval_pipeline.main() == 0
            assert resume_calls == []
        finally:
            rmtree(dataset_dir, ignore_errors=True)
            rmtree(run_dir, ignore_errors=True)


def test_full_pipeline_raises_clear_error_on_run_eval_schema_gap(monkeypatch) -> None:  # noqa: ANN001
    with _case_dir() as case_dir:
        run_id = f"pipeline_{uuid.uuid4().hex}"
        dataset_version = f"vtest_{uuid.uuid4().hex[:8]}"
        dataset_dir = _prepare_dataset(dataset_version)
        matrix_config = case_dir / "matrix.json"
        _make_matrix_config(matrix_config)
        run_dir = Path("eval/reports") / run_id

        fake_runner, _calls = _fake_subprocess_factory(missing_run_eval_field=True)
        args = _make_args(
            dataset_version=dataset_version,
            matrix_config=matrix_config,
            run_id=run_id,
            resume=False,
        )
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: args)
        monkeypatch.setattr(run_full_eval_pipeline, "_run_subprocess", fake_runner)

        try:
            with pytest.raises(ValueError, match=r"summary\.avg_recall_at_10"):
                run_full_eval_pipeline.main()
        finally:
            rmtree(dataset_dir, ignore_errors=True)
            rmtree(run_dir, ignore_errors=True)
