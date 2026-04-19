"""Unit tests for matrix eval runner helpers."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

import pytest

import eval.run_matrix_eval as run_matrix_eval


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_run_matrix_eval")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def test_load_matrix_groups_reads_default_matrix() -> None:
    matrix_path = Path("eval/experiment_matrix.json")
    groups = run_matrix_eval.load_matrix_groups(matrix_path)
    ids = [group.id for group in groups]

    assert len(groups) == 6
    assert ids[0] == "G0_baseline"
    assert ids[-1] == "G5_full_optimized"


def test_load_matrix_config_reads_baseline_and_default_args() -> None:
    matrix_path = Path("eval/experiment_matrix.json")
    config = run_matrix_eval.load_matrix_config(matrix_path)

    assert config.baseline_group == "G0_baseline"
    assert config.frozen_dataset_version
    assert "--dataset" in config.default_run_eval_args


def test_load_matrix_groups_rejects_duplicate_ids() -> None:
    with _case_dir() as case_dir:
        matrix_path = case_dir / "dup_matrix.json"
        matrix_path.write_text(
            json.dumps(
                {
                    "groups": [
                        {"id": "G0", "description": "a", "env": {}},
                        {"id": "G0", "description": "b", "env": {}},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Duplicate matrix group id"):
            run_matrix_eval.load_matrix_groups(matrix_path)


def test_resolve_forwarded_run_eval_args_defaults_and_conflict() -> None:
    defaults = run_matrix_eval.resolve_forwarded_run_eval_args([])
    assert defaults == ["--suite", "default", "--runs-per-question", "3"]

    merged = run_matrix_eval.resolve_forwarded_run_eval_args(
        ["--runs-per-question", "1"],
        default_run_eval_args=["--dataset", "eval/datasets/versions/v20260417_1642/regression.jsonl"],
    )
    assert merged[:2] == ["--dataset", "eval/datasets/versions/v20260417_1642/regression.jsonl"]
    assert merged[-2:] == ["--runs-per-question", "1"]

    with pytest.raises(ValueError, match="--output/--experiment-group"):
        run_matrix_eval.resolve_forwarded_run_eval_args(["--suite", "smoke", "--output", "x.json"])


def test_build_run_eval_command_includes_group_and_output() -> None:
    with _case_dir() as case_dir:
        run_eval_path = Path("eval/run_eval.py").resolve()
        output_path = case_dir / "g0.json"
        cmd = run_matrix_eval.build_run_eval_command(
            run_eval_path,
            ["--suite", "smoke", "--runs-per-question", "1"],
            output_path=output_path,
            group_id="G0_baseline",
        )

        assert str(run_eval_path) in cmd
        assert "--experiment-group" in cmd
        assert "G0_baseline" in cmd
        assert "--output" in cmd
        assert str(output_path) in cmd
