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
    matrix_path = Path("eval/config/matrix.json")
    groups = run_matrix_eval.load_matrix_groups(matrix_path)
    ids = [group.id for group in groups]

    assert len(groups) == 3
    assert ids == ["G0", "G1", "G2"]


def test_load_matrix_config_reads_baseline_and_default_args() -> None:
    matrix_path = Path("eval/config/matrix.json")
    config = run_matrix_eval.load_matrix_config(matrix_path)

    assert config.baseline_group == "G0"
    assert config.runner_script == "run_task_eval.py"
    assert config.default_runner_args == []


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


def test_resolve_forwarded_runner_args_defaults_and_conflict() -> None:
    defaults = run_matrix_eval.resolve_forwarded_runner_args([])
    assert defaults == ["--runs-per-case", "1"]

    merged = run_matrix_eval.resolve_forwarded_runner_args(
        ["--runs-per-case", "1"],
        default_runner_args=["--dataset", "eval/datasets/versions/v20260417_1642/regression.jsonl"],
    )
    assert merged[:2] == ["--dataset", "eval/datasets/versions/v20260417_1642/regression.jsonl"]
    assert merged[-2:] == ["--runs-per-case", "1"]

    with pytest.raises(ValueError, match="--output/--experiment-group"):
        run_matrix_eval.resolve_forwarded_runner_args(["--output", "x.json"])

    with pytest.raises(ValueError, match="Do not pass --output"):
        run_matrix_eval.resolve_forwarded_runner_args(["--experiment-group", "G0"])


def test_load_matrix_config_accepts_task_eval_runner() -> None:
    with _case_dir() as case_dir:
        matrix_path = case_dir / "task_eval_matrix.json"
        matrix_path.write_text(
            json.dumps(
                {
                    "runner": "task_eval",
                    "groups": [
                        {"id": "G0", "description": "base", "env": {}},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        config = run_matrix_eval.load_matrix_config(matrix_path)
        assert config.runner_script == "run_task_eval.py"


def test_load_matrix_config_rejects_non_task_runner() -> None:
    with _case_dir() as case_dir:
        matrix_path = case_dir / "legacy_matrix.json"
        matrix_path.write_text(
            json.dumps(
                {
                    "runner": "legacy_runner",
                    "groups": [{"id": "G0", "description": "legacy", "env": {}}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Matrix runner must be task_eval"):
            run_matrix_eval.load_matrix_config(matrix_path)


def test_build_runner_command_includes_output_only() -> None:
    with _case_dir() as case_dir:
        runner_path = Path("eval/run_task_eval.py").resolve()
        output_path = case_dir / "g0.json"
        cmd = run_matrix_eval.build_runner_command(
            runner_path,
            ["--runs-per-case", "1"],
            output_path=output_path,
        )

        assert str(runner_path) in cmd
        assert "--output" in cmd
        assert str(output_path) in cmd
        assert "--experiment-group" not in cmd
