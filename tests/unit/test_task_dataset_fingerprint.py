from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from eval import build_task_dataset as mod


def _args(task_path: Path, *, seed: int = 42) -> SimpleNamespace:
    return SimpleNamespace(
        task_types=task_path,
        seed=seed,
        pools_per_task=0,
        enforce_coverage_policy=True,
        enforce_scenario_retrieval_map=True,
        disable_audit=False,
        audit_max_regen_rounds=3,
        audit_regen_mode="failed_only",
        initial_cases_per_audit_call=0,
        regen_cases_per_audit_call=1,
        pools_per_generation_call=2,
        regen_pools_per_generation_call=1,
        provider="vertex",
        model="gemini-3.1-pro-preview",
        temperature=0.0,
        audit_temperature=0.0,
    )


def _task_rows() -> list[dict]:
    return [
        {
            "task_id": "search_news.normal",
            "skill": "search_news",
            "scenario": "normal",
            "retrieval_mode": "evaluable",
            "sampling": {"n_min": 10, "days": 30, "pool_size": 12, "candidate_limit": 300},
        },
        {
            "task_id": "search_news.empty",
            "skill": "search_news",
            "scenario": "empty",
            "retrieval_mode": "non_retrieval",
            "sampling": {"n_min": 4, "days": 30, "pool_size": 12, "candidate_limit": 300},
        },
    ]


def test_dataset_fingerprint_stable_and_seed_sensitive() -> None:
    tmp_root = Path.cwd() / ".tmp_pytest_fingerprint"
    tmp_root.mkdir(parents=True, exist_ok=True)
    task_path = tmp_root / f"tasks_{uuid.uuid4().hex}.json"
    task_path.write_text(json.dumps(_task_rows(), ensure_ascii=False), encoding="utf-8")
    try:
        fp_a, payload_a = mod.build_dataset_fingerprint(args=_args(task_path, seed=42), task_types=_task_rows())
        fp_b, payload_b = mod.build_dataset_fingerprint(args=_args(task_path, seed=42), task_types=_task_rows())
        fp_c, _ = mod.build_dataset_fingerprint(args=_args(task_path, seed=7), task_types=_task_rows())

        assert fp_a == fp_b
        assert payload_a == payload_b
        assert fp_a != fp_c
    finally:
        if task_path.exists():
            task_path.unlink()


def test_validate_scenario_retrieval_map_rejects_mismatch() -> None:
    mod.validate_scenario_retrieval_map(_task_rows())

    bad_rows = _task_rows()
    bad_rows[1] = dict(bad_rows[1], retrieval_mode="evaluable")
    with pytest.raises(ValueError):
        mod.validate_scenario_retrieval_map(bad_rows)
