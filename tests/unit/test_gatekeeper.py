"""Unit tests for eval/gatekeeper.py."""

from __future__ import annotations

import argparse
import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

from eval import gatekeeper


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_gatekeeper")
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


def _rule(
    *,
    name: str,
    metric: str,
    layer: str,
    min_value: float | None = None,
    max_value: float | None = None,
    n_min: int | None = None,
    ci_lower_bound_min: float | None = None,
    severity: str = gatekeeper.HARD_FAIL,
    missing_severity: str | None = None,
    action: str = "",
) -> gatekeeper.RuleSpec:
    return gatekeeper.RuleSpec(
        name=name,
        metric=gatekeeper.normalize_metric_name(metric),
        layer=layer,
        direction="higher_better",
        min_value=min_value,
        max_value=max_value,
        n_min=n_min,
        ci_lower_bound_min=ci_lower_bound_min,
        severity=severity,
        missing_severity=missing_severity or severity,
        n_severity=severity,
        ci_severity=severity,
        action=action,
        enabled=True,
    )


def _obs(
    *,
    metric: str,
    layer: str,
    value: float,
    n: int,
    samples: list[float],
    source: str = "test",
) -> gatekeeper.MetricObservation:
    return gatekeeper.MetricObservation(
        metric=gatekeeper.normalize_metric_name(metric),
        layer=layer,
        direction="higher_better",
        value=value,
        n=n,
        samples=samples,
        source=source,
    )


def test_threshold_boundary_pass_inclusive() -> None:
    rules = [
        _rule(
            name="retrieval_recall_boundary",
            metric="avg_recall_at_10",
            layer="retrieval",
            min_value=0.50,
            n_min=3,
            ci_lower_bound_min=0.50,
            severity=gatekeeper.HARD_FAIL,
        )
    ]
    metrics = {
        "avg_recall_at_10": _obs(
            metric="avg_recall_at_10",
            layer="retrieval",
            value=0.50,
            n=3,
            samples=[0.50, 0.50, 0.50],
        )
    }

    out = gatekeeper.evaluate_rules(
        rules,
        metrics,
        ci_confidence=0.95,
        ci_bootstrap_samples=500,
        ci_seed=17,
    )

    assert out["status"] == gatekeeper.PASS
    assert out["exit_code"] == 0
    assert out["summary"]["hard_failed_rules"] == 0
    assert out["summary"]["soft_failed_rules"] == 0


def test_missing_metric_triggers_soft_fail() -> None:
    rules = [
        _rule(
            name="judge_optional_metric",
            metric="avg_composite",
            layer="judge",
            min_value=4.0,
            severity=gatekeeper.SOFT_FAIL,
            missing_severity=gatekeeper.SOFT_FAIL,
            action="Backfill judge score first.",
        )
    ]

    out = gatekeeper.evaluate_rules(
        rules,
        metrics={},
        ci_confidence=0.95,
        ci_bootstrap_samples=500,
        ci_seed=17,
    )

    assert out["status"] == gatekeeper.SOFT_FAIL
    assert out["exit_code"] == 2
    assert out["summary"]["soft_failed_rules"] == 1
    assert out["failures"][0]["condition"] == "metric_missing"


def test_hard_fail_contains_metric_threshold_delta_and_action() -> None:
    rules = [
        _rule(
            name="retrieval_recall_hard_gate",
            metric="avg_recall_at_10",
            layer="retrieval",
            min_value=0.80,
            severity=gatekeeper.HARD_FAIL,
            action="Raise recall before release.",
        )
    ]
    metrics = {
        "avg_recall_at_10": _obs(
            metric="avg_recall_at_10",
            layer="retrieval",
            value=0.50,
            n=5,
            samples=[0.50, 0.50, 0.50, 0.50, 0.50],
        )
    }

    out = gatekeeper.evaluate_rules(
        rules,
        metrics,
        ci_confidence=0.95,
        ci_bootstrap_samples=500,
        ci_seed=17,
    )

    assert out["status"] == gatekeeper.HARD_FAIL
    assert out["exit_code"] == 3
    fail = out["failures"][0]
    assert fail["metric"] == "avg_recall_at_10"
    assert fail["threshold"] == 0.8
    assert fail["actual"] == 0.5
    assert fail["delta"] == -0.3
    assert fail["action"] == "Raise recall before release."


def test_ci_calculation_is_deterministic_with_fixed_seed() -> None:
    obs = _obs(
        metric="avg_composite",
        layer="judge",
        value=3.65,
        n=6,
        samples=[3.3, 3.6, 3.9, 3.7, 3.8, 3.6],
    )

    ci_a = gatekeeper.compute_metric_ci(
        obs,
        confidence=0.95,
        bootstrap_samples=600,
        seed=42,
    )
    ci_b = gatekeeper.compute_metric_ci(
        obs,
        confidence=0.95,
        bootstrap_samples=600,
        seed=42,
    )

    assert ci_a == ci_b


def test_main_can_read_leaderboard_and_follow_source_reports(monkeypatch) -> None:  # noqa: ANN001
    with _case_dir() as tmp_dir:
        run_eval_path = tmp_dir / "run_eval.json"
        judge_path = tmp_dir / "judge.json"
        leaderboard_path = tmp_dir / "leaderboard.json"
        config_path = tmp_dir / "gates.yaml"
        output_json = tmp_dir / "gate.json"
        output_md = tmp_dir / "gate.md"

        _write_json(
            run_eval_path,
            {
                "summary": {
                    "retrieval_case_count": 3,
                    "avg_recall_at_10": 0.50,
                    "avg_mrr_at_10": 0.40,
                },
                "cases": [
                    {"metrics": {"recall_at_10": 0.5, "mrr_at_10": 0.4}},
                    {"metrics": {"recall_at_10": 0.5, "mrr_at_10": 0.4}},
                    {"metrics": {"recall_at_10": 0.5, "mrr_at_10": 0.4}},
                ],
            },
        )
        _write_json(
            judge_path,
            {
                "row_count": 3,
                "summary": {"avg_composite": 4.00},
                "rows": [{"composite": 4.0}, {"composite": 4.0}, {"composite": 4.0}],
            },
        )

        _write_json(
            leaderboard_path,
            {
                "baseline_group": "G0_baseline",
                "groups": [
                    {
                        "group_id": "G1_candidate",
                        "metrics": {
                            "avg_recall_at_10": {
                                "current": 0.50,
                                "sample_n": {"current": 3},
                                "layer": "retrieval",
                                "direction": "higher_better",
                            }
                        },
                        "sources": {
                            "run_eval": str(run_eval_path),
                            "judge": str(judge_path),
                        },
                    }
                ],
            },
        )

        _write_json(
            config_path,
            {
                "ci": {"confidence": 0.95, "bootstrap_samples": 500, "seed": 17},
                "rules": [
                    {
                        "name": "recall_gate",
                        "metric": "avg_recall_at_10",
                        "layer": "retrieval",
                        "min_value": 0.50,
                        "n_min": 3,
                        "ci_lower_bound_min": 0.50,
                        "severity": "hard",
                    },
                    {
                        "name": "judge_gate",
                        "metric": "avg_composite",
                        "layer": "judge",
                        "min_value": 4.00,
                        "n_min": 3,
                        "ci_lower_bound_min": 4.00,
                        "severity": "hard",
                    },
                ],
            },
        )

        parser = gatekeeper._build_arg_parser(Path("eval"))
        args = parser.parse_args([])
        args.config = config_path
        args.leaderboard = leaderboard_path
        args.group_id = "G1_candidate"
        args.run_eval_report = None
        args.judge_report = None
        args.output_json = output_json
        args.output_md = output_md

        def _fake_parse_args(_self):
            return args

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _fake_parse_args)

        exit_code = gatekeeper.main()
        assert exit_code == 0
        assert output_json.exists()
        assert output_md.exists()

        report = json.loads(output_json.read_text(encoding="utf-8"))
        assert report["status"] == gatekeeper.PASS
        assert report["target_group"] == "G1_candidate"
        assert report["metrics"]["avg_recall_at_10"]["source"] == "run_eval"
