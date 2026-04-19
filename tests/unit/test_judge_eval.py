"""Unit tests for judge eval runner."""

from __future__ import annotations

import argparse
import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

from eval import run_judge_eval


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_judge_eval")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def test_compute_composite_uses_fixed_weights() -> None:
    scores = {
        "accuracy": 5,
        "groundedness": 4,
        "coherence": 3,
        "completeness": 2,
        "helpfulness": 1,
    }
    composite = run_judge_eval._compute_composite(scores)
    expected = (5 * 0.30) + (4 * 0.25) + (2 * 0.20) + (3 * 0.15) + (1 * 0.10)
    assert abs(composite - expected) < 1e-9


def test_build_cases_from_run_eval_merges_trace_contexts() -> None:
    source_report = {
        "cases": [
            {
                "id": "case_1",
                "question": "Q1",
                "outputs": ["A1"],
                "constraints": {
                    "ground_truth": "GT1",
                },
                "runs": [
                    {
                        "trace_summary": {
                            "tool_events": [
                                {
                                    "output_summary": {
                                        "context_docs": [
                                            {"summary": "trace context"},
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ],
            }
        ]
    }
    cases = run_judge_eval._build_cases(source_report)
    assert len(cases) == 1
    case = cases[0]
    assert case["case_id"] == "case_1"
    assert case["question"] == "Q1"
    assert case["answer"] == "A1"
    assert case["constraints"]["ground_truth"] == "GT1"
    assert "trace context" in case["contexts"]


def test_evaluate_case_with_retry_handles_empty_answer() -> None:
    config = run_judge_eval.JudgeRuntimeConfig()
    case = {"case_id": "c1", "question": "Q", "answer": ""}

    row = run_judge_eval._evaluate_case_with_retry(
        case=case,
        config=config,
        invoker=lambda _case: ({"accuracy": 5}, {"accuracy": "unused"}),
    )
    assert row["case_id"] == "c1"
    assert row["verdict"] == "failing"
    assert row["scores"]["composite"] == 1.0
    assert row["scores"]["accuracy"] == 1
    assert row["scores"]["groundedness"] == 1


def test_main_outputs_structured_scores(monkeypatch) -> None:  # noqa: ANN001
    with _case_dir() as tmp_dir:
        report_path = tmp_dir / "report.json"
        output_path = tmp_dir / "judge.json"
        report_path.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "id": "c1",
                            "question": "What changed?",
                            "outputs": ["OpenAI released updates. https://example.com/news"],
                            "constraints": {"expected_facts": ["openai"]},
                            "runs": [],
                        },
                        {
                            "id": "c2",
                            "question": "Any timeline?",
                            "outputs": [""],
                            "constraints": {},
                            "runs": [],
                        },
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        args = argparse.Namespace(
            report=report_path,
            output=output_path,
            config=tmp_dir / "judge.yaml",
            env_file=None,
            model=None,
            provider=None,
            backend="heuristic",
            temperature=0.0,
            batch_size=2,
            max_retries=0,
            retry_backoff_sec=0.0,
            max_cases=0,
            no_skip_failed_cases=False,
        )

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: args)

        exit_code = run_judge_eval.main()
        assert exit_code == 0
        assert output_path.exists()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["source_report"] == str(report_path.resolve())
        assert payload["case_count"] == 2
        assert "summary" in payload
        assert "avg_composite" in payload["summary"]
        assert len(payload["rows"]) == 2
        for row in payload["rows"]:
            assert "case_id" in row
            assert "scores" in row
            assert "evidence" in row
            assert row["verdict"] in {"excellent", "good", "adequate", "poor", "failing"}
            for metric in ("accuracy", "groundedness", "coherence", "completeness", "helpfulness"):
                value = row["scores"][metric]
                assert isinstance(value, int)
                assert 1 <= value <= 5
            composite = row["scores"]["composite"]
            assert isinstance(composite, float)
            assert 1.0 <= composite <= 5.0
