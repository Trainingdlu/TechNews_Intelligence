"""Unit tests for eval runner argument and gate behaviors."""

from __future__ import annotations

import argparse
import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

from eval import run_eval


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_run_eval")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def _parse_default_args() -> argparse.Namespace:
    parser = run_eval._build_arg_parser(Path("eval"))
    return parser.parse_args([])


def test_build_gate_specs_collects_only_configured_thresholds() -> None:
    args = _parse_default_args()
    args.fail_on_avg_error_rate = 0.05
    args.fail_on_avg_fact_hit_rate = 0.70
    args.fail_on_react_success_rate = 0.90

    specs = run_eval._build_gate_specs(args)
    names = {item["name"] for item in specs}

    assert "avg_error_rate_max" in names
    assert "avg_fact_hit_rate_min" in names
    assert "react_success_rate_min" in names
    assert "avg_source_domain_hit_rate_min" not in names


def test_resolve_dataset_path_prefers_explicit_dataset() -> None:
    with _case_dir() as tmp_dir:
        args = _parse_default_args()
        custom = tmp_dir / "custom.jsonl"
        custom.write_text('{"id":"c1","category":"general","question":"Q"}\n', encoding="utf-8")
        args.dataset = custom

        resolved = run_eval._resolve_dataset_path(Path("eval"), args)
        assert resolved == custom.resolve()


def test_resolve_dataset_path_from_suite_name() -> None:
    args = _parse_default_args()
    args.dataset = None
    args.suite = "smoke"

    resolved = run_eval._resolve_dataset_path(Path("eval"), args)
    assert resolved.name == "smoke.jsonl"


def test_build_experiment_context_captures_group_and_env(monkeypatch) -> None:  # noqa: ANN001
    args = _parse_default_args()
    args.experiment_group = "  G0_baseline  "
    monkeypatch.setenv("EVAL_RETRIEVAL_VARIANT", "baseline")
    monkeypatch.setenv("NEWS_RERANK_MODE", "none")

    context = run_eval._build_experiment_context(args)
    assert context["group"] == "G0_baseline"
    assert context["env"]["EVAL_RETRIEVAL_VARIANT"] == "baseline"
    assert context["env"]["NEWS_RERANK_MODE"] == "none"


def test_invoke_eval_payload_backward_compatibility() -> None:
    def _legacy_fn(_history, _question):
        return {"text": "ok", "tool_calls": ["query_news"]}

    payload = run_eval._invoke_eval_payload(
        _legacy_fn,
        "Q",
        request_id="req-1",
        case_id="case-1",
        experiment_group="G0",
        include_trace_summary=True,
    )
    assert payload["text"] == "ok"
    assert payload["tool_calls"] == ["query_news"]


def test_build_ragas_rows_extracts_trace_contexts() -> None:
    report = {
        "experiment": {"group": "G3_retrieval_full"},
        "cases": [
            {
                "id": "case_1",
                "question": "Q1",
                "outputs": ["A1"],
                "constraints": {
                    "ground_truth": "GT1",
                    "expected_facts": [],
                    "ragas_contexts": ["fallback context"],
                },
                "runs": [
                    {
                        "trace_summary": {
                            "tool_events": [
                                {
                                    "output_summary": {
                                        "context_docs": [
                                            {
                                                "url": "https://a.com",
                                                "title": "Doc A",
                                                "summary": "Doc summary A",
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ],
            }
        ],
    }
    rows = run_eval._build_ragas_rows(report)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "case_1"
    assert rows[0]["reference"] == "GT1"
    assert rows[0]["experiment_group"] == "G3_retrieval_full"
    assert rows[0]["contexts"] == ["Doc summary A"]


def test_main_returns_nonzero_when_quality_gate_fails(monkeypatch) -> None:  # noqa: ANN001
    with _case_dir() as tmp_dir:
        dataset = tmp_dir / "suite.jsonl"
        dataset.write_text(
            '{"id":"case_1","category":"brief","capability":"general_qa","question":"latest news?"}\n',
            encoding="utf-8",
        )
        output = tmp_dir / "report.json"

        args = _parse_default_args()
        args.dataset = dataset
        args.output = output
        args.runs_per_question = 1
        args.fail_on_react_success_rate = 0.90

        def _fake_parse_args(_self):
            return args

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _fake_parse_args)

        def _fake_bootstrap_imports():
            def _generate_response_eval_payload(_history, _question):
                return {"text": "ok", "tool_calls": []}

            def _get_last_tool_calls_snapshot():
                return []

            def _get_route_metrics_snapshot():
                return {
                    "react_attempts": 1,
                    "react_success": 0,
                    "react_error": 1,
                    "react_recursion_limit_hit": 0,
                    "react_success_rate": 0.0,
                    "react_error_rate": 1.0,
                    "react_recursion_limit_rate": 0.0,
                }

            def _reset_route_metrics():
                return None

            return (
                _generate_response_eval_payload,
                _get_last_tool_calls_snapshot,
                _get_route_metrics_snapshot,
                _reset_route_metrics,
            )

        monkeypatch.setattr(run_eval, "_bootstrap_imports", _fake_bootstrap_imports)

        exit_code = run_eval.main()
        assert exit_code == 2
        assert output.exists()

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["quality_gate"]["result"]["failed_count"] >= 1
