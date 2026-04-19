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


def test_arg_parser_does_not_expose_legacy_export_flags() -> None:
    args = _parse_default_args()
    keys = set(vars(args).keys())
    assert not any(key.startswith("export_") and key.endswith("_jsonl") for key in keys)


def test_run_case_includes_retrieval_metrics_from_payload_and_trace() -> None:
    case = {
        "id": "case_1",
        "category": "brief",
        "capability": "general_qa",
        "question": "Q1",
        "min_urls": 0,
        "retrieval_gold_urls": ["https://a.com/x", "https://b.com"],
        "difficulty": "medium",
        "priority": 2,
        "failure_tag": ["rerank"],
    }

    def _fake_generate_response_eval_payload(_history, _question, **_kwargs):
        return {
            "text": "Answer with extra link https://c.com",
            "request_id": "req-1",
            "tool_calls": ["query_news"],
            "valid_urls": ["https://A.com/x/"],
            "trace_summary": {
                "tool_events": [
                    {
                        "output_summary": {
                            "context_docs": [
                                {"url": "https://b.com", "title": "Doc B"},
                            ]
                        }
                    }
                ]
            },
        }

    out = run_eval._run_case(
        case=case,
        runs_per_question=1,
        sleep_seconds=0.0,
        generate_response_eval_payload=_fake_generate_response_eval_payload,
        get_last_tool_calls_snapshot=lambda: [],
        include_outputs=True,
        include_trace_summary=True,
        experiment_group="G0",
    )

    metrics = out["metrics"]
    assert metrics["retrieval_has_gold"] is True
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == 1.0

    run_meta = out["runs"][0]
    assert run_meta["retrieved_urls"] == ["https://A.com/x/", "https://b.com", "https://c.com"]
    assert out["constraints"]["retrieval_gold_urls"] == ["https://a.com/x", "https://b.com"]
    assert out["constraints"]["difficulty"] == "medium"
    assert out["constraints"]["priority"] == 2
    assert out["constraints"]["failure_tag"] == ["rerank"]


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
        assert report["summary"]["retrieval_case_count"] == 0
        assert "system" in report
        assert "citation_guard_block_rate" in report["system"]
        assert "citation_guard_block_rate" in report["route_metrics"]
        assert "avg_recall_at_5" in report["summary"]
        assert "avg_recall_at_10" in report["summary"]
        assert "avg_mrr_at_10" in report["summary"]
        assert "avg_ndcg_at_10" in report["summary"]
