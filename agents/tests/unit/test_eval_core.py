"""Unit tests for eval core metrics helpers."""

from __future__ import annotations

import unittest

try:
    from agents.tests.utils.bootstrap import ensure_agents_on_path
except ModuleNotFoundError:
    from utils.bootstrap import ensure_agents_on_path

ensure_agents_on_path()

from eval.eval_core import (  # noqa: E402  pylint: disable=wrong-import-position
    average_pairwise_similarity,
    build_baseline_comparison,
    evaluate_case_outputs,
    evaluate_quality_gates,
    extract_urls,
    normalize_text,
    summarize_case_results,
)


class EvalCoreTests(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("  A   B  "), "a b")
        self.assertEqual(normalize_text("**Hello**"), "hello")

    def test_extract_urls_dedup(self) -> None:
        text = "A https://a.com B https://b.com A2 https://a.com"
        self.assertEqual(extract_urls(text), ["https://a.com", "https://b.com"])

    def test_average_pairwise_similarity(self) -> None:
        same = average_pairwise_similarity(["Alpha", "Alpha", "Alpha"])
        diff = average_pairwise_similarity(["Alpha", "Beta", "Gamma"])
        self.assertAlmostEqual(same, 1.0, places=6)
        self.assertLess(diff, 1.0)

    def test_evaluate_case_outputs_constraints(self) -> None:
        outputs = [
            "## Sources\n- https://a.com",
            "No sources in this output",
        ]
        metrics = evaluate_case_outputs(outputs, min_urls=1, must_contain=["sources"])
        self.assertEqual(metrics["run_count"], 2)
        self.assertEqual(metrics["runs_with_min_urls"], 1)
        self.assertAlmostEqual(metrics["min_url_hit_rate"], 0.5, places=6)
        self.assertAlmostEqual(metrics["phrase_hit_rate"], 1.0, places=6)
        self.assertEqual(metrics["error_count"], 0)

    def test_summarize_case_results(self) -> None:
        case_a = {"metrics": evaluate_case_outputs(["A", "A"], min_urls=0)}
        case_b = {"metrics": evaluate_case_outputs(["A", "B"], min_urls=0)}
        summary = summarize_case_results([case_a, case_b])
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["run_count_total"], 4)
        self.assertGreaterEqual(summary["avg_pairwise_similarity"], 0.0)
        self.assertLessEqual(summary["avg_pairwise_similarity"], 1.0)

    def test_build_baseline_comparison(self) -> None:
        current = {
            "summary": {
                "avg_pairwise_similarity": 0.90,
                "avg_unique_response_ratio": 0.20,
                "avg_min_url_hit_rate": 0.95,
                "avg_phrase_hit_rate": 0.90,
                "avg_error_rate": 0.01,
            },
            "route_metrics": {
                "fallback_rate_total": 0.02,
                "fallback_rate_langchain": 0.03,
                "langchain_success_rate": 0.97,
                "forced_route_rate": 0.40,
                "landscape_low_evidence_rate": 0.10,
            },
        }
        baseline = {
            "summary": {
                "avg_pairwise_similarity": 0.85,
                "avg_unique_response_ratio": 0.25,
                "avg_min_url_hit_rate": 0.96,
                "avg_phrase_hit_rate": 0.88,
                "avg_error_rate": 0.02,
            },
            "route_metrics": {
                "fallback_rate_total": 0.03,
                "fallback_rate_langchain": 0.05,
                "langchain_success_rate": 0.95,
                "forced_route_rate": 0.35,
                "landscape_low_evidence_rate": 0.20,
            },
        }
        cmp_out = build_baseline_comparison(current, baseline)
        self.assertIn("items", cmp_out)
        self.assertGreater(len(cmp_out["items"]), 0)
        self.assertGreater(cmp_out["improved_count"], 0)

    def test_evaluate_quality_gates(self) -> None:
        report = {
            "summary": {
                "avg_error_rate": 0.01,
                "avg_min_url_hit_rate": 0.90,
            },
            "route_metrics": {
                "fallback_rate_total": 0.04,
            },
        }
        gates = [
            {
                "name": "error_max",
                "metric_path": "summary.avg_error_rate",
                "op": "max",
                "threshold": 0.02,
            },
            {
                "name": "url_min",
                "metric_path": "summary.avg_min_url_hit_rate",
                "op": "min",
                "threshold": 0.80,
            },
            {
                "name": "fallback_max",
                "metric_path": "route_metrics.fallback_rate_total",
                "op": "max",
                "threshold": 0.03,
            },
        ]
        out = evaluate_quality_gates(report, gates)
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["failed_count"], 1)
        self.assertFalse(out["ok"])
