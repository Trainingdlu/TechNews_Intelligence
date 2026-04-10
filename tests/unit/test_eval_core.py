"""Unit tests for eval core metrics helpers."""

from __future__ import annotations

import unittest

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

    def test_evaluate_case_outputs_accuracy_metrics(self) -> None:
        outputs = [
            "OpenAI and Anthropic both announced updates. No uncertainty here.",
            "Only OpenAI appears in this summary. rumor included.",
        ]
        run_tool_calls = [
            ["query_news", "compare_topics"],
            ["query_news"],
        ]
        metrics = evaluate_case_outputs(
            outputs=outputs,
            expected_facts=["openai", "anthropic"],
            required_tools=["compare_topics"],
            must_not_contain=["rumor"],
            run_tool_calls=run_tool_calls,
        )
        self.assertAlmostEqual(metrics["fact_hit_rate"], 0.75, places=6)
        self.assertAlmostEqual(metrics["tool_path_hit_rate"], 0.5, places=6)
        self.assertAlmostEqual(metrics["forbidden_claim_rate"], 0.5, places=6)
        self.assertEqual(metrics["runs_with_required_tools"], 1)
        self.assertEqual(metrics["forbidden_claim_violations"], 1)

    def test_evaluate_case_outputs_group_path_domain_metrics(self) -> None:
        outputs = [
            "OpenAI update with links: https://www.techcrunch.com/a https://news.ycombinator.com/item?id=1",
            "Anthropic update from https://example.com/post",
        ]
        run_tool_calls = [
            ["query_news", "read_news_content"],
            ["search_news"],
        ]
        metrics = evaluate_case_outputs(
            outputs=outputs,
            expected_fact_groups=[["openai", "oai"], ["anthropic", "claude"]],
            acceptable_tool_paths=[["query_news"], ["search_news", "read_news_content"]],
            expected_source_domains=["techcrunch.com", "news.ycombinator.com"],
            run_tool_calls=run_tool_calls,
        )
        self.assertAlmostEqual(metrics["fact_group_hit_rate"], 0.5, places=6)
        self.assertAlmostEqual(metrics["tool_path_accept_hit_rate"], 0.5, places=6)
        self.assertAlmostEqual(metrics["source_domain_hit_rate"], 0.5, places=6)
        self.assertEqual(metrics["runs_with_acceptable_tool_path"], 1)
        self.assertEqual(metrics["source_domain_hit_runs"], 1.0)

    def test_summarize_case_results(self) -> None:
        case_a = {"metrics": evaluate_case_outputs(["A", "A"], min_urls=0)}
        case_b = {"metrics": evaluate_case_outputs(["A", "B"], min_urls=0)}
        summary = summarize_case_results([case_a, case_b])
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["run_count_total"], 4)
        self.assertGreaterEqual(summary["avg_pairwise_similarity"], 0.0)
        self.assertLessEqual(summary["avg_pairwise_similarity"], 1.0)
        self.assertIn("avg_fact_hit_rate", summary)
        self.assertIn("avg_fact_group_hit_rate", summary)
        self.assertIn("avg_tool_path_hit_rate", summary)
        self.assertIn("avg_tool_path_accept_hit_rate", summary)
        self.assertIn("avg_source_domain_hit_rate", summary)
        self.assertIn("avg_forbidden_claim_rate", summary)

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
                "react_success_rate": 0.97,
                "react_error_rate": 0.03,
                "react_recursion_limit_rate": 0.02,
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
                "react_success_rate": 0.95,
                "react_error_rate": 0.05,
                "react_recursion_limit_rate": 0.03,
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
                "react_error_rate": 0.04,
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
                "name": "react_error_max",
                "metric_path": "route_metrics.react_error_rate",
                "op": "max",
                "threshold": 0.03,
            },
        ]
        out = evaluate_quality_gates(report, gates)
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["failed_count"], 1)
        self.assertFalse(out["ok"])

