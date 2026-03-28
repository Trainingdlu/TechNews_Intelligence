"""Unit tests for eval dataset normalization and capability filtering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from agents.tests.utils.bootstrap import ensure_agents_on_path
except ModuleNotFoundError:
    from utils.bootstrap import ensure_agents_on_path

ensure_agents_on_path()

from eval.dataset_loader import (  # noqa: E402  pylint: disable=wrong-import-position
    filter_eval_cases,
    load_eval_cases,
    parse_csv_filter_arg,
    summarize_case_matrix,
)


def _write_jsonl(lines: list[str]) -> Path:
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w", encoding="utf-8")
    try:
        for line in lines:
            fd.write(line)
            fd.write("\n")
        return Path(fd.name)
    finally:
        fd.close()


class EvalDatasetLoaderTests(unittest.TestCase):
    def test_load_cases_with_explicit_capability(self) -> None:
        path = _write_jsonl(
            [
                '{"id":"c1","category":"compare","capability":"compare_topics","question":"A vs B","min_urls":2}',
                '{"id":"c2","category":"timeline","capability":"timeline","question":"build timeline"}',
            ]
        )
        cases = load_eval_cases(path)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["capability"], "compare_topics")
        self.assertEqual(cases[1]["capability"], "timeline")

    def test_load_cases_resolve_capability_from_category(self) -> None:
        path = _write_jsonl(
            [
                '{"id":"c1","category":"trend","question":"OpenAI trend?"}',
                '{"id":"c2","category":"brief","question":"latest news?"}',
            ]
        )
        cases = load_eval_cases(path)
        self.assertEqual(cases[0]["capability"], "trend_analysis")
        self.assertEqual(cases[1]["capability"], "general_qa")

    def test_unknown_capability_strict_raises(self) -> None:
        path = _write_jsonl(
            [
                '{"id":"c1","category":"query","capability":"not_supported","question":"Q"}',
            ]
        )
        with self.assertRaises(ValueError):
            load_eval_cases(path, strict_capability_check=True)

    def test_unknown_capability_non_strict_fallbacks(self) -> None:
        path = _write_jsonl(
            [
                '{"id":"c1","category":"query","capability":"not_supported","question":"Q"}',
            ]
        )
        cases = load_eval_cases(path, strict_capability_check=False)
        self.assertEqual(cases[0]["capability"], "general_qa")

    def test_filter_and_matrix(self) -> None:
        cases = [
            {"id": "a", "category": "compare", "capability": "compare_topics"},
            {"id": "b", "category": "timeline", "capability": "timeline"},
            {"id": "c", "category": "query", "capability": "query_news"},
        ]
        filtered = filter_eval_cases(cases, categories={"compare", "timeline"}, capabilities={"timeline"})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "b")

        matrix = summarize_case_matrix(filtered)
        self.assertEqual(matrix["case_count"], 1)
        self.assertEqual(matrix["categories"]["timeline"], 1)
        self.assertEqual(matrix["capabilities"]["timeline"], 1)

    def test_parse_csv_filter_arg(self) -> None:
        parsed = parse_csv_filter_arg("compare, timeline , ,landscape")
        self.assertEqual(parsed, {"compare", "timeline", "landscape"})
