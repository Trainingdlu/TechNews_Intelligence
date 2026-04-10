"""Unit tests for eval dataset normalization and capability filtering."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

import pytest

from eval.dataset_loader import (
    filter_eval_cases,
    load_eval_cases,
    parse_csv_filter_arg,
    summarize_case_matrix,
)


@contextmanager
def _jsonl_case(lines: list[str]):
    root = Path("tests/unit/.tmp_eval_dataset_loader")
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"case_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        yield path
    finally:
        rmtree(case_dir, ignore_errors=True)


def test_load_cases_with_explicit_capability() -> None:
    with _jsonl_case(
        [
            '{"id":"c1","category":"compare","capability":"compare_topics","question":"A vs B","min_urls":2}',
            '{"id":"c2","category":"timeline","capability":"timeline","question":"build timeline"}',
        ],
    ) as path:
        cases = load_eval_cases(path)
    assert len(cases) == 2
    assert cases[0]["capability"] == "compare_topics"
    assert cases[1]["capability"] == "timeline"
    assert cases[0]["required_tools"] == ["compare_topics"]
    assert cases[1]["required_tools"] == ["build_timeline"]


def test_load_cases_resolve_capability_from_category() -> None:
    with _jsonl_case(
        [
            '{"id":"c1","category":"trend","question":"OpenAI trend?"}',
            '{"id":"c2","category":"brief","question":"latest news?"}',
        ],
    ) as path:
        cases = load_eval_cases(path)
    assert cases[0]["capability"] == "trend_analysis"
    assert cases[1]["capability"] == "general_qa"


def test_unknown_capability_strict_raises() -> None:
    with _jsonl_case(
        ['{"id":"c1","category":"query","capability":"not_supported","question":"Q"}'],
    ) as path:
        with pytest.raises(ValueError):
            load_eval_cases(path, strict_capability_check=True)


def test_unknown_capability_non_strict_fallbacks() -> None:
    with _jsonl_case(
        ['{"id":"c1","category":"query","capability":"not_supported","question":"Q"}'],
    ) as path:
        cases = load_eval_cases(path, strict_capability_check=False)
    assert cases[0]["capability"] == "general_qa"
    assert cases[0]["required_tools"] == []


def test_optional_accuracy_fields_are_normalized() -> None:
    with _jsonl_case(
        [
            (
                '{"id":"c1","category":"query","question":"Q",'
                '"expected_facts":"openai,anthropic",'
                '"expected_fact_groups":[["openai","oai"],"anthropic|claude"],'
                '"required_tools":"query_news",'
                '"acceptable_tool_paths":[["query_news"],["search_news","read_news_content"]],'
                '"must_not_contain":["hallucination"],'
                '"expected_source_domains":"techcrunch.com,news.ycombinator.com"}'
            ),
        ],
    ) as path:
        cases = load_eval_cases(path)

    assert cases[0]["expected_facts"] == ["openai", "anthropic"]
    assert cases[0]["expected_fact_groups"] == [["openai", "oai"], ["anthropic", "claude"]]
    assert cases[0]["required_tools"] == ["query_news"]
    assert cases[0]["acceptable_tool_paths"] == [["query_news"], ["search_news", "read_news_content"]]
    assert cases[0]["must_not_contain"] == ["hallucination"]
    assert cases[0]["expected_source_domains"] == ["techcrunch.com", "news.ycombinator.com"]


def test_filter_and_matrix() -> None:
    cases = [
        {"id": "a", "category": "compare", "capability": "compare_topics"},
        {"id": "b", "category": "timeline", "capability": "timeline"},
        {"id": "c", "category": "query", "capability": "query_news"},
    ]
    filtered = filter_eval_cases(cases, categories={"compare", "timeline"}, capabilities={"timeline"})
    assert len(filtered) == 1
    assert filtered[0]["id"] == "b"

    matrix = summarize_case_matrix(filtered)
    assert matrix["case_count"] == 1
    assert matrix["categories"]["timeline"] == 1
    assert matrix["capabilities"]["timeline"] == 1


def test_parse_csv_filter_arg() -> None:
    parsed = parse_csv_filter_arg("compare, timeline , ,landscape")
    assert parsed == {"compare", "timeline", "landscape"}
