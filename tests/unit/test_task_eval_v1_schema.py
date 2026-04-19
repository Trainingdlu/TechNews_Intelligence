from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.task_eval_v1_schema import (
    build_news_pool_hash,
    load_task_types,
    normalize_case,
    validate_case,
)


def _task_type() -> dict:
    return {
        "task_id": "query_news.filter.precise.normal",
        "skill": "query_news",
        "intent_label": "query_news",
        "retrieval_mode": "evaluable",
        "scenario": "normal",
        "example_question": "q",
        "parameter_template": {"query": "OpenAI"},
        "acceptable_tool_paths": [[{"tool": "query_news", "args": {"query": "OpenAI"}}]],
        "required_tools": ["query_news"],
        "forbidden_tools": [],
        "should_clarify": False,
        "difficulty": "easy",
        "sampling": {"days": 30, "n_min": 2, "pool_size": 2},
        "tags": ["normal"],
    }


def _pool() -> list[dict]:
    return [
        {
            "doc_id": "doc_1",
            "url": "https://a.example.com",
            "title": "A",
            "summary": "s1",
            "published_at": "2026-01-01T00:00:00Z",
            "source": "TechCrunch",
        },
        {
            "doc_id": "doc_2",
            "url": "https://b.example.com",
            "title": "B",
            "summary": "s2",
            "published_at": "2026-01-02T00:00:00Z",
            "source": "HackerNews",
        },
    ]


def test_load_task_types_accepts_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([_task_type()]), encoding="utf-8")
    rows = load_task_types(path, strict_skill=False, enforce_coverage_policy=False)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "query_news.filter.precise.normal"


def test_normalize_case_enforces_retrieval_gold_subset() -> None:
    task = _task_type()
    case = normalize_case(
        {
            "expected_question": "请总结 OpenAI 相关新闻",
            "expected_answer": "这是基于新闻池的中文参考答案。",
            "expected_tool_paths": [[{"tool": "query_news", "args": {"query": "OpenAI"}}]],
            "retrieval_evaluable": True,
            "retrieval_gold_doc_ids": ["doc_1"],
            "verifiable_claims": [
                {
                    "claim": "A happened",
                    "evidence_doc_ids": ["doc_1"],
                    "claim_type": "fact",
                }
            ],
        },
        task_type=task,
        case_id="case-1",
        pool_id="pool-1",
        input_news_pool=_pool(),
    )
    validate_case(case, strict_skill=False)
    assert case["retrieval_gold_urls"] == ["https://a.example.com"]
    assert case["input_news_pool_hash"] == build_news_pool_hash(_pool())


def test_normalize_case_rejects_invalid_gold_ids() -> None:
    task = _task_type()
    with pytest.raises(ValueError, match="retrieval_gold_doc_ids must be subset"):
        normalize_case(
            {
                "expected_question": "请给出问题",
                "expected_answer": "这是中文答案。",
                "expected_tool_paths": [[{"tool": "query_news", "args": {"query": "OpenAI"}}]],
                "retrieval_evaluable": True,
                "retrieval_gold_doc_ids": ["doc_x"],
                "verifiable_claims": [],
            },
            task_type=task,
            case_id="case-2",
            pool_id="pool-2",
            input_news_pool=_pool(),
        )


def test_normalize_case_backfills_gold_doc_ids_from_urls() -> None:
    task = _task_type()
    case = normalize_case(
        {
            "expected_question": "请总结 OpenAI 相关新闻",
            "expected_answer": "这是基于新闻池的中文参考答案。",
            "expected_tool_paths": [[{"tool": "query_news", "args": {"query": "OpenAI"}}]],
            "retrieval_evaluable": True,
            "retrieval_gold_doc_ids": [],
            "retrieval_gold_urls": ["https://a.example.com"],
            "verifiable_claims": [],
        },
        task_type=task,
        case_id="case-backfill-url",
        pool_id="pool-backfill-url",
        input_news_pool=_pool(),
    )
    assert case["retrieval_gold_doc_ids"] == ["doc_1"]
    assert case["retrieval_gold_urls"] == ["https://a.example.com"]


def test_normalize_case_backfills_gold_doc_ids_from_claim_evidence() -> None:
    task = _task_type()
    case = normalize_case(
        {
            "expected_question": "请总结 OpenAI 相关新闻",
            "expected_answer": "这是基于新闻池的中文参考答案。",
            "expected_tool_paths": [[{"tool": "query_news", "args": {"query": "OpenAI"}}]],
            "retrieval_evaluable": True,
            "retrieval_gold_doc_ids": [],
            "retrieval_gold_urls": [],
            "verifiable_claims": [
                {
                    "claim": "A happened",
                    "evidence_doc_ids": ["doc_2"],
                    "claim_type": "fact",
                }
            ],
        },
        task_type=task,
        case_id="case-backfill-claim",
        pool_id="pool-backfill-claim",
        input_news_pool=_pool(),
    )
    assert case["retrieval_gold_doc_ids"] == ["doc_2"]
    assert case["retrieval_gold_urls"] == ["https://b.example.com"]


def test_normalize_case_rejects_non_chinese_expected_question() -> None:
    task = _task_type()
    with pytest.raises(ValueError, match="expected_question must contain Chinese text"):
        normalize_case(
            {
                "expected_question": "What is new about OpenAI?",
                "expected_answer": "这是中文答案。",
                "expected_tool_paths": [[{"tool": "query_news", "args": {"query": "OpenAI"}}]],
                "retrieval_evaluable": True,
                "retrieval_gold_doc_ids": ["doc_1"],
                "verifiable_claims": [],
            },
            task_type=task,
            case_id="case-3",
            pool_id="pool-3",
            input_news_pool=_pool(),
        )


def test_normalize_case_rejects_path_drift_from_task_acceptable() -> None:
    task = _task_type()
    with pytest.raises(ValueError, match="expected_tool_paths must be subset of task acceptable_tool_paths"):
        normalize_case(
            {
                "expected_question": "请总结 OpenAI 相关新闻",
                "expected_answer": "这是中文答案。",
                "expected_tool_paths": [[{"tool": "query_news", "args": {"query": "Anthropic"}}]],
                "retrieval_evaluable": True,
                "retrieval_gold_doc_ids": ["doc_1"],
                "verifiable_claims": [],
            },
            task_type=task,
            case_id="case-4",
            pool_id="pool-4",
            input_news_pool=_pool(),
        )
