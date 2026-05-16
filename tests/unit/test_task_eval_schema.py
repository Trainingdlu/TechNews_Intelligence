from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.task_eval_schema import (
    build_news_pool_hash,
    load_task_types,
    normalize_case,
    validate_case,
)

def _expected_tool_paths(query: str = "OpenAI") -> list[list[dict]]:
    return [[{"tool": "query_news", "args": {"query": query}}]]


def _task_type() -> dict:
    return {
        "task_id": "query_news.filter.precise.normal",
        "tool": "query_news",
        "intent_label": "query_news",
        "retrieval_mode": "evaluable",
        "scenario": "normal",
        "example_question": "q",
        "parameter_template": {"query": "OpenAI"},
        "acceptable_tool_paths": _expected_tool_paths(),
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


def _claim(doc_id: str = "doc_1", quote: str = "s1", *, include_quotes: bool = True) -> dict:
    claim = {
        "claim": "A happened",
        "evidence_doc_ids": [doc_id],
        "claim_type": "fact",
    }
    if include_quotes:
        claim["evidence_quotes"] = [{"doc_id": doc_id, "quote": quote}]
    return claim


def _case(**overrides) -> dict:
    raw_case = {
        "expected_question": "请总结 OpenAI 相关新闻",
        "expected_answer": "这是基于新闻池的中文参考答案。",
        "expected_tool_paths": _expected_tool_paths(),
        "retrieval_evaluable": True,
        "retrieval_gold_doc_ids": ["doc_1"],
        "verifiable_claims": [_claim()],
    }
    raw_case.update(overrides)
    return raw_case


def _case_dir(tmp_path: Path) -> Path:
    path = tmp_path / "task_eval_schema_case"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_task_types_accepts_valid_file(tmp_path: Path) -> None:
    temp_dir = _case_dir(tmp_path)
    path = temp_dir / "tasks.json"
    path.write_text(json.dumps([_task_type()]), encoding="utf-8")
    rows = load_task_types(path, strict_tool=False, enforce_coverage_policy=False)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "query_news.filter.precise.normal"
    assert rows[0]["capability"]


def test_normalize_case_enforces_retrieval_gold_subset() -> None:
    task = _task_type()
    case = normalize_case(
        _case(),
        task_type=task,
        case_id="case-1",
        pool_id="pool-1",
        input_news_pool=_pool(),
    )
    validate_case(case, strict_tool=False)
    assert case["capability"]
    assert case["retrieval_gold_urls"] == ["https://a.example.com"]
    assert case["input_news_pool_hash"] == build_news_pool_hash(_pool())


def test_normalize_case_allows_clarification_without_tool_path() -> None:
    task = _task_type()
    task["retrieval_mode"] = "non_retrieval"
    task["scenario"] = "conflict"
    task["should_clarify"] = True
    case = normalize_case(
        _case(
            expected_question="请查 Claude 但不要限定是哪类信息",
            expected_answer="请先确认你想查 Claude 产品更新、价格变化还是公司动态？",
            expected_tool_paths=[],
            required_tools=[],
            retrieval_evaluable=False,
            retrieval_gold_doc_ids=[],
            verifiable_claims=[],
        ),
        task_type=task,
        case_id="case-clarify",
        pool_id="pool-1",
        input_news_pool=_pool(),
    )

    validate_case(case, strict_tool=False)
    assert case["should_clarify"] is True
    assert case["expected_tool_paths"] == []
    assert case["required_tools"] == []


def test_normalize_case_rejects_invalid_gold_ids() -> None:
    task = _task_type()
    with pytest.raises(ValueError, match="retrieval_gold_doc_ids must be subset"):
        normalize_case(
            _case(
                expected_question="请给出问题",
                expected_answer="这是中文答案。",
                retrieval_gold_doc_ids=["doc_x"],
            ),
            task_type=task,
            case_id="case-2",
            pool_id="pool-2",
            input_news_pool=_pool(),
        )


def test_normalize_case_backfills_gold_doc_ids_from_urls() -> None:
    task = _task_type()
    case = normalize_case(
        _case(
            retrieval_gold_doc_ids=[],
            retrieval_gold_urls=["https://a.example.com"],
        ),
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
        _case(
            retrieval_gold_doc_ids=[],
            retrieval_gold_urls=[],
            verifiable_claims=[_claim("doc_2", "s2")],
        ),
        task_type=task,
        case_id="case-backfill-claim",
        pool_id="pool-backfill-claim",
        input_news_pool=_pool(),
    )
    assert case["retrieval_gold_doc_ids"] == ["doc_2"]
    assert case["retrieval_gold_urls"] == ["https://b.example.com"]


def test_retrieval_evaluable_case_requires_evidence_quotes() -> None:
    task = _task_type()
    with pytest.raises(ValueError, match="requires evidence_quotes"):
        normalize_case(
            _case(verifiable_claims=[_claim(include_quotes=False)]),
            task_type=task,
            case_id="case-missing-quotes",
            pool_id="pool-missing-quotes",
            input_news_pool=_pool(),
        )


def test_normalize_case_rejects_non_chinese_expected_question() -> None:
    task = _task_type()
    with pytest.raises(ValueError, match="expected_question must contain Chinese text"):
        normalize_case(
            _case(
                expected_question="What is new about OpenAI?",
                expected_answer="这是中文答案。",
                verifiable_claims=[],
            ),
            task_type=task,
            case_id="case-3",
            pool_id="pool-3",
            input_news_pool=_pool(),
        )


def test_normalize_case_rejects_path_drift_from_task_acceptable() -> None:
    task = _task_type()
    with pytest.raises(ValueError, match="expected_tool_paths must be subset of task acceptable_tool_paths"):
        normalize_case(
            _case(
                expected_answer="这是中文答案。",
                expected_tool_paths=_expected_tool_paths("Anthropic"),
                verifiable_claims=[],
            ),
            task_type=task,
            case_id="case-4",
            pool_id="pool-4",
            input_news_pool=_pool(),
        )
