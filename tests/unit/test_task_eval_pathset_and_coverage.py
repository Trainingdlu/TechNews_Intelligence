from __future__ import annotations

import json
from pathlib import Path

from agent.core.tool_catalog import iter_tool_definitions
from eval.task_eval_schema import SCENARIO_COVERAGE_POLICY
from eval.task_eval_scoring import score_case


def test_task_types_cover_all_current_tools() -> None:
    task_types = json.loads(
        Path("eval/config/task_types_retrieval.json").read_text(encoding="utf-8-sig")
    )
    task_tools = {
        str(item.get("tool", "")).strip()
        for item in task_types
        if isinstance(item, dict) and str(item.get("tool", "")).strip()
    }
    all_tools = {row.name for row in iter_tool_definitions()}
    unknown = sorted(task_tools - all_tools)
    assert unknown == []


def test_task_types_match_risk_based_scenario_policy() -> None:
    task_types = json.loads(
        Path("eval/config/task_types_retrieval.json").read_text(encoding="utf-8-sig")
    )
    by_tool: dict[str, set[str]] = {}
    for item in task_types:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", "")).strip()
        scenario = str(item.get("scenario", "")).strip().lower()
        if not tool or not scenario:
            continue
        by_tool.setdefault(tool, set()).add(scenario)

    for tool, covered in by_tool.items():
        required = SCENARIO_COVERAGE_POLICY.get(tool)
        if not required:
            continue
        assert required.issubset(covered)


def test_path_set_best_match_uses_alternative_path() -> None:
    case = {
        "intent_label": "fulltext_batch",
        "should_clarify": False,
        "required_tools": ["fulltext_batch"],
        "forbidden_tools": [],
        "expected_tool_paths": [
            [
                {
                    "tool": "fulltext_batch",
                    "args": {
                        "urls": "OpenAI Voice Engine recent 14 days",
                        "max_chars_per_article": 4000,
                    },
                }
            ],
            [
                {
                    "tool": "search_news",
                    "args": {
                        "query": "OpenAI Voice Engine",
                        "days": 14,
                    },
                },
                {
                    "tool": "fulltext_batch",
                    "args": {
                        "urls": "OpenAI Voice Engine recent 14 days",
                        "max_chars_per_article": 4000,
                    },
                },
            ],
        ],
        "retrieval_evaluable": False,
        "verifiable_claims": [],
        "input_news_pool": [],
    }

    run = {
        "final_answer": "summary",
        "tool_calls": ["search_news", "fulltext_batch"],
        "tool_calls_detailed": [
            {
                "tool": "search_news",
                "args": {
                    "query": "OpenAI Voice Engine",
                    "days": 14,
                },
            },
            {
                "tool": "fulltext_batch",
                "args": {
                    "urls": "OpenAI Voice Engine recent 14 days",
                    "max_chars_per_article": 4000,
                },
            },
        ],
        "retrieved_urls": [],
        "citations": [],
        "clarification_triggered": False,
        "error": "",
        "latency_ms": 120.0,
        "final_status": "success",
    }

    layers = score_case(case, [run])
    assert layers["tool"]["acceptable_path_hit_rate"] == 1.0
    assert layers["tool"]["param_accuracy"] == 1.0


def test_clarification_case_does_not_require_tool_path_hit() -> None:
    case = {
        "intent_label": "search_news",
        "should_clarify": True,
        "required_tools": [],
        "forbidden_tools": [],
        "expected_tool_paths": [],
        "retrieval_evaluable": False,
        "verifiable_claims": [],
        "input_news_pool": [],
    }
    run = {
        "final_answer": "请先澄清你指的是 Claude Code 还是 Claude API 定价？",
        "tool_calls": [],
        "tool_calls_detailed": [],
        "retrieved_urls": [],
        "citations": [],
        "clarification_triggered": True,
        "error": "",
        "latency_ms": 120.0,
        "final_status": "clarification_required",
    }

    layers = score_case(case, [run])
    assert layers["intent"]["clarification_accuracy"] == 1.0
    assert layers["intent"]["top1_accuracy"] == 1.0
    assert layers["tool"]["case_count"] == 0
    assert layers["tool"]["acceptable_path_hit_rate"] is None
    assert layers["tool"]["param_accuracy"] is None
    assert layers["attribution"]["code"] == "PASS"


