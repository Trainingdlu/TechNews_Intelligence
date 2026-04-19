from __future__ import annotations

import json
from pathlib import Path

from agent.core.skill_catalog import iter_skill_definitions
from eval.task_eval_v1_schema import SCENARIO_COVERAGE_POLICY
from eval.task_eval_v1_scoring import score_case


def test_task_types_cover_all_current_skills() -> None:
    task_types = json.loads(
        Path("eval/config/task_types_v1.json").read_text(encoding="utf-8-sig")
    )
    task_skills = {
        str(item.get("skill", "")).strip()
        for item in task_types
        if isinstance(item, dict) and str(item.get("skill", "")).strip()
    }
    all_skills = {row.name for row in iter_skill_definitions()}
    missing = sorted(all_skills - task_skills)
    assert missing == []


def test_task_types_match_risk_based_scenario_policy() -> None:
    task_types = json.loads(
        Path("eval/config/task_types_v1.json").read_text(encoding="utf-8-sig")
    )
    by_skill: dict[str, set[str]] = {}
    for item in task_types:
        if not isinstance(item, dict):
            continue
        skill = str(item.get("skill", "")).strip()
        scenario = str(item.get("scenario", "")).strip().lower()
        if not skill or not scenario:
            continue
        by_skill.setdefault(skill, set()).add(scenario)

    for skill, required in SCENARIO_COVERAGE_POLICY.items():
        assert required.issubset(by_skill.get(skill, set()))


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
