"""Schema helpers for task-driven evaluation dataset (v2).

This module is intentionally independent from legacy capability pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from agent.core.skill_catalog import iter_skill_definitions
except Exception:  # pragma: no cover - keep schema usable in limited envs
    iter_skill_definitions = None  # type: ignore[assignment]


VALID_RETRIEVAL_MODES = {"evaluable", "non_retrieval"}
VALID_SCENARIOS = {"normal", "empty", "boundary"}

TASK_TYPE_REQUIRED_FIELDS = (
    "task_id",
    "skill",
    "retrieval_mode",
    "scenario",
    "example_question",
    "expected_tool_path",
)

CASE_REQUIRED_FIELDS = (
    "case_id",
    "task_id",
    "skill",
    "retrieval_mode",
    "scenario",
    "question",
    "expected_answer",
    "expected_tool_calls",
    "news_doc_ids",
    "retrieval_gold_doc_ids",
)


def _available_skills() -> set[str]:
    if iter_skill_definitions is None:
        return set()
    try:
        return {row.name for row in iter_skill_definitions()}
    except Exception:
        return set()


def _as_non_empty_str(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty.")
    return text


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
        return [item for item in items if item]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _normalize_sampling(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    days = raw.get("days", 30)
    limit = raw.get("limit", 30)
    try:
        days = max(1, int(days))
    except Exception:
        days = 30
    try:
        limit = max(1, int(limit))
    except Exception:
        limit = 30
    return {
        "days": days,
        "limit": limit,
        "keywords": _as_str_list(raw.get("keywords", [])),
        "sources": _as_str_list(raw.get("sources", [])),
    }


def normalize_task_type(raw: dict[str, Any], *, strict_skill: bool = True) -> dict[str, Any]:
    for field in TASK_TYPE_REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(f"task_type missing required field: {field}")

    task_id = _as_non_empty_str(raw.get("task_id"), "task_id")
    skill = _as_non_empty_str(raw.get("skill"), "skill")
    retrieval_mode = _as_non_empty_str(raw.get("retrieval_mode"), "retrieval_mode").lower()
    scenario = _as_non_empty_str(raw.get("scenario"), "scenario").lower()
    example_question = _as_non_empty_str(raw.get("example_question"), "example_question")
    expected_tool_path = _as_str_list(raw.get("expected_tool_path"))

    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"task_type {task_id}: invalid retrieval_mode={retrieval_mode}")
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"task_type {task_id}: invalid scenario={scenario}")
    if not expected_tool_path:
        raise ValueError(f"task_type {task_id}: expected_tool_path cannot be empty")

    available_skills = _available_skills()
    if strict_skill and available_skills and skill not in available_skills:
        raise ValueError(f"task_type {task_id}: unknown skill={skill}")
    if skill not in expected_tool_path:
        raise ValueError(
            f"task_type {task_id}: expected_tool_path must include skill={skill}"
        )

    return {
        "task_id": task_id,
        "skill": skill,
        "retrieval_mode": retrieval_mode,
        "scenario": scenario,
        "example_question": example_question,
        "expected_tool_path": expected_tool_path,
        "sampling": _normalize_sampling(raw.get("sampling", {})),
    }


def load_task_types(path: Path, *, strict_skill: bool = True) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("task_types file must be a JSON array.")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, row in enumerate(payload, 1):
        if not isinstance(row, dict):
            raise ValueError(f"task_types[{idx}] must be an object.")
        normalized = normalize_task_type(row, strict_skill=strict_skill)
        task_id = normalized["task_id"]
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        out.append(normalized)
    return out


def _normalize_expected_tool_calls(value: Any, fallback_path: list[str]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    calls.append({"tool": text, "args": {}})
                continue
            if isinstance(item, dict):
                tool = str(item.get("tool", "")).strip()
                args = item.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                if tool:
                    calls.append({"tool": tool, "args": args})
    if calls:
        return calls
    return [{"tool": tool, "args": {}} for tool in fallback_path]


def normalize_generated_case(
    raw_case: dict[str, Any],
    *,
    task_type: dict[str, Any],
    case_id: str,
    news_doc_ids: list[str],
) -> dict[str, Any]:
    question = _as_non_empty_str(raw_case.get("question"), "question")
    expected_answer = _as_non_empty_str(raw_case.get("expected_answer"), "expected_answer")

    expected_tool_calls = _normalize_expected_tool_calls(
        raw_case.get("expected_tool_calls"),
        fallback_path=task_type["expected_tool_path"],
    )
    expected_tools = [str(call.get("tool", "")).strip() for call in expected_tool_calls]
    if any(not item for item in expected_tools):
        raise ValueError(f"{case_id}: expected_tool_calls contains empty tool name.")

    allowed_skills = _available_skills()
    if allowed_skills:
        unknown = sorted({tool for tool in expected_tools if tool not in allowed_skills})
        if unknown:
            raise ValueError(f"{case_id}: unknown tools in expected_tool_calls={unknown}")

    gold_ids = _as_str_list(raw_case.get("retrieval_gold_doc_ids", []))
    gold_set = set(gold_ids)
    doc_set = set(news_doc_ids)
    if not gold_set.issubset(doc_set):
        raise ValueError(
            f"{case_id}: retrieval_gold_doc_ids must be subset of news_doc_ids."
        )
    if task_type["retrieval_mode"] == "non_retrieval":
        gold_ids = []

    case = {
        "case_id": case_id,
        "task_id": task_type["task_id"],
        "skill": task_type["skill"],
        "retrieval_mode": task_type["retrieval_mode"],
        "scenario": task_type["scenario"],
        "question": question,
        "expected_answer": expected_answer,
        "expected_tool_calls": expected_tool_calls,
        "news_doc_ids": list(news_doc_ids),
        "retrieval_gold_doc_ids": gold_ids,
    }
    validate_case(case)
    return case


def validate_case(case: dict[str, Any]) -> None:
    for field in CASE_REQUIRED_FIELDS:
        if field not in case:
            raise ValueError(f"case missing required field: {field}")

    _as_non_empty_str(case.get("case_id"), "case_id")
    _as_non_empty_str(case.get("task_id"), "task_id")
    _as_non_empty_str(case.get("skill"), "skill")
    retrieval_mode = _as_non_empty_str(case.get("retrieval_mode"), "retrieval_mode").lower()
    scenario = _as_non_empty_str(case.get("scenario"), "scenario").lower()
    _as_non_empty_str(case.get("question"), "question")
    _as_non_empty_str(case.get("expected_answer"), "expected_answer")

    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"invalid retrieval_mode={retrieval_mode}")
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"invalid scenario={scenario}")

    expected_tool_calls = case.get("expected_tool_calls")
    if not isinstance(expected_tool_calls, list) or not expected_tool_calls:
        raise ValueError("expected_tool_calls must be a non-empty list")
    for idx, call in enumerate(expected_tool_calls, 1):
        if not isinstance(call, dict):
            raise ValueError(f"expected_tool_calls[{idx}] must be an object")
        _as_non_empty_str(call.get("tool"), f"expected_tool_calls[{idx}].tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"expected_tool_calls[{idx}].args must be an object")

    for key in ("news_doc_ids", "retrieval_gold_doc_ids"):
        value = case.get(key)
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list")
        if any(not str(item).strip() for item in value):
            raise ValueError(f"{key} contains empty id")

    if retrieval_mode == "non_retrieval" and case.get("retrieval_gold_doc_ids"):
        raise ValueError("non_retrieval case cannot contain retrieval_gold_doc_ids")

    doc_ids = set(str(item).strip() for item in case.get("news_doc_ids", []))
    gold_ids = set(str(item).strip() for item in case.get("retrieval_gold_doc_ids", []))
    if not gold_ids.issubset(doc_ids):
        raise ValueError("retrieval_gold_doc_ids must be subset of news_doc_ids")

