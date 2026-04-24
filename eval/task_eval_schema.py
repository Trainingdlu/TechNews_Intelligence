"""Task-driven evaluation schema.

This module is intentionally isolated from legacy capability pipelines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from agent.core.skill_catalog import iter_skill_definitions
except Exception:  # pragma: no cover
    iter_skill_definitions = None  # type: ignore[assignment]


VALID_RETRIEVAL_MODES = {"evaluable", "non_retrieval"}
VALID_SCENARIOS = {"normal", "empty", "boundary", "conflict"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}
VALID_CLAIM_TYPES = {"fact", "number", "comparison"}

SCENARIO_COVERAGE_POLICY: dict[str, set[str]] = {
    # High-risk: require 3 scenarios.
    "query_news": {"normal", "empty", "conflict"},
    "search_news": {"normal", "empty", "conflict"},
    "build_timeline": {"normal", "empty", "conflict"},
    "fulltext_batch": {"normal", "empty", "conflict"},
    "compare_topics": {"normal", "empty", "conflict"},
    "compare_sources": {"normal", "empty", "conflict"},
    "trend_analysis": {"normal", "empty", "conflict"},
    "analyze_landscape": {"normal", "empty", "conflict"},
    # Medium-risk.
    "read_news_content": {"normal", "empty"},
    # Low-risk.
    "get_db_stats": {"normal"},
    "list_topics": {"normal"},
}

TASK_TYPE_REQUIRED_FIELDS = (
    "task_id",
    "skill",
    "intent_label",
    "retrieval_mode",
    "scenario",
    "example_question",
    "parameter_template",
    "acceptable_tool_paths",
)

CASE_REQUIRED_FIELDS = (
    "case_id",
    "pool_id",
    "input_news_pool_hash",
    "task_type",
    "skill",
    "intent_label",
    "input_news_pool",
    "expected_question",
    "expected_answer",
    "expected_tool_paths",
    "required_tools",
    "forbidden_tools",
    "retrieval_gold_doc_ids",
    "retrieval_gold_urls",
    "verifiable_claims",
    "should_clarify",
    "retrieval_evaluable",
    "difficulty",
    "tags",
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
        out = [str(item).strip() for item in value]
        return [item for item in out if item]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _contains_zh(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _canonical_path(path: list[dict[str, Any]]) -> str:
    return json.dumps(path, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _paths_subset(
    expected_paths: list[list[dict[str, Any]]],
    acceptable_paths: list[list[dict[str, Any]]],
) -> bool:
    acceptable_set = {_canonical_path(path) for path in acceptable_paths}
    return all(_canonical_path(path) in acceptable_set for path in expected_paths)


def _normalize_tool_call(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        tool = raw.strip()
        if not tool:
            raise ValueError("tool call name cannot be empty.")
        return {"tool": tool, "args": {}}

    if not isinstance(raw, dict):
        raise ValueError("tool call must be object or string.")

    tool = _as_non_empty_str(raw.get("tool"), "tool")
    args = raw.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"tool '{tool}' args must be object.")
    return {"tool": tool, "args": args}


def _normalize_tool_paths(value: Any, *, field: str) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list.")

    paths: list[list[dict[str, Any]]] = []
    for path_idx, path in enumerate(value, 1):
        if isinstance(path, str):
            call = _normalize_tool_call(path)
            paths.append([call])
            continue

        if not isinstance(path, list) or not path:
            raise ValueError(f"{field}[{path_idx}] must be a non-empty list.")
        normalized_path = [_normalize_tool_call(item) for item in path]
        paths.append(normalized_path)
    return paths


def _normalize_sampling(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    days = raw.get("days", 30)
    n_min = raw.get("n_min", 30)
    pool_size = raw.get("pool_size", 12)
    candidate_limit = raw.get("candidate_limit", 300)
    try:
        days = max(1, int(days))
    except Exception:
        days = 30
    try:
        n_min = max(1, int(n_min))
    except Exception:
        n_min = 30
    try:
        pool_size = max(1, int(pool_size))
    except Exception:
        pool_size = 12
    try:
        candidate_limit = max(pool_size, int(candidate_limit))
    except Exception:
        candidate_limit = max(pool_size, 300)

    return {
        "days": days,
        "n_min": n_min,
        "pool_size": pool_size,
        "candidate_limit": candidate_limit,
        "keywords": _as_str_list(raw.get("keywords", [])),
        "sources": _as_str_list(raw.get("sources", [])),
        "languages": _as_str_list(raw.get("languages", [])),
    }


def normalize_task_type(raw: dict[str, Any], *, strict_skill: bool = True) -> dict[str, Any]:
    for field in TASK_TYPE_REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(f"task_type missing required field: {field}")

    task_id = _as_non_empty_str(raw.get("task_id"), "task_id")
    skill = _as_non_empty_str(raw.get("skill"), "skill")
    intent_label = _as_non_empty_str(raw.get("intent_label"), "intent_label")
    retrieval_mode = _as_non_empty_str(raw.get("retrieval_mode"), "retrieval_mode").lower()
    scenario = _as_non_empty_str(raw.get("scenario"), "scenario").lower()
    example_question = _as_non_empty_str(raw.get("example_question"), "example_question")

    parameter_template = raw.get("parameter_template", {})
    if not isinstance(parameter_template, dict):
        raise ValueError(f"task_type {task_id}: parameter_template must be object.")

    acceptable_tool_paths = _normalize_tool_paths(
        raw.get("acceptable_tool_paths"),
        field=f"task_type {task_id}.acceptable_tool_paths",
    )

    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"task_type {task_id}: invalid retrieval_mode={retrieval_mode}")
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"task_type {task_id}: invalid scenario={scenario}")

    available_skills = _available_skills()
    if strict_skill and available_skills and skill not in available_skills:
        raise ValueError(f"task_type {task_id}: unknown skill={skill}")

    path_tools = {
        str(call.get("tool", "")).strip()
        for path in acceptable_tool_paths
        for call in path
    }
    if not path_tools:
        raise ValueError(f"task_type {task_id}: acceptable_tool_paths cannot be empty.")
    if skill not in path_tools:
        raise ValueError(f"task_type {task_id}: acceptable_tool_paths must include skill={skill}")
    if strict_skill and available_skills:
        unknown = sorted(tool for tool in path_tools if tool not in available_skills)
        if unknown:
            raise ValueError(f"task_type {task_id}: unknown tools in paths={unknown}")

    required_tools = _as_str_list(raw.get("required_tools", [skill]))
    if not required_tools:
        required_tools = [skill]
    forbidden_tools = _as_str_list(raw.get("forbidden_tools", []))
    if set(required_tools).intersection(forbidden_tools):
        raise ValueError(f"task_type {task_id}: required_tools overlaps forbidden_tools.")

    should_clarify = bool(raw.get("should_clarify", False))
    difficulty = str(raw.get("difficulty", "medium")).strip().lower()
    if difficulty not in VALID_DIFFICULTY:
        raise ValueError(f"task_type {task_id}: invalid difficulty={difficulty}")

    return {
        "task_id": task_id,
        "skill": skill,
        "intent_label": intent_label,
        "retrieval_mode": retrieval_mode,
        "scenario": scenario,
        "example_question": example_question,
        "parameter_template": parameter_template,
        "acceptable_tool_paths": acceptable_tool_paths,
        "required_tools": required_tools,
        "forbidden_tools": forbidden_tools,
        "should_clarify": should_clarify,
        "difficulty": difficulty,
        "sampling": _normalize_sampling(raw.get("sampling", {})),
        "tags": _as_str_list(raw.get("tags", [scenario])),
    }


def validate_task_type_scenario_coverage(task_types: list[dict[str, Any]]) -> None:
    by_skill: dict[str, set[str]] = {}
    for row in task_types:
        skill = str(row.get("skill", "")).strip()
        scenario = str(row.get("scenario", "")).strip().lower()
        if not skill or not scenario:
            continue
        by_skill.setdefault(skill, set()).add(scenario)

    missing: dict[str, list[str]] = {}
    for skill, required in SCENARIO_COVERAGE_POLICY.items():
        covered = by_skill.get(skill, set())
        missing_scenarios = sorted(required.difference(covered))
        if missing_scenarios:
            missing[skill] = missing_scenarios
    if missing:
        raise ValueError(f"task_type scenario coverage policy violated: {missing}")


def load_task_types(
    path: Path,
    *,
    strict_skill: bool = True,
    enforce_coverage_policy: bool = True,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("task_types file must be a JSON array.")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, row in enumerate(payload, 1):
        if not isinstance(row, dict):
            raise ValueError(f"task_types[{idx}] must be object.")
        normalized = normalize_task_type(row, strict_skill=strict_skill)
        task_id = normalized["task_id"]
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        out.append(normalized)
    if enforce_coverage_policy:
        validate_task_type_scenario_coverage(out)
    return out


def build_news_pool_hash(input_news_pool: list[dict[str, Any]]) -> str:
    canonical = []
    for doc in input_news_pool:
        canonical.append(
            {
                "doc_id": str(doc.get("doc_id", "")).strip(),
                "url": str(doc.get("url", "")).strip(),
                "title": str(doc.get("title", "")).strip(),
                "published_at": str(doc.get("published_at", "")).strip(),
                "source": str(doc.get("source", "")).strip(),
            }
        )
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize_claim(raw: Any, *, case_id: str, pool_doc_ids: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{case_id}: verifiable_claims item must be object.")
    claim = _as_non_empty_str(raw.get("claim"), f"{case_id}.claim")
    claim_type = str(raw.get("claim_type", "fact")).strip().lower()
    if claim_type not in VALID_CLAIM_TYPES:
        raise ValueError(f"{case_id}: invalid claim_type={claim_type}")
    evidence_doc_ids = _as_str_list(raw.get("evidence_doc_ids", []))
    if evidence_doc_ids and not set(evidence_doc_ids).issubset(pool_doc_ids):
        raise ValueError(f"{case_id}: claim evidence_doc_ids must be subset of input_news_pool doc_id.")
    return {
        "claim": claim,
        "evidence_doc_ids": evidence_doc_ids,
        "claim_type": claim_type,
    }


def normalize_case(
    raw_case: dict[str, Any],
    *,
    task_type: dict[str, Any],
    case_id: str,
    pool_id: str,
    input_news_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    input_news_pool_hash = build_news_pool_hash(input_news_pool)
    pool_doc_ids = {
        str(doc.get("doc_id", "")).strip()
        for doc in input_news_pool
        if str(doc.get("doc_id", "")).strip()
    }
    pool_urls = {
        str(doc.get("url", "")).strip()
        for doc in input_news_pool
        if str(doc.get("url", "")).strip()
    }
    id_to_url = {
        str(doc.get("doc_id", "")).strip(): str(doc.get("url", "")).strip()
        for doc in input_news_pool
        if str(doc.get("doc_id", "")).strip() and str(doc.get("url", "")).strip()
    }
    url_to_id = {url: doc_id for doc_id, url in id_to_url.items()}

    expected_question = _as_non_empty_str(raw_case.get("expected_question"), "expected_question")
    expected_answer = _as_non_empty_str(raw_case.get("expected_answer"), "expected_answer")
    if not _contains_zh(expected_question):
        raise ValueError(f"{case_id}: expected_question must contain Chinese text.")
    if not _contains_zh(expected_answer):
        raise ValueError(f"{case_id}: expected_answer must contain Chinese text.")

    expected_tool_paths = _normalize_tool_paths(
        raw_case.get("expected_tool_paths", task_type["acceptable_tool_paths"]),
        field=f"{case_id}.expected_tool_paths",
    )
    acceptable_tool_paths = _normalize_tool_paths(
        task_type.get("acceptable_tool_paths", []),
        field=f"{case_id}.acceptable_tool_paths",
    )
    if not _paths_subset(expected_tool_paths, acceptable_tool_paths):
        raise ValueError(f"{case_id}: expected_tool_paths must be subset of task acceptable_tool_paths.")
    required_tools = _as_str_list(raw_case.get("required_tools", task_type["required_tools"]))
    forbidden_tools = _as_str_list(raw_case.get("forbidden_tools", task_type["forbidden_tools"]))
    if set(required_tools).intersection(forbidden_tools):
        raise ValueError(f"{case_id}: required_tools overlaps forbidden_tools.")

    claims_raw = raw_case.get("verifiable_claims", [])
    if not isinstance(claims_raw, list):
        raise ValueError(f"{case_id}: verifiable_claims must be a list.")
    verifiable_claims = [
        _normalize_claim(item, case_id=case_id, pool_doc_ids=pool_doc_ids)
        for item in claims_raw
    ]
    claim_evidence_doc_ids = _dedupe_keep_order(
        [
            str(doc_id).strip()
            for claim in verifiable_claims
            for doc_id in claim.get("evidence_doc_ids", [])
            if str(doc_id).strip()
        ]
    )

    retrieval_evaluable = bool(raw_case.get("retrieval_evaluable", task_type["retrieval_mode"] == "evaluable"))
    retrieval_gold_doc_ids = _dedupe_keep_order(_as_str_list(raw_case.get("retrieval_gold_doc_ids", [])))
    retrieval_gold_urls = _dedupe_keep_order(_as_str_list(raw_case.get("retrieval_gold_urls", [])))

    if retrieval_evaluable:
        if retrieval_gold_urls and not set(retrieval_gold_urls).issubset(pool_urls):
            raise ValueError(f"{case_id}: retrieval_gold_urls must be subset of input_news_pool urls.")
        if retrieval_gold_doc_ids and not set(retrieval_gold_doc_ids).issubset(pool_doc_ids):
            raise ValueError(f"{case_id}: retrieval_gold_doc_ids must be subset of input_news_pool doc_id.")

        if not retrieval_gold_doc_ids and retrieval_gold_urls:
            retrieval_gold_doc_ids = [
                url_to_id[url]
                for url in retrieval_gold_urls
                if url in url_to_id
            ]
        if not retrieval_gold_doc_ids and claim_evidence_doc_ids:
            retrieval_gold_doc_ids = [
                doc_id
                for doc_id in claim_evidence_doc_ids
                if doc_id in pool_doc_ids
            ]
        if not retrieval_gold_doc_ids:
            raise ValueError(f"{case_id}: retrieval_evaluable case requires retrieval_gold_doc_ids.")
        if not set(retrieval_gold_doc_ids).issubset(pool_doc_ids):
            raise ValueError(f"{case_id}: retrieval_gold_doc_ids must be subset of input_news_pool doc_id.")
        if not retrieval_gold_urls:
            retrieval_gold_urls = [
                id_to_url.get(doc_id, "")
                for doc_id in retrieval_gold_doc_ids
                if id_to_url.get(doc_id, "")
            ]
    else:
        retrieval_gold_doc_ids = []
        retrieval_gold_urls = []

    difficulty = str(raw_case.get("difficulty", task_type.get("difficulty", "medium"))).strip().lower()
    if difficulty not in VALID_DIFFICULTY:
        raise ValueError(f"{case_id}: invalid difficulty={difficulty}")

    case = {
        "case_id": _as_non_empty_str(case_id, "case_id"),
        "pool_id": _as_non_empty_str(pool_id, "pool_id"),
        "input_news_pool_hash": input_news_pool_hash,
        "task_type": task_type["task_id"],
        "skill": task_type["skill"],
        "intent_label": str(raw_case.get("intent_label", task_type["intent_label"])).strip() or task_type["intent_label"],
        "input_news_pool": input_news_pool,
        "expected_question": expected_question,
        "expected_answer": expected_answer,
        "expected_tool_paths": expected_tool_paths,
        "required_tools": required_tools,
        "forbidden_tools": forbidden_tools,
        "retrieval_gold_doc_ids": retrieval_gold_doc_ids,
        "retrieval_gold_urls": retrieval_gold_urls,
        "verifiable_claims": verifiable_claims,
        "should_clarify": bool(raw_case.get("should_clarify", task_type["should_clarify"])),
        "retrieval_evaluable": retrieval_evaluable,
        "difficulty": difficulty,
        "tags": _as_str_list(raw_case.get("tags", task_type.get("tags", []))),
    }
    validate_case(case, strict_skill=False)
    return case


def validate_case(case: dict[str, Any], *, strict_skill: bool = True) -> None:
    for field in CASE_REQUIRED_FIELDS:
        if field not in case:
            raise ValueError(f"case missing required field: {field}")

    case_id = _as_non_empty_str(case.get("case_id"), "case_id")
    _as_non_empty_str(case.get("pool_id"), f"{case_id}.pool_id")
    _as_non_empty_str(case.get("input_news_pool_hash"), f"{case_id}.input_news_pool_hash")
    _as_non_empty_str(case.get("task_type"), f"{case_id}.task_type")
    skill = _as_non_empty_str(case.get("skill"), f"{case_id}.skill")
    _as_non_empty_str(case.get("intent_label"), f"{case_id}.intent_label")
    expected_question = _as_non_empty_str(case.get("expected_question"), f"{case_id}.expected_question")
    expected_answer = _as_non_empty_str(case.get("expected_answer"), f"{case_id}.expected_answer")
    if not _contains_zh(expected_question):
        raise ValueError(f"{case_id}: expected_question must contain Chinese text.")
    if not _contains_zh(expected_answer):
        raise ValueError(f"{case_id}: expected_answer must contain Chinese text.")

    available_skills = _available_skills()
    if strict_skill and available_skills and skill not in available_skills:
        raise ValueError(f"{case_id}: unknown skill={skill}")

    input_news_pool = case.get("input_news_pool")
    if not isinstance(input_news_pool, list):
        raise ValueError(f"{case_id}: input_news_pool must be list.")
    doc_ids: set[str] = set()
    urls: set[str] = set()
    for idx, doc in enumerate(input_news_pool, 1):
        if not isinstance(doc, dict):
            raise ValueError(f"{case_id}: input_news_pool[{idx}] must be object.")
        doc_id = _as_non_empty_str(doc.get("doc_id"), f"{case_id}.input_news_pool[{idx}].doc_id")
        url = _as_non_empty_str(doc.get("url"), f"{case_id}.input_news_pool[{idx}].url")
        _as_non_empty_str(doc.get("title"), f"{case_id}.input_news_pool[{idx}].title")
        _as_non_empty_str(doc.get("published_at"), f"{case_id}.input_news_pool[{idx}].published_at")
        _as_non_empty_str(doc.get("source"), f"{case_id}.input_news_pool[{idx}].source")
        doc_ids.add(doc_id)
        urls.add(url)

    expected_tool_paths = _normalize_tool_paths(
        case.get("expected_tool_paths"),
        field=f"{case_id}.expected_tool_paths",
    )
    if strict_skill and available_skills:
        unknown_tools = sorted(
            {
                str(call.get("tool", "")).strip()
                for path in expected_tool_paths
                for call in path
                if str(call.get("tool", "")).strip() not in available_skills
            }
        )
        if unknown_tools:
            raise ValueError(f"{case_id}: unknown tools in expected_tool_paths={unknown_tools}")
    if skill not in {
        call["tool"]
        for path in expected_tool_paths
        for call in path
    }:
        raise ValueError(f"{case_id}: expected_tool_paths must include skill={skill}.")

    required_tools = _as_str_list(case.get("required_tools"))
    if not required_tools:
        raise ValueError(f"{case_id}: required_tools must be non-empty.")
    forbidden_tools = _as_str_list(case.get("forbidden_tools"))
    if set(required_tools).intersection(forbidden_tools):
        raise ValueError(f"{case_id}: required_tools overlaps forbidden_tools.")

    retrieval_evaluable = bool(case.get("retrieval_evaluable"))
    retrieval_gold_doc_ids = _dedupe_keep_order(_as_str_list(case.get("retrieval_gold_doc_ids")))
    retrieval_gold_urls = _dedupe_keep_order(_as_str_list(case.get("retrieval_gold_urls")))
    pool_id_to_url = {
        str(doc.get("doc_id", "")).strip(): str(doc.get("url", "")).strip()
        for doc in input_news_pool
        if str(doc.get("doc_id", "")).strip() and str(doc.get("url", "")).strip()
    }
    pool_url_to_id = {url: doc_id for doc_id, url in pool_id_to_url.items()}
    if retrieval_evaluable:
        if not retrieval_gold_doc_ids and retrieval_gold_urls:
            retrieval_gold_doc_ids = [
                pool_url_to_id[url]
                for url in retrieval_gold_urls
                if url in pool_url_to_id
            ]
        if not retrieval_gold_urls and retrieval_gold_doc_ids:
            retrieval_gold_urls = [
                pool_id_to_url.get(doc_id, "")
                for doc_id in retrieval_gold_doc_ids
                if pool_id_to_url.get(doc_id, "")
            ]
        if not retrieval_gold_doc_ids:
            raise ValueError(f"{case_id}: retrieval_evaluable requires retrieval_gold_doc_ids.")
        if not set(retrieval_gold_doc_ids).issubset(doc_ids):
            raise ValueError(f"{case_id}: retrieval_gold_doc_ids must be subset of input_news_pool doc_id.")
        if retrieval_gold_urls and not set(retrieval_gold_urls).issubset(urls):
            raise ValueError(f"{case_id}: retrieval_gold_urls must be subset of input_news_pool urls.")
    else:
        if retrieval_gold_doc_ids or retrieval_gold_urls:
            raise ValueError(f"{case_id}: non-retrieval case must not contain retrieval gold labels.")

    claims = case.get("verifiable_claims")
    if not isinstance(claims, list):
        raise ValueError(f"{case_id}: verifiable_claims must be list.")
    for idx, item in enumerate(claims, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{case_id}: verifiable_claims[{idx}] must be object.")
        _as_non_empty_str(item.get("claim"), f"{case_id}.verifiable_claims[{idx}].claim")
        claim_type = str(item.get("claim_type", "")).strip().lower()
        if claim_type not in VALID_CLAIM_TYPES:
            raise ValueError(f"{case_id}: verifiable_claims[{idx}] invalid claim_type={claim_type}")
        claim_doc_ids = _as_str_list(item.get("evidence_doc_ids", []))
        if claim_doc_ids and not set(claim_doc_ids).issubset(doc_ids):
            raise ValueError(
                f"{case_id}: verifiable_claims[{idx}].evidence_doc_ids must be subset of input_news_pool doc_id."
            )

    difficulty = str(case.get("difficulty", "")).strip().lower()
    if difficulty not in VALID_DIFFICULTY:
        raise ValueError(f"{case_id}: invalid difficulty={difficulty}")
    if not isinstance(case.get("should_clarify"), bool):
        raise ValueError(f"{case_id}: should_clarify must be bool.")
    tags = case.get("tags")
    if not isinstance(tags, list):
        raise ValueError(f"{case_id}: tags must be list.")


def load_cases_jsonl(path: Path, *, strict_skill: bool = False) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {idx}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {idx}: case must be object.")
        validate_case(row, strict_skill=strict_skill)
        cases.append(row)
    if not cases:
        raise ValueError("empty case dataset.")
    return cases
