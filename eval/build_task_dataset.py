"""Build legacy task-driven eval dataset.

This script is kept as an experimental/compatibility path. The primary eval
dataset flow is now event-driven:

1) eval/build_event_cards.py
2) eval/build_event_eval_datasets.py
3) eval/run_retrieval_eval.py, eval/run_generation_eval.py,
   eval/run_e2e_event_eval.py

Flow:
1) Load task types (task-driven).
2) Sample news pools per task type.
3) One LLM call per task type to generate expected question/answer/path.
4) Contract validation + optional audit model.
5) Export JSONL cases + manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from services.llm_provider import (
    DEFAULT_DEEPSEEK_MODEL,
    build_chat_model as build_shared_chat_model,
    resolve_model_config,
)
from services.db import get_conn, put_conn

try:
    from audit_task_topics import audit_task_with_sample
    from corpus_sampler import build_eval_sample
    from evidence_validator import EvidenceValidationResult, validate_case_evidence
    from pool_quality import pool_quality_summary, topic_anchor_terms
    from task_eval_schema import (
        build_news_pool_hash,
        load_task_types,
        normalize_case,
        validate_case,
    )
except ImportError:  # package-style import fallback
    from .audit_task_topics import audit_task_with_sample
    from .corpus_sampler import build_eval_sample
    from .evidence_validator import EvidenceValidationResult, validate_case_evidence
    from .pool_quality import pool_quality_summary, topic_anchor_terms
    from .task_eval_schema import (
        build_news_pool_hash,
        load_task_types,
        normalize_case,
        validate_case,
    )


DEFAULT_MODEL = DEFAULT_DEEPSEEK_MODEL
DEFAULT_PROVIDER = "deepseek"
SCHEMA_VERSION = "task_eval_schema@2026-04-25-evidence-quotes"
BUILD_LOGIC_VERSION = "task_dataset_build@2026-05-17-pool-quality"
SCENARIO_RETRIEVAL_MODE_MAP: dict[str, str] = {
    "normal": "evaluable",
    "conflict": "non_retrieval",
    "boundary": "evaluable",
    "empty": "non_retrieval",
}
T = TypeVar("T")
_TASK_EVAL_MODEL_AUTO = "__task_eval_model_auto__"


@dataclass
class Pool:
    pool_id: str
    docs: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)


def _json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_sampling_for_fingerprint(task_types: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in task_types:
        sampling = task.get("sampling", {})
        if not isinstance(sampling, dict):
            sampling = {}
        rows.append(
            {
                "task_id": str(task.get("task_id", "")).strip(),
                "tool": str(task.get("tool", "")).strip(),
                "capability": str(task.get("capability", "")).strip(),
                "scenario": str(task.get("scenario", "")).strip().lower(),
                "retrieval_mode": str(task.get("retrieval_mode", "")).strip().lower(),
                "sampling": {
                    "n_min": int(sampling.get("n_min", 0) or 0),
                    "days": int(sampling.get("days", 0) or 0),
                    "pool_size": int(sampling.get("pool_size", 0) or 0),
                    "candidate_limit": int(sampling.get("candidate_limit", 0) or 0),
                    "keywords": sorted(str(item).strip() for item in sampling.get("keywords", []) if str(item).strip()),
                    "sources": sorted(str(item).strip() for item in sampling.get("sources", []) if str(item).strip()),
                    "languages": sorted(
                        str(item).strip().lower()
                        for item in sampling.get("languages", [])
                        if str(item).strip()
                    ),
                },
            }
        )
    rows.sort(key=lambda row: row.get("task_id", ""))
    return rows


def build_dataset_fingerprint_payload(
    *,
    args: argparse.Namespace,
    task_types: list[dict[str, Any]],
) -> dict[str, Any]:
    task_file = args.task_types.resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "build_logic_version": BUILD_LOGIC_VERSION,
        "task_type_file": str(task_file),
        "task_type_file_sha256": _file_sha256(task_file),
        "task_type_count": len(task_types),
        "seed": int(args.seed),
        "pools_per_task_override": int(args.pools_per_task),
        "coverage_policy_enforced": bool(args.enforce_coverage_policy),
        "scenario_retrieval_map_enforced": bool(getattr(args, "enforce_scenario_retrieval_map", False)),
        "audit_policy": {
            "enabled": not bool(args.disable_audit),
            "allow_topic_audit_fail": bool(getattr(args, "allow_topic_audit_fail", False)),
            "max_seed_attempts": int(getattr(args, "seed_max_attempts", 3)),
            "temperature_schedule": "feedback_descending",
            "legacy_max_regen_rounds": int(args.audit_max_regen_rounds),
            "legacy_regen_mode": str(args.audit_regen_mode),
            "initial_cases_per_audit_call": int(getattr(args, "initial_cases_per_audit_call", 0)),
            "regen_cases_per_audit_call": int(getattr(args, "regen_cases_per_audit_call", 1)),
            "pools_per_generation_call": int(args.pools_per_generation_call),
            "regen_pools_per_generation_call": int(args.regen_pools_per_generation_call),
        },
        "sampler_policy": {
            "candidate_source": "eval_corpus_sampler",
            "production_retrieval_topk_used": False,
            "constraint_aware_packing": True,
            "topic_preflight_gate": True,
            "points_used": False,
        },
        "generator_profile": {
            "provider": str(args.provider),
            "model": str(args.model),
            "temperature": float(args.temperature),
            "audit_temperature": float(args.audit_temperature),
        },
        "sampling_controls": _normalize_sampling_for_fingerprint(task_types),
    }


def build_dataset_fingerprint(
    *,
    args: argparse.Namespace,
    task_types: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    payload = build_dataset_fingerprint_payload(args=args, task_types=task_types)
    return _json_sha256(payload), payload


def validate_scenario_retrieval_map(task_types: list[dict[str, Any]]) -> None:
    violations: list[str] = []
    for task in task_types:
        task_id = str(task.get("task_id", "")).strip() or "<unknown>"
        scenario = str(task.get("scenario", "")).strip().lower()
        retrieval_mode = str(task.get("retrieval_mode", "")).strip().lower()
        expected = SCENARIO_RETRIEVAL_MODE_MAP.get(scenario)
        if not expected:
            continue
        if retrieval_mode != expected:
            violations.append(
                f"{task_id}: scenario={scenario} requires retrieval_mode={expected}, got={retrieval_mode}"
            )
    if violations:
        joined = "; ".join(violations)
        raise ValueError(f"scenario->retrieval_mode policy violated: {joined}")


def _resolve_preferred_provider() -> str:
    explicit = str(os.getenv("TASK_EVAL_PROVIDER", "")).strip()
    if explicit:
        return explicit
    return DEFAULT_PROVIDER


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
    if env_file:
        candidate = env_file.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Env file not found: {candidate}")
        load_dotenv(dotenv_path=candidate, override=True)

    # Dataset generation defaults to DeepSeek unless TASK_EVAL_PROVIDER is explicit.
    if not str(os.getenv("TASK_EVAL_PROVIDER", "")).strip():
        os.environ["TASK_EVAL_PROVIDER"] = _resolve_preferred_provider()


def _extract_first_json_object(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("LLM output is empty.")
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM output.")
    depth = 0
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Incomplete JSON object in LLM output.")


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    chunks.append(str(txt))
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content)


def _build_chat_model(provider: str, model_name: str, temperature: float) -> Any:
    return build_shared_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=float(temperature),
        default_provider=DEFAULT_PROVIDER,
        default_model=DEFAULT_MODEL,
    )


def _is_retryable_llm_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    retry_markers = (
        "429",
        "rate limit",
        "resource exhausted",
        "quota",
        "too many requests",
        "deadline exceeded",
        "timeout",
        "timed out",
        "service unavailable",
        "503",
    )
    return any(marker in text for marker in retry_markers)


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}")
QUESTION_STOPWORDS = {
    "请", "帮我", "过去", "最近", "新闻", "分析", "比较", "构建", "时间线", "趋势", "关于",
    "the", "and", "for", "with", "over", "last", "days", "news", "compare", "analyze", "build",
}
ANCHOR_STOPWORDS = QUESTION_STOPWORDS.union(
    {
        "a",
        "an",
        "about",
        "around",
        "day",
        "english",
        "from",
        "http",
        "https",
        "latest",
        "com",
        "net",
        "org",
        "recent",
        "related",
        "search",
        "updates",
        "vs",
        "week",
    }
)
SOURCE_ALIASES = {
    "techcrunch": "techcrunch",
    "hackernews": "hackernews",
    "hacker news": "hackernews",
    "wsj": "wsj",
    "wall street journal": "wsj",
    "arstechnica": "arstechnica",
    "ars technica": "arstechnica",
}


def _contains_zh(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _canonical_path(path: list[dict[str, Any]]) -> str:
    return json.dumps(path, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_path_list(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list):
        return []
    out: list[list[dict[str, Any]]] = []
    for item in value:
        if not isinstance(item, list) or not item:
            continue
        path: list[dict[str, Any]] = []
        for step in item:
            if isinstance(step, str):
                tool = step.strip()
                if not tool:
                    continue
                path.append({"tool": tool, "args": {}})
                continue
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool", "")).strip()
            if not tool:
                continue
            args = step.get("args", {})
            if not isinstance(args, dict):
                args = {}
            path.append({"tool": tool, "args": args})
        if path:
            out.append(path)
    return out


def _ordered_tool_matches(actual_tools: list[str], expected_tools: list[str]) -> int:
    if not actual_tools or not expected_tools:
        return 0
    cursor = 0
    for tool in actual_tools:
        if cursor < len(expected_tools) and tool == expected_tools[cursor]:
            cursor += 1
            if cursor == len(expected_tools):
                return cursor
    return cursor


def _select_best_acceptable_path(
    raw_paths: list[list[dict[str, Any]]],
    acceptable_paths: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not acceptable_paths:
        return []
    if not raw_paths:
        return acceptable_paths[0]

    raw_tools = [str(step.get("tool", "")).strip() for step in raw_paths[0] if isinstance(step, dict)]
    best_idx = 0
    best_score: tuple[int, int] | None = None
    for idx, path in enumerate(acceptable_paths):
        exp_tools = [str(step.get("tool", "")).strip() for step in path if isinstance(step, dict)]
        score = (_ordered_tool_matches(raw_tools, exp_tools), -len(exp_tools))
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
    return acceptable_paths[best_idx]


def _coerce_expected_paths_to_acceptable(
    raw_paths: Any,
    acceptable_paths: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    acceptable = _coerce_path_list(acceptable_paths)
    if not acceptable:
        return _coerce_path_list(raw_paths)

    raw = _coerce_path_list(raw_paths)
    acceptable_set = {_canonical_path(path) for path in acceptable}
    exact = [path for path in raw if _canonical_path(path) in acceptable_set]
    if exact:
        return exact
    return [_select_best_acceptable_path(raw, acceptable)]


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def _question_grounded(question: str, pool_docs: list[dict[str, Any]]) -> bool:
    q_tokens = [token for token in _tokenize(question) if token not in QUESTION_STOPWORDS]
    if not q_tokens:
        return False

    pool_text = " ".join(
        f"{doc.get('title', '')} {doc.get('summary', '')}"
        for doc in pool_docs
        if isinstance(doc, dict)
    )
    pool_tokens = set(_tokenize(pool_text))
    if not pool_tokens:
        return False

    overlap = [token for token in q_tokens if token in pool_tokens]
    if len(overlap) >= 2:
        return True
    return any(len(token) >= 4 for token in overlap)


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        text = str(token or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _anchor_tokens(text: Any) -> list[str]:
    tokens: list[str] = []
    for token in _tokenize(str(text or "")):
        cleaned = token.strip("_").lower()
        if len(cleaned) < 2:
            continue
        if cleaned.isdigit():
            continue
        if cleaned in ANCHOR_STOPWORDS:
            continue
        if re.fullmatch(r"20\d{2}|\d+d?", cleaned):
            continue
        tokens.append(cleaned)
    return _dedupe_tokens(tokens)


def _path_anchor_groups(expected_paths: list[list[dict[str, Any]]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in expected_paths:
        if not isinstance(path, list):
            continue
        for step in path:
            if not isinstance(step, dict):
                continue
            args = step.get("args", {})
            if not isinstance(args, dict):
                continue
            for key in ("query", "topic", "topic_a", "topic_b", "url", "urls"):
                value = args.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                tokens = _anchor_tokens(text)
                if tokens:
                    groups[key] = _dedupe_tokens(groups.get(key, []) + tokens)
    return groups


def _doc_text(doc: dict[str, Any]) -> str:
    return " ".join(
        str(doc.get(key, "") or "")
        for key in ("title", "title_cn", "summary", "evidence_text", "url", "source")
    )


def _docs_by_id(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(doc.get("doc_id", "")).strip(): doc
        for doc in docs
        if isinstance(doc, dict) and str(doc.get("doc_id", "")).strip()
    }


def _required_anchor_hits(tokens: list[str]) -> int:
    if len(tokens) >= 3:
        return 2
    if len(tokens) >= 1:
        return 1
    return 0


def _anchor_overlap_count(tokens: list[str], text: str) -> int:
    text_tokens = set(_anchor_tokens(text))
    return sum(1 for token in _dedupe_tokens(tokens) if token in text_tokens)


def _source_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if text in {"hackernews", "hn"}:
        return "hackernews"
    if text == "techcrunch":
        return "techcrunch"
    if text in {"wsj", "wallstreetjournal"}:
        return "wsj"
    if text in {"arstechnica", "ars"}:
        return "arstechnica"
    return text


def _mentioned_sources(text: str) -> set[str]:
    lowered = str(text or "").lower()
    found: set[str] = set()
    for alias, canonical in SOURCE_ALIASES.items():
        if alias in lowered:
            found.add(canonical)
    return found


def _validate_generated_case_alignment(
    case: dict[str, Any],
    task: dict[str, Any],
) -> None:
    if not bool(case.get("retrieval_evaluable")):
        return

    case_id = str(case.get("case_id", "")).strip() or str(task.get("task_id", "")).strip()
    expected_paths = _coerce_path_list(case.get("expected_tool_paths", []))
    anchor_groups = _path_anchor_groups(expected_paths)
    topic_tokens = _dedupe_tokens(
        anchor_groups.get("query", [])
        + anchor_groups.get("topic", [])
        + anchor_groups.get("url", [])
        + anchor_groups.get("urls", [])
    )
    side_topic_tokens = _dedupe_tokens(anchor_groups.get("topic_a", []) + anchor_groups.get("topic_b", []))
    if not topic_tokens and side_topic_tokens:
        topic_tokens = side_topic_tokens
    if not topic_tokens:
        topic_tokens = _anchor_tokens(json.dumps(task.get("parameter_template", {}), ensure_ascii=False))

    question = str(case.get("expected_question", "") or "")
    if topic_tokens and _anchor_overlap_count(topic_tokens, question) < _required_anchor_hits(topic_tokens):
        raise ValueError(
            f"{case_id}: expected_question is not aligned with tool parameter anchors={topic_tokens}."
        )

    docs = case.get("input_news_pool", [])
    if not isinstance(docs, list):
        docs = []
    by_id = _docs_by_id(docs)
    gold_doc_ids = [
        str(doc_id).strip()
        for doc_id in case.get("retrieval_gold_doc_ids", [])
        if str(doc_id).strip()
    ]
    gold_docs = [by_id[doc_id] for doc_id in gold_doc_ids if doc_id in by_id]
    gold_text = "\n".join(_doc_text(doc) for doc in gold_docs)
    if topic_tokens and _anchor_overlap_count(topic_tokens, gold_text) < _required_anchor_hits(topic_tokens):
        raise ValueError(
            f"{case_id}: retrieval_gold_doc_ids do not match tool parameter anchors={topic_tokens}."
        )

    if str(task.get("tool", "")).strip() == "compare_topics":
        for side_key in ("topic_a", "topic_b"):
            side_tokens = anchor_groups.get(side_key, [])
            if side_tokens and _anchor_overlap_count(side_tokens, question) < _required_anchor_hits(side_tokens):
                raise ValueError(f"{case_id}: compare_topics question missing {side_key}.")
            if side_tokens and _anchor_overlap_count(side_tokens, gold_text) < _required_anchor_hits(side_tokens):
                raise ValueError(f"{case_id}: compare_topics missing gold evidence for {side_key}.")

    if str(task.get("tool", "")).strip() == "compare_sources":
        expected_sources = {
            _source_key(source)
            for source in task.get("sampling", {}).get("sources", [])
            if _source_key(source)
        }
        if expected_sources:
            gold_sources = {_source_key(doc.get("source")) for doc in gold_docs if _source_key(doc.get("source"))}
            missing_sources = expected_sources.difference(gold_sources)
            if missing_sources:
                raise ValueError(f"{case_id}: compare_sources missing gold evidence sources={sorted(missing_sources)}.")

    answer_sources = _mentioned_sources(str(case.get("expected_answer", "") or ""))
    if answer_sources:
        gold_sources = {_source_key(doc.get("source")) for doc in gold_docs if _source_key(doc.get("source"))}
        if gold_sources and not answer_sources.issubset(gold_sources):
            raise ValueError(
                f"{case_id}: expected_answer mentions sources={sorted(answer_sources)} "
                f"not covered by gold evidence sources={sorted(gold_sources)}."
            )


def _title_topic(pool_docs: list[dict[str, Any]]) -> str:
    for doc in pool_docs:
        if not isinstance(doc, dict):
            continue
        title = str(doc.get("title", "")).strip()
        if not title:
            continue
        title = re.sub(r"^\[[^\]]+\]\s*", "", title).strip()
        if title:
            return title[:28]
    return "相关新闻"


def _first_arg(expected_paths: list[list[dict[str, Any]]], key: str) -> str:
    for path in expected_paths:
        if not isinstance(path, list):
            continue
        for step in path:
            if not isinstance(step, dict):
                continue
            args = step.get("args", {})
            if not isinstance(args, dict):
                continue
            value = args.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return ""


def _build_grounded_question(
    task: dict[str, Any],
    pool_docs: list[dict[str, Any]],
    expected_paths: list[list[dict[str, Any]]],
) -> str:
    tool = str(task.get("tool", "")).strip()
    days = _safe_int(
        _first_arg(expected_paths, "days") or task.get("sampling", {}).get("days", 30),
        30,
    )
    limit = _safe_int(_first_arg(expected_paths, "limit") or 10, 10)

    topic = (
        _first_arg(expected_paths, "query")
        or _first_arg(expected_paths, "topic")
        or _title_topic(pool_docs)
    )
    topic_a = _first_arg(expected_paths, "topic_a")
    topic_b = _first_arg(expected_paths, "topic_b")

    if tool == "compare_sources":
        return f"请比较过去{days}天「{topic}」在 HackerNews 与 TechCrunch 的覆盖和情绪差异。"
    if tool == "compare_topics":
        a = topic_a or topic or "主题A"
        b = topic_b or "主题B"
        return f"请比较过去{days}天「{a}」与「{b}」的热度、情绪与来源结构。"
    if tool == "build_timeline":
        return f"请构建过去{days}天「{topic}」的关键事件时间线，最多{limit}条。"
    if tool == "trend_analysis":
        return f"请分析「{topic}」最近{days}天相对前{days}天的趋势变化。"
    if tool == "analyze_landscape":
        return f"请分析过去{days}天「{topic}」相关赛道的竞争格局。"
    if tool == "fulltext_batch":
        return f"请围绕「{topic}」筛选最近{days}天相关新闻并批量读取全文，提炼关键结论。"
    if tool == "search_news":
        return f"请检索最近{days}天与「{topic}」相关的新闻，并返回最相关结果。"
    if tool == "query_news":
        return f"请查询最近{days}天与「{topic}」相关的新闻，并按相关性返回结果。"
    if tool == "read_news_content":
        url = _first_arg(expected_paths, "url")
        if url:
            return f"请读取该新闻链接全文并提炼要点：{url}"
        return "请读取指定新闻全文并提炼关键要点。"
    if tool == "list_topics":
        return "请给出最近一段时间的主题分布与数量统计。"
    if tool == "get_db_stats":
        return "请返回当前新闻库总量与最新数据时间。"
    return f"请基于最近{days}天新闻，围绕「{topic}」完成分析。"


def _repair_generated_case(
    raw_case: dict[str, Any],
    task: dict[str, Any],
    pool_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(raw_case)
    task_retrieval_evaluable = str(task.get("retrieval_mode", "")).strip().lower() == "evaluable"
    if bool(task.get("should_clarify", False)):
        out["expected_tool_paths"] = []
        out["required_tools"] = []
        out["retrieval_gold_doc_ids"] = []
        out["retrieval_gold_urls"] = []
        out["verifiable_claims"] = []
        out["retrieval_evaluable"] = False
        out["should_clarify"] = True
        return out

    acceptable_paths = _coerce_path_list(task.get("acceptable_tool_paths", []))
    out["should_clarify"] = False
    out["retrieval_evaluable"] = task_retrieval_evaluable
    out["required_tools"] = list(task.get("required_tools", []))
    out["forbidden_tools"] = list(task.get("forbidden_tools", []))
    out["expected_tool_paths"] = _coerce_expected_paths_to_acceptable(
        out.get("expected_tool_paths"),
        acceptable_paths,
    )
    if not task_retrieval_evaluable:
        out["retrieval_gold_doc_ids"] = []
        out["retrieval_gold_urls"] = []
        out["verifiable_claims"] = []

    scenario = str(task.get("scenario", "")).strip().lower()
    question = str(out.get("expected_question", "")).strip()
    requires_grounding = scenario != "empty"
    if (not _contains_zh(question)) or (requires_grounding and not _question_grounded(question, pool_docs)):
        out["expected_question"] = _build_grounded_question(task, pool_docs, out["expected_tool_paths"])

    answer = str(out.get("expected_answer", "")).strip()
    if not _contains_zh(answer):
        out["expected_answer"] = f"参考答案（中文）：{answer}"
    return out


def _doc_language(title: str, title_cn: str) -> str:
    if title_cn.strip():
        return "zh"
    if any("\u4e00" <= ch <= "\u9fff" for ch in title):
        return "zh"
    return "en"


def _doc_id_from_url(url: str) -> str:
    digest = hashlib.sha1(str(url).strip().encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _sample_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = task.get("sampling", {})
    days = int(sampling.get("days", 30))
    candidate_limit = int(sampling.get("candidate_limit", 300))
    keywords = [str(item).strip() for item in sampling.get("keywords", []) if str(item).strip()]
    sources = [str(item).strip() for item in sampling.get("sources", []) if str(item).strip()]
    languages = {str(item).strip().lower() for item in sampling.get("languages", []) if str(item).strip()}

    where_parts = ["created_at >= NOW() - %s::interval"]
    params: list[Any] = [f"{days} days"]

    if keywords:
        keyword_clauses: list[str] = []
        for token in keywords:
            keyword_clauses.append(
                "(title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)"
            )
            token_like = f"%{token}%"
            params.extend([token_like, token_like, token_like])
        where_parts.append("(" + " OR ".join(keyword_clauses) + ")")

    if sources:
        where_parts.append("COALESCE(source_type,'') = ANY(%s)")
        params.append(sources)

    sql = f"""
        SELECT
            COALESCE(title_cn, title) AS title_norm,
            title,
            COALESCE(summary, '') AS summary,
            url,
            source_type,
            created_at,
            COALESCE(sentiment, '') AS sentiment,
            COALESCE(points, 0) AS points,
            COALESCE(title_cn, '') AS title_cn
        FROM view_dashboard_news
        WHERE {' AND '.join(where_parts)}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(candidate_limit)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()
    finally:
        put_conn(conn)

    docs: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for row in rows:
        title_norm, title, summary, url, source_type, created_at, sentiment, points, title_cn = row
        url_text = str(url or "").strip()
        if not url_text or url_text in seen_urls:
            continue
        seen_urls.add(url_text)
        language = _doc_language(str(title or ""), str(title_cn or ""))
        if languages and language not in languages:
            continue
        docs.append(
            {
                "doc_id": _doc_id_from_url(url_text),
                "url": url_text,
                "title": str(title_norm or "").strip() or str(title or "").strip() or "(untitled)",
                "summary": str(summary or "").strip(),
                "published_at": created_at.isoformat() if created_at else "",
                "source": str(source_type or "").strip() or "unknown",
                "sentiment": str(sentiment or "").strip(),
                "points": int(points or 0),
                "language": language,
            }
        )
    return docs


def _round_robin_by_source(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        source = str(item.get("source", "unknown")).strip() or "unknown"
        groups.setdefault(source, []).append(item)
    ordered_sources = sorted(groups.keys())
    cursors = {source: 0 for source in ordered_sources}
    output: list[dict[str, Any]] = []
    while True:
        progressed = False
        for source in ordered_sources:
            idx = cursors[source]
            bucket = groups[source]
            if idx >= len(bucket):
                continue
            output.append(bucket[idx])
            cursors[source] = idx + 1
            progressed = True
        if not progressed:
            break
    return output


def _build_pools(task: dict[str, Any], candidates: list[dict[str, Any]], pools_per_task: int, rng: random.Random) -> list[Pool]:
    pool_size = int(task.get("sampling", {}).get("pool_size", 12))
    ordered = _round_robin_by_source(candidates)
    if ordered:
        # Keep deterministic but avoid repeated same-front slices.
        shift = rng.randint(0, max(0, len(ordered) - 1))
        ordered = ordered[shift:] + ordered[:shift]

    pools: list[Pool] = []
    for idx in range(1, pools_per_task + 1):
        pool_id = f"{task['task_id']}.pool.{idx:03d}"
        if not ordered:
            pools.append(Pool(pool_id=pool_id, docs=[]))
            continue
        start = ((idx - 1) * pool_size) % len(ordered)
        docs: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()
        for offset in range(len(ordered)):
            item = ordered[(start + offset) % len(ordered)]
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            docs.append(item)
            if len(docs) >= pool_size:
                break
        pools.append(Pool(pool_id=pool_id, docs=docs))
    return pools


def _chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    size = int(chunk_size)
    if size <= 0 or len(items) <= size:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _pools_for_prompt(pools: list[Pool], *, summary_chars: int = 700) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pool in pools:
        docs = []
        for item in pool.docs:
            docs.append(
                {
                    "doc_id": item["doc_id"],
                    "url": item["url"],
                    "title": item["title"],
                    "summary": str(item.get("summary", ""))[:summary_chars],
                    "evidence_text": str(item.get("evidence_text", item.get("summary", "")))[:summary_chars],
                    "published_at": item["published_at"],
                    "source": item["source"],
                    "sentiment": item.get("sentiment", ""),
                    "language": item.get("language", ""),
                    "channels": item.get("channels", []),
                    "seed_similarity": item.get("seed_similarity", 0.0),
                    "topic_match_score": item.get("topic_match_score", 0.0),
                    "matched_anchor_terms": item.get("matched_anchor_terms", []),
                    "anchor_hit_fields": item.get("anchor_hit_fields", {}),
                }
            )
        out.append(
            {
                "pool_id": pool.pool_id,
                "input_news_pool_hash": build_news_pool_hash(pool.docs),
                "pool_quality": pool.meta.get("pool_quality", {}) if isinstance(pool.meta, dict) else {},
                "docs": docs,
            }
        )
    return out


def _generator_prompts(
    task: dict[str, Any],
    pools: list[Pool],
    *,
    rejection_reasons: dict[str, str] | None = None,
    attempt_feedback: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    clarify_task = bool(task["should_clarify"])
    schema_hint = {
        "task_id": task["task_id"],
        "cases": [
            {
                "pool_id": "string",
                "expected_question": "string",
                "expected_answer": "string",
                "expected_tool_paths": []
                if clarify_task
                else [[{"tool": task["tool"], "args": task["parameter_template"]}]],
                "required_tools": [] if clarify_task else task["required_tools"],
                "forbidden_tools": task["forbidden_tools"],
                "retrieval_gold_doc_ids": [] if clarify_task else ["doc_id from same pool"],
                "retrieval_gold_urls": [] if clarify_task else ["url from same pool"],
                "verifiable_claims": []
                if clarify_task
                else [
                    {
                        "claim": "verifiable claim from expected_answer",
                        "evidence_doc_ids": ["doc_id from same pool"],
                        "evidence_quotes": [
                            {
                                "doc_id": "doc_id from same pool",
                                "quote": "single continuous exact excerpt from title/summary/evidence_text",
                            }
                        ],
                        "claim_type": "fact|number|comparison",
                    }
                ],
                "should_clarify": task["should_clarify"],
                "retrieval_evaluable": task["retrieval_mode"] == "evaluable",
                "difficulty": task["difficulty"],
                "tags": task["tags"],
            }
        ],
    }

    system_prompt = (
        "You generate task-driven evaluation cases.\n"
        "Think step by step:\n"
        "1) Read the task definition and understand what kind of test case is needed.\n"
        "2) Study the news pool to identify key entities, events, and facts.\n"
        "3) Craft a realistic Chinese question that a user would ask.\n"
        "4) Write a grounded expected_answer using only evidence from the pool.\n"
        "5) Specify tool paths and verifiable claims only for cases that should be answered directly.\n\n"
        "Rules:\n"
        "1) Return strict JSON object only.\n"
        "2) One case per pool_id, no missing and no extra pools.\n"
        "3) Use only tools from acceptable_tool_paths.\n"
        "4) retrieval_gold_doc_ids/retrieval_gold_urls must come from the SAME input_news_pool.\n"
        "5) If retrieval_evaluable=true, retrieval_gold_doc_ids must be non-empty.\n"
        "6) expected_answer must be grounded in pool docs.\n"
        "7) verifiable_claims must be checkable and linked to evidence_doc_ids in pool.\n"
        "8) Each non-empty retrieval-evaluable claim must include evidence_quotes. A quote must be a single, "
        "continuous, exact excerpt from the same doc's title/summary/evidence_text. Do not paraphrase. "
        "Do not combine non-adjacent sentences. Do not use ellipses (... or \\u2026).\n"
        "9) expected_question and expected_answer must be written in Chinese (entity/product names may stay in English).\n"
        "10) If should_clarify=false, expected_tool_paths must be an exact subset of acceptable_tool_paths; do not alter tool args.\n"
        "11) If should_clarify=true, the case must ask for clarification instead of tool execution: "
        "expected_tool_paths=[], required_tools=[], retrieval gold empty, verifiable_claims=[], and expected_answer "
        "should be the clarification question the agent should ask.\n"
        "12) For non-empty non-clarification scenarios, expected_question must be answerable by the pool and mention pool entities/topics.\n"
        "13) For empty/non_retrieval scenarios, do not invent facts, leave retrieval gold empty, and verifiable_claims may be empty.\n"
        "14) Do not change fixed evaluation attributes from the task definition: should_clarify, retrieval_evaluable, "
        "required_tools, forbidden_tools, and acceptable tool args are configuration-owned.\n"
        "15) The expected_question, tool args, retrieval gold docs, evidence quotes, and expected_answer must describe "
        "the same topic. Do not ask about one entity/product and answer with another.\n"
        "16) Each pool contains topic_anchor and pool_quality metadata. Generate the case only around that anchor; "
        "do not switch to adjacent news just because it appears in the pool.\n"
        "17) Do not output markdown fences."
    )

    # Feedback loop: inject previous rejection reasons
    if rejection_reasons:
        feedback_lines = ["\n\nPrevious attempt was rejected. Fix these specific issues:"]
        for case_id, reason in rejection_reasons.items():
            feedback_lines.append(f"- {case_id}: {reason}")
        system_prompt += "\n".join(feedback_lines)
    if attempt_feedback:
        system_prompt += (
            "\n\nPrevious seed attempt failed validation. Keep already valid fields when possible, "
            "and specifically repair the failed fields described below."
        )

    payload = {
        "task_definition": {
            "task_id": task["task_id"],
            "tool": task["tool"],
            "capability": task.get("capability", ""),
            "intent_label": task["intent_label"],
            "retrieval_mode": task["retrieval_mode"],
            "scenario": task["scenario"],
            "example_question": task["example_question"],
            "parameter_template": task["parameter_template"],
            "topic_anchor": topic_anchor_terms(task),
            "acceptable_tool_paths": task["acceptable_tool_paths"],
            "required_tools": task["required_tools"],
            "forbidden_tools": task["forbidden_tools"],
            "should_clarify": task["should_clarify"],
            "difficulty": task["difficulty"],
            "tags": task["tags"],
        },
        "news_pools": _pools_for_prompt(pools),
        "output_schema_example": schema_hint,
        "attempt_feedback": attempt_feedback or [],
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def _audit_prompts(task: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[str, str]:
    system_prompt = (
        "You are an evaluation-case auditor.\n"
        "Think step by step:\n"
        "1) Read the task definition and understand the expected contract.\n"
        "2) For each case, verify language (Chinese), tool path validity, and evidence grounding.\n"
        "3) Provide a clear accept/reject verdict with specific reasoning.\n\n"
        "Check each case for contract consistency and evidence grounding.\n"
        "Reject any case whose expected_question or expected_answer is not Chinese.\n"
        "For should_clarify=false, reject any case whose expected_tool_paths is not an exact subset of task.acceptable_tool_paths.\n"
        "For should_clarify=true, reject any case that defines expected_tool_paths, required_tools, retrieval gold labels, or verifiable claims.\n"
        "For non-empty scenarios, reject if expected_question is not grounded in input_news_pool topics/entities.\n"
        "If rejected for language, set reason to non_chinese_expected_text.\n"
        "If rejected for path drift, set reason to path_not_in_acceptable.\n"
        "If rejected for question grounding, set reason to question_not_grounded.\n"
        "Return strict JSON object only."
    )
    payload = {
        "task": {
            "task_id": task["task_id"],
            "tool": task["tool"],
            "capability": task.get("capability", ""),
            "retrieval_mode": task["retrieval_mode"],
            "scenario": task["scenario"],
            "acceptable_tool_paths": task["acceptable_tool_paths"],
            "required_tools": task["required_tools"],
            "forbidden_tools": task["forbidden_tools"],
        },
        "cases": cases,
        "output_schema": {
            "task_id": task["task_id"],
            "verdicts": [
                {
                    "case_id": "string",
                    "accepted": True,
                    "reason": "string",
                }
            ],
        },
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def _invoke_json(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    max_retries: int,
    backoff_sec: float,
) -> dict[str, Any]:
    attempts = max(1, int(max_retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            text = _coerce_text_content(getattr(result, "content", result))
            payload = json.loads(_extract_first_json_object(text))
            if not isinstance(payload, dict):
                raise ValueError("LLM output JSON root must be object.")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts or not _is_retryable_llm_error(exc):
                raise
            sleep_sec = max(0.0, float(backoff_sec)) * (2 ** (attempt - 1))
            print(
                "[TaskDataset][Retry] attempt=%s/%s backoff=%.2fs error=%s"
                % (attempt, attempts, sleep_sec, exc)
            )
            time.sleep(sleep_sec)

    raise RuntimeError(f"LLM invocation failed after retries: {last_exc}")


def _generate_for_task(
    llm: Any,
    task: dict[str, Any],
    pools: list[Pool],
    *,
    llm_max_retries: int,
    llm_backoff_sec: float,
    pools_per_generation_call: int,
    inter_llm_call_sleep_sec: float,
    rejection_reasons: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    pool_map = {pool.pool_id: pool.docs for pool in pools}
    out_cases: list[dict[str, Any]] = []
    pool_chunks = _chunk_list(pools, int(pools_per_generation_call))

    for chunk_idx, pools_chunk in enumerate(pool_chunks, 1):
        try:
            system_prompt, user_prompt = _generator_prompts(
                task, pools_chunk, rejection_reasons=rejection_reasons,
            )
            payload = _invoke_json(
                llm,
                system_prompt,
                user_prompt,
                max_retries=llm_max_retries,
                backoff_sec=llm_backoff_sec,
            )

            rows = payload.get("cases", [])
            if not isinstance(rows, list):
                raise ValueError(f"{task['task_id']}: generator output missing cases list.")

            by_pool: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pool_id = str(row.get("pool_id", "")).strip()
                if not pool_id:
                    continue
                by_pool[pool_id] = row

            missing = [pool.pool_id for pool in pools_chunk if pool.pool_id not in by_pool]
            if missing:
                raise ValueError(f"{task['task_id']}: missing generated cases for pools={missing}")

            for pool in pools_chunk:
                suffix_match = re.search(r"\.pool\.(\d+)$", pool.pool_id)
                if suffix_match:
                    suffix = f"{int(suffix_match.group(1)):03d}"
                else:
                    suffix = hashlib.sha1(pool.pool_id.encode("utf-8")).hexdigest()[:8]
                case_id = f"{task['task_id']}.{suffix}"
                raw_case = _repair_generated_case(
                    by_pool[pool.pool_id],
                    task,
                    pool_map[pool.pool_id],
                )
                normalized = normalize_case(
                    raw_case,
                    task_type=task,
                    case_id=case_id,
                    pool_id=pool.pool_id,
                    input_news_pool=pool_map[pool.pool_id],
                )
                _validate_generated_case_alignment(normalized, task)
                out_cases.append(normalized)
        except Exception as exc:
            # Cost/stability trade-off: try larger chunk first, fallback to single-pool calls on failure.
            if int(pools_per_generation_call) > 1 and len(pools_chunk) > 1:
                print(
                    "[TaskDataset][Fallback] task=%s generation chunk=%s/%s size=%s failed=%s -> retry single-pool mode"
                    % (task["task_id"], chunk_idx, len(pool_chunks), len(pools_chunk), exc)
                )
                fallback_cases = _generate_for_task(
                    llm,
                    task,
                    pools_chunk,
                    llm_max_retries=llm_max_retries,
                    llm_backoff_sec=llm_backoff_sec,
                    pools_per_generation_call=1,
                    inter_llm_call_sleep_sec=inter_llm_call_sleep_sec,
                    rejection_reasons=rejection_reasons,
                )
                out_cases.extend(fallback_cases)
            else:
                raise

        if chunk_idx < len(pool_chunks) and float(inter_llm_call_sleep_sec) > 0:
            sleep_sec = float(inter_llm_call_sleep_sec)
            print(
                "[TaskDataset][Throttle] task=%s phase=generation chunk=%s/%s sleep=%.2fs"
                % (task["task_id"], chunk_idx, len(pool_chunks), sleep_sec)
            )
            time.sleep(sleep_sec)
    return out_cases


def _audit_cases(
    llm: Any,
    task: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    llm_max_retries: int,
    llm_backoff_sec: float,
    cases_per_audit_call: int,
    inter_llm_call_sleep_sec: float,
) -> dict[str, str]:
    if not cases:
        return {}

    rejected: dict[str, str] = {}
    case_chunks = _chunk_list(cases, int(cases_per_audit_call))
    for chunk_idx, cases_chunk in enumerate(case_chunks, 1):
        system_prompt, user_prompt = _audit_prompts(task, cases_chunk)
        payload = _invoke_json(
            llm,
            system_prompt,
            user_prompt,
            max_retries=llm_max_retries,
            backoff_sec=llm_backoff_sec,
        )
        verdicts = payload.get("verdicts", [])
        if not isinstance(verdicts, list):
            continue
        for row in verdicts:
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case_id", "")).strip()
            accepted = bool(row.get("accepted", False))
            reason = str(row.get("reason", "")).strip() or "audit_rejected"
            if case_id and not accepted:
                rejected[case_id] = reason

        if chunk_idx < len(case_chunks) and float(inter_llm_call_sleep_sec) > 0:
            sleep_sec = float(inter_llm_call_sleep_sec)
            print(
                "[TaskDataset][Throttle] task=%s phase=audit chunk=%s/%s sleep=%.2fs"
                % (task["task_id"], chunk_idx, len(case_chunks), sleep_sec)
            )
            time.sleep(sleep_sec)
    return rejected


def _case_id_for_pool(task: dict[str, Any], pool: Pool) -> str:
    suffix_match = re.search(r"\.pool\.(\d+)$", pool.pool_id)
    if suffix_match:
        suffix = f"{int(suffix_match.group(1)):03d}"
    else:
        suffix = hashlib.sha1(pool.pool_id.encode("utf-8")).hexdigest()[:8]
    return f"{task['task_id']}.{suffix}"


def _seed_attempt_temperatures(base: float, max_attempts: int) -> list[float]:
    attempts = max(1, int(max_attempts))
    schedule = [float(base), min(float(base), 0.4), 0.1]
    if attempts <= len(schedule):
        return schedule[:attempts]
    return schedule + [0.1 for _ in range(attempts - len(schedule))]


def _audit_single_case_with_evidence(
    llm: Any,
    task: dict[str, Any],
    case: dict[str, Any],
    evidence_result: EvidenceValidationResult,
    *,
    llm_max_retries: int,
    llm_backoff_sec: float,
) -> tuple[bool, str]:
    system_prompt = (
        "You are a strict evaluation-case auditor.\n"
        "Return strict JSON only.\n"
        "Decide whether the generated case is usable for an automated retrieval/generation evaluation.\n"
        "For non-empty cases, verify that each claim is supported by the quoted packed context and that "
        "the answer does not exceed the evidence. For borderline fuzzy matches, decide whether this is "
        "only a formatting difference or a semantic drift. For numeric uncertain matches, decide whether "
        "the quantities are equivalent. For empty cases, reject invented concrete facts.\n"
    )
    docs = []
    for doc in case.get("input_news_pool", []):
        if not isinstance(doc, dict):
            continue
        docs.append(
            {
                "doc_id": doc.get("doc_id"),
                "url": doc.get("url"),
                "title": doc.get("title"),
                "summary": doc.get("summary"),
                "evidence_text": doc.get("evidence_text", doc.get("summary", "")),
                "source": doc.get("source"),
                "published_at": doc.get("published_at"),
            }
        )
    payload = {
        "task": {
            "task_id": task["task_id"],
            "tool": task["tool"],
            "capability": task.get("capability", ""),
            "retrieval_mode": task["retrieval_mode"],
            "scenario": task["scenario"],
            "parameter_template": task["parameter_template"],
            "acceptable_tool_paths": task["acceptable_tool_paths"],
        },
        "case": {
            "case_id": case.get("case_id"),
            "expected_question": case.get("expected_question"),
            "expected_answer": case.get("expected_answer"),
            "expected_tool_paths": case.get("expected_tool_paths"),
            "retrieval_gold_doc_ids": case.get("retrieval_gold_doc_ids"),
            "verifiable_claims": case.get("verifiable_claims"),
            "retrieval_evaluable": case.get("retrieval_evaluable"),
        },
        "evidence_validation": evidence_result.as_case_metadata(),
        "packed_context": docs,
        "output_schema": {
            "case_id": str(case.get("case_id", "")),
            "accepted": True,
            "reason": "short reason string",
        },
    }
    result = _invoke_json(
        llm,
        system_prompt,
        json.dumps(payload, ensure_ascii=False, indent=2),
        max_retries=llm_max_retries,
        backoff_sec=llm_backoff_sec,
    )
    accepted = bool(result.get("accepted", False))
    reason = str(result.get("reason", "")).strip() or ("accepted" if accepted else "audit_rejected")
    print(
        "[Audit] case=%s status=%s score=%.1f verdict=%s"
        % (
            case.get("case_id"),
            evidence_result.status,
            float(evidence_result.score),
            "accepted" if accepted else "rejected",
        )
    )
    return accepted, reason


def _generate_single_case_attempt(
    llm: Any,
    task: dict[str, Any],
    pool: Pool,
    *,
    llm_max_retries: int,
    llm_backoff_sec: float,
    attempt_feedback: list[dict[str, Any]],
) -> dict[str, Any]:
    system_prompt, user_prompt = _generator_prompts(task, [pool], attempt_feedback=attempt_feedback)
    payload = _invoke_json(
        llm,
        system_prompt,
        user_prompt,
        max_retries=llm_max_retries,
        backoff_sec=llm_backoff_sec,
    )
    rows = payload.get("cases", [])
    if not isinstance(rows, list):
        raise ValueError(f"{task['task_id']}: generator output missing cases list.")
    by_pool = {
        str(row.get("pool_id", "")).strip(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("pool_id", "")).strip()
    }
    if pool.pool_id not in by_pool:
        raise ValueError(f"{task['task_id']}: missing generated case for pool={pool.pool_id}")
    case_id = _case_id_for_pool(task, pool)
    raw_case = _repair_generated_case(by_pool[pool.pool_id], task, pool.docs)
    normalized = normalize_case(
        raw_case,
        task_type=task,
        case_id=case_id,
        pool_id=pool.pool_id,
        input_news_pool=pool.docs,
    )
    _validate_generated_case_alignment(normalized, task)
    return normalized


def _generate_case_for_seed(
    *,
    generation_llms: dict[float, Any],
    audit_llm: Any,
    task: dict[str, Any],
    pool: Pool,
    temperatures: list[float],
    audit_enabled: bool,
    llm_max_retries: int,
    llm_backoff_sec: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    feedback: list[dict[str, Any]] = []
    attempts_meta: list[dict[str, Any]] = []
    last_stage = "unknown"
    last_reason = "unknown"
    last_score = 0.0
    api_error = False

    for attempt_idx, temp in enumerate(temperatures, 1):
        feedback_label = feedback[-1]["reason"] if feedback else "none"
        print(
            "[SeedAttempt] task=%s pool=%s attempt=%s/%s temp=%.2f feedback=%s"
            % (task["task_id"], pool.pool_id, attempt_idx, len(temperatures), float(temp), feedback_label)
        )
        try:
            case = _generate_single_case_attempt(
                generation_llms[temp],
                task,
                pool,
                llm_max_retries=llm_max_retries,
                llm_backoff_sec=llm_backoff_sec,
                attempt_feedback=feedback,
            )
        except Exception as exc:  # noqa: BLE001
            last_stage = "llm_generation_error" if _is_retryable_llm_error(exc) else "schema"
            last_reason = str(exc)
            api_error = api_error or _is_retryable_llm_error(exc)
            next_feedback = {
                "failed_claim_index": 0,
                "stage": last_stage,
                "reason": type(exc).__name__,
                "instruction": "Regenerate a valid strict JSON case for this exact pool.",
                "detail": str(exc),
            }
            feedback.append(next_feedback)
            attempts_meta.append({"attempt": attempt_idx, "temperature": temp, "stage": last_stage, "reason": last_reason})
            continue

        evidence_result = validate_case_evidence(case)
        last_score = float(evidence_result.score)
        if not evidence_result.accepted:
            last_stage = "evidence_match"
            last_reason = "; ".join(item.reason for item in evidence_result.feedback) or "evidence_validation_failed"
            feedback.extend(item.as_dict() for item in evidence_result.feedback)
            attempts_meta.append(
                {
                    "attempt": attempt_idx,
                    "temperature": temp,
                    "stage": last_stage,
                    "reason": last_reason,
                    "score": last_score,
                }
            )
            continue

        audit_required = bool(audit_enabled)
        if evidence_result.audit_required and not audit_enabled:
            last_stage = "audit"
            last_reason = "audit_required_but_disabled"
            feedback.append(
                {
                    "failed_claim_index": 0,
                    "stage": "audit",
                    "reason": last_reason,
                    "instruction": "Replace borderline evidence with exact continuous quotes.",
                }
            )
            attempts_meta.append(
                {"attempt": attempt_idx, "temperature": temp, "stage": last_stage, "reason": last_reason}
            )
            continue

        if audit_required:
            try:
                audit_ok, audit_reason = _audit_single_case_with_evidence(
                    audit_llm,
                    task,
                    case,
                    evidence_result,
                    llm_max_retries=llm_max_retries,
                    llm_backoff_sec=llm_backoff_sec,
                )
            except Exception as exc:  # noqa: BLE001
                last_stage = "llm_audit_error" if _is_retryable_llm_error(exc) else "audit"
                last_reason = str(exc)
                api_error = api_error or _is_retryable_llm_error(exc)
                feedback.append(
                    {
                        "failed_claim_index": 0,
                        "stage": last_stage,
                        "reason": type(exc).__name__,
                        "instruction": "Regenerate a case with exact non-borderline evidence quotes.",
                        "detail": str(exc),
                    }
                )
                attempts_meta.append(
                    {"attempt": attempt_idx, "temperature": temp, "stage": last_stage, "reason": last_reason}
                )
                continue
            if not audit_ok:
                last_stage = "audit"
                last_reason = audit_reason
                feedback.append(
                    {
                        "failed_claim_index": 0,
                        "stage": "audit",
                        "reason": audit_reason,
                        "instruction": "Revise only unsupported claims/quotes and keep valid fields stable.",
                    }
                )
                attempts_meta.append(
                    {"attempt": attempt_idx, "temperature": temp, "stage": last_stage, "reason": last_reason}
                )
                continue

        case.update(evidence_result.as_case_metadata())
        if evidence_result.audit_required:
            case["evidence_match_status"] = "borderline_audit_pass"
        case["seed_attempts"] = attempts_meta + [
            {
                "attempt": attempt_idx,
                "temperature": temp,
                "stage": "accepted",
                "reason": "accepted",
                "score": last_score,
            }
        ]
        return case, None, api_error

    drop = {
        "task_id": task["task_id"],
        "pool_id": pool.pool_id,
        "pool_hash": build_news_pool_hash(pool.docs),
        "attempts": len(temperatures),
        "temperatures": temperatures,
        "failure_stage": last_stage,
        "reason": last_reason,
        "last_score": round(last_score, 4),
        "candidate_doc_ids": [str(doc.get("doc_id", "")) for doc in pool.docs],
        "feedback_reasons": [str(item.get("reason", "")) for item in feedback],
        "attempt_log": attempts_meta,
    }
    return None, drop, api_error


def _generate_cases_seed_level(
    *,
    generation_llms: dict[float, Any],
    audit_llm: Any,
    task: dict[str, Any],
    pools: list[Pool],
    temperatures: list[float],
    audit_enabled: bool,
    llm_max_retries: int,
    llm_backoff_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    generated: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    consecutive_api_errors = 0
    for pool in pools:
        case, drop, api_error = _generate_case_for_seed(
            generation_llms=generation_llms,
            audit_llm=audit_llm,
            task=task,
            pool=pool,
            temperatures=temperatures,
            audit_enabled=audit_enabled,
            llm_max_retries=llm_max_retries,
            llm_backoff_sec=llm_backoff_sec,
        )
        if api_error:
            consecutive_api_errors += 1
        else:
            consecutive_api_errors = 0
        if case is not None:
            generated.append(case)
        if drop is not None:
            dropped.append(drop)
            print(
                "[TaskDataset][DropSeed] task=%s pool=%s stage=%s reason=%s"
                % (task["task_id"], drop.get("pool_id"), drop.get("failure_stage"), drop.get("reason"))
            )
        if consecutive_api_errors >= 3:
            print(
                "[TaskDataset][CircuitBreaker] task=%s aborting after %d consecutive API-error seeds."
                % (task["task_id"], consecutive_api_errors)
            )
            break
    return generated, dropped, consecutive_api_errors


def _audit_regen_failed_only(
    *,
    llm: Any,
    audit_llm: Any,
    task: dict[str, Any],
    pools: list[Pool],
    generated_cases: list[dict[str, Any]],
    llm_max_retries: int,
    llm_backoff_sec: float,
    max_regen_rounds: int,
    initial_cases_per_audit_call: int,
    regen_cases_per_audit_call: int,
    pools_per_generation_call: int,
    regen_pools_per_generation_call: int,
    inter_llm_call_sleep_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Audit + regeneration loop that only retries rejected cases."""
    if not generated_cases:
        return [], {}

    ordered_case_ids = [str(case.get("case_id", "")).strip() for case in generated_cases]
    case_by_id = {str(case.get("case_id", "")).strip(): case for case in generated_cases}
    pool_by_id = {pool.pool_id: pool for pool in pools}

    rejected = _audit_cases(
        audit_llm,
        task,
        generated_cases,
        llm_max_retries=llm_max_retries,
        llm_backoff_sec=llm_backoff_sec,
        cases_per_audit_call=int(initial_cases_per_audit_call),
        inter_llm_call_sleep_sec=inter_llm_call_sleep_sec,
    )
    if not rejected:
        kept = [case_by_id[cid] for cid in ordered_case_ids if cid in case_by_id]
        return kept, {}

    pending_rejected = dict(rejected)
    consecutive_llm_errors = 0
    regen_chunk_size = (
        int(regen_pools_per_generation_call)
        if int(regen_pools_per_generation_call) > 0
        else int(pools_per_generation_call)
    )
    if regen_chunk_size <= 0:
        regen_chunk_size = 1
    for regen_round in range(1, max(0, int(max_regen_rounds)) + 1):
        if not pending_rejected:
            break

        regen_pools: list[Pool] = []
        seen_pool_ids: set[str] = set()
        for case_id in pending_rejected.keys():
            case = case_by_id.get(str(case_id).strip())
            if not isinstance(case, dict):
                continue
            pool_id = str(case.get("pool_id", "")).strip()
            if not pool_id or pool_id in seen_pool_ids:
                continue
            pool = pool_by_id.get(pool_id)
            if pool is None:
                continue
            seen_pool_ids.add(pool_id)
            regen_pools.append(pool)

        if not regen_pools:
            break

        print(
            "[TaskDataset][AuditRegen][FailedOnly] task=%s round=%s/%s rejected=%s regen_pools=%s reasons=%s"
            % (
                task["task_id"],
                regen_round,
                max_regen_rounds,
                len(pending_rejected),
                len(regen_pools),
                pending_rejected,
            )
        )

        try:
            regenerated = _generate_for_task(
                llm,
                task,
                regen_pools,
                llm_max_retries=llm_max_retries,
                llm_backoff_sec=llm_backoff_sec,
                pools_per_generation_call=regen_chunk_size,
                inter_llm_call_sleep_sec=inter_llm_call_sleep_sec,
                rejection_reasons=pending_rejected,
            )
            consecutive_llm_errors = 0
        except Exception as regen_exc:
            consecutive_llm_errors += 1
            print(
                "[TaskDataset][CircuitBreaker] task=%s regen LLM error #%d: %s"
                % (task["task_id"], consecutive_llm_errors, regen_exc)
            )
            if consecutive_llm_errors >= 3:
                print(
                    "[TaskDataset][CircuitBreaker] task=%s aborting regen after %d consecutive LLM errors."
                    % (task["task_id"], consecutive_llm_errors)
                )
                break
            continue

        regenerated_cases: list[dict[str, Any]] = []
        for row in regenerated:
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            case_by_id[case_id] = row
            regenerated_cases.append(row)

        if not regenerated_cases:
            break

        pending_rejected = _audit_cases(
            audit_llm,
            task,
            regenerated_cases,
            llm_max_retries=llm_max_retries,
            llm_backoff_sec=llm_backoff_sec,
            cases_per_audit_call=int(regen_cases_per_audit_call),
            inter_llm_call_sleep_sec=inter_llm_call_sleep_sec,
        )

    if pending_rejected:
        print(
            "[TaskDataset][Warning] %s: reached max retries. Dropping rejected cases=%s"
            % (task["task_id"], pending_rejected)
        )
        for case_id in pending_rejected.keys():
            case_by_id.pop(str(case_id).strip(), None)

    kept = [case_by_id[cid] for cid in ordered_case_ids if cid in case_by_id]
    return kept, pending_rejected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    eval_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build task-driven eval dataset.")
    parser.add_argument(
        "--task-types",
        type=Path,
        default=eval_dir / "config" / "task_types_retrieval.json",
        help="Task type config JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=eval_dir / "datasets" / "task_eval_cases_retrieval.jsonl",
        help="Output dataset JSONL path.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=eval_dir / "datasets" / "task_eval_manifest_retrieval.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--strict-tool-check",
        action="store_true",
        default=True,
        help="Validate task/case tools against live tool catalog.",
    )
    parser.add_argument(
        "--no-strict-tool-check",
        dest="strict_tool_check",
        action="store_false",
        help="Skip strict tool catalog validation.",
    )
    parser.add_argument(
        "--enforce-coverage-policy",
        action="store_true",
        default=True,
        help="Enforce scenario coverage policy for task type files.",
    )
    parser.add_argument(
        "--no-enforce-coverage-policy",
        dest="enforce_coverage_policy",
        action="store_false",
        help="Allow partial task subsets that do not satisfy full scenario coverage policy.",
    )
    parser.add_argument(
        "--pools-per-task",
        type=int,
        default=0,
        help="Override n_min per task. If 0, use task sampling.n_min.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic pool construction.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=_resolve_preferred_provider(),
        help="LLM provider: gemini_api|vertex|deepseek.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_TASK_EVAL_MODEL_AUTO,
        help="LLM model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TASK_EVAL_TEMPERATURE", "0.7")),
        help="LLM temperature for generation (LAAG). Default 0.7 for diversity.",
    )
    parser.add_argument(
        "--audit-temperature",
        type=float,
        default=float(os.getenv("TASK_EVAL_AUDIT_TEMPERATURE", "0.0")),
        help="LLM temperature for audit (LAAJ). Default 0.0 for determinism.",
    )
    parser.add_argument(
        "--disable-audit",
        action="store_true",
        help="Disable the second-pass audit model validation.",
    )
    parser.add_argument(
        "--allow-topic-audit-fail",
        action="store_true",
        help="Allow generation to continue when the topic preflight audit fails. Intended for smoke/debug only.",
    )
    parser.add_argument(
        "--audit-max-regen-rounds",
        type=int,
        default=int(os.getenv("AUDIT_MAX_REGEN_ROUNDS", os.getenv("TASK_EVAL_AUDIT_MAX_REGEN_ROUNDS", "3"))),
        help=(
            "Max regeneration rounds after audit rejection (0 means no regen, only one audit pass). Default 3."
        ),
    )
    parser.add_argument(
        "--audit-regen-mode",
        type=str,
        default=str(os.getenv("AUDIT_REGEN_MODE", "failed_only")).strip().lower() or "failed_only",
        help="Audit regeneration mode: failed_only|full. Default failed_only.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=int(os.getenv("TASK_EVAL_LLM_MAX_RETRIES", "1")),
        help="Retry times for retryable LLM failures (429/quota/timeout).",
    )
    parser.add_argument(
        "--seed-max-attempts",
        type=int,
        default=int(os.getenv("TASK_EVAL_SEED_MAX_ATTEMPTS", "3")),
        help="Max quality attempts per packed pool seed. Default 3.",
    )
    parser.add_argument(
        "--llm-backoff-sec",
        type=float,
        default=float(os.getenv("TASK_EVAL_LLM_BACKOFF_SEC", "2.0")),
        help="Base backoff seconds for LLM retries (exponential).",
    )
    parser.add_argument(
        "--pools-per-generation-call",
        type=int,
        default=int(os.getenv("TASK_EVAL_POOLS_PER_GENERATION_CALL", "1")),
        help="If > 0, split one task's pools into multiple generation calls. Default 1.",
    )
    parser.add_argument(
        "--regen-pools-per-generation-call",
        type=int,
        default=int(os.getenv("REGEN_POOLS_PER_GENERATION_CALL", "1")),
        help="If > 0, split failed-case regeneration pools into multiple generation calls. Default 1 (single-case regen).",
    )
    parser.add_argument(
        "--initial-cases-per-audit-call",
        type=int,
        default=int(os.getenv("INITIAL_CASES_PER_AUDIT_CALL", os.getenv("TASK_EVAL_CASES_PER_AUDIT_CALL", "0"))),
        help="Chunk size for initial full-batch audit call (0 means one call per task).",
    )
    parser.add_argument(
        "--regen-cases-per-audit-call",
        type=int,
        default=int(os.getenv("REGEN_CASES_PER_AUDIT_CALL", "1")),
        help="Chunk size for failed-case re-audit calls during regeneration (default 1).",
    )
    parser.add_argument(
        "--cases-per-audit-call",
        type=int,
        default=None,
        help="Deprecated alias; if set, applies to both initial/regen audit chunk size.",
    )
    parser.add_argument(
        "--inter-llm-call-sleep-sec",
        type=float,
        default=float(os.getenv("TASK_EVAL_INTER_LLM_CALL_SLEEP_SEC", "15.0")),
        help="Sleep seconds between chunked generation/audit calls for throttling. Default 15.0.",
    )
    parser.add_argument(
        "--inter-task-sleep-sec",
        type=float,
        default=float(os.getenv("TASK_EVAL_INTER_TASK_SLEEP_SEC", "0.0")),
        help="Sleep seconds between task types to reduce sustained quota pressure.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint JSON path for restore-point resume.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        default=True,
        help="Resume from checkpoint if exists.",
    )
    parser.add_argument(
        "--no-resume-from-checkpoint",
        dest="resume_from_checkpoint",
        action="store_false",
        help="Ignore existing checkpoint and rebuild from scratch.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file loaded after agent/.env.",
    )
    parser.add_argument(
        "--enforce-scenario-retrieval-map",
        action="store_true",
        default=bool(int(os.getenv("TASK_EVAL_ENFORCE_SCENARIO_RETRIEVAL_MAP", "0") or 0)),
        help="Enforce fixed scenario->retrieval_mode mapping (normal/boundary=evaluable, conflict/empty=non_retrieval).",
    )
    parser.add_argument(
        "--no-enforce-scenario-retrieval-map",
        dest="enforce_scenario_retrieval_map",
        action="store_false",
        help="Disable scenario->retrieval_mode enforcement.",
    )
    parser.add_argument(
        "--print-fingerprint-only",
        action="store_true",
        help="Print dataset fingerprint JSON and exit without generation.",
    )
    args = parser.parse_args(argv)
    provider_value = str(getattr(args, "provider", "") or "").strip() or _resolve_preferred_provider()
    explicit_model = str(getattr(args, "model", "") or "").strip()
    model_from_env = str(os.getenv("TASK_EVAL_MODEL", "")).strip()

    if explicit_model and explicit_model != _TASK_EVAL_MODEL_AUTO:
        resolved = resolve_model_config(
            provider=provider_value,
            model_name=explicit_model,
            default_provider=DEFAULT_PROVIDER,
            default_model=DEFAULT_MODEL,
        )
    else:
        resolved = resolve_model_config(
            provider=provider_value,
            model_name=model_from_env or None,
            default_provider=DEFAULT_PROVIDER,
            default_model=DEFAULT_MODEL,
        )

    args.provider = resolved.provider
    args.model = resolved.model
    return args


def _preparse_env_file(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", type=Path, default=None)
    parsed, _unknown = parser.parse_known_args(argv)
    return parsed


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _pool_quality_manifest(
    pools: list[Pool],
    pool_meta_by_id: dict[str, dict[str, Any]],
    task: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    passed = 0
    failed = 0
    for pool in pools:
        meta = pool_meta_by_id.get(pool.pool_id, pool.meta if isinstance(pool.meta, dict) else {})
        if not isinstance(meta, dict):
            meta = {}
        quality = meta.get("pool_quality", {}) if isinstance(meta.get("pool_quality", {}), dict) else {}
        if not quality:
            quality = pool_quality_summary(pool.docs, task)
        quality_passed = bool(quality.get("pool_quality_passed", meta.get("pool_quality_passed", False)))
        if quality_passed:
            passed += 1
        else:
            failed += 1
        reasons = quality.get("pool_quality_reasons", meta.get("pool_quality_reasons", [])) or []
        for reason in reasons:
            reason_text = str(reason or "unknown")
            reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1
        rows.append(
            {
                "pool_id": pool.pool_id,
                "pool_quality_passed": quality_passed,
                "pool_quality_reasons": reasons,
                "topic_anchor_terms": quality.get("topic_anchor_terms", []),
                "topic_match_ratio": quality.get("topic_match_ratio", 0.0),
                "avg_topic_match_score": quality.get("avg_topic_match_score", 0.0),
                "embedding_coherence": quality.get("embedding_coherence"),
                "source_count": quality.get("source_count", 0),
                "time_span_days": quality.get("time_span_days"),
                "selected_docs": len(pool.docs),
            }
        )
    return {
        "pool_count": len(pools),
        "passed": passed,
        "failed": failed,
        "failed_reasons": reason_counts,
        "pools": rows,
    }


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        return None
    return payload


def main(argv: list[str] | None = None) -> int:
    pre_args = _preparse_env_file(argv)
    _load_eval_env(pre_args.env_file)
    args = _parse_args(argv)

    task_types = load_task_types(
        args.task_types.resolve(),
        strict_tool=bool(args.strict_tool_check),
        enforce_coverage_policy=bool(args.enforce_coverage_policy),
    )
    if bool(getattr(args, "enforce_scenario_retrieval_map", False)):
        validate_scenario_retrieval_map(task_types)
    dataset_fingerprint, dataset_fingerprint_payload = build_dataset_fingerprint(
        args=args,
        task_types=task_types,
    )
    if bool(getattr(args, "print_fingerprint_only", False)):
        print(
            json.dumps(
                {
                    "dataset_fingerprint": dataset_fingerprint,
                    "dataset_fingerprint_payload": dataset_fingerprint_payload,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    seed_temperatures = _seed_attempt_temperatures(
        float(args.temperature),
        max(1, int(getattr(args, "seed_max_attempts", 3))),
    )
    generation_llms: dict[float, Any] = {}
    for temp in seed_temperatures:
        if temp not in generation_llms:
            generation_llms[temp] = _build_chat_model(
                provider=str(args.provider),
                model_name=str(args.model),
                temperature=float(temp),
            )
    # Separate audit LLM with deterministic temperature for LAAJ
    audit_temperature = float(getattr(args, "audit_temperature", 0.0))
    matching_generation_temp = next(
        (temp for temp in generation_llms if abs(audit_temperature - temp) < 1e-6),
        None,
    )
    if matching_generation_temp is not None:
        audit_llm = generation_llms[matching_generation_temp]  # reuse if temperatures match
    else:
        audit_llm = _build_chat_model(
            provider=str(args.provider),
            model_name=str(args.model),
            temperature=audit_temperature,
        )

    rng = random.Random(int(args.seed))
    output_path = args.output.resolve()
    checkpoint_path = (
        args.checkpoint_path.resolve()
        if args.checkpoint_path
        else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    )

    all_cases: list[dict[str, Any]] = []
    manifest_tasks: list[dict[str, Any]] = []
    dropped_seeds: list[dict[str, Any]] = []
    completed_task_ids: set[str] = set()
    audit_regen_mode = str(getattr(args, "audit_regen_mode", "failed_only")).strip().lower() or "failed_only"
    if audit_regen_mode not in {"failed_only", "full"}:
        print(f"[TaskDataset][Warn] unknown audit_regen_mode={audit_regen_mode}, fallback=failed_only")
        audit_regen_mode = "failed_only"

    initial_cases_per_audit_call = max(0, int(getattr(args, "initial_cases_per_audit_call", 0)))
    regen_cases_per_audit_call = max(0, int(getattr(args, "regen_cases_per_audit_call", 1)))
    if getattr(args, "cases_per_audit_call", None) is not None:
        legacy_chunk = max(0, int(args.cases_per_audit_call))
        initial_cases_per_audit_call = legacy_chunk
        regen_cases_per_audit_call = legacy_chunk

    if bool(args.resume_from_checkpoint):
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint:
            cp_task_file = str(checkpoint.get("task_type_file", "")).strip()
            cp_fingerprint = str(checkpoint.get("dataset_fingerprint", "")).strip()
            expected_task_file = str(args.task_types.resolve())
            if cp_task_file == expected_task_file and (not cp_fingerprint or cp_fingerprint == dataset_fingerprint):
                cp_cases = checkpoint.get("cases", [])
                cp_tasks = checkpoint.get("tasks", [])
                cp_dropped = checkpoint.get("dropped_seeds", [])
                cp_completed = checkpoint.get("completed_task_ids", [])
                if isinstance(cp_cases, list):
                    all_cases = [row for row in cp_cases if isinstance(row, dict)]
                if isinstance(cp_tasks, list):
                    manifest_tasks = [row for row in cp_tasks if isinstance(row, dict)]
                if isinstance(cp_dropped, list):
                    dropped_seeds = [row for row in cp_dropped if isinstance(row, dict)]
                if isinstance(cp_completed, list):
                    completed_task_ids = {
                        str(item).strip() for item in cp_completed if str(item).strip()
                    }
                print(
                    "[TaskDataset][Resume] checkpoint=%s completed_tasks=%s cases=%s"
                    % (checkpoint_path, len(completed_task_ids), len(all_cases))
                )
            else:
                print(
                    "[TaskDataset][Resume] skip checkpoint due to config mismatch: cp_task=%s current_task=%s cp_fingerprint=%s current_fingerprint=%s"
                    % (cp_task_file, expected_task_file, cp_fingerprint or "-", dataset_fingerprint)
                )

    total_tasks = len(task_types)
    for task_idx, task in enumerate(task_types, 1):
        task_id = str(task.get("task_id", "")).strip()
        if task_id in completed_task_ids:
            print("[TaskDataset][Skip] task already completed in checkpoint: %s" % task_id)
            continue

        n_min = int(task.get("sampling", {}).get("n_min", 30))
        pools_per_task = int(args.pools_per_task) if int(args.pools_per_task) > 0 else n_min

        try:
            sample = build_eval_sample(task, pools_per_task=pools_per_task, rng=rng)
            candidates = sample.candidates
            pools = [Pool(pool_id=pool.pool_id, docs=pool.docs, meta=pool.meta) for pool in sample.pools]
            sample_meta = sample.meta
            pool_meta_by_id = {pool.pool_id: pool.meta for pool in sample.pools}
        except Exception as exc:  # noqa: BLE001
            print(f"[TaskDataset][SamplerFallback] task={task_id} independent sampler failed: {exc}")
            candidates = _sample_candidates(task)
            pools = _build_pools(task, candidates, pools_per_task=pools_per_task, rng=rng)
            sample_meta = {
                "candidate_source": "legacy_eval_sampler_fallback",
                "sampler_error": str(exc),
                "candidate_docs": len(candidates),
                "pool_count": len(pools),
                "cluster_mode": "legacy_round_robin",
                "cluster_fallback": True,
                "cluster_fallback_reason": "independent_sampler_exception",
            }
            pool_meta_by_id = {pool.pool_id: {} for pool in pools}

        topic_audit = audit_task_with_sample(task, candidates, sample_meta, pools)
        if topic_audit.get("verdict") != "pass":
            print(
                "[TaskDataset][TopicAudit] task=%s verdict=%s issues=%s warnings=%s"
                % (
                    task["task_id"],
                    topic_audit.get("verdict"),
                    len(topic_audit.get("issues", [])),
                    len(topic_audit.get("warnings", [])),
                )
            )
        topic_audit_blocked = topic_audit.get("verdict") == "fail" and not bool(
            getattr(args, "allow_topic_audit_fail", False)
        )
        if topic_audit_blocked:
            feedback_reasons = [
                str(issue.get("code", "")).strip()
                for issue in topic_audit.get("issues", [])
                if isinstance(issue, dict) and str(issue.get("code", "")).strip()
            ] or ["topic_audit_failed"]
            generated_cases = []
            dropped_for_task = [
                {
                    "task_id": task["task_id"],
                    "pool_id": "",
                    "pool_hash": "",
                    "attempts": 0,
                    "temperatures": seed_temperatures,
                    "failure_stage": "topic_audit",
                    "reason": "topic_audit_failed",
                    "last_score": 0.0,
                    "candidate_doc_ids": [str(doc.get("doc_id", "")) for doc in candidates],
                    "pool_ids": [str(pool.pool_id) for pool in pools],
                    "feedback_reasons": feedback_reasons,
                    "attempt_log": [],
                }
            ]
            print("[TaskDataset][TopicAuditBlock] task=%s reason=topic_audit_failed" % task["task_id"])
        elif pools:
            generated_cases, dropped_for_task, _task_api_errors = _generate_cases_seed_level(
                generation_llms=generation_llms,
                audit_llm=audit_llm,
                task=task,
                pools=pools,
                temperatures=seed_temperatures,
                audit_enabled=not bool(args.disable_audit),
                llm_max_retries=int(args.llm_max_retries),
                llm_backoff_sec=float(args.llm_backoff_sec),
            )
        else:
            generated_cases = []
            dropped_for_task = [
                {
                    "task_id": task["task_id"],
                    "pool_id": "",
                    "pool_hash": "",
                    "attempts": 0,
                    "temperatures": seed_temperatures,
                    "failure_stage": "packing",
                    "reason": "no_valid_packed_pools",
                    "last_score": 0.0,
                    "candidate_doc_ids": [str(doc.get("doc_id", "")) for doc in candidates],
                    "feedback_reasons": ["no_valid_packed_pools"],
                    "attempt_log": [],
                }
            ]
        dropped_seeds.extend(dropped_for_task)
        for case in generated_cases:
            pool_meta = pool_meta_by_id.get(str(case.get("pool_id", "")), {})
            if pool_meta:
                case["packing_meta"] = pool_meta

        for case in generated_cases:
            validate_case(case, strict_tool=bool(args.strict_tool_check))
        all_cases.extend(generated_cases)

        manifest_tasks.append(
            {
                "task_id": task["task_id"],
                "tool": task["tool"],
                "capability": task.get("capability", ""),
                "retrieval_mode": task["retrieval_mode"],
                "scenario": task["scenario"],
                "candidate_docs": len(candidates),
                "embedding_docs": int(sample_meta.get("embedding_docs", 0) or 0),
                "candidate_source": sample_meta.get("candidate_source", ""),
                "candidate_channel_counts": sample_meta.get("candidate_channel_counts", {}),
                "cluster_mode": sample_meta.get("cluster_mode", ""),
                "cluster_count": sample_meta.get("cluster_count", 0),
                "cluster_fallback": bool(sample_meta.get("cluster_fallback", False)),
                "cluster_fallback_reason": sample_meta.get("cluster_fallback_reason", ""),
                "topic_strong_candidate_docs": int(sample_meta.get("topic_strong_candidate_docs", 0) or 0),
                "pool_quality_failed_count": int(sample_meta.get("pool_quality_failed_count", 0) or 0),
                "pool_quality_failed_reasons": sample_meta.get("pool_quality_failed_reasons", {}),
                "pool_count": len(pools),
                "pool_quality": _pool_quality_manifest(pools, pool_meta_by_id, task),
                "generated_cases": len(generated_cases),
                "dropped_seed_count": len(dropped_for_task),
                "seed_attempt_temperatures": seed_temperatures,
                "topic_audit": topic_audit,
            }
        )
        completed_task_ids.add(task["task_id"])
        checkpoint_payload = {
            "status": "in_progress",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_type_file": str(args.task_types.resolve()),
            "dataset_path": str(output_path),
            "coverage_policy_enforced": bool(args.enforce_coverage_policy),
            "provider": str(args.provider),
            "model": str(args.model),
            "temperature": float(args.temperature),
            "audit_temperature": float(args.audit_temperature),
            "seed_max_attempts": int(args.seed_max_attempts),
            "seed_attempt_temperatures": seed_temperatures,
            "audit_enabled": not bool(args.disable_audit),
            "allow_topic_audit_fail": bool(getattr(args, "allow_topic_audit_fail", False)),
            "llm_max_retries": int(args.llm_max_retries),
            "llm_backoff_sec": float(args.llm_backoff_sec),
            "audit_max_regen_rounds": int(args.audit_max_regen_rounds),
            "audit_regen_mode": audit_regen_mode,
            "pools_per_generation_call": int(args.pools_per_generation_call),
            "regen_pools_per_generation_call": int(args.regen_pools_per_generation_call),
            "cases_per_audit_call": int(initial_cases_per_audit_call),
            "initial_cases_per_audit_call": int(initial_cases_per_audit_call),
            "regen_cases_per_audit_call": int(regen_cases_per_audit_call),
            "inter_llm_call_sleep_sec": float(args.inter_llm_call_sleep_sec),
            "inter_task_sleep_sec": float(args.inter_task_sleep_sec),
            "scenario_retrieval_map_enforced": bool(getattr(args, "enforce_scenario_retrieval_map", False)),
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_fingerprint_payload": dataset_fingerprint_payload,
            "completed_task_ids": sorted(completed_task_ids),
            "tasks": manifest_tasks,
            "dropped_seeds": dropped_seeds,
            "cases": all_cases,
        }
        _write_json_atomic(checkpoint_path, checkpoint_payload)
        print(
            "[TaskDataset] task=%s pools=%s candidates=%s generated=%s"
            % (task["task_id"], len(pools), len(candidates), len(generated_cases))
        )
        if float(args.inter_task_sleep_sec) > 0 and task_idx < total_tasks:
            sleep_sec = float(args.inter_task_sleep_sec)
            print(
                "[TaskDataset][Throttle] after task=%s sleep=%.2fs"
                % (task["task_id"], sleep_sec)
            )
            time.sleep(sleep_sec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    topic_audit_summary = {
        "points_used": False,
        "pass": sum(1 for row in manifest_tasks if row.get("topic_audit", {}).get("verdict") == "pass"),
        "warn": sum(1 for row in manifest_tasks if row.get("topic_audit", {}).get("verdict") == "warn"),
        "fail": sum(1 for row in manifest_tasks if row.get("topic_audit", {}).get("verdict") == "fail"),
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_type_file": str(args.task_types.resolve()),
        "dataset_path": str(output_path),
        "coverage_policy_enforced": bool(args.enforce_coverage_policy),
        "case_count": len(all_cases),
        "provider": str(args.provider),
        "model": str(args.model),
        "temperature": float(args.temperature),
        "audit_temperature": float(args.audit_temperature),
        "seed_max_attempts": int(args.seed_max_attempts),
        "seed_attempt_temperatures": seed_temperatures,
        "audit_enabled": not bool(args.disable_audit),
        "allow_topic_audit_fail": bool(getattr(args, "allow_topic_audit_fail", False)),
        "llm_max_retries": int(args.llm_max_retries),
        "llm_backoff_sec": float(args.llm_backoff_sec),
        "audit_max_regen_rounds": int(args.audit_max_regen_rounds),
        "audit_regen_mode": audit_regen_mode,
        "pools_per_generation_call": int(args.pools_per_generation_call),
        "regen_pools_per_generation_call": int(args.regen_pools_per_generation_call),
        "cases_per_audit_call": int(initial_cases_per_audit_call),
        "initial_cases_per_audit_call": int(initial_cases_per_audit_call),
        "regen_cases_per_audit_call": int(regen_cases_per_audit_call),
        "inter_llm_call_sleep_sec": float(args.inter_llm_call_sleep_sec),
        "inter_task_sleep_sec": float(args.inter_task_sleep_sec),
        "scenario_retrieval_map_enforced": bool(getattr(args, "enforce_scenario_retrieval_map", False)),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_fingerprint_payload": dataset_fingerprint_payload,
        "topic_audit": topic_audit_summary,
        "tasks": manifest_tasks,
        "dropped_seeds": dropped_seeds,
        "dropped_seed_count": len(dropped_seeds),
    }
    manifest_path = args.manifest_output.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_json_atomic(
        checkpoint_path,
        {
            "status": "completed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_type_file": str(args.task_types.resolve()),
            "dataset_path": str(output_path),
            "coverage_policy_enforced": bool(args.enforce_coverage_policy),
            "provider": str(args.provider),
            "model": str(args.model),
            "temperature": float(args.temperature),
            "audit_temperature": float(args.audit_temperature),
            "seed_max_attempts": int(args.seed_max_attempts),
            "seed_attempt_temperatures": seed_temperatures,
            "audit_enabled": not bool(args.disable_audit),
            "allow_topic_audit_fail": bool(getattr(args, "allow_topic_audit_fail", False)),
            "llm_max_retries": int(args.llm_max_retries),
            "llm_backoff_sec": float(args.llm_backoff_sec),
            "audit_max_regen_rounds": int(args.audit_max_regen_rounds),
            "audit_regen_mode": audit_regen_mode,
            "pools_per_generation_call": int(args.pools_per_generation_call),
            "regen_pools_per_generation_call": int(args.regen_pools_per_generation_call),
            "cases_per_audit_call": int(initial_cases_per_audit_call),
            "initial_cases_per_audit_call": int(initial_cases_per_audit_call),
            "regen_cases_per_audit_call": int(regen_cases_per_audit_call),
            "inter_llm_call_sleep_sec": float(args.inter_llm_call_sleep_sec),
            "inter_task_sleep_sec": float(args.inter_task_sleep_sec),
            "scenario_retrieval_map_enforced": bool(getattr(args, "enforce_scenario_retrieval_map", False)),
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_fingerprint_payload": dataset_fingerprint_payload,
            "topic_audit": topic_audit_summary,
            "completed_task_ids": sorted(completed_task_ids),
            "tasks": manifest_tasks,
            "dropped_seeds": dropped_seeds,
            "cases": all_cases,
            "manifest_path": str(manifest_path),
        },
    )

    print("[TaskDataset] output=%s cases=%s" % (output_path, len(all_cases)))
    print("[TaskDataset] manifest=%s" % manifest_path)
    print("[TaskDataset] checkpoint=%s" % checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
