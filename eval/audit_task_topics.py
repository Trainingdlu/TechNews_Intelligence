"""Topic quality preflight for task-eval dataset generation."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from corpus_sampler import build_eval_sample, topic_side_for_doc
    from pool_quality import pool_quality_summary, quality_required
    from task_eval_schema import load_task_types
except ImportError:
    from .corpus_sampler import build_eval_sample, topic_side_for_doc
    from .pool_quality import pool_quality_summary, quality_required
    from .task_eval_schema import load_task_types


PLACEHOLDER_TOPICS = {
    "low frequency topic",
    "newly emerged keyword",
    "rare timeline keyword",
    "rare source compare keyword",
    "ultra-rare trend keyword",
    "niche micro domain",
}
PLACEHOLDER_PREFIXES = ("need clarification",)

MIN_CANDIDATE_DOCS = 60
MIN_EMBEDDING_DOCS = 24
MIN_VALID_POOLS = 3
MIN_SOURCE_DOCS = 6
MIN_TOPIC_SIDE_DOCS = 20
MAX_EMPTY_POSITIVE_MATCHES = 3
EMPTY_HIGH_SIM_CEILING = 0.45


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;|，、\n]+", value) if item.strip()]
    return []


def _params(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("parameter_template", {})
    return value if isinstance(value, dict) else {}


def _sampling(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("sampling", {})
    return value if isinstance(value, dict) else {}


def task_seed_texts(task: dict[str, Any]) -> list[str]:
    params = _params(task)
    sampling = _sampling(task)
    texts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in texts:
            texts.append(text)

    for key in ("query", "urls", "topic", "topic_a", "topic_b"):
        add(params.get(key))
    for entity in _as_list(params.get("entities")):
        add(entity)
    for keyword in _as_list(sampling.get("keywords", [])):
        add(keyword)
    return texts


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"code": code, "message": message}
    if details:
        row["details"] = details
    return row


def _base_result(task: dict[str, Any]) -> dict[str, Any]:
    sampling = _sampling(task)
    return {
        "task_id": str(task.get("task_id", "")),
        "tool": str(task.get("tool", "")),
        "scenario": str(task.get("scenario", "")).strip().lower(),
        "retrieval_mode": str(task.get("retrieval_mode", "")).strip().lower(),
        "should_clarify": bool(task.get("should_clarify", False)),
        "seed_queries": task_seed_texts(task),
        "keywords": _as_list(sampling.get("keywords", [])),
        "sources": _as_list(sampling.get("sources", [])),
        "points_used": False,
        "issues": [],
        "warnings": [],
        "coverage": {},
        "verdict": "pass",
    }


def _finalize(result: dict[str, Any]) -> dict[str, Any]:
    if result["issues"]:
        result["verdict"] = "fail"
    elif result["warnings"]:
        result["verdict"] = "warn"
    else:
        result["verdict"] = "pass"
    return result


def _is_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if normalized in PLACEHOLDER_TOPICS:
        return True
    return any(normalized.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def _main_tool_args_match(task: dict[str, Any]) -> list[dict[str, Any]]:
    tool = str(task.get("tool", "")).strip()
    expected_args = _params(task)
    mismatches: list[dict[str, Any]] = []
    paths = task.get("acceptable_tool_paths", [])
    if not isinstance(paths, list):
        return [{"path_index": 0, "step_index": 0, "reason": "acceptable_tool_paths_not_list"}]
    for path_idx, path in enumerate(paths, 1):
        if not isinstance(path, list):
            continue
        for step_idx, step in enumerate(path, 1):
            if not isinstance(step, dict) or str(step.get("tool", "")).strip() != tool:
                continue
            args = step.get("args", {})
            if args != expected_args:
                mismatches.append({"path_index": path_idx, "step_index": step_idx, "args": args})
    return mismatches


def audit_task_config(task: dict[str, Any]) -> dict[str, Any]:
    result = _base_result(task)
    scenario = result["scenario"]
    retrieval_mode = result["retrieval_mode"]
    seeds = result["seed_queries"]

    if retrieval_mode == "evaluable":
        placeholders = [text for text in seeds if _is_placeholder(text)]
        if placeholders:
            result["issues"].append(
                _issue(
                    "topic_placeholder",
                    "retrieval-evaluable task uses placeholder topic text.",
                    placeholders=placeholders,
                )
            )

    if scenario == "conflict":
        if retrieval_mode != "non_retrieval" or not result["should_clarify"]:
            result["issues"].append(
                _issue(
                    "conflict_policy_mismatch",
                    "conflict tasks must be non_retrieval and should_clarify=true.",
                )
            )

    if scenario == "empty" and retrieval_mode != "non_retrieval":
        result["issues"].append(_issue("empty_policy_mismatch", "empty tasks must be non_retrieval."))

    path_mismatches = _main_tool_args_match(task)
    if path_mismatches:
        result["issues"].append(
            _issue(
                "path_topic_mismatch",
                "main tool args in acceptable_tool_paths must equal parameter_template.",
                mismatches=path_mismatches,
            )
        )

    return _finalize(result)


def _date_key(doc: dict[str, Any]) -> str:
    return str(doc.get("published_at", "") or "")[:10]


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _pool_docs(pool: Any) -> list[dict[str, Any]]:
    if isinstance(pool, dict):
        docs = pool.get("docs", [])
    else:
        docs = getattr(pool, "docs", [])
    return [doc for doc in docs if isinstance(doc, dict)]


def _pool_meta(pool: Any) -> dict[str, Any]:
    if isinstance(pool, dict):
        meta = pool.get("meta", {})
    else:
        meta = getattr(pool, "meta", {})
    return meta if isinstance(meta, dict) else {}


def _pool_quality(pool: Any, task: dict[str, Any]) -> dict[str, Any]:
    meta = _pool_meta(pool)
    quality = meta.get("pool_quality", {})
    if isinstance(quality, dict) and quality:
        return quality
    return pool_quality_summary(_pool_docs(pool), task)


def _topic_side_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"A": 0, "B": 0}
    for doc in candidates:
        side = topic_side_for_doc(doc)
        if side in counts:
            counts[side] += 1
    return counts


def _mentions(text: str, term: str) -> bool:
    return term.strip().lower() in text.lower()


def _entity_coverage(task: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, int]:
    params = _params(task)
    terms = _as_list(params.get("entities")) or _as_list(_sampling(task).get("keywords", []))
    coverage: dict[str, int] = {}
    for term in terms:
        if not term or term.lower() in {"zzqv"}:
            continue
        count = 0
        for doc in candidates:
            haystack = f"{doc.get('title', '')} {doc.get('summary', '')} {doc.get('evidence_text', '')}"
            if _mentions(haystack, term):
                count += 1
        coverage[term] = count
    return coverage


def _trend_window_counts(task: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, int]:
    window = int(_params(task).get("window", _sampling(task).get("days", 14)) or 14)
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(days=window)
    previous_start = now - timedelta(days=window * 2)
    current = 0
    previous = 0
    for doc in candidates:
        dt = _parse_datetime(doc.get("published_at"))
        if not dt:
            continue
        if dt >= current_start:
            current += 1
        elif previous_start <= dt < current_start:
            previous += 1
    return {"current_window_docs": current, "previous_window_docs": previous, "window_days": window}


def audit_task_with_sample(
    task: dict[str, Any],
    candidates: list[dict[str, Any]],
    sample_meta: dict[str, Any],
    pools: list[Any],
) -> dict[str, Any]:
    result = audit_task_config(task)
    issues = result["issues"]
    warnings = result["warnings"]
    tool = result["tool"]
    scenario = result["scenario"]
    retrieval_evaluable = result["retrieval_mode"] == "evaluable"

    candidate_docs = int(sample_meta.get("candidate_docs", len(candidates)) or 0)
    embedding_docs = int(
        sample_meta.get("embedding_docs", sum(1 for doc in candidates if doc.get("embedding_available"))) or 0
    )
    source_counts = Counter(str(doc.get("source", "unknown")) or "unknown" for doc in candidates)
    date_count = len({_date_key(doc) for doc in candidates if _date_key(doc)})
    channel_counts = sample_meta.get("candidate_channel_counts", {})
    if not isinstance(channel_counts, dict):
        channel_counts = {}

    result["coverage"] = {
        "candidate_docs": candidate_docs,
        "embedding_docs": embedding_docs,
        "pool_count": len(pools),
        "source_counts": dict(source_counts),
        "date_count": date_count,
        "candidate_channel_counts": channel_counts,
        "points_used": False,
    }

    pool_quality_rows = [_pool_quality(pool, task) for pool in pools]
    pool_quality_passed = sum(1 for row in pool_quality_rows if bool(row.get("pool_quality_passed", False)))
    pool_quality_failed = len(pool_quality_rows) - pool_quality_passed
    result["coverage"]["pool_quality"] = {
        "required": quality_required(task),
        "passed": pool_quality_passed,
        "failed": pool_quality_failed,
        "failed_reasons": dict(
            Counter(
                str(reason)
                for row in pool_quality_rows
                for reason in (row.get("pool_quality_reasons", []) or [])
            )
        ),
        "topic_match_ratio_min": min(
            (float(row.get("topic_match_ratio", 0.0) or 0.0) for row in pool_quality_rows),
            default=0.0,
        ),
        "topic_match_ratio_avg": (
            round(
                sum(float(row.get("topic_match_ratio", 0.0) or 0.0) for row in pool_quality_rows)
                / len(pool_quality_rows),
                4,
            )
            if pool_quality_rows
            else 0.0
        ),
    }

    if retrieval_evaluable and scenario in {"normal", "boundary"}:
        if candidate_docs < MIN_CANDIDATE_DOCS:
            issues.append(
                _issue(
                    "candidate_coverage_insufficient",
                    "normal/boundary task does not have enough candidate documents.",
                    actual=candidate_docs,
                    minimum=MIN_CANDIDATE_DOCS,
                )
            )
        if embedding_docs < MIN_EMBEDDING_DOCS:
            issues.append(
                _issue(
                    "embedding_coverage_insufficient",
                    "normal/boundary task does not have enough embedded documents.",
                    actual=embedding_docs,
                    minimum=MIN_EMBEDDING_DOCS,
                )
            )
        if len(pools) < MIN_VALID_POOLS:
            issues.append(
                _issue(
                    "valid_pool_insufficient",
                    "normal/boundary task must produce at least three valid packed pools.",
                    actual=len(pools),
                    minimum=MIN_VALID_POOLS,
                )
            )
        if quality_required(task) and pool_quality_passed < MIN_VALID_POOLS:
            issues.append(
                _issue(
                    "pool_quality_insufficient",
                    "normal/boundary task must produce at least three topic-consistent pools.",
                    actual=pool_quality_passed,
                    failed=pool_quality_failed,
                    minimum=MIN_VALID_POOLS,
                    failed_reasons=result["coverage"]["pool_quality"]["failed_reasons"],
                )
            )

    if retrieval_evaluable and tool == "compare_sources":
        missing = {
            source: source_counts.get(source, 0)
            for source in result["sources"]
            if source_counts.get(source, 0) < MIN_SOURCE_DOCS
        }
        if missing:
            issues.append(
                _issue(
                    "source_coverage_insufficient",
                    "compare_sources does not have enough candidates for each requested source.",
                    source_counts=missing,
                    minimum=MIN_SOURCE_DOCS,
                )
            )

    if retrieval_evaluable and tool == "compare_topics":
        side_counts = _topic_side_counts(candidates)
        result["coverage"]["topic_side_counts"] = side_counts
        if side_counts["A"] < MIN_TOPIC_SIDE_DOCS or side_counts["B"] < MIN_TOPIC_SIDE_DOCS:
            issues.append(
                _issue(
                    "topic_side_coverage_insufficient",
                    "compare_topics requires enough candidates on both topic sides.",
                    topic_side_counts=side_counts,
                    minimum=MIN_TOPIC_SIDE_DOCS,
                )
            )
        unbalanced_pools = []
        for idx, pool in enumerate(pools, 1):
            pool_counts = _topic_side_counts(_pool_docs(pool))
            if abs(pool_counts["A"] - pool_counts["B"]) > 1:
                unbalanced_pools.append({"pool_index": idx, **pool_counts})
        if unbalanced_pools:
            issues.append(
                _issue(
                    "topic_packing_unbalanced",
                    "compare_topics packed pools must keep topic A/B counts close.",
                    pools=unbalanced_pools,
                )
            )

    if retrieval_evaluable and tool in {"build_timeline", "trend_analysis"}:
        if date_count < 3:
            issues.append(
                _issue(
                    "time_spread_insufficient",
                    "timeline/trend tasks require at least three distinct publication dates.",
                    actual=date_count,
                    minimum=3,
                )
            )
        if tool == "trend_analysis":
            trend_counts = _trend_window_counts(task, candidates)
            result["coverage"]["trend_window_counts"] = trend_counts
            if trend_counts["current_window_docs"] == 0 or trend_counts["previous_window_docs"] == 0:
                issues.append(
                    _issue(
                        "trend_window_coverage_insufficient",
                        "trend_analysis needs candidates in both current and previous windows.",
                        **trend_counts,
                    )
                )

    if retrieval_evaluable and tool == "analyze_landscape":
        entity_counts = _entity_coverage(task, candidates)
        active = sum(1 for count in entity_counts.values() if count > 0)
        result["coverage"]["entity_counts"] = entity_counts
        if active < 3:
            issues.append(
                _issue(
                    "entity_coverage_insufficient",
                    "analyze_landscape must cover at least three core entities.",
                    active_entities=active,
                    entity_counts=entity_counts,
                    minimum=3,
                )
            )

    if scenario == "empty":
        lexical_alias = int(channel_counts.get("lexical", 0) or 0) + int(channel_counts.get("alias", 0) or 0)
        high_sim_docs = sum(
            1 for doc in candidates if float(doc.get("seed_similarity", 0.0) or 0.0) >= EMPTY_HIGH_SIM_CEILING
        )
        result["coverage"]["empty_positive_channel_matches"] = lexical_alias
        result["coverage"]["empty_high_similarity_docs"] = high_sim_docs
        if lexical_alias > MAX_EMPTY_POSITIVE_MATCHES:
            issues.append(
                _issue(
                    "empty_positive_match_too_high",
                    "empty topic has too many lexical/alias positive matches.",
                    actual=lexical_alias,
                    maximum=MAX_EMPTY_POSITIVE_MATCHES,
                )
            )
        if high_sim_docs > 0:
            warnings.append(
                _issue(
                    "empty_semantic_similarity_high",
                    "empty topic has high semantic-similarity candidates; verify negative sampling.",
                    actual=high_sim_docs,
                    ceiling=EMPTY_HIGH_SIM_CEILING,
                )
            )

    return _finalize(result)


def run_topic_audit(
    task_types: list[dict[str, Any]],
    *,
    runtime: bool,
    seed: int,
    pools_per_task: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for task in task_types:
        if not runtime:
            result = audit_task_config(task)
        else:
            try:
                sample = build_eval_sample(
                    task,
                    pools_per_task=max(MIN_VALID_POOLS, int(pools_per_task)),
                    rng=rng,
                )
                result = audit_task_with_sample(task, sample.candidates, sample.meta, sample.pools)
            except Exception as exc:  # noqa: BLE001
                result = audit_task_config(task)
                result["issues"].append(
                    _issue(
                        "runtime_sampler_error",
                        "runtime topic sampling failed.",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                _finalize(result)
        results.append(result)

    counts = Counter(row["verdict"] for row in results)
    return {
        "summary": {
            "task_count": len(results),
            "pass": counts.get("pass", 0),
            "warn": counts.get("warn", 0),
            "fail": counts.get("fail", 0),
            "points_used": False,
        },
        "tasks": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit task topic quality before dataset generation.")
    parser.add_argument("--task-file", type=Path, default=Path("eval/config/task_types_retrieval.json"))
    parser.add_argument("--strict", action="store_true", help="Treat warnings as blocking unless allowed by env.")
    parser.add_argument("--static-only", action="store_true", help="Skip DB-backed candidate coverage checks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pools-per-task", type=int, default=MIN_VALID_POOLS)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    task_types = load_task_types(args.task_file, strict_tool=False, enforce_coverage_policy=False)
    report = run_topic_audit(
        task_types,
        runtime=not bool(args.static_only),
        seed=int(args.seed),
        pools_per_task=int(args.pools_per_task),
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for row in report["tasks"]:
        coverage = row.get("coverage", {})
        print(
            "[TopicAudit] task=%s verdict=%s candidates=%s embeddings=%s pools=%s issues=%s warnings=%s"
            % (
                row.get("task_id"),
                row.get("verdict"),
                coverage.get("candidate_docs", ""),
                coverage.get("embedding_docs", ""),
                coverage.get("pool_count", ""),
                len(row.get("issues", [])),
                len(row.get("warnings", [])),
            )
        )
        for issue in row.get("issues", []):
            print("[TopicAudit][Issue] task=%s code=%s message=%s" % (row.get("task_id"), issue["code"], issue["message"]))
        for warning in row.get("warnings", []):
            print(
                "[TopicAudit][Warn] task=%s code=%s message=%s"
                % (row.get("task_id"), warning["code"], warning["message"])
            )

    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    has_fail = report["summary"]["fail"] > 0
    has_warn = report["summary"]["warn"] > 0
    allow_warnings = str(os.getenv("ALLOW_TOPIC_PREFLIGHT_WARNINGS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if has_fail:
        return 2
    if args.strict and has_warn and not allow_warnings:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
