"""Inspect generated task-eval datasets.

This script is intentionally read-only. It catches structural issues plus the
common semantic drift where the question, tool args, gold docs, and evidence
quotes describe different topics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from pool_quality import required_anchor_hits, score_doc_topic_match
except ImportError:
    from .pool_quality import required_anchor_hits, score_doc_topic_match


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}")
STOPWORDS = {
    "a",
    "an",
    "about",
    "analyze",
    "around",
    "build",
    "compare",
    "day",
    "days",
    "english",
    "for",
    "from",
    "http",
    "https",
    "com",
    "net",
    "org",
    "latest",
    "last",
    "news",
    "recent",
    "related",
    "search",
    "the",
    "updates",
    "vs",
    "week",
    "with",
}
SOURCE_ALIASES = {
    "techcrunch": "techcrunch",
    "hackernews": "hackernews",
    "hacker news": "hackernews",
    "wsj": "wsj",
    "wall street journal": "wsj",
    "arstechnica": "arstechnica",
    "ars technica": "arstechnica",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            rows.append({"case_id": f"line-{idx}", "_load_error": f"invalid_json: {exc}"})
            continue
        if not isinstance(row, dict):
            rows.append({"case_id": f"line-{idx}", "_load_error": "case_must_be_object"})
            continue
        rows.append(row)
    return rows


def _load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _tokenize(text: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _anchor_tokens(text: Any) -> list[str]:
    tokens: list[str] = []
    for token in _tokenize(text):
        cleaned = token.strip("_").lower()
        if len(cleaned) < 2:
            continue
        if cleaned.isdigit():
            continue
        if cleaned in STOPWORDS:
            continue
        if re.fullmatch(r"20\d{2}|\d+d?", cleaned):
            continue
        tokens.append(cleaned)
    return _dedupe(tokens)


def _normalize_paths(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list):
        return []
    out: list[list[dict[str, Any]]] = []
    for path in value:
        if not isinstance(path, list):
            continue
        normalized: list[dict[str, Any]] = []
        for call in path:
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool", "")).strip()
            args = call.get("args", {})
            if tool:
                normalized.append({"tool": tool, "args": args if isinstance(args, dict) else {}})
        if normalized:
            out.append(normalized)
    return out


def _path_anchor_groups(paths: list[list[dict[str, Any]]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        for call in path:
            args = call.get("args", {})
            if not isinstance(args, dict):
                continue
            for key in ("query", "topic", "topic_a", "topic_b", "url", "urls"):
                value = args.get(key)
                if value is None:
                    continue
                tokens = _anchor_tokens(value)
                if tokens:
                    groups[key] = _dedupe(groups.get(key, []) + tokens)
    return groups


def _doc_text(doc: dict[str, Any]) -> str:
    return " ".join(
        str(doc.get(key, "") or "")
        for key in ("title", "title_cn", "summary", "evidence_text", "url", "source")
    )


def _docs_by_id(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    docs = case.get("input_news_pool", [])
    if not isinstance(docs, list):
        return {}
    return {
        str(doc.get("doc_id", "")).strip(): doc
        for doc in docs
        if isinstance(doc, dict) and str(doc.get("doc_id", "")).strip()
    }


def _required_hits(tokens: list[str]) -> int:
    if len(tokens) >= 3:
        return 2
    if tokens:
        return 1
    return 0


def _overlap(tokens: list[str], text: str) -> int:
    text_tokens = set(_anchor_tokens(text))
    return sum(1 for token in _dedupe(tokens) if token in text_tokens)


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


def _case_issues(case: dict[str, Any], *, min_per_task: int, task_counts: Counter[str]) -> list[dict[str, str]]:
    case_id = str(case.get("case_id", "")).strip() or "(missing-case-id)"
    issues: list[dict[str, str]] = []
    if case.get("_load_error"):
        return [{"case_id": case_id, "code": str(case["_load_error"]), "detail": ""}]

    question = str(case.get("expected_question", "") or "").strip()
    answer = str(case.get("expected_answer", "") or "").strip()
    if not question:
        issues.append({"case_id": case_id, "code": "missing_question", "detail": ""})
    if not answer:
        issues.append({"case_id": case_id, "code": "missing_answer", "detail": ""})

    should_clarify = bool(case.get("should_clarify"))
    retrieval_evaluable = bool(case.get("retrieval_evaluable"))
    expected_paths = _normalize_paths(case.get("expected_tool_paths", []))
    gold_doc_ids = [str(item).strip() for item in case.get("retrieval_gold_doc_ids", []) if str(item).strip()]
    gold_urls = [str(item).strip() for item in case.get("retrieval_gold_urls", []) if str(item).strip()]
    claims = case.get("verifiable_claims", [])
    if not isinstance(claims, list):
        claims = []
        issues.append({"case_id": case_id, "code": "claims_not_list", "detail": ""})

    if should_clarify:
        if retrieval_evaluable:
            issues.append({"case_id": case_id, "code": "clarification_is_retrieval_evaluable", "detail": ""})
        if expected_paths:
            issues.append({"case_id": case_id, "code": "clarification_has_tool_path", "detail": ""})
        if gold_doc_ids or gold_urls:
            issues.append({"case_id": case_id, "code": "clarification_has_gold", "detail": ""})
        if claims:
            issues.append({"case_id": case_id, "code": "clarification_has_claims", "detail": ""})

    if retrieval_evaluable:
        if not gold_urls:
            issues.append({"case_id": case_id, "code": "retrieval_missing_gold_urls", "detail": ""})
        if not gold_doc_ids:
            issues.append({"case_id": case_id, "code": "retrieval_missing_gold_doc_ids", "detail": ""})
        if not claims:
            issues.append({"case_id": case_id, "code": "retrieval_missing_claims", "detail": ""})
    else:
        if gold_urls or gold_doc_ids:
            issues.append({"case_id": case_id, "code": "non_retrieval_has_gold", "detail": ""})
        if claims:
            issues.append({"case_id": case_id, "code": "non_retrieval_has_claims", "detail": ""})

    task_type = str(case.get("task_type", "")).strip()
    if min_per_task > 0 and task_type and task_counts[task_type] < min_per_task:
        issues.append(
            {
                "case_id": case_id,
                "code": "task_below_min_cases",
                "detail": f"{task_type}={task_counts[task_type]} < {min_per_task}",
            }
        )

    if not retrieval_evaluable:
        return issues

    anchors = _path_anchor_groups(expected_paths)
    topic_tokens = _dedupe(
        anchors.get("query", []) + anchors.get("topic", []) + anchors.get("url", []) + anchors.get("urls", [])
    )
    side_topic_tokens = _dedupe(anchors.get("topic_a", []) + anchors.get("topic_b", []))
    if not topic_tokens and side_topic_tokens:
        topic_tokens = side_topic_tokens
    if topic_tokens and _overlap(topic_tokens, question) < _required_hits(topic_tokens):
        issues.append(
            {
                "case_id": case_id,
                "code": "question_tool_topic_mismatch",
                "detail": ",".join(topic_tokens),
            }
        )

    by_id = _docs_by_id(case)
    pool_docs = list(by_id.values())
    if topic_tokens and pool_docs:
        matched = 0
        scores: list[float] = []
        for doc in pool_docs:
            if "topic_match_passed" in doc:
                passed = bool(doc.get("topic_match_passed"))
                score = float(doc.get("topic_match_score", 0.0) or 0.0)
            else:
                metrics = score_doc_topic_match(doc, topic_tokens)
                passed = bool(metrics.get("topic_match_passed", False))
                score = float(metrics.get("topic_match_score", 0.0) or 0.0)
            matched += 1 if passed else 0
            scores.append(score)
        ratio = matched / len(pool_docs) if pool_docs else 0.0
        if ratio < 0.50:
            issues.append(
                {
                    "case_id": case_id,
                    "code": "pool_topic_match_low",
                    "detail": "ratio=%.2f anchors=%s" % (ratio, ",".join(topic_tokens)),
                }
            )
        if scores and max(scores) < 0.40 and matched < required_anchor_hits(topic_tokens):
            issues.append(
                {
                    "case_id": case_id,
                    "code": "pool_topic_score_low",
                    "detail": "max_score=%.2f anchors=%s" % (max(scores), ",".join(topic_tokens)),
                }
            )
    gold_docs = [by_id[doc_id] for doc_id in gold_doc_ids if doc_id in by_id]
    gold_text = "\n".join(_doc_text(doc) for doc in gold_docs)
    if topic_tokens and _overlap(topic_tokens, gold_text) < _required_hits(topic_tokens):
        issues.append(
            {"case_id": case_id, "code": "gold_docs_topic_mismatch", "detail": ",".join(topic_tokens)}
        )

    if str(case.get("tool", "")).strip() == "compare_topics":
        for side in ("topic_a", "topic_b"):
            side_tokens = anchors.get(side, [])
            if side_tokens and _overlap(side_tokens, question) < _required_hits(side_tokens):
                issues.append({"case_id": case_id, "code": f"question_missing_{side}", "detail": ",".join(side_tokens)})
            if side_tokens and _overlap(side_tokens, gold_text) < _required_hits(side_tokens):
                issues.append({"case_id": case_id, "code": f"missing_{side}_gold_evidence", "detail": ",".join(side_tokens)})

    answer_sources = _mentioned_sources(answer)
    if answer_sources and gold_docs:
        gold_sources = {_source_key(doc.get("source")) for doc in gold_docs if _source_key(doc.get("source"))}
        if gold_sources and not answer_sources.issubset(gold_sources):
            issues.append(
                {
                    "case_id": case_id,
                    "code": "source_claim_mismatch",
                    "detail": f"answer={sorted(answer_sources)} gold={sorted(gold_sources)}",
                }
            )

    quote_text = "\n".join(
        str(quote.get("quote", "") or "")
        for claim in claims
        if isinstance(claim, dict)
        for quote in claim.get("evidence_quotes", [])
        if isinstance(quote, dict)
    )
    if topic_tokens and quote_text and _overlap(topic_tokens, quote_text) < _required_hits(topic_tokens):
        issues.append(
            {"case_id": case_id, "code": "quote_topic_mismatch", "detail": ",".join(topic_tokens)}
        )
    return issues


def _print_manifest(manifest: dict[str, Any]) -> None:
    if not manifest:
        print("\nmanifest: (not loaded)")
        return
    print("\nmanifest_topic_audit:", manifest.get("topic_audit"))
    print("\ntask_summary:")
    for task in manifest.get("tasks", []) or []:
        if not isinstance(task, dict):
            continue
        audit = task.get("topic_audit", {}) or {}
        issues = [row.get("code") for row in (audit.get("issues", []) or []) if isinstance(row, dict)]
        warnings = [row.get("code") for row in (audit.get("warnings", []) or []) if isinstance(row, dict)]
        print(
            "- {task_id} generated={generated} dropped={dropped} audit={audit} pool_quality={pool_quality} issues={issues} warnings={warnings}".format(
                task_id=task.get("task_id"),
                generated=task.get("generated_cases"),
                dropped=task.get("dropped_seed_count"),
                audit=audit.get("verdict"),
                pool_quality=(
                    "pass:{passed}/fail:{failed}".format(
                        passed=(task.get("pool_quality", {}) or {}).get("passed", 0),
                        failed=(task.get("pool_quality", {}) or {}).get("failed", 0),
                    )
                    if isinstance(task.get("pool_quality"), dict)
                    else "-"
                ),
                issues=issues,
                warnings=warnings,
            )
        )
        pool_quality = task.get("pool_quality", {}) if isinstance(task.get("pool_quality"), dict) else {}
        failed_reasons = pool_quality.get("failed_reasons", {}) if isinstance(pool_quality, dict) else {}
        if failed_reasons:
            print(f"  pool_quality_failed_reasons={failed_reasons}")

    dropped = manifest.get("dropped_seeds", []) or []
    if dropped:
        print("\ndropped_reasons:")
        for reason, count in Counter(str(row.get("reason", "")) for row in dropped if isinstance(row, dict)).most_common():
            print(f"  {count} x {reason[:180]}")


def _print_samples(cases: list[dict[str, Any]], samples_per_task: int) -> None:
    if samples_per_task <= 0:
        return
    print("\nsamples:")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get("task_type", "(missing-task)"))].append(case)
    for task_type in sorted(grouped):
        print("\n" + "=" * 100)
        print(f"TASK: {task_type} count={len(grouped[task_type])}")
        for case in grouped[task_type][:samples_per_task]:
            print("-" * 100)
            print("case_id:", case.get("case_id"))
            print("should_clarify:", case.get("should_clarify"))
            print("retrieval_evaluable:", case.get("retrieval_evaluable"))
            print("question:", case.get("expected_question"))
            answer = str(case.get("expected_answer", "") or "").replace("\n", " ")
            print("answer:", answer[:320] + ("..." if len(answer) > 320 else ""))
            print("expected_tool_paths:", json.dumps(case.get("expected_tool_paths", []), ensure_ascii=False))
            print("gold_urls:", case.get("retrieval_gold_urls", []))
            claims = case.get("verifiable_claims", [])
            print("claims_count:", len(claims) if isinstance(claims, list) else "invalid")
            if isinstance(claims, list) and claims:
                first = claims[0]
                print("first_claim:", first.get("claim") if isinstance(first, dict) else first)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect generated task-eval dataset quality.")
    parser.add_argument("--dataset", type=Path, required=True, help="Generated JSONL dataset path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON path.")
    parser.add_argument("--samples-per-task", type=int, default=2, help="How many cases to print per task.")
    parser.add_argument("--max-issues", type=int, default=80, help="Max issues to print.")
    parser.add_argument("--min-per-task", type=int, default=0, help="Optional minimum generated cases per task.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = _load_jsonl(args.dataset)
    manifest = _load_manifest(args.manifest)
    task_counts = Counter(str(case.get("task_type", "(missing-task)")) for case in cases)

    print("dataset:", args.dataset)
    print("case_count:", len(cases))
    print("task_count:", len(task_counts))
    print("should_clarify:", Counter(bool(case.get("should_clarify")) for case in cases))
    print("retrieval_evaluable:", Counter(bool(case.get("retrieval_evaluable")) for case in cases))
    print("\ncase_count_by_task:")
    for task_type, count in task_counts.most_common():
        print(f"  {task_type}: {count}")

    _print_manifest(manifest)

    issues: list[dict[str, str]] = []
    for case in cases:
        issues.extend(_case_issues(case, min_per_task=int(args.min_per_task), task_counts=task_counts))

    print("\nissue_count:", len(issues))
    for issue in issues[: max(0, int(args.max_issues))]:
        print(f"- {issue['case_id']}: {issue['code']} {issue.get('detail', '')}".rstrip())
    if len(issues) > int(args.max_issues):
        print(f"... {len(issues) - int(args.max_issues)} more issues omitted")

    _print_samples(cases, int(args.samples_per_task))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
