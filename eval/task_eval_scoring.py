"""Layered scoring for task-driven eval."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median
from typing import Any

try:
    from eval_core import (
        extract_urls,
        hit_rate_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )
except ImportError:  # package-style import fallback
    from .eval_core import (
        extract_urls,
        hit_rate_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )


TIMEOUT_MARKERS = ("timeout", "timed out", "deadline", "resource exhausted")
NEGATION_MARKERS = (" not ", " no ", " never ", "没有", "无", "并非", "不是")
TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
EMPTY_ACK_MARKERS = (
    "无结果",
    "没有找到",
    "未找到",
    "暂无",
    "no relevant",
    "no results",
    "not found",
)
RECOVERY_SUGGESTION_MARKERS = (
    "建议",
    "可尝试",
    "放宽",
    "扩大",
    "调整",
    "try",
    "consider",
    "broaden",
    "widen",
)
RCS_RECALL_W = 0.4
RCS_MRR_W = 0.3
RCS_NDCG_W = 0.3

ATTRIBUTION_CODES = (
    "INTENT_FAIL",
    "TOOL_PATH_FAIL",
    "TOOL_ARG_FAIL",
    "RETRIEVAL_FAIL",
    "ANALYSIS_UNSUPPORTED",
    "ANALYSIS_CONTRADICT",
    "SYSTEM_FAIL",
)


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def _mean(values: list[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * q) - 1)))
    return float(ordered[idx])


def _normalize_text(value: str) -> str:
    return " " + re.sub(r"\s+", " ", str(value or "").strip().lower()) + " "


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(value or ""))]


def _extract_numbers(value: str) -> set[str]:
    return {n for n in NUMBER_RE.findall(str(value or ""))}


def _scenario_from_task_type(task_type: str) -> str:
    parts = [part.strip().lower() for part in str(task_type or "").split(".") if part.strip()]
    if not parts:
        return ""
    return parts[-1]


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    norm = str(text or "").strip().lower()
    if not norm:
        return False
    return any(marker in norm for marker in markers)


def _path_tools(path: list[dict[str, Any]]) -> list[str]:
    return [str(step.get("tool", "")).strip() for step in path if str(step.get("tool", "")).strip()]


def _ordered_tool_matches(actual_tools: list[str], expected_tools: list[str]) -> int:
    if not expected_tools or not actual_tools:
        return 0
    cursor = 0
    for tool in actual_tools:
        if cursor < len(expected_tools) and tool == expected_tools[cursor]:
            cursor += 1
            if cursor == len(expected_tools):
                return cursor
    return cursor


def _path_match(actual_tools: list[str], expected_tools: list[str]) -> bool:
    if not expected_tools:
        return True
    return _ordered_tool_matches(actual_tools, expected_tools) == len(expected_tools)


def _align_expected_path_to_actual(
    expected_path: list[dict[str, Any]],
    actual_detailed_calls: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    matched = 0
    aligned: list[dict[str, Any]] = []
    cursor = 0

    for expected_step in expected_path:
        expected_tool = str(expected_step.get("tool", "")).strip()
        chosen: dict[str, Any] | None = None
        while cursor < len(actual_detailed_calls):
            actual_step = actual_detailed_calls[cursor]
            cursor += 1
            actual_tool = str(actual_step.get("tool", "")).strip()
            if expected_tool and actual_tool == expected_tool:
                chosen = actual_step if isinstance(actual_step, dict) else {}
                matched += 1
                break
        aligned.append(chosen or {})
    return matched, aligned


def _param_accuracy_for_aligned_path(
    expected_path: list[dict[str, Any]],
    aligned_actual_steps: list[dict[str, Any]],
) -> float:
    if not expected_path:
        return 1.0

    total = 0
    matched = 0
    for idx, expected_step in enumerate(expected_path):
        expected_tool = str(expected_step.get("tool", "")).strip()
        expected_args = expected_step.get("args", {})
        if not isinstance(expected_args, dict):
            expected_args = {}

        actual_step = aligned_actual_steps[idx] if idx < len(aligned_actual_steps) else {}
        actual_tool = str(actual_step.get("tool", "")).strip()
        actual_args = actual_step.get("args", {})
        if not isinstance(actual_args, dict):
            actual_args = {}

        if expected_tool and expected_tool != actual_tool:
            for _ in expected_args.keys():
                total += 1
            continue

        for key, expected_value in expected_args.items():
            total += 1
            actual_value = actual_args.get(key)
            if str(actual_value).strip().lower() == str(expected_value).strip().lower():
                matched += 1

    if total == 0:
        return 1.0
    return _safe_div(float(matched), float(total))


def _score_candidate_path(
    expected_path: list[dict[str, Any]],
    actual_tools: list[str],
    actual_detailed_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_tools = _path_tools(expected_path)
    if not expected_tools:
        return {
            "path_hit": 1.0,
            "coverage": 1.0,
            "param_accuracy": 1.0,
            "expected_tool_count": 0,
        }

    ordered_matches = _ordered_tool_matches(actual_tools, expected_tools)
    path_hit = 1.0 if ordered_matches == len(expected_tools) else 0.0
    coverage = _safe_div(float(ordered_matches), float(len(expected_tools)))

    if actual_detailed_calls:
        matched_steps, aligned_steps = _align_expected_path_to_actual(expected_path, actual_detailed_calls)
        if matched_steps > ordered_matches:
            # Keep name-level sequence score as source of truth, align only for args.
            matched_steps = ordered_matches
        param_accuracy = _param_accuracy_for_aligned_path(expected_path, aligned_steps)
    else:
        # No detailed args available: only score params when no explicit expected args.
        aligned_steps = [{"tool": tool, "args": {}} for tool in expected_tools]
        param_accuracy = _param_accuracy_for_aligned_path(expected_path, aligned_steps)

    return {
        "path_hit": path_hit,
        "coverage": coverage,
        "param_accuracy": param_accuracy,
        "expected_tool_count": len(expected_tools),
    }


def _best_path_match(
    expected_paths: list[list[dict[str, Any]]],
    actual_tools: list[str],
    actual_detailed_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    if not expected_paths:
        return {
            "best_path_index": -1,
            "path_hit": 0.0,
            "coverage": 0.0,
            "param_accuracy": 0.0,
        }

    best_index = 0
    best_score: tuple[float, float, float, float] | None = None
    best_metrics: dict[str, Any] | None = None

    for idx, path in enumerate(expected_paths):
        metrics = _score_candidate_path(path, actual_tools, actual_detailed_calls)
        # Priority: full path hit > higher coverage > higher param accuracy > shorter expected path.
        score = (
            float(metrics["path_hit"]),
            float(metrics["coverage"]),
            float(metrics["param_accuracy"]),
            -float(metrics["expected_tool_count"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_index = idx
            best_metrics = metrics

    if best_metrics is None:
        return {
            "best_path_index": -1,
            "path_hit": 0.0,
            "coverage": 0.0,
            "param_accuracy": 0.0,
        }
    return {
        "best_path_index": best_index,
        "path_hit": float(best_metrics["path_hit"]),
        "coverage": float(best_metrics["coverage"]),
        "param_accuracy": float(best_metrics["param_accuracy"]),
    }


def _claim_mentioned(answer: str, claim: str) -> bool:
    answer_tokens = set(_tokenize(answer))
    claim_tokens = [token for token in _tokenize(claim) if len(token) >= 3]
    if not claim_tokens:
        return False
    overlap = sum(1 for token in claim_tokens if token in answer_tokens)
    if overlap >= 2:
        return True
    return _normalize_text(claim).strip() in _normalize_text(answer)


def _claim_negated(answer: str, claim: str) -> bool:
    answer_norm = _normalize_text(answer)
    if not any(marker in answer_norm for marker in NEGATION_MARKERS):
        return False
    claim_tokens = [token for token in _tokenize(claim) if len(token) >= 4][:3]
    return any(f" {token} " in answer_norm for token in claim_tokens)


def _macro_f1(expected_labels: list[str], predicted_labels: list[str]) -> float:
    labels = sorted(set(expected_labels) | set(predicted_labels))
    if not labels:
        return 0.0
    f1_values: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for expected, predicted in zip(expected_labels, predicted_labels):
            if predicted == label and expected == label:
                tp += 1
            elif predicted == label and expected != label:
                fp += 1
            elif predicted != label and expected == label:
                fn += 1
        precision = _safe_div(float(tp), float(tp + fp))
        recall = _safe_div(float(tp), float(tp + fn))
        if precision + recall <= 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_values.append(f1)
    return _safe_div(sum(f1_values), len(f1_values))


def score_case(
    case: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_intent = str(case.get("intent_label", "")).strip()
    should_clarify = bool(case.get("should_clarify", False))
    required_tools = {str(item).strip() for item in case.get("required_tools", []) if str(item).strip()}
    forbidden_tools = {str(item).strip() for item in case.get("forbidden_tools", []) if str(item).strip()}
    expected_paths = case.get("expected_tool_paths", [])
    if not isinstance(expected_paths, list):
        expected_paths = []
    expected_paths = [path for path in expected_paths if isinstance(path, list)]

    retrieval_evaluable = bool(case.get("retrieval_evaluable", False))
    retrieval_gold_urls = [str(item).strip() for item in case.get("retrieval_gold_urls", []) if str(item).strip()]
    claims = case.get("verifiable_claims", [])
    if not isinstance(claims, list):
        claims = []
    doc_url_map = {
        str(doc.get("doc_id", "")).strip(): str(doc.get("url", "")).strip()
        for doc in case.get("input_news_pool", [])
        if isinstance(doc, dict)
    }

    run_rows: list[dict[str, Any]] = []
    expected_labels: list[str] = []
    predicted_labels: list[str] = []

    for run in runs:
        final_answer = str(run.get("final_answer", ""))
        tool_calls = [str(item).strip() for item in run.get("tool_calls", []) if str(item).strip()]
        actual_tools_set = set(tool_calls)
        actual_detailed_calls = run.get("tool_calls_detailed", [])
        if not isinstance(actual_detailed_calls, list):
            actual_detailed_calls = []
        retrieved_urls = [str(item).strip() for item in run.get("retrieved_urls", []) if str(item).strip()]
        citations = set([str(item).strip() for item in run.get("citations", []) if str(item).strip()])
        citations.update(extract_urls(final_answer))
        clarification_triggered = bool(run.get("clarification_triggered", False))
        error_text = str(run.get("error", "")).strip().lower()
        latency_ms = float(run.get("latency_ms", 0.0) or 0.0)

        predicted_intent = tool_calls[0] if tool_calls else "none"
        expected_labels.append(expected_intent)
        predicted_labels.append(predicted_intent)
        intent_hit = 1.0 if predicted_intent == expected_intent else 0.0
        clarify_hit = 1.0 if clarification_triggered == should_clarify else 0.0

        tool_precision = _safe_div(
            float(len(actual_tools_set.intersection(required_tools))),
            float(len(actual_tools_set) or 1),
        )
        tool_recall = _safe_div(
            float(len(actual_tools_set.intersection(required_tools))),
            float(len(required_tools) or 1),
        )
        best_path = _best_path_match(expected_paths, tool_calls, actual_detailed_calls)
        path_hit = float(best_path["path_hit"])
        forbidden_hit = 1.0 if actual_tools_set.intersection(forbidden_tools) else 0.0
        param_acc = float(best_path["param_accuracy"])

        if retrieval_evaluable:
            recall5 = recall_at_k(retrieved_urls, retrieval_gold_urls, 5)
            recall10 = recall_at_k(retrieved_urls, retrieval_gold_urls, 10)
            mrr5 = mrr_at_k(retrieved_urls, retrieval_gold_urls, 5)
            mrr10 = mrr_at_k(retrieved_urls, retrieval_gold_urls, 10)
            ndcg5 = ndcg_at_k(retrieved_urls, retrieval_gold_urls, 5)
            ndcg10 = ndcg_at_k(retrieved_urls, retrieval_gold_urls, 10)
            hit5 = hit_rate_at_k(retrieved_urls, retrieval_gold_urls, 5)
            hit10 = hit_rate_at_k(retrieved_urls, retrieval_gold_urls, 10)
            prec5 = precision_at_k(retrieved_urls, retrieval_gold_urls, 5)
            gold_hit = 1.0 if set(retrieved_urls).intersection(retrieval_gold_urls) else 0.0
            retrieval_rcs = (
                (RCS_RECALL_W * recall10)
                + (RCS_MRR_W * mrr10)
                + (RCS_NDCG_W * ndcg10)
            )
            depth_gain = max(0.0, float(hit10) - float(hit5))
        else:
            recall5 = recall10 = mrr5 = mrr10 = ndcg5 = ndcg10 = None
            hit5 = hit10 = prec5 = gold_hit = None
            retrieval_rcs = depth_gain = None

        supported_count = 0
        mentioned_count = 0
        contradiction_count = 0
        numeric_expected = set()
        for claim_row in claims:
            if not isinstance(claim_row, dict):
                continue
            claim_text = str(claim_row.get("claim", "")).strip()
            if not claim_text:
                continue
            evidence_doc_ids = [
                str(item).strip()
                for item in claim_row.get("evidence_doc_ids", [])
                if str(item).strip()
            ]
            evidence_urls = {
                doc_url_map.get(doc_id, "")
                for doc_id in evidence_doc_ids
                if doc_url_map.get(doc_id, "")
            }

            mentioned = _claim_mentioned(final_answer, claim_text)
            if mentioned:
                mentioned_count += 1
                citation_hit = bool(evidence_urls.intersection(citations)) if evidence_urls else True
                if citation_hit:
                    supported_count += 1
                if _claim_negated(final_answer, claim_text):
                    contradiction_count += 1

            claim_type = str(claim_row.get("claim_type", "")).strip().lower()
            if claim_type == "number":
                numeric_expected.update(_extract_numbers(claim_text))

        claim_count = len([row for row in claims if isinstance(row, dict)])
        if claim_count > 0:
            claim_support_rate = _safe_div(float(supported_count), float(claim_count))
            unsupported_claim_rate = _safe_div(float(max(mentioned_count - supported_count, 0)), float(claim_count))
            contradiction_rate = _safe_div(float(contradiction_count), float(claim_count))
        else:
            claim_support_rate = None
            unsupported_claim_rate = None
            contradiction_rate = None

        answer_numbers = _extract_numbers(final_answer)
        if numeric_expected:
            numeric_consistency = _safe_div(
                float(len(numeric_expected.intersection(answer_numbers))),
                float(len(numeric_expected)),
            )
        else:
            numeric_consistency = None

        is_error = 1.0 if error_text else 0.0
        timeout_hit = 1.0 if any(marker in error_text for marker in TIMEOUT_MARKERS) else 0.0
        final_status = str(run.get("final_status", "")).strip().lower()
        fallback = 1.0 if final_status in {"blocked", "clarification_required"} or not tool_calls else 0.0

        run_rows.append(
            {
                "intent_hit": intent_hit,
                "clarification_hit": clarify_hit,
                "predicted_intent_label": predicted_intent,
                "tool_set_precision": tool_precision,
                "tool_set_recall": tool_recall,
                "path_hit": path_hit,
                "path_coverage": float(best_path["coverage"]),
                "best_path_index": int(best_path["best_path_index"]),
                "forbidden_tool_hit": forbidden_hit,
                "param_accuracy": param_acc,
                "recall_at_5": recall5,
                "recall_at_10": recall10,
                "mrr_at_5": mrr5,
                "mrr_at_10": mrr10,
                "ndcg_at_5": ndcg5,
                "ndcg_at_10": ndcg10,
                "hit_rate_at_5": hit5,
                "hit_rate_at_10": hit10,
                "precision_at_5": prec5,
                "gold_hit": gold_hit,
                "retrieval_rcs": retrieval_rcs,
                "depth_gain": depth_gain,
                "claim_support_rate": claim_support_rate,
                "unsupported_claim_rate": unsupported_claim_rate,
                "contradiction_rate": contradiction_rate,
                "numeric_consistency": numeric_consistency,
                "error_hit": is_error,
                "timeout_hit": timeout_hit,
                "fallback_hit": fallback,
                "latency_ms": latency_ms,
                # Generation-layer judge scores (populated externally via --enable-llm-judge)
                "faithfulness_score": run.get("faithfulness_score"),
                "answer_relevancy_score": run.get("answer_relevancy_score"),
            }
        )

    intent_top1 = _mean([row["intent_hit"] for row in run_rows]) or 0.0
    clarification_accuracy = _mean([row["clarification_hit"] for row in run_rows]) or 0.0
    intent_macro_f1 = _macro_f1(expected_labels, predicted_labels)

    latencies = [float(row["latency_ms"]) for row in run_rows if float(row["latency_ms"]) > 0]

    layer = {
        "intent": {
            "top1_accuracy": intent_top1,
            "macro_f1": intent_macro_f1,
            "clarification_accuracy": clarification_accuracy,
        },
        "tool": {
            "tool_set_precision": _mean([row["tool_set_precision"] for row in run_rows]) or 0.0,
            "tool_set_recall": _mean([row["tool_set_recall"] for row in run_rows]) or 0.0,
            "acceptable_path_hit_rate": _mean([row["path_hit"] for row in run_rows]) or 0.0,
            "forbidden_tool_rate": _mean([row["forbidden_tool_hit"] for row in run_rows]) or 0.0,
            "param_accuracy": _mean([row["param_accuracy"] for row in run_rows]) or 0.0,
        },
        "retrieval": {
            "evaluable": retrieval_evaluable,
            "recall_at_5": _mean([row["recall_at_5"] for row in run_rows]),
            "recall_at_10": _mean([row["recall_at_10"] for row in run_rows]),
            "mrr_at_5": _mean([row["mrr_at_5"] for row in run_rows]),
            "mrr_at_10": _mean([row["mrr_at_10"] for row in run_rows]),
            "ndcg_at_5": _mean([row["ndcg_at_5"] for row in run_rows]),
            "ndcg_at_10": _mean([row["ndcg_at_10"] for row in run_rows]),
            "hit_rate_at_5": _mean([row["hit_rate_at_5"] for row in run_rows]),
            "hit_rate_at_10": _mean([row["hit_rate_at_10"] for row in run_rows]),
            "precision_at_5": _mean([row["precision_at_5"] for row in run_rows]),
            "gold_hit_rate": _mean([row["gold_hit"] for row in run_rows]),
            "rcs": _mean([row["retrieval_rcs"] for row in run_rows]),
            "depth_gain": _mean([row["depth_gain"] for row in run_rows]),
        },
        "analysis": {
            "claim_support_rate": _mean([row["claim_support_rate"] for row in run_rows]),
            "unsupported_claim_rate": _mean([row["unsupported_claim_rate"] for row in run_rows]),
            "contradiction_rate": _mean([row["contradiction_rate"] for row in run_rows]),
            "numeric_consistency": _mean([row["numeric_consistency"] for row in run_rows]),
        },
        "generation": {
            "faithfulness_score": _mean([row["faithfulness_score"] for row in run_rows]),
            "answer_relevancy_score": _mean([row["answer_relevancy_score"] for row in run_rows]),
        },
        "system": {
            "error_rate": _mean([row["error_hit"] for row in run_rows]) or 0.0,
            "timeout_rate": _mean([row["timeout_hit"] for row in run_rows]) or 0.0,
            "fallback_rate": _mean([row["fallback_hit"] for row in run_rows]) or 0.0,
            "latency_p50_ms": median(latencies) if latencies else 0.0,
            "latency_p95_ms": _quantile(latencies, 0.95),
        },
        "run_count": len(run_rows),
    }
    layer["attribution"] = derive_case_attribution(layer)
    return layer


def derive_case_attribution(layer: dict[str, Any]) -> dict[str, Any]:
    intent = layer.get("intent", {})
    tool = layer.get("tool", {})
    retrieval = layer.get("retrieval", {})
    analysis = layer.get("analysis", {})
    system = layer.get("system", {})

    if float(intent.get("top1_accuracy", 0.0) or 0.0) < 1.0:
        return {"code": "INTENT_FAIL", "layer": "intent"}
    if float(tool.get("acceptable_path_hit_rate", 0.0) or 0.0) < 1.0:
        return {"code": "TOOL_PATH_FAIL", "layer": "tool"}
    if float(tool.get("param_accuracy", 0.0) or 0.0) < 1.0:
        return {"code": "TOOL_ARG_FAIL", "layer": "tool"}
    if bool(retrieval.get("evaluable")) and float(retrieval.get("gold_hit_rate", 0.0) or 0.0) <= 0.0:
        return {"code": "RETRIEVAL_FAIL", "layer": "retrieval"}

    contradiction_rate = analysis.get("contradiction_rate")
    if contradiction_rate is not None and float(contradiction_rate) > 0.0:
        return {"code": "ANALYSIS_CONTRADICT", "layer": "analysis"}

    unsupported_rate = analysis.get("unsupported_claim_rate")
    if unsupported_rate is not None and float(unsupported_rate) > 0.0:
        return {"code": "ANALYSIS_UNSUPPORTED", "layer": "analysis"}

    if float(system.get("error_rate", 0.0) or 0.0) > 0.0:
        return {"code": "SYSTEM_FAIL", "layer": "system"}
    return {"code": "PASS", "layer": "pass"}


def aggregate_layer_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_rows:
        return {
            "intent": {"top1_accuracy": 0.0, "macro_f1": 0.0, "clarification_accuracy": 0.0},
            "tool": {
                "tool_set_precision": 0.0,
                "tool_set_recall": 0.0,
                "acceptable_path_hit_rate": 0.0,
                "forbidden_tool_rate": 0.0,
                "param_accuracy": 0.0,
            },
            "retrieval": {
                "case_count": 0,
                "recall_at_5": None,
                "recall_at_10": None,
                "mrr_at_10": None,
                "ndcg_at_10": None,
                "hit_rate_at_5": None,
                "hit_rate_at_10": None,
                "gold_hit_rate": None,
                "rcs": None,
                "depth_gain": None,
            },
            "analysis": {
                "claim_support_rate": None,
                "unsupported_claim_rate": None,
                "contradiction_rate": None,
                "numeric_consistency": None,
            },
            "system": {
                "error_rate": 0.0,
                "timeout_rate": 0.0,
                "fallback_rate": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
            },
            "empty_risk": {
                "case_count": 0,
                "run_count": 0,
                "empty_response_accuracy": None,
                "empty_hallucination_rate": None,
                "empty_recovery_suggestion_rate": None,
            },
            "attribution_breakdown": {},
        }

    intent_expected: list[str] = []
    intent_predicted: list[str] = []
    intent_top1_vals: list[float] = []
    clarify_vals: list[float] = []

    tool_precision_vals: list[float] = []
    tool_recall_vals: list[float] = []
    path_hit_vals: list[float] = []
    forbidden_vals: list[float] = []
    param_vals: list[float] = []

    retrieval_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    empty_case_count = 0
    empty_run_count = 0
    empty_ack_count = 0
    empty_hallucination_count = 0
    empty_recovery_count = 0
    latencies: list[float] = []
    attribution_count: dict[str, int] = defaultdict(int)

    for row in case_rows:
        case = row.get("case", {})
        runs = row.get("runs", [])
        layers = row.get("layers", {})

        intent = layers.get("intent", {})
        tool = layers.get("tool", {})
        retrieval = layers.get("retrieval", {})
        analysis = layers.get("analysis", {})
        system = layers.get("system", {})
        attribution = row.get("attribution", {})

        intent_top1_vals.append(float(intent.get("top1_accuracy", 0.0) or 0.0))
        clarify_vals.append(float(intent.get("clarification_accuracy", 0.0) or 0.0))
        for run in runs:
            expected_intent = str(case.get("intent_label", "")).strip()
            predicted = str(run.get("predicted_intent_label", "")).strip()
            if expected_intent and predicted:
                intent_expected.append(expected_intent)
                intent_predicted.append(predicted)

        tool_precision_vals.append(float(tool.get("tool_set_precision", 0.0) or 0.0))
        tool_recall_vals.append(float(tool.get("tool_set_recall", 0.0) or 0.0))
        path_hit_vals.append(float(tool.get("acceptable_path_hit_rate", 0.0) or 0.0))
        forbidden_vals.append(float(tool.get("forbidden_tool_rate", 0.0) or 0.0))
        param_vals.append(float(tool.get("param_accuracy", 0.0) or 0.0))

        if retrieval.get("evaluable"):
            retrieval_rows.append(retrieval)
        analysis_rows.append(analysis)
        system_rows.append(system)
        scenario = _scenario_from_task_type(str(case.get("task_type", "")))
        if scenario == "empty":
            empty_case_count += 1
            for run in runs:
                final_answer = str(run.get("final_answer", ""))
                final_status = str(run.get("final_status", "")).strip().lower()
                error_text = str(run.get("error", "")).strip()
                empty_run_count += 1
                empty_ack = _contains_any_marker(final_answer, EMPTY_ACK_MARKERS)
                if (
                    not empty_ack
                    and final_status in {"clarification_required", "blocked"}
                    and not error_text
                ):
                    empty_ack = True
                if empty_ack:
                    empty_ack_count += 1
                if _contains_any_marker(final_answer, RECOVERY_SUGGESTION_MARKERS):
                    empty_recovery_count += 1
                if (not empty_ack) and (not error_text):
                    empty_hallucination_count += 1
        if float(system.get("latency_p50_ms", 0.0) or 0.0) > 0:
            latencies.append(float(system.get("latency_p50_ms", 0.0)))
        if float(system.get("latency_p95_ms", 0.0) or 0.0) > 0:
            latencies.append(float(system.get("latency_p95_ms", 0.0)))

        code = str(attribution.get("code", "PASS"))
        attribution_count[code] += 1

    summary = {
        "intent": {
            "top1_accuracy": _safe_div(sum(intent_top1_vals), len(intent_top1_vals)),
            "macro_f1": _macro_f1(intent_expected, intent_predicted),
            "clarification_accuracy": _safe_div(sum(clarify_vals), len(clarify_vals)),
        },
        "tool": {
            "tool_set_precision": _safe_div(sum(tool_precision_vals), len(tool_precision_vals)),
            "tool_set_recall": _safe_div(sum(tool_recall_vals), len(tool_recall_vals)),
            "acceptable_path_hit_rate": _safe_div(sum(path_hit_vals), len(path_hit_vals)),
            "forbidden_tool_rate": _safe_div(sum(forbidden_vals), len(forbidden_vals)),
            "param_accuracy": _safe_div(sum(param_vals), len(param_vals)),
        },
        "retrieval": {
            "case_count": len(retrieval_rows),
            "recall_at_5": _mean([row.get("recall_at_5") for row in retrieval_rows]),
            "recall_at_10": _mean([row.get("recall_at_10") for row in retrieval_rows]),
            "mrr_at_10": _mean([row.get("mrr_at_10") for row in retrieval_rows]),
            "ndcg_at_10": _mean([row.get("ndcg_at_10") for row in retrieval_rows]),
            "hit_rate_at_5": _mean([row.get("hit_rate_at_5") for row in retrieval_rows]),
            "hit_rate_at_10": _mean([row.get("hit_rate_at_10") for row in retrieval_rows]),
            "gold_hit_rate": _mean([row.get("gold_hit_rate") for row in retrieval_rows]),
            "rcs": _mean([row.get("rcs") for row in retrieval_rows]),
            "depth_gain": _mean([row.get("depth_gain") for row in retrieval_rows]),
        },
        "analysis": {
            "claim_support_rate": _mean([row.get("claim_support_rate") for row in analysis_rows]),
            "unsupported_claim_rate": _mean([row.get("unsupported_claim_rate") for row in analysis_rows]),
            "contradiction_rate": _mean([row.get("contradiction_rate") for row in analysis_rows]),
            "numeric_consistency": _mean([row.get("numeric_consistency") for row in analysis_rows]),
        },
        "system": {
            "error_rate": _mean([row.get("error_rate") for row in system_rows]) or 0.0,
            "timeout_rate": _mean([row.get("timeout_rate") for row in system_rows]) or 0.0,
            "fallback_rate": _mean([row.get("fallback_rate") for row in system_rows]) or 0.0,
            "latency_p50_ms": median(latencies) if latencies else 0.0,
            "latency_p95_ms": _quantile(latencies, 0.95),
        },
        "empty_risk": {
            "case_count": empty_case_count,
            "run_count": empty_run_count,
            "empty_response_accuracy": _safe_div(float(empty_ack_count), float(empty_run_count or 1))
            if empty_run_count > 0
            else None,
            "empty_hallucination_rate": _safe_div(float(empty_hallucination_count), float(empty_run_count or 1))
            if empty_run_count > 0
            else None,
            "empty_recovery_suggestion_rate": _safe_div(float(empty_recovery_count), float(empty_run_count or 1))
            if empty_run_count > 0
            else None,
        },
        "attribution_breakdown": dict(sorted(attribution_count.items(), key=lambda x: x[0])),
    }
    return summary
