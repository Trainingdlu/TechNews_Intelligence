"""Core metrics and quality gates for agent stability evaluation."""

from __future__ import annotations

import itertools
import re
from difflib import SequenceMatcher
from typing import Any, Iterable

URL_RE = re.compile(r"https?://[^\s)\]]+")


def normalize_text(text: str) -> str:
    """Normalize text for rough stability comparison."""
    if not text:
        return ""
    lowered = text.strip().lower()
    # Normalize whitespace and markdown emphasis noise.
    lowered = lowered.replace("**", "")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def extract_urls(text: str) -> list[str]:
    """Extract unique URLs in appearance order."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in URL_RE.findall(text):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def average_pairwise_similarity(texts: Iterable[str]) -> float:
    """Average pairwise text similarity in [0, 1]."""
    items = [normalize_text(t) for t in texts]
    if len(items) < 2:
        return 1.0

    sims: list[float] = []
    for a, b in itertools.combinations(items, 2):
        sims.append(SequenceMatcher(None, a, b).ratio())
    return (sum(sims) / len(sims)) if sims else 1.0


def evaluate_case_outputs(
    outputs: list[str],
    min_urls: int = 0,
    must_contain: list[str] | None = None,
    expected_facts: list[str] | None = None,
    required_tools: list[str] | None = None,
    must_not_contain: list[str] | None = None,
    run_tool_calls: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Compute per-case stability and compliance metrics."""
    run_count = len(outputs)
    normalized = [normalize_text(x) for x in outputs]
    unique_count = len(set(normalized))
    unique_ratio = (unique_count / run_count) if run_count else 0.0

    url_counts = [len(extract_urls(x)) for x in outputs]
    min_urls = max(0, int(min_urls))
    if run_count == 0:
        min_url_hits = 0
    elif min_urls == 0:
        min_url_hits = run_count
    else:
        min_url_hits = sum(1 for n in url_counts if n >= min_urls)
    min_url_hit_rate = (min_url_hits / run_count) if run_count else 0.0

    phrases = [normalize_text(x) for x in (must_contain or []) if str(x).strip()]
    if run_count == 0:
        phrase_hits = 0
    elif not phrases:
        phrase_hits = run_count
    else:
        phrase_hits = sum(
            1 for raw in normalized if all(p in raw for p in phrases)
        )
    phrase_hit_rate = (phrase_hits / run_count) if run_count else 0.0

    expected = [normalize_text(x) for x in (expected_facts or []) if str(x).strip()]
    if run_count == 0:
        fact_hit_runs = 0.0
        fact_hit_rate = 0.0
    elif not expected:
        fact_hit_runs = float(run_count)
        fact_hit_rate = 1.0
    else:
        hit_scores: list[float] = []
        for raw in normalized:
            matched = sum(1 for fact in expected if fact in raw)
            hit_scores.append(matched / len(expected))
        fact_hit_rate = sum(hit_scores) / len(hit_scores)
        fact_hit_runs = float(sum(1 for score in hit_scores if score >= 0.999999))

    forbidden = [normalize_text(x) for x in (must_not_contain or []) if str(x).strip()]
    if run_count == 0:
        forbidden_violations = 0
        forbidden_claim_rate = 0.0
    elif not forbidden:
        forbidden_violations = 0
        forbidden_claim_rate = 0.0
    else:
        forbidden_violations = sum(
            1 for raw in normalized if any(token in raw for token in forbidden)
        )
        forbidden_claim_rate = forbidden_violations / run_count

    required = {normalize_text(x) for x in (required_tools or []) if str(x).strip()}
    tool_call_runs = run_tool_calls or [[] for _ in outputs]
    if len(tool_call_runs) < run_count:
        tool_call_runs = tool_call_runs + ([[]] * (run_count - len(tool_call_runs)))

    if run_count == 0:
        runs_with_required_tools = 0
        tool_path_hit_rate = 0.0
    elif not required:
        runs_with_required_tools = run_count
        tool_path_hit_rate = 1.0
    else:
        runs_with_required_tools = 0
        for tools in tool_call_runs[:run_count]:
            used = {
                normalize_text(x)
                for x in (tools or [])
                if str(x).strip()
            }
            if required.issubset(used):
                runs_with_required_tools += 1
        tool_path_hit_rate = runs_with_required_tools / run_count

    error_count = sum(1 for x in outputs if "[EVAL_ERROR]" in x)
    error_rate = (error_count / run_count) if run_count else 0.0

    return {
        "run_count": run_count,
        "unique_output_count": unique_count,
        "unique_response_ratio": unique_ratio,
        "avg_pairwise_similarity": average_pairwise_similarity(outputs),
        "avg_url_count": (sum(url_counts) / run_count) if run_count else 0.0,
        "runs_with_min_urls": min_url_hits,
        "min_url_hit_rate": min_url_hit_rate,
        "phrase_hit_rate": phrase_hit_rate,
        "fact_hit_runs": fact_hit_runs,
        "fact_hit_rate": fact_hit_rate,
        "runs_with_required_tools": runs_with_required_tools,
        "tool_path_hit_rate": tool_path_hit_rate,
        "forbidden_claim_violations": forbidden_violations,
        "forbidden_claim_rate": forbidden_claim_rate,
        "error_count": error_count,
        "error_rate": error_rate,
    }


def summarize_case_results(case_results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics over all evaluated cases."""
    if not case_results:
        return {
            "case_count": 0,
            "run_count_total": 0,
            "avg_pairwise_similarity": 0.0,
            "avg_unique_response_ratio": 0.0,
            "avg_min_url_hit_rate": 0.0,
            "avg_phrase_hit_rate": 0.0,
            "avg_fact_hit_rate": 0.0,
            "avg_tool_path_hit_rate": 0.0,
            "avg_forbidden_claim_rate": 0.0,
            "avg_error_rate": 0.0,
        }

    metrics = [item.get("metrics", item) for item in case_results]
    case_count = len(metrics)
    run_count_total = int(sum(float(m.get("run_count", 0)) for m in metrics))

    def _avg(key: str) -> float:
        vals = [float(m.get(key, 0.0)) for m in metrics]
        return (sum(vals) / len(vals)) if vals else 0.0

    return {
        "case_count": case_count,
        "run_count_total": run_count_total,
        "avg_pairwise_similarity": _avg("avg_pairwise_similarity"),
        "avg_unique_response_ratio": _avg("unique_response_ratio"),
        "avg_min_url_hit_rate": _avg("min_url_hit_rate"),
        "avg_phrase_hit_rate": _avg("phrase_hit_rate"),
        "avg_fact_hit_rate": _avg("fact_hit_rate"),
        "avg_tool_path_hit_rate": _avg("tool_path_hit_rate"),
        "avg_forbidden_claim_rate": _avg("forbidden_claim_rate"),
        "avg_error_rate": _avg("error_rate"),
    }


DEFAULT_BASELINE_METRIC_SPECS: list[tuple[str, str]] = [
    ("summary.avg_pairwise_similarity", "higher_better"),
    ("summary.avg_unique_response_ratio", "lower_better"),
    ("summary.avg_min_url_hit_rate", "higher_better"),
    ("summary.avg_phrase_hit_rate", "higher_better"),
    ("summary.avg_fact_hit_rate", "higher_better"),
    ("summary.avg_tool_path_hit_rate", "higher_better"),
    ("summary.avg_forbidden_claim_rate", "lower_better"),
    ("summary.avg_error_rate", "lower_better"),
    ("route_metrics.react_success_rate", "higher_better"),
    ("route_metrics.react_error_rate", "lower_better"),
    ("route_metrics.react_recursion_limit_rate", "lower_better"),
]


def _get_nested(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def build_baseline_comparison(
    current_report: dict[str, Any],
    baseline_report: dict[str, Any],
    metric_specs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build baseline comparison table between current and baseline reports."""
    specs = metric_specs or DEFAULT_BASELINE_METRIC_SPECS
    items: list[dict[str, Any]] = []
    improved = 0
    regressed = 0
    unchanged = 0
    missing = 0
    eps = 1e-12

    for path, direction in specs:
        current_value = _get_nested(current_report, path)
        baseline_value = _get_nested(baseline_report, path)
        if current_value is None or baseline_value is None:
            status = "missing"
            delta = None
            missing += 1
        else:
            c = float(current_value)
            b = float(baseline_value)
            delta = c - b
            if abs(delta) <= eps:
                status = "unchanged"
                unchanged += 1
            elif direction == "higher_better":
                if delta > 0:
                    status = "improved"
                    improved += 1
                else:
                    status = "regressed"
                    regressed += 1
            else:
                if delta < 0:
                    status = "improved"
                    improved += 1
                else:
                    status = "regressed"
                    regressed += 1

        items.append(
            {
                "metric": path,
                "direction": direction,
                "baseline": baseline_value,
                "current": current_value,
                "delta": delta,
                "status": status,
            }
        )

    return {
        "baseline_generated_at_utc": baseline_report.get("generated_at_utc"),
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_count": unchanged,
        "missing_count": missing,
        "items": items,
    }


def evaluate_quality_gates(
    current_report: dict[str, Any],
    gates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate threshold gates against a report."""
    failures: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []

    for gate in gates:
        name = str(gate.get("name", "gate"))
        path = str(gate.get("metric_path", ""))
        op = str(gate.get("op", "max"))
        threshold = float(gate.get("threshold", 0.0))
        value_raw = _get_nested(current_report, path)

        if value_raw is None:
            failures.append(
                {
                    "name": name,
                    "metric_path": path,
                    "reason": "metric_missing",
                }
            )
            continue

        value = float(value_raw)
        if op == "min":
            ok = value >= threshold
        elif op == "max":
            ok = value <= threshold
        else:
            failures.append(
                {
                    "name": name,
                    "metric_path": path,
                    "reason": "invalid_operator",
                    "op": op,
                }
            )
            continue

        item = {
            "name": name,
            "metric_path": path,
            "op": op,
            "threshold": threshold,
            "value": value,
        }
        if ok:
            passes.append(item)
        else:
            failures.append(item)

    return {
        "total": len(gates),
        "passed_count": len(passes),
        "failed_count": len(failures),
        "passed": passes,
        "failed": failures,
        "ok": len(failures) == 0,
    }
