"""Evaluation release gatekeeper.

This module converts eval outputs into release-gate decisions with three levels:
PASS / SOFT_FAIL / HARD_FAIL.

Exit codes:
- 0: PASS
- 2: SOFT_FAIL
- 3: HARD_FAIL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

try:
    from common import read_json_object, to_float, to_int
except ImportError:  # package-style import fallback
    from .common import read_json_object, to_float, to_int


PASS = "PASS"
SOFT_FAIL = "SOFT_FAIL"
HARD_FAIL = "HARD_FAIL"
MISSING = "missing"

EXIT_CODE_BY_STATUS = {
    PASS: 0,
    SOFT_FAIL: 2,
    HARD_FAIL: 3,
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    layer: str
    direction: str
    bounded_01: bool = True


@dataclass
class MetricObservation:
    metric: str
    layer: str
    direction: str
    value: float | None
    n: int | None
    samples: list[float]
    source: str
    ci_95: dict[str, float | str] | None = None


@dataclass(frozen=True)
class RuleSpec:
    name: str
    metric: str
    layer: str
    direction: str
    min_value: float | None
    max_value: float | None
    n_min: int | None
    ci_lower_bound_min: float | None
    severity: str
    missing_severity: str
    n_severity: str
    ci_severity: str
    action: str
    enabled: bool


METRIC_SPECS: dict[str, MetricSpec] = {
    "avg_recall_at_10": MetricSpec("avg_recall_at_10", "retrieval", "higher_better", bounded_01=True),
    "avg_mrr_at_10": MetricSpec("avg_mrr_at_10", "retrieval", "higher_better", bounded_01=True),
    "avg_composite": MetricSpec("avg_composite", "judge", "higher_better", bounded_01=False),
}


_METRIC_ALIAS_TOKEN_MAP: dict[str, str] = {
    # retrieval
    "avg_recall_at_10": "avg_recall_at_10",
    "recall_at_10": "avg_recall_at_10",
    "recall_10": "avg_recall_at_10",
    "recall10": "avg_recall_at_10",
    "avg_mrr_at_10": "avg_mrr_at_10",
    "mrr_at_10": "avg_mrr_at_10",
    "mrr_10": "avg_mrr_at_10",
    "mrr10": "avg_mrr_at_10",
    "mrr": "avg_mrr_at_10",
    # judge
    "avg_composite": "avg_composite",
    "composite": "avg_composite",
    "composite_score": "avg_composite",
    "judge_composite": "avg_composite",
}


def _normalize_metric_token(name: str) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("@", "_at_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_metric_name(name: str) -> str:
    token = _normalize_metric_token(name)
    if token in _METRIC_ALIAS_TOKEN_MAP:
        return _METRIC_ALIAS_TOKEN_MAP[token]
    return token


def _metric_spec(metric: str) -> MetricSpec:
    normalized = normalize_metric_name(metric)
    spec = METRIC_SPECS.get(normalized)
    if spec is not None:
        return spec
    return MetricSpec(
        name=normalized,
        layer="unknown",
        direction="higher_better",
        bounded_01=False,
    )


def _to_float(value: Any) -> float | None:
    return to_float(value, parse_string=True, missing_token=MISSING)


def _to_int(value: Any) -> int | None:
    return to_int(value, parse_string=True, missing_token=MISSING)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    items = sorted(values)
    pos = (len(items) - 1) * (pct / 100.0)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(items[low])
    frac = pos - low
    return float(items[low] * (1.0 - frac) + items[high] * frac)


def _round_or_missing(value: float | None, digits: int = 6) -> float | str:
    if value is None or not math.isfinite(value):
        return MISSING
    return round(value, digits)


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(path, encoding="utf-8-sig")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        return payload

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Failed to parse gate config. Install PyYAML or use JSON syntax in eval/config/gates.yaml."
        ) from exc

    parsed = yaml.safe_load(raw)
    if not parsed:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return parsed


def _extract_case_metric_samples(report: dict[str, Any] | None, key: str) -> list[float]:
    if not isinstance(report, dict):
        return []
    samples: list[float] = []
    for case in report.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        metrics = case.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        value = _to_float(metrics.get(key))
        if value is None:
            continue
        samples.append(value)
    return samples


def _extract_judge_samples(report: dict[str, Any] | None) -> list[float]:
    if not isinstance(report, dict):
        return []

    rows: Any = report.get("rows")
    if not isinstance(rows, list):
        rows = report.get("results")
    if not isinstance(rows, list):
        rows = report.get("items")
    if not isinstance(rows, list):
        return []

    samples: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidates = [
            row.get("avg_composite"),
            row.get("composite"),
            row.get("composite_score"),
            row.get("score"),
            (row.get("scores") or {}).get("avg_composite") if isinstance(row.get("scores"), dict) else None,
            (row.get("scores") or {}).get("composite") if isinstance(row.get("scores"), dict) else None,
        ]
        parsed = next((x for x in (_to_float(item) for item in candidates) if x is not None), None)
        if parsed is not None:
            samples.append(parsed)
    return samples

def _observe_from_run_eval(run_eval: dict[str, Any], source: str) -> dict[str, MetricObservation]:
    out: dict[str, MetricObservation] = {}

    retrieval_n = _to_int((run_eval.get("summary") or {}).get("retrieval_case_count"))

    def _build(metric_name: str, summary_key: str, case_key: str) -> None:
        metric = normalize_metric_name(metric_name)
        spec = _metric_spec(metric)
        summary = run_eval.get("summary", {})
        value = _to_float(summary.get(summary_key)) if isinstance(summary, dict) else None
        samples = _extract_case_metric_samples(run_eval, case_key)
        n = retrieval_n if retrieval_n is not None else (len(samples) if samples else None)
        if value is None:
            value = _mean(samples)
        out[metric] = MetricObservation(
            metric=metric,
            layer=spec.layer,
            direction=spec.direction,
            value=value,
            n=n,
            samples=samples,
            source=source,
        )

    _build("avg_recall_at_10", "avg_recall_at_10", "recall_at_10")
    _build("avg_mrr_at_10", "avg_mrr_at_10", "mrr_at_10")
    return out


def _observe_from_judge(judge: dict[str, Any], source: str) -> dict[str, MetricObservation]:
    samples = _extract_judge_samples(judge)
    summary = judge.get("summary", {})
    value = None
    if isinstance(summary, dict):
        value = _to_float(summary.get("avg_composite"))
        if value is None:
            value = _to_float(summary.get("composite"))
    if value is None:
        value = _to_float(judge.get("avg_composite"))
    if value is None:
        value = _to_float(judge.get("composite"))
    if value is None:
        value = _mean(samples)

    n = len(samples) if samples else _to_int(judge.get("row_count"))
    spec = _metric_spec("avg_composite")
    return {
        "avg_composite": MetricObservation(
            metric="avg_composite",
            layer=spec.layer,
            direction=spec.direction,
            value=value,
            n=n,
            samples=samples,
            source=source,
        )
    }


def _observe_from_leaderboard_group(group: dict[str, Any], source: str) -> dict[str, MetricObservation]:
    out: dict[str, MetricObservation] = {}
    metrics = group.get("metrics", {})
    if not isinstance(metrics, dict):
        return out

    for metric_name, record in metrics.items():
        if not isinstance(record, dict):
            continue
        normalized = normalize_metric_name(metric_name)
        spec = _metric_spec(normalized)
        sample_n = record.get("sample_n", {})
        n = None
        if isinstance(sample_n, dict):
            n = _to_int(sample_n.get("current"))
        if n is None:
            n = _to_int(record.get("n"))
        out[normalized] = MetricObservation(
            metric=normalized,
            layer=str(record.get("layer", spec.layer)),
            direction=str(record.get("direction", spec.direction)),
            value=_to_float(record.get("current")),
            n=n,
            samples=[],
            source=source,
        )

    return out


def _observation_quality(obs: MetricObservation) -> tuple[int, int, int]:
    value_score = 1 if obs.value is not None else 0
    n_score = 1 if obs.n is not None else 0
    sample_score = len(obs.samples)
    return (value_score, n_score, sample_score)


def _merge_observation(current: MetricObservation, incoming: MetricObservation) -> MetricObservation:
    if _observation_quality(incoming) > _observation_quality(current):
        return incoming
    return current


def merge_metric_maps(*maps: dict[str, MetricObservation]) -> dict[str, MetricObservation]:
    merged: dict[str, MetricObservation] = {}
    for metric_map in maps:
        for metric, obs in metric_map.items():
            if metric not in merged:
                merged[metric] = obs
                continue
            merged[metric] = _merge_observation(merged[metric], obs)
    return merged


def _stable_metric_seed(base_seed: int, metric: str) -> int:
    digest = hashlib.sha256(metric.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return int(base_seed) + offset


def _bootstrap_mean_ci(
    samples: list[float],
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if len(samples) < 2:
        return None
    rounds = max(200, int(bootstrap_samples))
    n = len(samples)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(rounds):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)

    alpha = 1.0 - confidence
    lower = _percentile(means, (alpha / 2.0) * 100.0)
    upper = _percentile(means, (1.0 - alpha / 2.0) * 100.0)
    if lower is None or upper is None:
        return None
    return (lower, upper)


def _normal_rate_ci(value: float, n: int, confidence: float) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = max(0.0, min(1.0, value))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    se = math.sqrt((p * (1.0 - p)) / n)
    margin = z * se
    return (p - margin, p + margin)


def compute_metric_ci(
    observation: MetricObservation,
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | str]:
    metric_spec = _metric_spec(observation.metric)

    ci_bootstrap = _bootstrap_mean_ci(
        observation.samples,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=_stable_metric_seed(seed, observation.metric),
    )
    if ci_bootstrap is not None:
        return {
            "lower": _round_or_missing(ci_bootstrap[0]),
            "upper": _round_or_missing(ci_bootstrap[1]),
            "method": "bootstrap_mean",
        }

    if observation.value is not None and observation.n is not None and metric_spec.bounded_01:
        ci_normal = _normal_rate_ci(observation.value, observation.n, confidence)
        if ci_normal is not None:
            return {
                "lower": _round_or_missing(ci_normal[0]),
                "upper": _round_or_missing(ci_normal[1]),
                "method": "normal_approx",
            }

    return {"lower": MISSING, "upper": MISSING, "method": MISSING}


def _normalize_severity(value: str | None, default: str = HARD_FAIL) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"hard", "hard_fail", "hardfail"}:
        return HARD_FAIL
    if token in {"soft", "soft_fail", "softfail"}:
        return SOFT_FAIL
    if token in {"pass"}:
        return PASS
    return default


def _severity_rank(severity: str) -> int:
    if severity == HARD_FAIL:
        return 2
    if severity == SOFT_FAIL:
        return 1
    return 0


def _resolve_rule_action(layer: str, condition: str, explicit_action: str) -> str:
    action = str(explicit_action or "").strip()
    if action:
        return action

    if condition in {"n_min", "sample_size_missing"}:
        return "Increase evaluation sample size and rerun the pipeline."
    if condition in {"ci_lower_bound_min", "ci_missing"}:
        return "Reduce variance and/or increase sample size before release decision."

    layer_lower = str(layer or "").strip().lower()
    if layer_lower == "retrieval":
        return "Tune recall/rerank strategy, then rerun run_eval + leaderboard."
    if layer_lower == "judge":
        return "Improve end-to-end answer quality and rerun judge evaluation."
    return "Fix the failing metric and rerun the full evaluation pipeline."


def _parse_rule(raw: dict[str, Any], index: int) -> RuleSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"Rule at index {index} must be an object.")

    name = str(raw.get("name", f"rule_{index + 1}")).strip() or f"rule_{index + 1}"
    metric_raw = str(raw.get("metric", "")).strip()
    if not metric_raw:
        raise ValueError(f"Rule '{name}' missing metric.")
    metric = normalize_metric_name(metric_raw)

    metric_spec = _metric_spec(metric)

    threshold = raw.get("threshold", {})
    if not isinstance(threshold, dict):
        threshold = {}

    ci_cfg = raw.get("ci", {})
    if not isinstance(ci_cfg, dict):
        ci_cfg = {}

    min_value = _to_float(raw.get("min_value"))
    if min_value is None:
        min_value = _to_float(threshold.get("min"))

    max_value = _to_float(raw.get("max_value"))
    if max_value is None:
        max_value = _to_float(threshold.get("max"))

    n_min = _to_int(raw.get("n_min"))

    ci_lower_bound_min = _to_float(raw.get("ci_lower_bound_min"))
    if ci_lower_bound_min is None:
        ci_lower_bound_min = _to_float(ci_cfg.get("lower_bound_min"))

    severity = _normalize_severity(raw.get("severity"), default=HARD_FAIL)
    missing_severity = _normalize_severity(raw.get("missing_severity"), default=severity)
    n_severity = _normalize_severity(raw.get("n_severity"), default=severity)
    ci_severity = _normalize_severity(raw.get("ci_severity"), default=severity)

    layer = str(raw.get("layer", metric_spec.layer)).strip() or metric_spec.layer
    direction = str(raw.get("direction", metric_spec.direction)).strip() or metric_spec.direction
    enabled = bool(raw.get("enabled", True))

    action = str(raw.get("action", "")).strip()

    return RuleSpec(
        name=name,
        metric=metric,
        layer=layer,
        direction=direction,
        min_value=min_value,
        max_value=max_value,
        n_min=n_min,
        ci_lower_bound_min=ci_lower_bound_min,
        severity=severity,
        missing_severity=missing_severity,
        n_severity=n_severity,
        ci_severity=ci_severity,
        action=action,
        enabled=enabled,
    )


def parse_rules(config: dict[str, Any]) -> list[RuleSpec]:
    rules_raw = config.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError("Config field 'rules' must be a list.")

    rules: list[RuleSpec] = []
    for idx, raw in enumerate(rules_raw):
        rule = _parse_rule(raw, idx)
        if rule.enabled:
            rules.append(rule)
    if not rules:
        raise ValueError("No enabled gate rules configured.")
    return rules

def _pick_target_group(
    leaderboard: dict[str, Any],
    requested_group_id: str,
) -> tuple[str, dict[str, Any]]:
    groups = leaderboard.get("groups", [])
    if not isinstance(groups, list) or not groups:
        raise ValueError("Leaderboard missing non-empty groups list.")

    wanted = str(requested_group_id or "").strip()
    if wanted:
        for group in groups:
            if isinstance(group, dict) and str(group.get("group_id", "")).strip() == wanted:
                return (wanted, group)
        raise ValueError(f"Group id '{wanted}' not found in leaderboard.")

    baseline = str(leaderboard.get("baseline_group", "")).strip()
    for group in groups:
        if not isinstance(group, dict):
            continue
        gid = str(group.get("group_id", "")).strip()
        if gid and gid != baseline:
            return (gid, group)

    first = groups[0]
    if not isinstance(first, dict):
        raise ValueError("Leaderboard first group is invalid.")
    gid = str(first.get("group_id", "")).strip() or "group_0"
    return (gid, first)


def _resolve_path(path: str | None, base_dir: Path) -> Path | None:
    if not path:
        return None
    token = str(path).strip()
    if not token or token.lower() == MISSING:
        return None
    candidate = Path(token)
    if candidate.is_absolute():
        return candidate
    direct = candidate.resolve()
    if direct.exists():
        return direct
    return (base_dir / candidate).resolve()


def _leaderboard_source_path(
    group: dict[str, Any],
    *,
    key: str,
    base_dir: Path,
) -> Path | None:
    sources = group.get("sources", {})
    if not isinstance(sources, dict):
        return None
    raw = sources.get(key)
    if not isinstance(raw, str):
        return None
    return _resolve_path(raw, base_dir=base_dir)


def evaluate_rules(
    rules: list[RuleSpec],
    metrics: dict[str, MetricObservation],
    *,
    ci_confidence: float,
    ci_bootstrap_samples: int,
    ci_seed: int,
) -> dict[str, Any]:
    rule_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    recommendations: list[str] = []

    for rule in rules:
        obs = metrics.get(rule.metric)
        if obs is not None:
            obs.ci_95 = compute_metric_ci(
                obs,
                confidence=ci_confidence,
                bootstrap_samples=ci_bootstrap_samples,
                seed=ci_seed,
            )

        checks: list[dict[str, Any]] = []
        rule_failures: list[dict[str, Any]] = []

        if obs is None or obs.value is None:
            action = _resolve_rule_action(rule.layer, "metric_missing", rule.action)
            failure = {
                "rule": rule.name,
                "metric": rule.metric,
                "layer": rule.layer,
                "condition": "metric_missing",
                "severity": rule.missing_severity,
                "actual": MISSING,
                "threshold": MISSING,
                "delta": MISSING,
                "action": action,
                "message": f"Metric '{rule.metric}' is missing.",
            }
            checks.append({**failure, "passed": False})
            rule_failures.append(failure)
        else:
            value = float(obs.value)
            n = obs.n
            ci = obs.ci_95 or {"lower": MISSING, "upper": MISSING, "method": MISSING}

            if rule.min_value is not None:
                delta = value - float(rule.min_value)
                passed = delta >= 0
                check = {
                    "rule": rule.name,
                    "metric": rule.metric,
                    "layer": rule.layer,
                    "condition": "min_value",
                    "severity": rule.severity,
                    "actual": _round_or_missing(value),
                    "threshold": _round_or_missing(rule.min_value),
                    "delta": _round_or_missing(delta),
                    "action": _resolve_rule_action(rule.layer, "min_value", rule.action),
                    "message": (
                        f"Metric '{rule.metric}'={value:.6f} is below min threshold {float(rule.min_value):.6f}."
                        if not passed
                        else f"Metric '{rule.metric}' passes min threshold."
                    ),
                    "passed": passed,
                }
                checks.append(check)
                if not passed:
                    rule_failures.append({k: check[k] for k in check if k != "passed"})

            if rule.max_value is not None:
                delta = value - float(rule.max_value)
                passed = delta <= 0
                check = {
                    "rule": rule.name,
                    "metric": rule.metric,
                    "layer": rule.layer,
                    "condition": "max_value",
                    "severity": rule.severity,
                    "actual": _round_or_missing(value),
                    "threshold": _round_or_missing(rule.max_value),
                    "delta": _round_or_missing(delta),
                    "action": _resolve_rule_action(rule.layer, "max_value", rule.action),
                    "message": (
                        f"Metric '{rule.metric}'={value:.6f} is above max threshold {float(rule.max_value):.6f}."
                        if not passed
                        else f"Metric '{rule.metric}' passes max threshold."
                    ),
                    "passed": passed,
                }
                checks.append(check)
                if not passed:
                    rule_failures.append({k: check[k] for k in check if k != "passed"})

            if rule.n_min is not None:
                if n is None:
                    check = {
                        "rule": rule.name,
                        "metric": rule.metric,
                        "layer": rule.layer,
                        "condition": "sample_size_missing",
                        "severity": rule.n_severity,
                        "actual": MISSING,
                        "threshold": int(rule.n_min),
                        "delta": MISSING,
                        "action": _resolve_rule_action(rule.layer, "sample_size_missing", rule.action),
                        "message": f"Metric '{rule.metric}' sample size is missing.",
                        "passed": False,
                    }
                    checks.append(check)
                    rule_failures.append({k: check[k] for k in check if k != "passed"})
                else:
                    delta_n = int(n) - int(rule.n_min)
                    passed = delta_n >= 0
                    check = {
                        "rule": rule.name,
                        "metric": rule.metric,
                        "layer": rule.layer,
                        "condition": "n_min",
                        "severity": rule.n_severity,
                        "actual": int(n),
                        "threshold": int(rule.n_min),
                        "delta": int(delta_n),
                        "action": _resolve_rule_action(rule.layer, "n_min", rule.action),
                        "message": (
                            f"Metric '{rule.metric}' sample size {n} is below n_min {rule.n_min}."
                            if not passed
                            else f"Metric '{rule.metric}' sample size passes n_min."
                        ),
                        "passed": passed,
                    }
                    checks.append(check)
                    if not passed:
                        rule_failures.append({k: check[k] for k in check if k != "passed"})

            if rule.ci_lower_bound_min is not None:
                lower = _to_float(ci.get("lower"))
                if lower is None:
                    check = {
                        "rule": rule.name,
                        "metric": rule.metric,
                        "layer": rule.layer,
                        "condition": "ci_missing",
                        "severity": rule.ci_severity,
                        "actual": MISSING,
                        "threshold": _round_or_missing(rule.ci_lower_bound_min),
                        "delta": MISSING,
                        "action": _resolve_rule_action(rule.layer, "ci_missing", rule.action),
                        "message": f"Metric '{rule.metric}' CI lower bound is missing.",
                        "passed": False,
                    }
                    checks.append(check)
                    rule_failures.append({k: check[k] for k in check if k != "passed"})
                else:
                    delta_ci = float(lower) - float(rule.ci_lower_bound_min)
                    passed = delta_ci >= 0
                    check = {
                        "rule": rule.name,
                        "metric": rule.metric,
                        "layer": rule.layer,
                        "condition": "ci_lower_bound_min",
                        "severity": rule.ci_severity,
                        "actual": _round_or_missing(lower),
                        "threshold": _round_or_missing(rule.ci_lower_bound_min),
                        "delta": _round_or_missing(delta_ci),
                        "action": _resolve_rule_action(rule.layer, "ci_lower_bound_min", rule.action),
                        "message": (
                            f"Metric '{rule.metric}' CI lower bound {lower:.6f} is below required {float(rule.ci_lower_bound_min):.6f}."
                            if not passed
                            else f"Metric '{rule.metric}' CI lower bound passes threshold."
                        ),
                        "passed": passed,
                    }
                    checks.append(check)
                    if not passed:
                        rule_failures.append({k: check[k] for k in check if k != "passed"})

        if not rule_failures:
            status = PASS
        else:
            max_rank = max(_severity_rank(item["severity"]) for item in rule_failures)
            status = HARD_FAIL if max_rank >= 2 else SOFT_FAIL
            for item in rule_failures:
                failures.append(item)
                action = str(item.get("action", "")).strip()
                if action and action not in recommendations:
                    recommendations.append(action)

        rule_results.append(
            {
                "name": rule.name,
                "metric": rule.metric,
                "layer": rule.layer,
                "status": status,
                "value": _round_or_missing(obs.value) if obs and obs.value is not None else MISSING,
                "n": obs.n if (obs and obs.n is not None) else MISSING,
                "ci_95": obs.ci_95 if (obs and obs.ci_95) else {"lower": MISSING, "upper": MISSING, "method": MISSING},
                "checks": checks,
            }
        )

    hard_count = sum(1 for row in rule_results if row["status"] == HARD_FAIL)
    soft_count = sum(1 for row in rule_results if row["status"] == SOFT_FAIL)
    pass_count = sum(1 for row in rule_results if row["status"] == PASS)

    if hard_count > 0:
        overall = HARD_FAIL
    elif soft_count > 0:
        overall = SOFT_FAIL
    else:
        overall = PASS

    return {
        "status": overall,
        "exit_code": EXIT_CODE_BY_STATUS[overall],
        "rules": rule_results,
        "failures": failures,
        "recommendations": recommendations,
        "summary": {
            "total_rules": len(rule_results),
            "passed_rules": pass_count,
            "soft_failed_rules": soft_count,
            "hard_failed_rules": hard_count,
        },
    }

def _fmt_num(value: Any, digits: int = 6) -> str:
    if value == MISSING:
        return MISSING
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_markdown_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Eval Gatekeeper Summary")
    lines.append("")
    lines.append(f"- Verdict: `{report.get('status', MISSING)}`")
    lines.append(f"- Exit code: `{report.get('exit_code', MISSING)}`")
    lines.append(f"- Generated at (UTC): `{report.get('generated_at_utc', MISSING)}`")
    lines.append(f"- Target group: `{report.get('target_group', MISSING)}`")
    lines.append("")

    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines.append("## Rule Summary")
    lines.append("")
    lines.append(
        "- "
        f"total={summary.get('total_rules', 0)} "
        f"passed={summary.get('passed_rules', 0)} "
        f"soft_failed={summary.get('soft_failed_rules', 0)} "
        f"hard_failed={summary.get('hard_failed_rules', 0)}"
    )
    lines.append("")

    lines.append("## Rule Results")
    lines.append("")
    lines.append("| Rule | Layer | Metric | Status | Value | n | CI95 Lower |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for row in report.get("rules", []) or []:
        if not isinstance(row, dict):
            continue
        ci = row.get("ci_95", {})
        if not isinstance(ci, dict):
            ci = {}
        lines.append(
            "| "
            f"{row.get('name', MISSING)} | "
            f"{row.get('layer', MISSING)} | "
            f"{row.get('metric', MISSING)} | "
            f"{row.get('status', MISSING)} | "
            f"{_fmt_num(row.get('value', MISSING), digits=4)} | "
            f"{row.get('n', MISSING)} | "
            f"{_fmt_num(ci.get('lower', MISSING), digits=4)} |"
        )
    lines.append("")

    failures = report.get("failures", [])
    if isinstance(failures, list) and failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("| Severity | Rule | Metric | Condition | Actual | Threshold | Delta | Action |")
        lines.append("|---|---|---|---|---:|---:|---:|---|")
        for item in failures:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{item.get('severity', MISSING)} | "
                f"{item.get('rule', MISSING)} | "
                f"{item.get('metric', MISSING)} | "
                f"{item.get('condition', MISSING)} | "
                f"{_fmt_num(item.get('actual', MISSING), digits=4)} | "
                f"{_fmt_num(item.get('threshold', MISSING), digits=4)} | "
                f"{_fmt_num(item.get('delta', MISSING), digits=4)} | "
                f"{item.get('action', '')} |"
            )
        lines.append("")
    else:
        lines.append("## Failures")
        lines.append("")
        lines.append("No failing gate conditions.")
        lines.append("")

    recommendations = report.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        lines.append("## Suggested Actions")
        for idx, item in enumerate(recommendations, 1):
            lines.append(f"{idx}. {item}")
        lines.append("")

    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("## Warnings")
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def _build_arg_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Release gatekeeper for eval metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=eval_dir / "config" / "gates.yaml",
        help="Gate config path (YAML or JSON syntax).",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=eval_dir / "reports" / "leaderboard" / "latest.json",
        help="Optional leaderboard JSON input.",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="",
        help="Target group id in leaderboard; default picks first non-baseline group.",
    )
    parser.add_argument(
        "--run-eval-report",
        type=Path,
        default=None,
        help="Optional run_eval output JSON.",
    )
    parser.add_argument(
        "--judge-report",
        type=Path,
        default=None,
        help="Optional judge output JSON.",
    )
    parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=None,
        help="Override bootstrap sample count for CI estimation.",
    )
    parser.add_argument(
        "--ci-seed",
        type=int,
        default=None,
        help="Override CI random seed.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=eval_dir / "reports" / "gatekeeper" / "latest.json",
        help="Machine-readable gate output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=eval_dir / "reports" / "gatekeeper" / "latest.md",
        help="Human-readable gate summary markdown path.",
    )
    return parser


def _serialize_metrics(metrics: dict[str, MetricObservation]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in sorted(metrics.keys()):
        obs = metrics[metric]
        out[metric] = {
            "layer": obs.layer,
            "direction": obs.direction,
            "value": _round_or_missing(obs.value),
            "n": obs.n if obs.n is not None else MISSING,
            "ci_95": obs.ci_95 if obs.ci_95 is not None else {"lower": MISSING, "upper": MISSING, "method": MISSING},
            "source": obs.source,
        }
    return out


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_arg_parser(eval_dir)
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Gate config not found: {config_path}")
    config = _read_yaml_or_json(config_path)
    rules = parse_rules(config)

    ci_cfg = config.get("ci", {})
    if not isinstance(ci_cfg, dict):
        ci_cfg = {}
    ci_confidence = _to_float(ci_cfg.get("confidence")) or 0.95
    ci_bootstrap_samples = _to_int(ci_cfg.get("bootstrap_samples")) or 1000
    ci_seed = _to_int(ci_cfg.get("seed")) or 17

    if args.ci_bootstrap_samples is not None:
        ci_bootstrap_samples = max(200, int(args.ci_bootstrap_samples))
    if args.ci_seed is not None:
        ci_seed = int(args.ci_seed)

    warnings: list[str] = []

    requested_group = str(args.group_id or config.get("target_group") or "").strip()
    target_group = MISSING

    leaderboard_payload: dict[str, Any] | None = None
    leaderboard_group: dict[str, Any] | None = None
    leaderboard_path: Path | None = args.leaderboard.resolve() if args.leaderboard else None

    if leaderboard_path is not None and leaderboard_path.exists():
        leaderboard_payload = _read_json(leaderboard_path)
        target_group, leaderboard_group = _pick_target_group(leaderboard_payload, requested_group)
    elif leaderboard_path is not None:
        warnings.append(f"Leaderboard not found: {leaderboard_path}")

    run_eval_path = args.run_eval_report.resolve() if args.run_eval_report else None
    judge_path = args.judge_report.resolve() if args.judge_report else None

    if leaderboard_group is not None and leaderboard_path is not None:
        if run_eval_path is None:
            run_eval_path = _leaderboard_source_path(leaderboard_group, key="run_eval", base_dir=leaderboard_path.parent)
        if judge_path is None:
            judge_path = _leaderboard_source_path(leaderboard_group, key="judge", base_dir=leaderboard_path.parent)

    run_eval_payload = _read_json(run_eval_path) if run_eval_path and run_eval_path.exists() else None
    if run_eval_path is not None and not run_eval_path.exists():
        warnings.append(f"run_eval report not found: {run_eval_path}")

    judge_payload = _read_json(judge_path) if judge_path and judge_path.exists() else None
    if judge_path is not None and not judge_path.exists():
        warnings.append(f"judge report not found: {judge_path}")

    observation_maps: list[dict[str, MetricObservation]] = []
    if leaderboard_group is not None:
        observation_maps.append(_observe_from_leaderboard_group(leaderboard_group, source="leaderboard"))
    if run_eval_payload is not None:
        observation_maps.append(_observe_from_run_eval(run_eval_payload, source="run_eval"))
    if judge_payload is not None:
        observation_maps.append(_observe_from_judge(judge_payload, source="judge"))

    if not observation_maps:
        raise ValueError(
            "No usable metric source found. Provide leaderboard and/or run_eval/judge reports."
        )

    metrics = merge_metric_maps(*observation_maps)

    gate_result = evaluate_rules(
        rules,
        metrics,
        ci_confidence=float(ci_confidence),
        ci_bootstrap_samples=max(200, int(ci_bootstrap_samples)),
        ci_seed=int(ci_seed),
    )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate_result["status"],
        "exit_code": gate_result["exit_code"],
        "target_group": target_group,
        "inputs": {
            "config": str(config_path),
            "leaderboard": str(leaderboard_path) if leaderboard_path is not None else MISSING,
            "run_eval": str(run_eval_path) if run_eval_path is not None else MISSING,
            "judge": str(judge_path) if judge_path is not None else MISSING,
        },
        "ci": {
            "confidence": ci_confidence,
            "bootstrap_samples": ci_bootstrap_samples,
            "seed": ci_seed,
        },
        "summary": gate_result["summary"],
        "rules": gate_result["rules"],
        "failures": gate_result["failures"],
        "recommendations": gate_result["recommendations"],
        "metrics": _serialize_metrics(metrics),
        "warnings": warnings,
    }

    markdown = build_markdown_summary(report)

    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    print(f"[Gatekeeper] status={report['status']} exit_code={report['exit_code']}")
    print(f"[Gatekeeper] json={output_json}")
    print(f"[Gatekeeper] markdown={output_md}")

    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
