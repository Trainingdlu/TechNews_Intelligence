"""Build matrix leaderboard reports with baseline deltas and CI estimates.

This script merges matrix `run_eval` outputs with optional `judge`
results, aligned by `group_id`, then emits:

- JSON report: `eval/reports/leaderboard/latest.json`
- Markdown report: `eval/reports/leaderboard/latest.md`
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from common import get_nested, to_float, to_int
except ImportError:  # package-style import fallback
    from .common import get_nested, to_float, to_int

MISSING = "missing"
DEFAULT_BASELINE_GROUP = "G0_baseline"
DEFAULT_TOP_K = 5
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 7

_LATENCY_KEYS_MS = ("latency_ms", "elapsed_ms", "duration_ms")
_LATENCY_KEYS_SEC = ("latency_seconds", "elapsed_seconds", "duration_seconds")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    layer: str
    direction: str
    bounded_01: bool = True


@dataclass
class MetricObservation:
    value: float | None
    n: int | None
    samples: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupArtifacts:
    group_id: str
    description: str
    run_eval_path: Path | None
    judge_path: Path | None
    run_eval: dict[str, Any] | None
    judge: dict[str, Any] | None


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec("avg_recall_at_10", "retrieval", "higher_better"),
    MetricSpec("avg_mrr_at_10", "retrieval", "higher_better"),
    MetricSpec("avg_ndcg_at_10", "retrieval", "higher_better"),
    MetricSpec("avg_error_rate", "system", "lower_better"),
    MetricSpec("react_success_rate", "system", "higher_better"),
    MetricSpec("p95_latency", "system", "lower_better", bounded_01=False),
    MetricSpec("avg_composite", "judge", "higher_better"),
]


def _warn(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _to_float(value: Any) -> float | None:
    return to_float(value)


def _to_int(value: Any) -> int | None:
    return to_int(value)


def _round(value: float | None, digits: int = 6) -> float | str:
    if value is None or not math.isfinite(value):
        return MISSING
    return round(value, digits)


def _get_nested(data: dict[str, Any] | None, path: str) -> Any:
    return get_nested(data, path)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    items = sorted(values)
    pos = (len(items) - 1) * (p / 100.0)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(items[low])
    frac = pos - low
    return float(items[low] * (1.0 - frac) + items[high] * frac)


def _load_json(path: Path, *, warnings: list[str], label: str) -> dict[str, Any] | None:
    if not path.exists():
        _warn(warnings, f"[Warn] missing {label} file: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _warn(warnings, f"[Warn] failed to parse {label} JSON ({path}): {type(exc).__name__}: {exc}")
        return None
    if not isinstance(payload, dict):
        _warn(warnings, f"[Warn] invalid {label} JSON root (expect object): {path}")
        return None
    return payload


def _resolve_path(raw: str | None, *, base_dir: Path) -> Path | None:
    if not raw:
        return None
    token = str(raw).strip()
    if not token:
        return None
    path = Path(token)
    if path.is_absolute():
        return path
    direct = path.resolve()
    if direct.exists():
        return direct
    return (base_dir / path).resolve()


def _latest_manifest_path(matrix_dir: Path) -> Path:
    manifests = sorted(matrix_dir.glob("*_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No matrix manifest found under: {matrix_dir}")
    return manifests[-1]


def _read_config(config_path: Path, *, warnings: list[str]) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    raw = config_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    try:
        import yaml  # type: ignore[import-not-found]

        payload = yaml.safe_load(raw)  # type: ignore[attr-defined]
        if isinstance(payload, dict):
            return payload
    except Exception as exc:  # noqa: BLE001
        _warn(
            warnings,
            f"[Warn] failed to parse config ({config_path}); fallback to defaults: "
            f"{type(exc).__name__}: {exc}",
        )
    return {}


def _extract_dataset_version(dataset_path: str | None) -> str:
    if not dataset_path:
        return MISSING
    path = Path(str(dataset_path))
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part == "versions" and idx + 1 < len(parts):
            return parts[idx + 1]
    stem = path.stem.strip()
    return stem or MISSING


def _extract_case_metric_samples(report: dict[str, Any] | None, key: str) -> list[float]:
    if not isinstance(report, dict):
        return []
    out: list[float] = []
    for case in report.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        metrics = case.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        value = _to_float(metrics.get(key))
        if value is None:
            continue
        out.append(value)
    return out


def _extract_latency_samples(report: dict[str, Any] | None) -> list[float]:
    if not isinstance(report, dict):
        return []

    def _extract_ms(obj: dict[str, Any]) -> float | None:
        for key in _LATENCY_KEYS_MS:
            value = _to_float(obj.get(key))
            if value is not None and value >= 0:
                return value
        for key in _LATENCY_KEYS_SEC:
            value = _to_float(obj.get(key))
            if value is not None and value >= 0:
                return value * 1000.0
        return None

    latencies: list[float] = []
    for case in report.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        for run in case.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            direct = _extract_ms(run)
            if direct is not None:
                latencies.append(direct)
                continue
            trace_summary = run.get("trace_summary", {})
            if not isinstance(trace_summary, dict):
                continue
            traced = _extract_ms(trace_summary)
            if traced is not None:
                latencies.append(traced)
    return latencies


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

    out: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidates = [
            row.get("avg_composite"),
            row.get("composite"),
            row.get("composite_score"),
            row.get("score"),
            _get_nested(row, "scores.avg_composite"),
            _get_nested(row, "scores.composite"),
            _get_nested(row, "judge.avg_composite"),
            _get_nested(row, "judge.composite"),
        ]
        parsed = next((x for x in (_to_float(item) for item in candidates) if x is not None), None)
        if parsed is not None:
            out.append(parsed)
    return out


def _observe_metric(spec: MetricSpec, group: GroupArtifacts) -> MetricObservation:
    if spec.name == "avg_recall_at_10":
        value = _to_float(_get_nested(group.run_eval, "summary.avg_recall_at_10"))
        samples = _extract_case_metric_samples(group.run_eval, "recall_at_10")
        n = _to_int(_get_nested(group.run_eval, "summary.retrieval_case_count"))
        if n is None and samples:
            n = len(samples)
        if value is None:
            value = _mean(samples)
        return MetricObservation(value=value, n=n, samples=samples)

    if spec.name == "avg_mrr_at_10":
        value = _to_float(_get_nested(group.run_eval, "summary.avg_mrr_at_10"))
        samples = _extract_case_metric_samples(group.run_eval, "mrr_at_10")
        n = _to_int(_get_nested(group.run_eval, "summary.retrieval_case_count"))
        if n is None and samples:
            n = len(samples)
        if value is None:
            value = _mean(samples)
        return MetricObservation(value=value, n=n, samples=samples)

    if spec.name == "avg_ndcg_at_10":
        value = _to_float(_get_nested(group.run_eval, "summary.avg_ndcg_at_10"))
        samples = _extract_case_metric_samples(group.run_eval, "ndcg_at_10")
        n = _to_int(_get_nested(group.run_eval, "summary.retrieval_case_count"))
        if n is None and samples:
            n = len(samples)
        if value is None:
            value = _mean(samples)
        return MetricObservation(value=value, n=n, samples=samples)

    if spec.name == "avg_error_rate":
        value = _to_float(_get_nested(group.run_eval, "summary.avg_error_rate"))
        samples = _extract_case_metric_samples(group.run_eval, "error_rate")
        n = _to_int(_get_nested(group.run_eval, "summary.case_count"))
        if n is None and samples:
            n = len(samples)
        if value is None:
            value = _mean(samples)
        return MetricObservation(value=value, n=n, samples=samples)

    if spec.name == "react_success_rate":
        value = _to_float(_get_nested(group.run_eval, "route_metrics.react_success_rate"))
        attempts = _to_int(_get_nested(group.run_eval, "route_metrics.react_attempts"))
        success = _to_int(_get_nested(group.run_eval, "route_metrics.react_success"))
        n = attempts if attempts is not None and attempts > 0 else None
        if value is None and success is not None and n:
            value = success / n
        extras = {"attempts": attempts, "success": success}
        return MetricObservation(value=value, n=n, samples=[], extras=extras)

    if spec.name == "p95_latency":
        samples = _extract_latency_samples(group.run_eval)
        value = _percentile(samples, 95.0)
        n = len(samples) if samples else None
        return MetricObservation(value=value, n=n, samples=samples)

    if spec.name == "avg_composite":
        samples = _extract_judge_samples(group.judge)
        value_candidates = [
            _get_nested(group.judge, "summary.avg_composite"),
            _get_nested(group.judge, "summary.composite"),
            _get_nested(group.judge, "avg_composite"),
            _get_nested(group.judge, "composite"),
        ]
        value = next((x for x in (_to_float(item) for item in value_candidates) if x is not None), None)
        if value is None:
            value = _mean(samples)
        n = len(samples) if samples else _to_int(_get_nested(group.judge, "row_count"))
        return MetricObservation(value=value, n=n, samples=samples)

    return MetricObservation(value=None, n=None, samples=[])


def _bootstrap_delta_mean(
    current: list[float],
    baseline: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if len(current) < 2 or len(baseline) < 2:
        return None
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(samples):
        cur_mean = sum(rng.choice(current) for _ in range(len(current))) / len(current)
        base_mean = sum(rng.choice(baseline) for _ in range(len(baseline))) / len(baseline)
        diffs.append(cur_mean - base_mean)
    lower = _percentile(diffs, 2.5)
    upper = _percentile(diffs, 97.5)
    if lower is None or upper is None:
        return None
    return (lower, upper)


def _bootstrap_delta_p95(
    current: list[float],
    baseline: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if len(current) < 5 or len(baseline) < 5:
        return None
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(samples):
        cur_sample = [rng.choice(current) for _ in range(len(current))]
        base_sample = [rng.choice(baseline) for _ in range(len(baseline))]
        cur_p95 = _percentile(cur_sample, 95.0)
        base_p95 = _percentile(base_sample, 95.0)
        if cur_p95 is None or base_p95 is None:
            continue
        diffs.append(cur_p95 - base_p95)
    if not diffs:
        return None
    lower = _percentile(diffs, 2.5)
    upper = _percentile(diffs, 97.5)
    if lower is None or upper is None:
        return None
    return (lower, upper)


def _normal_approx_delta_ci(
    cur_value: float,
    cur_n: int,
    base_value: float,
    base_n: int,
) -> tuple[float, float] | None:
    if cur_n <= 0 or base_n <= 0:
        return None
    p_cur = max(0.0, min(1.0, cur_value))
    p_base = max(0.0, min(1.0, base_value))
    se = math.sqrt((p_cur * (1.0 - p_cur) / cur_n) + (p_base * (1.0 - p_base) / base_n))
    margin = 1.96 * se
    delta = p_cur - p_base
    return (delta - margin, delta + margin)


def _estimate_ci_95(
    spec: MetricSpec,
    current: MetricObservation,
    baseline: MetricObservation,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | str]:
    if current.value is None or baseline.value is None:
        return {"lower": MISSING, "upper": MISSING, "method": MISSING}

    if spec.name == "p95_latency":
        ci = _bootstrap_delta_p95(
            current.samples,
            baseline.samples,
            samples=bootstrap_samples,
            seed=seed,
        )
        if ci is not None:
            return {
                "lower": _round(ci[0]),
                "upper": _round(ci[1]),
                "method": "bootstrap_delta_p95",
            }

    ci_mean = _bootstrap_delta_mean(
        current.samples,
        baseline.samples,
        samples=bootstrap_samples,
        seed=seed,
    )
    if ci_mean is not None:
        return {
            "lower": _round(ci_mean[0]),
            "upper": _round(ci_mean[1]),
            "method": "bootstrap_delta_mean",
        }

    if spec.bounded_01 and current.n and baseline.n:
        ci_normal = _normal_approx_delta_ci(current.value, current.n, baseline.value, baseline.n)
        if ci_normal is not None:
            return {
                "lower": _round(ci_normal[0]),
                "upper": _round(ci_normal[1]),
                "method": "normal_approx",
            }

    return {"lower": MISSING, "upper": MISSING, "method": MISSING}


def _delta_pct(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    delta = current - baseline
    if abs(baseline) <= 1e-12:
        return 0.0 if abs(delta) <= 1e-12 else None
    return (delta / abs(baseline)) * 100.0


def _improvement_score(direction: str, delta_abs: float | None) -> float | None:
    if delta_abs is None:
        return None
    return delta_abs if direction == "higher_better" else -delta_abs


def _fmt_num(value: float | str, digits: int = 4) -> str:
    if value == MISSING:
        return MISSING
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_pct(value: float | str) -> str:
    if value == MISSING:
        return MISSING
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}%"
    return str(value)


def _fmt_ci(ci: dict[str, Any]) -> str:
    lower = ci.get("lower", MISSING)
    upper = ci.get("upper", MISSING)
    method = str(ci.get("method", MISSING))
    if lower == MISSING or upper == MISSING:
        return MISSING
    return f"[{_fmt_num(lower)}, {_fmt_num(upper)}] ({method})"


def _build_metric_record(
    *,
    spec: MetricSpec,
    current: MetricObservation,
    baseline: MetricObservation,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    delta_abs = None
    if current.value is not None and baseline.value is not None:
        delta_abs = current.value - baseline.value
    delta_pct = _delta_pct(current.value, baseline.value)
    ci = _estimate_ci_95(
        spec,
        current,
        baseline,
        bootstrap_samples=bootstrap_samples,
        seed=seed + abs(hash(spec.name)) % 10000,
    )
    improvement = _improvement_score(spec.direction, delta_abs)

    return {
        "layer": spec.layer,
        "direction": spec.direction,
        "current": _round(current.value),
        "baseline": _round(baseline.value),
        "delta_abs": _round(delta_abs),
        "delta_pct": _round(delta_pct, digits=4),
        "sample_n": {
            "current": current.n if current.n is not None else MISSING,
            "baseline": baseline.n if baseline.n is not None else MISSING,
        },
        "n": current.n if current.n is not None else MISSING,
        "ci_95": ci,
        "improvement_score": _round(improvement),
    }


def _discover_aux_from_group_record(group: dict[str, Any], kind: str, *, base_dir: Path) -> Path | None:
    keys = (
        f"{kind}_output",
        f"{kind}_report",
        kind,
    )
    for key in keys:
        value = group.get(key)
        if isinstance(value, str):
            path = _resolve_path(value, base_dir=base_dir)
            if path is not None:
                return path
    outputs = group.get("outputs")
    if isinstance(outputs, dict):
        for key in keys:
            value = outputs.get(key)
            if isinstance(value, str):
                path = _resolve_path(value, base_dir=base_dir)
                if path is not None:
                    return path
    return None


def _discover_aux_report(
    *,
    kind: str,
    group_id: str,
    group_record: dict[str, Any],
    run_eval_path: Path | None,
    eval_dir: Path,
) -> Path | None:
    manifest_dir = Path(group_record.get("_manifest_dir", "."))
    explicit = _discover_aux_from_group_record(group_record, kind, base_dir=manifest_dir)
    if explicit is not None:
        return explicit

    candidates: list[Path] = []
    if run_eval_path is not None:
        stem = run_eval_path.stem
        parent = run_eval_path.parent
        candidates.extend(
            [
                parent / f"{stem}_{kind}.json",
                parent / f"{stem}.{kind}.json",
                parent / f"{stem}-{kind}.json",
            ]
        )

    aux_dir = (eval_dir / "reports" / kind).resolve()
    if run_eval_path is not None:
        stem = run_eval_path.stem
        candidates.extend(
            [
                aux_dir / f"{stem}.json",
                aux_dir / f"{stem}_{kind}.json",
                aux_dir / f"{stem}.{kind}.json",
            ]
        )
    candidates.extend(
        [
            aux_dir / f"{group_id}.json",
            aux_dir / f"{group_id}_{kind}.json",
            aux_dir / f"{group_id}.{kind}.json",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    if not aux_dir.exists():
        return None

    name_hits = sorted(
        [p for p in aux_dir.glob("*.json") if group_id.lower() in p.stem.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if name_hits:
        return name_hits[0]
    return None


def _load_group_artifacts(
    manifest: dict[str, Any],
    *,
    eval_dir: Path,
    warnings: list[str],
) -> list[GroupArtifacts]:
    groups_raw = manifest.get("groups")
    if not isinstance(groups_raw, list):
        raise ValueError("Manifest missing 'groups' list.")

    manifest_path = _resolve_path(str(manifest.get("_manifest_path", "")), base_dir=eval_dir) or eval_dir
    manifest_dir = manifest_path.parent

    out: list[GroupArtifacts] = []
    for item in groups_raw:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("id", "")).strip()
        if not group_id:
            continue
        description = str(item.get("description", "")).strip()
        output_raw = item.get("output")
        run_eval_path = _resolve_path(str(output_raw), base_dir=manifest_dir) if output_raw else None
        run_eval = None
        if run_eval_path is not None:
            run_eval = _load_json(run_eval_path, warnings=warnings, label=f"run_eval[{group_id}]")
        else:
            _warn(warnings, f"[Warn] missing run_eval output path for group {group_id} in manifest.")

        shadow = dict(item)
        shadow["_manifest_dir"] = str(manifest_dir)
        judge_path = _discover_aux_report(
            kind="judge",
            group_id=group_id,
            group_record=shadow,
            run_eval_path=run_eval_path,
            eval_dir=eval_dir,
        )

        if judge_path is None:
            _warn(warnings, f"[Warn] missing judge file for group {group_id}.")

        judge = _load_json(judge_path, warnings=warnings, label=f"judge[{group_id}]") if judge_path else None

        out.append(
            GroupArtifacts(
                group_id=group_id,
                description=description,
                run_eval_path=run_eval_path,
                judge_path=judge_path,
                run_eval=run_eval,
                judge=judge,
            )
        )
    return out


def _infer_dataset_info(groups: list[GroupArtifacts], *, warnings: list[str]) -> dict[str, str]:
    picked: GroupArtifacts | None = None
    for group in groups:
        if isinstance(group.run_eval, dict):
            picked = group
            break
    if picked is None:
        return {"path": MISSING, "version": MISSING, "suite": MISSING}

    dataset_path = str(_get_nested(picked.run_eval, "dataset") or "").strip()
    suite = str(_get_nested(picked.run_eval, "selection.suite") or "").strip()
    version = _extract_dataset_version(dataset_path)

    dataset_paths = {
        str(_get_nested(group.run_eval, "dataset") or "").strip()
        for group in groups
        if isinstance(group.run_eval, dict)
    }
    dataset_paths = {x for x in dataset_paths if x}
    if len(dataset_paths) > 1:
        _warn(
            warnings,
            "[Warn] dataset path mismatch across groups: "
            + ", ".join(sorted(dataset_paths)),
        )

    return {
        "path": dataset_path or MISSING,
        "version": version,
        "suite": suite or MISSING,
    }


def _collect_top_changes(
    group_records: list[dict[str, Any]],
    *,
    baseline_group: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for group in group_records:
        group_id = str(group.get("group_id", ""))
        if group_id == baseline_group:
            continue
        metrics = group.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric_name, record in metrics.items():
            if not isinstance(record, dict):
                continue
            delta_abs = record.get("delta_abs")
            improvement = record.get("improvement_score")
            if not isinstance(delta_abs, (int, float)):
                continue
            if not isinstance(improvement, (int, float)):
                continue
            candidates.append(
                {
                    "group_id": group_id,
                    "metric": metric_name,
                    "layer": record.get("layer", MISSING),
                    "direction": record.get("direction", MISSING),
                    "delta_abs": delta_abs,
                    "delta_pct": record.get("delta_pct", MISSING),
                    "n": record.get("n", MISSING),
                    "ci_95": record.get("ci_95", {}),
                    "improvement_score": improvement,
                }
            )

    gains = sorted(candidates, key=lambda item: float(item["improvement_score"]), reverse=True)
    regressions = sorted(candidates, key=lambda item: float(item["improvement_score"]))

    top_gains = [item for item in gains if float(item["improvement_score"]) > 0][:top_k]
    top_regressions = [item for item in regressions if float(item["improvement_score"]) < 0][:top_k]
    return (top_gains, top_regressions)


def _rule_based_recommendations(report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    groups = report.get("groups", [])
    if not isinstance(groups, list):
        return recommendations

    missing_judge = 0
    for group in groups:
        sources = group.get("sources", {})
        if not isinstance(sources, dict):
            continue
        if sources.get("judge") == MISSING:
            missing_judge += 1

    if missing_judge > 0:
        recommendations.append("Backfill missing Judge outputs so subjective quality and composite score stay comparable.")

    top_regressions = report.get("top_regressions", [])
    if isinstance(top_regressions, list):
        for item in top_regressions:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", ""))
            if metric in {"avg_error_rate", "react_success_rate"}:
                recommendations.append("System-layer regression detected: prioritize tool error paths, retry policy, and fallback handling.")
                break
            if metric in {"avg_composite"}:
                recommendations.append("Judge-layer regression detected: prioritize answer quality, grounding, and completeness.")
                break
            if metric in {"avg_recall_at_10", "avg_mrr_at_10", "avg_ndcg_at_10"}:
                recommendations.append("Retrieval-layer regression detected: prioritize rollback/tuning for recall and rerank strategy.")
                break
            if metric == "p95_latency":
                recommendations.append("Latency regression detected: optimize recall fanout, rerank model choice, and timeout thresholds.")
                break

    top_gains = report.get("top_gains", [])
    if isinstance(top_gains, list) and top_gains:
        first = top_gains[0]
        if isinstance(first, dict):
            group_id = str(first.get("group_id", ""))
            if group_id:
                recommendations.append(
                    f"Use `{group_id}` as the next optimization branch and validate on regression suite before broad rollout."
                )

    if not recommendations:
        recommendations.append("Signal is insufficient; backfill missing evaluations and increase sample size before decision.")
    return recommendations[:4]


def _render_markdown_builtin(report: dict[str, Any], metric_order: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Matrix Leaderboard Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{report.get('generated_at_utc', MISSING)}`")
    lines.append(f"- Manifest: `{report.get('manifest_path', MISSING)}`")
    dataset = report.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    lines.append(f"- Dataset version: `{dataset.get('version', MISSING)}`")
    lines.append(f"- Dataset suite: `{dataset.get('suite', MISSING)}`")
    lines.append(f"- 基线组: `{report.get('baseline_group', MISSING)}`")
    lines.append("")

    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("## Warnings")
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Group Metrics (vs Baseline)")
    for group in report.get("groups", []):
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id", ""))
        description = str(group.get("description", "")).strip()
        title = f"### {group_id}" if not description else f"### {group_id} - {description}"
        lines.append(title)
        lines.append("")
        sources = group.get("sources", {})
        if not isinstance(sources, dict):
            sources = {}
        lines.append(
            "- Sources: "
            f"run_eval=`{sources.get('run_eval', MISSING)}`, "
            f"judge=`{sources.get('judge', MISSING)}`"
        )
        lines.append("")
        lines.append("| Layer | Metric | Current | Baseline | Delta | Delta % | n(cur/base) | CI95(delta) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        metrics = group.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        for metric_name in metric_order:
            row = metrics.get(metric_name)
            if not isinstance(row, dict):
                continue
            sample_n = row.get("sample_n", {})
            if not isinstance(sample_n, dict):
                sample_n = {}
            lines.append(
                "| "
                f"{row.get('layer', MISSING)} | "
                f"{metric_name} | "
                f"{_fmt_num(row.get('current', MISSING))} | "
                f"{_fmt_num(row.get('baseline', MISSING))} | "
                f"{_fmt_num(row.get('delta_abs', MISSING))} | "
                f"{_fmt_pct(row.get('delta_pct', MISSING))} | "
                f"{sample_n.get('current', MISSING)}/{sample_n.get('baseline', MISSING)} | "
                f"{_fmt_ci(row.get('ci_95', {}))} |"
            )
        lines.append("")

    lines.append("## Top Gains")
    lines.append("")
    lines.append("| Group | Metric | Layer | Delta | Delta % | n | CI95(delta) |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    gains = report.get("top_gains", [])
    if isinstance(gains, list) and gains:
        for item in gains:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{item.get('group_id', MISSING)} | "
                f"{item.get('metric', MISSING)} | "
                f"{item.get('layer', MISSING)} | "
                f"{_fmt_num(item.get('delta_abs', MISSING))} | "
                f"{_fmt_pct(item.get('delta_pct', MISSING))} | "
                f"{item.get('n', MISSING)} | "
                f"{_fmt_ci(item.get('ci_95', {}))} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - |")
    lines.append("")

    lines.append("## Top Regressions")
    lines.append("")
    lines.append("| Group | Metric | Layer | Delta | Delta % | n | CI95(delta) |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    regressions = report.get("top_regressions", [])
    if isinstance(regressions, list) and regressions:
        for item in regressions:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{item.get('group_id', MISSING)} | "
                f"{item.get('metric', MISSING)} | "
                f"{item.get('layer', MISSING)} | "
                f"{_fmt_num(item.get('delta_abs', MISSING))} | "
                f"{_fmt_pct(item.get('delta_pct', MISSING))} | "
                f"{item.get('n', MISSING)} | "
                f"{_fmt_ci(item.get('ci_95', {}))} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - |")
    lines.append("")

    lines.append("## Recommended Next Steps")
    for item in report.get("recommendations", []) or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _render_markdown(
    report: dict[str, Any],
    *,
    metric_order: list[str],
    template_path: Path | None,
    warnings: list[str],
) -> str:
    if template_path and template_path.exists():
        try:
            from jinja2 import Template  # type: ignore[import-not-found]

            raw = template_path.read_text(encoding="utf-8")
            tpl = Template(raw)
            return tpl.render(report=report, metric_order=metric_order, MISSING=MISSING)
        except Exception as exc:  # noqa: BLE001
            _warn(
                warnings,
                f"[Warn] failed to render markdown template ({template_path}); "
                f"fallback built-in renderer: {type(exc).__name__}: {exc}",
            )
    return _render_markdown_builtin(report, metric_order)


def build_leaderboard_report(
    manifest_path: Path,
    *,
    eval_dir: Path,
    baseline_group: str,
    top_k: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    manifest = _load_json(manifest_path, warnings=warnings, label="matrix_manifest")
    if manifest is None:
        raise ValueError(f"Cannot load manifest: {manifest_path}")
    manifest["_manifest_path"] = str(manifest_path)

    groups = _load_group_artifacts(manifest, eval_dir=eval_dir, warnings=warnings)
    if not groups:
        raise ValueError(f"No valid groups in manifest: {manifest_path}")

    baseline = next((g for g in groups if g.group_id == baseline_group), None)
    if baseline is None:
        baseline = groups[0]
        _warn(
            warnings,
            f"[Warn] baseline group '{baseline_group}' not found; fallback to '{baseline.group_id}'.",
        )
        baseline_group = baseline.group_id

    metric_order = [spec.name for spec in METRIC_SPECS]
    group_records: list[dict[str, Any]] = []

    for group in groups:
        metric_payload: dict[str, Any] = {}
        for spec in METRIC_SPECS:
            current_obs = _observe_metric(spec, group)
            baseline_obs = _observe_metric(spec, baseline)
            record = _build_metric_record(
                spec=spec,
                current=current_obs,
                baseline=baseline_obs,
                bootstrap_samples=max(200, int(bootstrap_samples)),
                seed=int(bootstrap_seed),
            )
            metric_payload[spec.name] = record

        group_records.append(
            {
                "group_id": group.group_id,
                "description": group.description,
                "sources": {
                    "run_eval": str(group.run_eval_path) if group.run_eval_path is not None else MISSING,
                    "judge": str(group.judge_path) if group.judge_path is not None else MISSING,
                },
                "metrics": metric_payload,
            }
        )

    top_gains, top_regressions = _collect_top_changes(
        group_records,
        baseline_group=baseline_group,
        top_k=max(1, int(top_k)),
    )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "matrix_generated_at_utc": str(manifest.get("generated_at_utc", MISSING)),
        "baseline_group": baseline_group,
        "dataset": _infer_dataset_info(groups, warnings=warnings),
        "metric_order": metric_order,
        "groups": group_records,
        "top_gains": top_gains,
        "top_regressions": top_regressions,
        "recommendations": [],
        "warnings": warnings,
    }

    report["recommendations"] = _rule_based_recommendations(report)
    return report


def _build_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build matrix leaderboard report (run_eval + judge).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Explicit matrix manifest path; default uses latest in eval/reports/matrix.",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=eval_dir / "reports" / "matrix",
        help="Matrix report directory used when --manifest is omitted.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=eval_dir / "config" / "leaderboard.yaml",
        help="Optional leaderboard config path (JSON/YAML).",
    )
    parser.add_argument(
        "--baseline-group",
        type=str,
        default="",
        help=f"基线组 id (default: {DEFAULT_BASELINE_GROUP}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top K gains/regressions to include.",
    )
    parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap sample count for CI estimation.",
    )
    parser.add_argument(
        "--ci-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for CI bootstrap.",
    )
    parser.add_argument(
        "--markdown-template",
        type=Path,
        default=None,
        help="Optional Jinja2 markdown template path (fallback to built-in renderer when omitted).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=eval_dir / "reports" / "leaderboard" / "latest.json",
        help="Output path for leaderboard JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=eval_dir / "reports" / "leaderboard" / "latest.md",
        help="Output path for leaderboard markdown.",
    )
    return parser


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_parser(eval_dir)
    args = parser.parse_args()

    warnings: list[str] = []
    config = _read_config(args.config.resolve(), warnings=warnings)
    baseline_group = (
        str(args.baseline_group).strip()
        or str(config.get("baseline_group", "")).strip()
        or DEFAULT_BASELINE_GROUP
    )
    top_k = int(config.get("top_k", args.top_k))
    bootstrap_cfg = config.get("ci", {})
    if not isinstance(bootstrap_cfg, dict):
        bootstrap_cfg = {}
    bootstrap_samples = int(bootstrap_cfg.get("bootstrap_samples", args.ci_bootstrap_samples))
    bootstrap_seed = int(bootstrap_cfg.get("seed", args.ci_seed))

    manifest_path = args.manifest.resolve() if args.manifest else _latest_manifest_path(args.matrix_dir.resolve())

    report = build_leaderboard_report(
        manifest_path=manifest_path,
        eval_dir=eval_dir,
        baseline_group=baseline_group,
        top_k=top_k,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        config=config,
    )

    report_warnings = report.get("warnings", [])
    if isinstance(report_warnings, list):
        for item in warnings:
            _warn(report_warnings, item)

    metric_order = report.get("metric_order", [])
    if not isinstance(metric_order, list):
        metric_order = []
    markdown = _render_markdown(
        report,
        metric_order=[str(x) for x in metric_order],
        template_path=args.markdown_template.resolve() if args.markdown_template else None,
        warnings=report.get("warnings", []),
    )

    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    group_count = len(report.get("groups", []) or [])
    warning_count = len(report.get("warnings", []) or [])
    print(f"[Leaderboard] manifest={manifest_path}")
    print(f"[Leaderboard] groups={group_count} baseline={report.get('baseline_group', MISSING)}")
    print(f"[Leaderboard] warnings={warning_count}")
    print(f"[Leaderboard] json={output_json}")
    print(f"[Leaderboard] markdown={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
