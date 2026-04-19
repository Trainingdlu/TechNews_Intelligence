"""Build matrix leaderboard for task_eval_v1 layered reports.

Input:
- matrix manifest from eval/run_matrix_eval.py (runner=task_eval_v1)

Output:
- JSON leaderboard report
- Markdown leaderboard report
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
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
DEFAULT_BOOTSTRAP_SEED = 11


@dataclass(frozen=True)
class MetricSpec:
    name: str
    layer: str
    direction: str
    bounded_01: bool = True
    summary_path: str = ""
    case_layer: str = ""
    case_key: str = ""
    n_scope: str = "case"  # case|retrieval


@dataclass
class MetricObservation:
    value: float | None
    n: int | None
    samples: list[float]


@dataclass
class GroupArtifacts:
    group_id: str
    description: str
    report_path: Path | None
    report: dict[str, Any] | None


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        name="intent_top1_accuracy",
        layer="intent",
        direction="higher_better",
        summary_path="summary.intent.top1_accuracy",
        case_layer="intent",
        case_key="top1_accuracy",
    ),
    MetricSpec(
        name="intent_macro_f1",
        layer="intent",
        direction="higher_better",
        summary_path="summary.intent.macro_f1",
        case_layer="intent",
        case_key="macro_f1",
    ),
    MetricSpec(
        name="intent_clarification_accuracy",
        layer="intent",
        direction="higher_better",
        summary_path="summary.intent.clarification_accuracy",
        case_layer="intent",
        case_key="clarification_accuracy",
    ),
    MetricSpec(
        name="tool_path_hit_rate",
        layer="tool",
        direction="higher_better",
        summary_path="summary.tool.acceptable_path_hit_rate",
        case_layer="tool",
        case_key="acceptable_path_hit_rate",
    ),
    MetricSpec(
        name="tool_param_accuracy",
        layer="tool",
        direction="higher_better",
        summary_path="summary.tool.param_accuracy",
        case_layer="tool",
        case_key="param_accuracy",
    ),
    MetricSpec(
        name="tool_forbidden_tool_rate",
        layer="tool",
        direction="lower_better",
        summary_path="summary.tool.forbidden_tool_rate",
        case_layer="tool",
        case_key="forbidden_tool_rate",
    ),
    MetricSpec(
        name="avg_recall_at_10",
        layer="retrieval",
        direction="higher_better",
        summary_path="summary.retrieval.recall_at_10",
        case_layer="retrieval",
        case_key="recall_at_10",
        n_scope="retrieval",
    ),
    MetricSpec(
        name="avg_mrr_at_10",
        layer="retrieval",
        direction="higher_better",
        summary_path="summary.retrieval.mrr_at_10",
        case_layer="retrieval",
        case_key="mrr_at_10",
        n_scope="retrieval",
    ),
    MetricSpec(
        name="avg_ndcg_at_10",
        layer="retrieval",
        direction="higher_better",
        summary_path="summary.retrieval.ndcg_at_10",
        case_layer="retrieval",
        case_key="ndcg_at_10",
        n_scope="retrieval",
    ),
    MetricSpec(
        name="retrieval_gold_hit_rate",
        layer="retrieval",
        direction="higher_better",
        summary_path="summary.retrieval.gold_hit_rate",
        case_layer="retrieval",
        case_key="gold_hit_rate",
        n_scope="retrieval",
    ),
    MetricSpec(
        name="analysis_claim_support_rate",
        layer="analysis",
        direction="higher_better",
        summary_path="summary.analysis.claim_support_rate",
        case_layer="analysis",
        case_key="claim_support_rate",
    ),
    MetricSpec(
        name="analysis_unsupported_claim_rate",
        layer="analysis",
        direction="lower_better",
        summary_path="summary.analysis.unsupported_claim_rate",
        case_layer="analysis",
        case_key="unsupported_claim_rate",
    ),
    MetricSpec(
        name="analysis_contradiction_rate",
        layer="analysis",
        direction="lower_better",
        summary_path="summary.analysis.contradiction_rate",
        case_layer="analysis",
        case_key="contradiction_rate",
    ),
    MetricSpec(
        name="analysis_numeric_consistency",
        layer="analysis",
        direction="higher_better",
        summary_path="summary.analysis.numeric_consistency",
        case_layer="analysis",
        case_key="numeric_consistency",
    ),
    MetricSpec(
        name="avg_error_rate",
        layer="system",
        direction="lower_better",
        summary_path="summary.system.error_rate",
        case_layer="system",
        case_key="error_rate",
    ),
    MetricSpec(
        name="system_timeout_rate",
        layer="system",
        direction="lower_better",
        summary_path="summary.system.timeout_rate",
        case_layer="system",
        case_key="timeout_rate",
    ),
    MetricSpec(
        name="system_fallback_rate",
        layer="system",
        direction="lower_better",
        summary_path="summary.system.fallback_rate",
        case_layer="system",
        case_key="fallback_rate",
    ),
    MetricSpec(
        name="p95_latency",
        layer="system",
        direction="lower_better",
        bounded_01=False,
        summary_path="summary.system.latency_p95_ms",
        case_layer="system",
        case_key="latency_p95_ms",
    ),
]


def _to_float(value: Any) -> float | None:
    return to_float(value)


def _to_int(value: Any) -> int | None:
    return to_int(value)


def _round(value: float | None, digits: int = 6) -> float | str:
    if value is None or not math.isfinite(value):
        return MISSING
    return round(value, digits)


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


def _warn(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


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


def _extract_case_layer_samples(
    report: dict[str, Any] | None,
    *,
    layer: str,
    key: str,
) -> list[float]:
    if not isinstance(report, dict):
        return []
    out: list[float] = []
    for case in report.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        layers = case.get("layers", {})
        if not isinstance(layers, dict):
            continue
        layer_obj = layers.get(layer, {})
        if not isinstance(layer_obj, dict):
            continue
        value = _to_float(layer_obj.get(key))
        if value is None:
            continue
        out.append(value)
    return out


def _observe_metric(spec: MetricSpec, group: GroupArtifacts) -> MetricObservation:
    report = group.report if isinstance(group.report, dict) else {}
    summary_value = _to_float(get_nested(report, spec.summary_path))
    samples = _extract_case_layer_samples(
        report,
        layer=spec.case_layer,
        key=spec.case_key,
    )
    if summary_value is None:
        summary_value = _mean(samples)

    case_count = _to_int(report.get("case_count"))
    retrieval_case_count = _to_int(get_nested(report, "summary.retrieval.case_count"))
    n = retrieval_case_count if spec.n_scope == "retrieval" else case_count
    if n is None and samples:
        n = len(samples)
    return MetricObservation(value=summary_value, n=n, samples=samples)


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

    ci_mean = _bootstrap_delta_mean(
        current.samples,
        baseline.samples,
        samples=bootstrap_samples,
        seed=seed + abs(hash(spec.name)) % 10000,
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
        seed=seed,
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
        output_raw = item.get("output")
        report_path = _resolve_path(str(output_raw), base_dir=manifest_dir) if output_raw else None
        report = (
            _load_json(report_path, warnings=warnings, label=f"runner_report[{group_id}]")
            if report_path is not None
            else None
        )
        out.append(
            GroupArtifacts(
                group_id=group_id,
                description=str(item.get("description", "")).strip(),
                report_path=report_path,
                report=report,
            )
        )
    return out


def _infer_dataset_info(groups: list[GroupArtifacts]) -> dict[str, str]:
    picked: GroupArtifacts | None = None
    for group in groups:
        if isinstance(group.report, dict):
            picked = group
            break
    if picked is None:
        return {"path": MISSING, "version": MISSING, "suite": MISSING}
    dataset_path = str(get_nested(picked.report, "dataset") or "").strip()
    version = _extract_dataset_version(dataset_path)
    return {
        "path": dataset_path or MISSING,
        "version": version,
        "suite": "task_eval_v1",
    }


def _build_recommendations(report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    regressions = report.get("top_regressions", [])
    if isinstance(regressions, list):
        for item in regressions:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", ""))
            if metric in {"avg_recall_at_10", "avg_mrr_at_10", "avg_ndcg_at_10"}:
                recommendations.append("Retrieval regression detected: tune recall/rerank settings and re-run matrix.")
                break
            if metric in {"tool_path_hit_rate", "tool_param_accuracy", "tool_forbidden_tool_rate"}:
                recommendations.append("Tool-path regression detected: verify routing policy, path-set matching, and arg extraction.")
                break
            if metric in {"analysis_unsupported_claim_rate", "analysis_contradiction_rate"}:
                recommendations.append("Analysis regression detected: tighten evidence grounding and contradiction handling.")
                break
            if metric in {"avg_error_rate", "system_timeout_rate", "system_fallback_rate", "p95_latency"}:
                recommendations.append("System regression detected: inspect timeouts/fallbacks and trace-level failures.")
                break
    gains = report.get("top_gains", [])
    if isinstance(gains, list) and gains:
        first = gains[0]
        if isinstance(first, dict):
            gid = str(first.get("group_id", "")).strip()
            if gid:
                recommendations.append(f"Use `{gid}` as candidate branch and validate on a larger frozen set.")
    if not recommendations:
        recommendations.append("Signal is limited; increase case count and repeat matrix run.")
    return recommendations[:4]


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


def _render_markdown(report: dict[str, Any], metric_order: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Task Eval v1 Matrix Leaderboard")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{report.get('generated_at_utc', MISSING)}`")
    lines.append(f"- Manifest: `{report.get('manifest_path', MISSING)}`")
    lines.append(f"- Baseline group: `{report.get('baseline_group', MISSING)}`")
    lines.append("")
    dataset = report.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    lines.append(f"- Dataset version: `{dataset.get('version', MISSING)}`")
    lines.append(f"- Dataset path: `{dataset.get('path', MISSING)}`")
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
        gid = str(group.get("group_id", ""))
        desc = str(group.get("description", "")).strip()
        title = f"### {gid}" if not desc else f"### {gid} - {desc}"
        lines.append(title)
        lines.append("")
        sources = group.get("sources", {})
        if not isinstance(sources, dict):
            sources = {}
        lines.append(f"- Source: report=`{sources.get('runner_report', MISSING)}`")
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
    return "\n".join(lines)


def build_leaderboard_report(
    *,
    manifest_path: Path,
    eval_dir: Path,
    baseline_group: str,
    top_k: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    manifest = _load_json(manifest_path, warnings=warnings, label="matrix_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"Failed to load manifest: {manifest_path}")
    manifest["_manifest_path"] = str(manifest_path)

    runner = str(manifest.get("runner", "")).strip().lower()
    if runner and runner != "task_eval_v1":
        _warn(warnings, f"[Warn] manifest runner is `{runner}`; expected `task_eval_v1`.")

    groups = _load_group_artifacts(manifest, eval_dir=eval_dir, warnings=warnings)
    if not groups:
        raise ValueError("No groups found in matrix manifest.")

    baseline = next((g for g in groups if g.group_id == baseline_group), None)
    if baseline is None:
        raise ValueError(f"Baseline group not found in manifest groups: {baseline_group}")

    group_records: list[dict[str, Any]] = []
    metric_order = [spec.name for spec in METRIC_SPECS]
    for group in groups:
        metrics: dict[str, Any] = {}
        for spec in METRIC_SPECS:
            cur = _observe_metric(spec, group)
            base = _observe_metric(spec, baseline)
            metrics[spec.name] = _build_metric_record(
                spec=spec,
                current=cur,
                baseline=base,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        group_records.append(
            {
                "group_id": group.group_id,
                "description": group.description,
                "sources": {
                    "runner_report": str(group.report_path) if group.report_path else MISSING,
                },
                "metrics": metrics,
            }
        )

    top_gains, top_regressions = _collect_top_changes(
        group_records,
        baseline_group=baseline_group,
        top_k=top_k,
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "matrix_generated_at_utc": str(manifest.get("generated_at_utc", MISSING)),
        "runner": str(manifest.get("runner", "task_eval_v1")),
        "baseline_group": baseline_group,
        "dataset": _infer_dataset_info(groups),
        "metric_order": metric_order,
        "groups": group_records,
        "top_gains": top_gains,
        "top_regressions": top_regressions,
        "warnings": warnings,
        "recommendations": [],
    }
    report["recommendations"] = _build_recommendations(report)
    return report


def _build_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build task_eval_v1 matrix leaderboard report.")
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
        "--baseline-group",
        type=str,
        default=DEFAULT_BASELINE_GROUP,
        help=f"Baseline group id (default: {DEFAULT_BASELINE_GROUP}).",
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
        "--output-json",
        type=Path,
        default=eval_dir / "reports" / "leaderboard" / "task_eval_v1_latest.json",
        help="Output path for leaderboard JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=eval_dir / "reports" / "leaderboard" / "task_eval_v1_latest.md",
        help="Output path for leaderboard markdown.",
    )
    return parser


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_parser(eval_dir)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve() if args.manifest else _latest_manifest_path(args.matrix_dir.resolve())
    report = build_leaderboard_report(
        manifest_path=manifest_path,
        eval_dir=eval_dir,
        baseline_group=str(args.baseline_group).strip() or DEFAULT_BASELINE_GROUP,
        top_k=max(1, int(args.top_k)),
        bootstrap_samples=max(200, int(args.ci_bootstrap_samples)),
        bootstrap_seed=int(args.ci_seed),
    )
    metric_order = report.get("metric_order", [])
    if not isinstance(metric_order, list):
        metric_order = []
    markdown = _render_markdown(report, [str(x) for x in metric_order])

    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    print(f"[TaskEvalV1Leaderboard] manifest={manifest_path}")
    print(f"[TaskEvalV1Leaderboard] groups={len(report.get('groups', []) or [])}")
    print(f"[TaskEvalV1Leaderboard] json={output_json}")
    print(f"[TaskEvalV1Leaderboard] markdown={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
