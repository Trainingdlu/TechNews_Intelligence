"""Export CV-ready metric conclusions from leaderboard + matrix + failure evidence.

Inputs:
- leaderboard JSON (`eval/build_leaderboard.py` output)
- matrix manifest JSON (`eval/run_matrix_eval.py` / pipeline manifest output)
- key failure samples (JSON/JSONL files)
- optional LangSmith trace exports (to map request_id -> run_id)

Outputs:
- cv_metrics.json
- summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from common import parse_csv_tokens, read_json_object, to_float, to_int
except ImportError:  # package-style import fallback
    from .common import parse_csv_tokens, read_json_object, to_float, to_int

MISSING = "missing"
DEFAULT_BASELINE = "G0_baseline"
DEFAULT_BOOTSTRAP_SAMPLES = 800
DEFAULT_BOOTSTRAP_SEED = 17

LEADERBOARD_METRIC_LAYER: dict[str, str] = {
    "avg_recall_at_10": "retrieval",
    "avg_mrr_at_10": "retrieval",
    "avg_ndcg_at_10": "retrieval",
    "avg_error_rate": "tool_path",
    "react_success_rate": "tool_path",
    "p95_latency": "tool_path",
    "avg_composite": "judge",
}

DEFAULT_DIRECTION_BY_METRIC: dict[str, str] = {
    "avg_recall_at_5": "higher_better",
    "avg_recall_at_10": "higher_better",
    "avg_mrr_at_10": "higher_better",
    "avg_ndcg_at_10": "higher_better",
    "avg_error_rate": "lower_better",
    "react_success_rate": "higher_better",
    "p95_latency": "lower_better",
    "avg_composite": "higher_better",
    "avg_tool_path_hit_rate": "higher_better",
    "avg_tool_path_accept_hit_rate": "higher_better",
}

EXTRA_METRICS: dict[str, dict[str, str]] = {
    "avg_recall_at_5": {
        "layer": "retrieval",
        "direction": "higher_better",
        "summary_key": "avg_recall_at_5",
        "case_key": "recall_at_5",
        "n_key": "retrieval_case_count",
    },
    "avg_tool_path_hit_rate": {
        "layer": "tool_path",
        "direction": "higher_better",
        "summary_key": "avg_tool_path_hit_rate",
        "case_key": "tool_path_hit_rate",
        "n_key": "case_count",
    },
    "avg_tool_path_accept_hit_rate": {
        "layer": "tool_path",
        "direction": "higher_better",
        "summary_key": "avg_tool_path_accept_hit_rate",
        "case_key": "tool_path_accept_hit_rate",
        "n_key": "case_count",
    },
}

LAYER_METRIC_MAP: dict[str, tuple[str, ...]] = {
    "retrieval": ("avg_recall_at_5", "avg_recall_at_10"),
    "rerank": ("avg_mrr_at_10", "avg_ndcg_at_10"),
    "judge": ("avg_composite",),
    "tool_path": ("react_success_rate", "avg_error_rate", "avg_tool_path_hit_rate", "avg_tool_path_accept_hit_rate"),
}

RETRIEVAL_ENV_KEYS = ("EVAL_RETRIEVAL_VARIANT",)
RERANK_ENV_KEYS = ("NEWS_RERANK_MODE", "SEARCH_NEWS_RERANK_MODE", "FULLTEXT_BATCH_RERANK_MODE")
GENERATION_ENV_KEYS = ("EVAL_AGENT_VARIANT", "AGENT_PROMPT_VARIANT")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    return to_float(value)


def _to_int(value: Any) -> int | None:
    return to_int(value)


def _round(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return MISSING
    if not math.isfinite(value):
        return MISSING
    return round(value, digits)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    if mean is None:
        return None
    return sum((x - mean) ** 2 for x in values) / len(values)


def _delta_pct(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    delta = current - baseline
    if abs(baseline) <= 1e-12:
        return 0.0 if abs(delta) <= 1e-12 else None
    return (delta / abs(baseline)) * 100.0


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (p / 100.0)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_vals[low])
    frac = pos - low
    return float(sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac)


def _bootstrap_delta_mean_ci(
    current: list[float],
    baseline: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | str]:
    if len(current) < 2 or len(baseline) < 2:
        return {"lower": MISSING, "upper": MISSING, "method": MISSING}
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        cur_mean = sum(rng.choice(current) for _ in range(len(current))) / len(current)
        base_mean = sum(rng.choice(baseline) for _ in range(len(baseline))) / len(baseline)
        deltas.append(cur_mean - base_mean)
    lower = _percentile(deltas, 2.5)
    upper = _percentile(deltas, 97.5)
    if lower is None or upper is None:
        return {"lower": MISSING, "upper": MISSING, "method": MISSING}
    return {
        "lower": _round(lower),
        "upper": _round(upper),
        "method": "bootstrap_delta_mean",
    }


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _parse_csv(value: str) -> list[str]:
    return parse_csv_tokens(value)


def _load_json(path: Path) -> dict[str, Any]:
    return read_json_object(path, encoding="utf-8-sig")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no} in {path}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_path(raw: str | None, *, base_dir: Path) -> Path | None:
    if not raw:
        return None
    token = str(raw).strip()
    if not token:
        return None
    path = Path(token)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _first_present_dict(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            out: dict[str, str] = {}
            for k, v in value.items():
                k_token = str(k).strip()
                v_token = str(v).strip()
                if k_token and v_token:
                    out[k_token] = v_token
            if out:
                return out
    return {}


def _safe_group_env(group_record: dict[str, Any]) -> dict[str, str]:
    env = _first_present_dict(group_record, ("env_overrides", "env"))
    if env:
        return env
    outputs = group_record.get("outputs")
    if isinstance(outputs, dict):
        env_from_outputs = _first_present_dict(outputs, ("env_overrides", "env"))
        if env_from_outputs:
            return env_from_outputs
    return {}


def _extract_manifest_groups(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    groups = manifest.get("groups", [])
    if not isinstance(groups, list):
        return out
    for item in groups:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("id", "")).strip()
        if group_id:
            out[group_id] = item
    return out


def _extract_request_ids_from_run(run_payload: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("run_id", "request_id", "id"):
        value = run_payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    trace_summary = run_payload.get("trace_summary")
    if isinstance(trace_summary, dict):
        req = trace_summary.get("request_id")
        if isinstance(req, str) and req.strip():
            candidates.append(req.strip())
    return _dedupe_keep_order(candidates)


def _build_case_request_index(run_eval_report: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cases = run_eval_report.get("cases", [])
    if not isinstance(cases, list):
        return out
    for idx, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id", "")).strip() or f"case_{idx}"
        run_ids: list[str] = []
        runs = case.get("runs", [])
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                run_ids.extend(_extract_request_ids_from_run(run))
        out[case_id] = _dedupe_keep_order(run_ids)
    return out


def _collect_all_request_ids(run_eval_report: dict[str, Any]) -> list[str]:
    index = _build_case_request_index(run_eval_report)
    merged: list[str] = []
    for run_ids in index.values():
        merged.extend(run_ids)
    return _dedupe_keep_order(merged)


def _extract_case_metric_samples(run_eval_report: dict[str, Any], metric_key: str) -> list[float]:
    out: list[float] = []
    cases = run_eval_report.get("cases", [])
    if not isinstance(cases, list):
        return out
    for case in cases:
        if not isinstance(case, dict):
            continue
        metrics = case.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        value = _to_float(metrics.get(metric_key))
        if value is not None:
            out.append(value)
    return out


def _extract_latency_samples(run_eval_report: dict[str, Any]) -> list[float]:
    latencies: list[float] = []
    cases = run_eval_report.get("cases", [])
    if not isinstance(cases, list):
        return latencies
    for case in cases:
        if not isinstance(case, dict):
            continue
        runs = case.get("runs", [])
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            direct = _to_float(run.get("latency_ms"))
            if direct is not None:
                latencies.append(direct)
                continue
            trace_summary = run.get("trace_summary")
            if isinstance(trace_summary, dict):
                traced = _to_float(trace_summary.get("latency_ms"))
                if traced is not None:
                    latencies.append(traced)
    return latencies


def _extract_judge_samples(judge_report: dict[str, Any]) -> list[float]:
    out: list[float] = []
    rows = judge_report.get("rows", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        scores = row.get("scores", {})
        if isinstance(scores, dict):
            composite = _to_float(scores.get("composite"))
            if composite is not None:
                out.append(composite)
                continue
        composite = _to_float(row.get("composite"))
        if composite is not None:
            out.append(composite)
    return out


def _extract_case_ids_for_metric(
    metric_name: str,
    *,
    run_eval_report: dict[str, Any] | None,
    judge_report: dict[str, Any] | None,
) -> list[str]:
    if run_eval_report is None:
        return []
    case_ids: list[str] = []

    if metric_name in {"avg_recall_at_5", "avg_recall_at_10", "avg_mrr_at_10", "avg_ndcg_at_10"}:
        metric_key = {
            "avg_recall_at_5": "recall_at_5",
            "avg_recall_at_10": "recall_at_10",
            "avg_mrr_at_10": "mrr_at_10",
            "avg_ndcg_at_10": "ndcg_at_10",
        }[metric_name]
        for case in run_eval_report.get("cases", []) or []:
            if not isinstance(case, dict):
                continue
            metrics = case.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            if _to_float(metrics.get(metric_key)) is not None:
                case_id = str(case.get("id", "")).strip()
                if case_id:
                    case_ids.append(case_id)
        return _dedupe_keep_order(case_ids)

    if metric_name in {"avg_error_rate", "react_success_rate", "p95_latency", "avg_tool_path_hit_rate", "avg_tool_path_accept_hit_rate"}:
        for case in run_eval_report.get("cases", []) or []:
            if isinstance(case, dict):
                case_id = str(case.get("id", "")).strip()
                if case_id:
                    case_ids.append(case_id)
        return _dedupe_keep_order(case_ids)

    if metric_name == "avg_composite" and isinstance(judge_report, dict):
        rows = judge_report.get("rows", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                scores = row.get("scores", {})
                has_composite = False
                if isinstance(scores, dict) and _to_float(scores.get("composite")) is not None:
                    has_composite = True
                if not has_composite and _to_float(row.get("composite")) is not None:
                    has_composite = True
                if not has_composite:
                    continue
                case_id = str(row.get("case_id", "")).strip()
                if case_id:
                    case_ids.append(case_id)
        return _dedupe_keep_order(case_ids)

    return []


def _load_objects_from_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            for key in ("rows", "items", "failures", "samples", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    return []


def _collect_candidate_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".json", ".jsonl"}:
                    out.append(child)
        elif path.exists() and path.suffix.lower() in {".json", ".jsonl"}:
            out.append(path)
    deduped = _dedupe_keep_order([str(p.resolve()) for p in out])
    return [Path(p) for p in deduped]


def _extract_request_id_from_tags(tags: Any) -> str | None:
    if not isinstance(tags, list):
        return None
    for item in tags:
        token = str(item).strip()
        if token.startswith("request:") and len(token) > len("request:"):
            return token[len("request:") :].strip() or None
    return None


def _safe_get_nested(payload: dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur[part]
    return cur


def _extract_trace_mapping_item(item: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    run_id = None
    for key in ("run_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            run_id = value.strip()
            break
    trace_id = None
    for key in ("trace_id",):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            trace_id = value.strip()
            break

    request_id: str | None = None
    request_candidates = (
        "request_id",
        "metadata.request_id",
        "extra.metadata.request_id",
        "inputs.request_id",
        "run.inputs.request_id",
    )
    for path in request_candidates:
        value = _safe_get_nested(item, path) if "." in path else item.get(path)
        if isinstance(value, str) and value.strip():
            request_id = value.strip()
            break
    if request_id is None:
        tags = item.get("tags")
        if tags is None:
            tags = _safe_get_nested(item, "extra.tags")
        request_id = _extract_request_id_from_tags(tags)
    return (request_id, run_id, trace_id)


def _build_trace_index(trace_paths: list[Path]) -> dict[str, dict[str, list[str]]]:
    index: dict[str, dict[str, set[str]]] = {}
    for path in _collect_candidate_files(trace_paths):
        for item in _load_objects_from_path(path):
            request_id, run_id, trace_id = _extract_trace_mapping_item(item)
            if request_id is None:
                continue
            entry = index.setdefault(request_id, {"run_ids": set(), "trace_ids": set()})
            if run_id:
                entry["run_ids"].add(run_id)
            if trace_id:
                entry["trace_ids"].add(trace_id)

    normalized: dict[str, dict[str, list[str]]] = {}
    for request_id, entry in index.items():
        normalized[request_id] = {
            "run_ids": sorted(entry["run_ids"]),
            "trace_ids": sorted(entry["trace_ids"]),
        }
    return normalized


def _normalize_layer(value: Any, metric_name: str | None = None, reason: str | None = None) -> str:
    token = str(value or "").strip().lower()
    mapping = {
        "retrieval": "retrieval",
        "search": "retrieval",
        "rerank": "rerank",
        "re_rank": "rerank",
        "generation": "judge",
        "judge": "judge",
        "system": "tool_path",
        "tool_path": "tool_path",
        "tool": "tool_path",
        "route": "tool_path",
    }
    if token in mapping:
        return mapping[token]

    metric_token = str(metric_name or "").strip()
    if metric_token in LAYER_METRIC_MAP["retrieval"]:
        return "retrieval"
    if metric_token in LAYER_METRIC_MAP["rerank"]:
        return "rerank"
    if metric_token in LAYER_METRIC_MAP["judge"]:
        return "judge"
    if metric_token in LAYER_METRIC_MAP["tool_path"]:
        return "tool_path"

    text = str(reason or "").strip().lower()
    if "rerank" in text:
        return "rerank"
    if "retriev" in text or "recall" in text:
        return "retrieval"
    if "judge" in text or "ground" in text or "answer" in text:
        return "judge"
    if "tool" in text or "route" in text or "error" in text:
        return "tool_path"
    return "unknown"


def _extract_multi_ids(item: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            token = value.strip()
            if token:
                out.append(token)
        elif isinstance(value, list):
            for nested in value:
                token = str(nested).strip()
                if token:
                    out.append(token)
    return _dedupe_keep_order(out)


def _normalize_failure_item(item: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    metric = str(item.get("metric") or item.get("metric_name") or "").strip()
    reason = str(
        item.get("reason")
        or item.get("message")
        or item.get("error")
        or item.get("failure_reason")
        or ""
    ).strip()
    layer = _normalize_layer(item.get("layer"), metric_name=metric, reason=reason)
    group_id = str(
        item.get("group_id")
        or item.get("experiment_group")
        or item.get("candidate_group")
        or item.get("group")
        or ""
    ).strip()
    case_id = str(item.get("case_id") or item.get("id") or "").strip()
    run_ids = _extract_multi_ids(item, ("run_id", "run_ids", "request_id", "request_ids", "trace_id", "trace_ids"))
    return {
        "group_id": group_id,
        "case_id": case_id,
        "metric": metric,
        "layer": layer,
        "reason": reason,
        "run_refs": run_ids,
        "source_file": str(source_path.resolve()),
    }


def _load_failure_samples(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _collect_candidate_files(paths):
        for item in _load_objects_from_path(path):
            rows.append(_normalize_failure_item(item, source_path=path))
    return rows


def _resolve_metric_layer(metric_name: str, candidate_record: dict[str, Any]) -> str:
    explicit = str(candidate_record.get("layer", "")).strip().lower()
    if explicit:
        normalized = _normalize_layer(explicit, metric_name=metric_name, reason="")
        if normalized != "unknown":
            return normalized
    return LEADERBOARD_METRIC_LAYER.get(metric_name, "unknown")


def _resolve_metric_direction(metric_name: str, candidate_record: dict[str, Any]) -> str:
    explicit = str(candidate_record.get("direction", "")).strip().lower()
    if explicit in {"higher_better", "lower_better"}:
        return explicit
    return DEFAULT_DIRECTION_BY_METRIC.get(metric_name, "higher_better")


def _improvement_score(direction: str, delta_abs: float | None) -> float | None:
    if delta_abs is None:
        return None
    if direction == "lower_better":
        return -delta_abs
    return delta_abs


def _is_ci_proven_improvement(direction: str, ci_95: dict[str, Any]) -> bool:
    lower = _to_float(ci_95.get("lower"))
    upper = _to_float(ci_95.get("upper"))
    if lower is None or upper is None:
        return False
    if direction == "lower_better":
        return upper < 0.0
    return lower > 0.0


def _format_ci(ci_95: dict[str, Any]) -> str:
    lower = _to_float(ci_95.get("lower"))
    upper = _to_float(ci_95.get("upper"))
    method = str(ci_95.get("method", MISSING))
    if lower is None or upper is None:
        return "CI=missing"
    return f"95%CI(delta)=[{lower:+.4f}, {upper:+.4f}] ({method})"


def _resolve_request_ids_to_run_refs(
    request_ids: list[str],
    *,
    trace_index: dict[str, dict[str, list[str]]],
    max_count: int,
) -> tuple[list[str], str]:
    resolved_run_ids: list[str] = []
    for request_id in request_ids:
        mapping = trace_index.get(request_id)
        if not mapping:
            continue
        resolved_run_ids.extend(mapping.get("run_ids", []))
    resolved_run_ids = _dedupe_keep_order(resolved_run_ids)
    if resolved_run_ids:
        return (resolved_run_ids[:max_count], "run_id")
    return (_dedupe_keep_order(request_ids)[:max_count], "request_id")


def _build_group_paths(
    group_id: str,
    *,
    leaderboard_group: dict[str, Any],
    manifest_group: dict[str, Any] | None,
    leaderboard_base_dir: Path,
    manifest_base_dir: Path | None,
) -> dict[str, Path | None]:
    sources = leaderboard_group.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}

    def _resolve_from_sources(key: str) -> Path | None:
        raw = sources.get(key)
        if isinstance(raw, str) and raw.strip() and raw != MISSING:
            return _resolve_path(raw, base_dir=leaderboard_base_dir)
        return None

    run_eval_path = _resolve_from_sources("run_eval")
    judge_path = _resolve_from_sources("judge")

    if manifest_group is not None:
        base_dir = manifest_base_dir or leaderboard_base_dir
        if run_eval_path is None:
            raw_output = manifest_group.get("output")
            if isinstance(raw_output, str) and raw_output.strip():
                run_eval_path = _resolve_path(raw_output, base_dir=base_dir)
        if judge_path is None:
            raw_judge = manifest_group.get("judge_output")
            if isinstance(raw_judge, str) and raw_judge.strip():
                judge_path = _resolve_path(raw_judge, base_dir=base_dir)

    return {
        "run_eval": run_eval_path,
        "judge": judge_path,
    }


def _safe_load_optional_json(path: Path | None, warnings: list[str], label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        warnings.append(f"[Warn] missing {label} file: {path}")
        return None
    try:
        payload = _load_json(path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"[Warn] failed to read {label} file {path}: {type(exc).__name__}: {exc}")
        return None
    return payload


def _get_metric_samples(
    metric_name: str,
    *,
    run_eval_report: dict[str, Any] | None,
    judge_report: dict[str, Any] | None,
) -> list[float]:
    if metric_name == "avg_recall_at_5":
        return _extract_case_metric_samples(run_eval_report or {}, "recall_at_5")
    if metric_name == "avg_recall_at_10":
        return _extract_case_metric_samples(run_eval_report or {}, "recall_at_10")
    if metric_name == "avg_mrr_at_10":
        return _extract_case_metric_samples(run_eval_report or {}, "mrr_at_10")
    if metric_name == "avg_ndcg_at_10":
        return _extract_case_metric_samples(run_eval_report or {}, "ndcg_at_10")
    if metric_name == "avg_error_rate":
        return _extract_case_metric_samples(run_eval_report or {}, "error_rate")
    if metric_name == "avg_tool_path_hit_rate":
        return _extract_case_metric_samples(run_eval_report or {}, "tool_path_hit_rate")
    if metric_name == "avg_tool_path_accept_hit_rate":
        return _extract_case_metric_samples(run_eval_report or {}, "tool_path_accept_hit_rate")
    if metric_name == "p95_latency":
        return _extract_latency_samples(run_eval_report or {})
    if metric_name == "avg_composite":
        return _extract_judge_samples(judge_report or {})
    return []


def _summary_value(report: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(report, dict):
        return None
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return None
    return _to_float(summary.get(key))


def _build_metric_record(
    *,
    metric_name: str,
    layer: str,
    direction: str,
    current: float | None,
    baseline: float | None,
    n: int | None,
    ci_95: dict[str, Any],
    current_samples: list[float],
    baseline_samples: list[float],
    source_files: list[str],
    request_ids: list[str],
    trace_index: dict[str, dict[str, list[str]]],
    max_run_refs: int,
) -> dict[str, Any]:
    delta_abs = None
    if current is not None and baseline is not None:
        delta_abs = current - baseline
    delta_pct = _delta_pct(current, baseline)
    score = _improvement_score(direction, delta_abs)
    observed_positive = bool(score is not None and score > 0)
    proven_positive = _is_ci_proven_improvement(direction, ci_95)

    run_refs, run_ref_field = _resolve_request_ids_to_run_refs(
        request_ids,
        trace_index=trace_index,
        max_count=max_run_refs,
    )
    return {
        "metric": metric_name,
        "layer": layer,
        "direction": direction,
        "current": _round(current),
        "baseline": _round(baseline),
        "delta_abs": _round(delta_abs),
        "delta_pct": _round(delta_pct, digits=4),
        "n": n if n is not None else MISSING,
        "ci_95": {
            "lower": _round(_to_float(ci_95.get("lower"))),
            "upper": _round(_to_float(ci_95.get("upper"))),
            "method": str(ci_95.get("method", MISSING)),
        },
        "variance": {
            "current": _round(_variance(current_samples)),
            "baseline": _round(_variance(baseline_samples)),
        },
        "improvement_score": _round(score),
        "observed_positive": observed_positive,
        "proven_positive": proven_positive,
        "evidence": {
            "metric_files": _dedupe_keep_order(source_files),
            "run_id_field": run_ref_field,
            "run_ids": run_refs,
            "request_ids": _dedupe_keep_order(request_ids)[:max_run_refs],
        },
    }


def _collect_metric_request_ids(
    metric_name: str,
    *,
    run_eval_report: dict[str, Any] | None,
    judge_report: dict[str, Any] | None,
    case_request_index: dict[str, list[str]],
) -> list[str]:
    case_ids = _extract_case_ids_for_metric(
        metric_name,
        run_eval_report=run_eval_report,
        judge_report=judge_report,
    )
    if not case_ids and run_eval_report is not None:
        return _collect_all_request_ids(run_eval_report)
    out: list[str] = []
    for case_id in case_ids:
        out.extend(case_request_index.get(case_id, []))
    return _dedupe_keep_order(out)


def _collect_group_metrics(
    *,
    group_id: str,
    leaderboard_group: dict[str, Any],
    run_eval_report: dict[str, Any] | None,
    baseline_run_eval_report: dict[str, Any] | None,
    judge_report: dict[str, Any] | None,
    baseline_judge_report: dict[str, Any] | None,
    leaderboard_path: Path,
    group_paths: dict[str, Path | None],
    trace_index: dict[str, dict[str, list[str]]],
    max_run_refs: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metric_rows = leaderboard_group.get("metrics", {})
    if not isinstance(metric_rows, dict):
        metric_rows = {}

    case_request_index = _build_case_request_index(run_eval_report or {})

    for metric_name, raw_record in metric_rows.items():
        if not isinstance(raw_record, dict):
            continue
        layer = _resolve_metric_layer(metric_name, raw_record)
        direction = _resolve_metric_direction(metric_name, raw_record)
        current = _to_float(raw_record.get("current"))
        baseline = _to_float(raw_record.get("baseline"))
        n = _to_int(raw_record.get("n"))
        ci_95 = raw_record.get("ci_95", {})
        if not isinstance(ci_95, dict):
            ci_95 = {"lower": MISSING, "upper": MISSING, "method": MISSING}

        current_samples = _get_metric_samples(
            metric_name,
            run_eval_report=run_eval_report,
            judge_report=judge_report,
        )
        baseline_samples = _get_metric_samples(
            metric_name,
            run_eval_report=baseline_run_eval_report,
            judge_report=baseline_judge_report,
        )

        source_files = [str(leaderboard_path.resolve())]
        if metric_name == "avg_composite" and group_paths.get("judge"):
            source_files.append(str(group_paths["judge"].resolve()))
        elif group_paths.get("run_eval"):
            source_files.append(str(group_paths["run_eval"].resolve()))

        request_ids = _collect_metric_request_ids(
            metric_name,
            run_eval_report=run_eval_report,
            judge_report=judge_report,
            case_request_index=case_request_index,
        )

        records.append(
            _build_metric_record(
                metric_name=metric_name,
                layer=layer,
                direction=direction,
                current=current,
                baseline=baseline,
                n=n,
                ci_95=ci_95,
                current_samples=current_samples,
                baseline_samples=baseline_samples,
                source_files=source_files,
                request_ids=request_ids,
                trace_index=trace_index,
                max_run_refs=max_run_refs,
            )
        )

    # Add extra metrics directly from run_eval summary/case samples.
    for metric_name, spec in EXTRA_METRICS.items():
        if any(str(row.get("metric")) == metric_name for row in records):
            continue
        current = _summary_value(run_eval_report, spec["summary_key"])
        baseline = _summary_value(baseline_run_eval_report, spec["summary_key"])
        if current is None or baseline is None:
            continue
        current_samples = _extract_case_metric_samples(run_eval_report or {}, spec["case_key"])
        baseline_samples = _extract_case_metric_samples(baseline_run_eval_report or {}, spec["case_key"])

        n = _to_int(_safe_get_nested(run_eval_report or {}, f"summary.{spec['n_key']}"))
        if n is None and current_samples:
            n = len(current_samples)

        ci_95 = _bootstrap_delta_mean_ci(
            current_samples,
            baseline_samples,
            samples=DEFAULT_BOOTSTRAP_SAMPLES,
            seed=DEFAULT_BOOTSTRAP_SEED + abs(hash(f"{group_id}:{metric_name}")) % 10000,
        )
        request_ids = _collect_metric_request_ids(
            metric_name,
            run_eval_report=run_eval_report,
            judge_report=judge_report,
            case_request_index=case_request_index,
        )

        source_files = [str(leaderboard_path.resolve())]
        if group_paths.get("run_eval"):
            source_files.append(str(group_paths["run_eval"].resolve()))

        records.append(
            _build_metric_record(
                metric_name=metric_name,
                layer=spec["layer"],
                direction=spec["direction"],
                current=current,
                baseline=baseline,
                n=n,
                ci_95=ci_95,
                current_samples=current_samples,
                baseline_samples=baseline_samples,
                source_files=source_files,
                request_ids=request_ids,
                trace_index=trace_index,
                max_run_refs=max_run_refs,
            )
        )

    # Ensure deterministic order.
    records.sort(key=lambda row: str(row.get("metric")))
    return records


def _diff_env(baseline_env: dict[str, str], candidate_env: dict[str, str]) -> dict[str, dict[str, str]]:
    diff: dict[str, dict[str, str]] = {}
    keys = sorted(set(baseline_env.keys()).union(candidate_env.keys()))
    for key in keys:
        base = str(baseline_env.get(key, "")).strip()
        cur = str(candidate_env.get(key, "")).strip()
        if base == cur:
            continue
        diff[key] = {"baseline": base, "candidate": cur}
    return diff


def _layer_changed_flags(env_diff: dict[str, dict[str, str]]) -> dict[str, bool]:
    changed_keys = set(env_diff.keys())
    retrieval_changed = any(key in changed_keys for key in RETRIEVAL_ENV_KEYS)
    rerank_changed = any(key in changed_keys for key in RERANK_ENV_KEYS)
    judge_changed = any(key in changed_keys for key in GENERATION_ENV_KEYS)
    tool_path_changed = judge_changed or any(("TOOL" in key or "ROUTE" in key or "RETRY" in key) for key in changed_keys)
    return {
        "retrieval": retrieval_changed,
        "rerank": rerank_changed,
        "judge": judge_changed,
        "tool_path": tool_path_changed,
    }


def _summarize_failures(
    failure_rows: list[dict[str, Any]],
    *,
    candidate_group: str,
    layer: str,
) -> dict[str, Any]:
    scoped = [
        row
        for row in failure_rows
        if str(row.get("layer", "")) == layer
        and (not str(row.get("group_id", "")).strip() or str(row.get("group_id", "")).strip() == candidate_group)
    ]
    run_refs: list[str] = []
    sources: list[str] = []
    for row in scoped:
        run_refs.extend([str(x).strip() for x in row.get("run_refs", []) if str(x).strip()])
        source_file = str(row.get("source_file", "")).strip()
        if source_file:
            sources.append(source_file)
    return {
        "count": len(scoped),
        "run_refs": _dedupe_keep_order(run_refs),
        "source_files": _dedupe_keep_order(sources),
    }


def _build_layer_attribution(
    *,
    candidate_group: str,
    metric_rows: list[dict[str, Any]],
    baseline_env: dict[str, str],
    candidate_env: dict[str, str],
    failure_rows: list[dict[str, Any]],
    manifest_path: Path | None,
    max_run_refs: int,
) -> list[dict[str, Any]]:
    env_diff = _diff_env(baseline_env, candidate_env)
    changed_flags = _layer_changed_flags(env_diff)
    out: list[dict[str, Any]] = []

    for layer in ("retrieval", "rerank", "judge", "tool_path"):
        relevant_metrics = [
            row
            for row in metric_rows
            if str(row.get("metric")) in set(LAYER_METRIC_MAP[layer])
        ]
        observed_positive = [row for row in relevant_metrics if bool(row.get("observed_positive"))]
        proven_positive = [row for row in relevant_metrics if bool(row.get("proven_positive"))]
        observed_negative = [
            row
            for row in relevant_metrics
            if isinstance(_to_float(row.get("improvement_score")), float) and float(_to_float(row.get("improvement_score")) or 0.0) < 0
        ]

        failure_summary = _summarize_failures(
            failure_rows,
            candidate_group=candidate_group,
            layer=layer,
        )

        if proven_positive and not observed_negative:
            status = "supported"
        elif proven_positive and observed_negative:
            status = "mixed"
        elif observed_positive and not observed_negative:
            status = "observed"
        elif observed_negative:
            status = "regressed"
        else:
            status = "insufficient"

        score_candidates = [float(_to_float(row.get("improvement_score")) or 0.0) for row in relevant_metrics]
        layer_score = _mean(score_candidates) if score_candidates else None

        run_refs: list[str] = []
        run_ref_field = "request_id"
        metric_files: list[str] = []
        for row in relevant_metrics:
            evidence = row.get("evidence", {})
            if not isinstance(evidence, dict):
                continue
            run_ref_field = str(evidence.get("run_id_field", run_ref_field))
            refs = evidence.get("run_ids", [])
            if isinstance(refs, list):
                run_refs.extend([str(item).strip() for item in refs if str(item).strip()])
            files = evidence.get("metric_files", [])
            if isinstance(files, list):
                metric_files.extend([str(item).strip() for item in files if str(item).strip()])

        run_refs = _dedupe_keep_order(run_refs)
        metric_files = _dedupe_keep_order(metric_files)
        if manifest_path is not None:
            metric_files.append(str(manifest_path.resolve()))
            metric_files = _dedupe_keep_order(metric_files)
        if failure_summary["source_files"]:
            metric_files.extend(failure_summary["source_files"])
            metric_files = _dedupe_keep_order(metric_files)
        if failure_summary["run_refs"]:
            run_refs.extend(failure_summary["run_refs"])
            run_refs = _dedupe_keep_order(run_refs)

        out.append(
            {
                "layer": layer,
                "env_changed": bool(changed_flags[layer]),
                "status": status,
                "metric_count": len(relevant_metrics),
                "observed_positive_metrics": [str(row.get("metric")) for row in observed_positive],
                "proven_positive_metrics": [str(row.get("metric")) for row in proven_positive],
                "observed_negative_metrics": [str(row.get("metric")) for row in observed_negative],
                "improvement_score_mean": _round(layer_score),
                "failure_samples": {
                    "count": failure_summary["count"],
                    "source_files": failure_summary["source_files"],
                },
                "evidence": {
                    "run_id_field": run_ref_field,
                    "run_ids": run_refs[:max_run_refs],
                    "metric_files": metric_files,
                },
            }
        )

    return out


def _pick_metric_for_relative(metric_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_order = (
        "avg_recall_at_5",
        "avg_recall_at_10",
        "avg_mrr_at_10",
        "avg_ndcg_at_10",
        "react_success_rate",
        "avg_tool_path_hit_rate",
    )
    candidates = [
        row
        for row in metric_rows
        if isinstance(_to_float(row.get("delta_pct")), float)
        and bool(row.get("observed_positive"))
    ]
    if not candidates:
        return None
    by_metric = {str(row.get("metric")): row for row in candidates}
    for metric in preferred_order:
        if metric in by_metric:
            return by_metric[metric]
    candidates.sort(key=lambda row: abs(float(_to_float(row.get("delta_pct")) or 0.0)), reverse=True)
    return candidates[0]


def _pick_metric_for_absolute(metric_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_order = (
        "avg_composite",
        "avg_recall_at_10",
        "avg_mrr_at_10",
        "avg_error_rate",
    )
    candidates = [
        row
        for row in metric_rows
        if isinstance(_to_float(row.get("delta_abs")), float)
        and bool(row.get("observed_positive"))
    ]
    if not candidates:
        return None
    by_metric = {str(row.get("metric")): row for row in candidates}
    for metric in preferred_order:
        if metric in by_metric:
            return by_metric[metric]
    candidates.sort(key=lambda row: abs(float(_to_float(row.get("delta_abs")) or 0.0)), reverse=True)
    return candidates[0]


def _pick_metric_for_stability(metric_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    with_ci = []
    for row in metric_rows:
        ci_95 = row.get("ci_95", {})
        if not isinstance(ci_95, dict):
            continue
        if _to_float(ci_95.get("lower")) is None or _to_float(ci_95.get("upper")) is None:
            continue
        with_ci.append(row)
    if with_ci:
        with_ci.sort(
            key=lambda row: (
                0 if bool(row.get("proven_positive")) else 1,
                -abs(float(_to_float(row.get("delta_abs")) or 0.0)),
            )
        )
        return with_ci[0]
    with_var = [
        row
        for row in metric_rows
        if _to_float(_safe_get_nested(row, "variance.current")) is not None
        and _to_float(_safe_get_nested(row, "variance.baseline")) is not None
    ]
    if with_var:
        with_var.sort(key=lambda row: -abs(float(_to_float(row.get("delta_abs")) or 0.0)))
        return with_var[0]
    return None


def _conclusion_from_metric(
    *,
    conclusion_id: str,
    conclusion_type: str,
    baseline_group: str,
    candidate_group: str,
    metric_row: dict[str, Any],
    experiment_meta: dict[str, Any],
) -> dict[str, Any]:
    metric = str(metric_row.get("metric", ""))
    layer = str(metric_row.get("layer", "unknown"))
    current = _to_float(metric_row.get("current"))
    baseline = _to_float(metric_row.get("baseline"))
    delta_abs = _to_float(metric_row.get("delta_abs"))
    delta_pct = _to_float(metric_row.get("delta_pct"))
    n = _to_int(metric_row.get("n"))
    ci_95 = metric_row.get("ci_95", {})
    if not isinstance(ci_95, dict):
        ci_95 = {"lower": MISSING, "upper": MISSING, "method": MISSING}
    evidence = metric_row.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}

    base_text = (
        f"{candidate_group} vs {baseline_group}: {metric} "
        f"{'from' if baseline is not None and current is not None else 'delta'}"
    )
    if baseline is not None and current is not None:
        base_text = (
            f"{candidate_group} 相比 {baseline_group}，{metric} 从 {baseline:.4f} 到 {current:.4f}，"
            f"绝对变化 {delta_abs:+.4f}。"
        )
    if conclusion_type == "relative_improvement" and delta_pct is not None:
        base_text = (
            f"{candidate_group} 相比 {baseline_group}，{metric} 相对变化 {delta_pct:+.2f}% "
            f"（绝对变化 {delta_abs:+.4f}，n={n if n is not None else MISSING}）。"
        )
    if conclusion_type == "stability":
        var_cur = _to_float(_safe_get_nested(metric_row, "variance.current"))
        var_base = _to_float(_safe_get_nested(metric_row, "variance.baseline"))
        base_text = (
            f"{candidate_group} 的 {metric} 稳定性：{_format_ci(ci_95)}，"
            f"方差 current={var_cur if var_cur is not None else MISSING}，"
            f"baseline={var_base if var_base is not None else MISSING}。"
        )

    ci_text = _format_ci(ci_95)
    if bool(metric_row.get("proven_positive")):
        evidence_note = "CI 下界/上界支持正向结论。"
    elif _to_float(ci_95.get("lower")) is not None and _to_float(ci_95.get("upper")) is not None:
        evidence_note = "CI 跨 0，不做显著性断言，仅报告观测差异。"
    else:
        evidence_note = "CI 缺失，仅报告观测差异。"

    statement = f"{base_text} {ci_text} {evidence_note}"
    return {
        "id": conclusion_id,
        "type": conclusion_type,
        "group_id": candidate_group,
        "layer": layer,
        "statement": statement,
        "metric": metric,
        "values": {
            "current": _round(current),
            "baseline": _round(baseline),
            "delta_abs": _round(delta_abs),
            "delta_pct": _round(delta_pct, digits=4),
            "n": n if n is not None else MISSING,
            "ci_95": ci_95,
        },
        "experiment": experiment_meta,
        "evidence": {
            "run_id_field": str(evidence.get("run_id_field", "request_id")),
            "run_ids": list(evidence.get("run_ids", [])) if isinstance(evidence.get("run_ids"), list) else [],
            "metric_files": list(evidence.get("metric_files", [])) if isinstance(evidence.get("metric_files"), list) else [],
        },
    }


def _conclusion_from_attribution(
    *,
    conclusion_id: str,
    baseline_group: str,
    candidate_group: str,
    layer_row: dict[str, Any],
    experiment_meta: dict[str, Any],
) -> dict[str, Any]:
    layer = str(layer_row.get("layer", "unknown"))
    status = str(layer_row.get("status", "insufficient"))
    changed = bool(layer_row.get("env_changed"))
    proven_metrics = layer_row.get("proven_positive_metrics", [])
    if not isinstance(proven_metrics, list):
        proven_metrics = []
    observed_metrics = layer_row.get("observed_positive_metrics", [])
    if not isinstance(observed_metrics, list):
        observed_metrics = []
    failures = layer_row.get("failure_samples", {})
    if not isinstance(failures, dict):
        failures = {}
    failure_count = int(failures.get("count", 0) or 0)
    evidence = layer_row.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}

    if status in {"supported", "observed"}:
        metrics = proven_metrics or observed_metrics
        metric_text = ", ".join(metrics[:3]) if metrics else "无可用指标"
        statement = (
            f"{candidate_group} 相比 {baseline_group} 的 {layer} 层结论：status={status}，"
            f"env_changed={'yes' if changed else 'no'}，正向指标={metric_text}，"
            f"失败样本数={failure_count}。"
        )
    elif status == "regressed":
        neg_metrics = layer_row.get("observed_negative_metrics", [])
        if not isinstance(neg_metrics, list):
            neg_metrics = []
        statement = (
            f"{candidate_group} 相比 {baseline_group} 在 {layer} 层存在回归："
            f"负向指标={', '.join(neg_metrics[:3]) if neg_metrics else MISSING}，"
            f"失败样本数={failure_count}。"
        )
    else:
        statement = (
            f"{candidate_group} 相比 {baseline_group} 在 {layer} 层证据不足："
            f"status={status}，env_changed={'yes' if changed else 'no'}，失败样本数={failure_count}。"
        )

    return {
        "id": conclusion_id,
        "type": "attribution",
        "group_id": candidate_group,
        "layer": layer,
        "statement": statement,
        "experiment": experiment_meta,
        "evidence": {
            "run_id_field": str(evidence.get("run_id_field", "request_id")),
            "run_ids": list(evidence.get("run_ids", [])) if isinstance(evidence.get("run_ids"), list) else [],
            "metric_files": list(evidence.get("metric_files", [])) if isinstance(evidence.get("metric_files"), list) else [],
        },
        "status": status,
        "env_changed": changed,
        "failure_count": failure_count,
    }


def _conclusion_from_reproducibility(
    *,
    conclusion_id: str,
    baseline_group: str,
    candidate_group: str,
    dataset_info: dict[str, Any],
    sample_size: dict[str, Any],
    source_files: list[str],
    run_ids: list[str],
    run_id_field: str,
    matrix_manifest_path: Path | None,
    experiment_meta: dict[str, Any],
) -> dict[str, Any]:
    dataset_version = str(dataset_info.get("version", MISSING))
    dataset_path = str(dataset_info.get("path", MISSING))
    suite = str(dataset_info.get("suite", MISSING))
    case_n = sample_size.get("case_count", MISSING)
    retrieval_n = sample_size.get("retrieval_case_count", MISSING)
    judge_n = sample_size.get("judge_case_count", MISSING)

    statement = (
        f"可复现实验描述：dataset_version={dataset_version}，dataset={dataset_path}，suite={suite}，"
        f"baseline={baseline_group}，candidate={candidate_group}，"
        f"样本量(case={case_n}, retrieval={retrieval_n}, judge={judge_n})。"
    )

    files = list(source_files)
    if matrix_manifest_path is not None:
        files.append(str(matrix_manifest_path.resolve()))
    files = _dedupe_keep_order(files)

    return {
        "id": conclusion_id,
        "type": "reproducibility",
        "group_id": candidate_group,
        "layer": "system",
        "statement": statement,
        "experiment": experiment_meta,
        "sample_size": {
            "case_count": case_n,
            "retrieval_case_count": retrieval_n,
            "judge_case_count": judge_n,
        },
        "evidence": {
            "run_id_field": run_id_field,
            "run_ids": run_ids,
            "metric_files": files,
        },
    }


def _build_summary_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# CV Metrics Summary")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{report.get('generated_at_utc', MISSING)}`")
    exp = report.get("experiment", {})
    if not isinstance(exp, dict):
        exp = {}
    dataset = exp.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    lines.append(f"- Dataset version: `{dataset.get('version', MISSING)}`")
    lines.append(f"- Dataset path: `{dataset.get('path', MISSING)}`")
    lines.append(f"- Baseline group: `{exp.get('baseline_group', MISSING)}`")
    lines.append(f"- Candidate groups: `{', '.join(exp.get('candidate_groups', [])) if isinstance(exp.get('candidate_groups'), list) else MISSING}`")
    lines.append("")

    lines.append("## Quantitative Conclusions")
    conclusions = report.get("conclusions", [])
    if not isinstance(conclusions, list):
        conclusions = []
    quant_rows = [row for row in conclusions if isinstance(row, dict) and str(row.get("type")) in {"relative_improvement", "absolute_improvement", "stability"}]
    if quant_rows:
        for row in quant_rows:
            lines.append(f"- [{row.get('id', MISSING)}] {row.get('statement', '')}")
    else:
        lines.append("- No quantitative conclusion with usable evidence.")
    lines.append("")

    lines.append("## Attribution Conclusions")
    attr_rows = [row for row in conclusions if isinstance(row, dict) and str(row.get("type")) == "attribution"]
    if attr_rows:
        for row in attr_rows:
            lines.append(f"- [{row.get('id', MISSING)}] {row.get('statement', '')}")
    else:
        lines.append("- No attribution conclusion with usable evidence.")
    lines.append("")

    lines.append("## Reproducibility")
    repro_rows = [row for row in conclusions if isinstance(row, dict) and str(row.get("type")) == "reproducibility"]
    if repro_rows:
        for row in repro_rows:
            lines.append(f"- [{row.get('id', MISSING)}] {row.get('statement', '')}")
    else:
        lines.append("- Missing reproducibility statement.")
    lines.append("")

    lines.append("## Evidence Index")
    lines.append("| Conclusion ID | Type | Run ID Field | Run IDs | Metric Files |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in conclusions:
        if not isinstance(row, dict):
            continue
        evidence = row.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        run_field = str(evidence.get("run_id_field", MISSING))
        run_ids = evidence.get("run_ids", [])
        files = evidence.get("metric_files", [])
        run_text = ", ".join([str(item) for item in run_ids[:6]]) if isinstance(run_ids, list) else MISSING
        file_text = ", ".join([str(item) for item in files[:3]]) if isinstance(files, list) else MISSING
        lines.append(
            f"| {row.get('id', MISSING)} | {row.get('type', MISSING)} | {run_field} | {run_text or MISSING} | {file_text or MISSING} |"
        )

    failures = report.get("failure_samples", {})
    if isinstance(failures, dict):
        lines.append("")
        lines.append("## Failure Sample Coverage")
        lines.append(f"- Total normalized failure samples: `{failures.get('total_count', 0)}`")
        by_layer = failures.get("by_layer", {})
        if isinstance(by_layer, dict) and by_layer:
            for layer, count in sorted(by_layer.items()):
                lines.append(f"- {layer}: `{count}`")
    lines.append("")
    return "\n".join(lines)


def build_cv_metrics_report(
    *,
    leaderboard_path: Path,
    matrix_manifest_path: Path | None,
    failure_sample_paths: list[Path],
    trace_artifact_paths: list[Path],
    baseline_group: str | None,
    candidate_groups: list[str] | None,
    max_run_refs: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    leaderboard = _load_json(leaderboard_path.resolve())
    leaderboard_base_dir = leaderboard_path.resolve().parent

    baseline = str(baseline_group or leaderboard.get("baseline_group") or DEFAULT_BASELINE).strip()
    if not baseline:
        baseline = DEFAULT_BASELINE

    selected_manifest_path = matrix_manifest_path
    if selected_manifest_path is None:
        raw_manifest = str(leaderboard.get("manifest_path", "")).strip()
        if raw_manifest:
            selected_manifest_path = _resolve_path(raw_manifest, base_dir=leaderboard_base_dir)

    matrix_manifest: dict[str, Any] | None = None
    manifest_group_lookup: dict[str, dict[str, Any]] = {}
    manifest_base_dir: Path | None = None
    if selected_manifest_path is not None and selected_manifest_path.exists():
        matrix_manifest = _load_json(selected_manifest_path.resolve())
        manifest_group_lookup = _extract_manifest_groups(matrix_manifest)
        manifest_base_dir = selected_manifest_path.resolve().parent
    elif selected_manifest_path is not None:
        warnings.append(f"[Warn] matrix manifest not found: {selected_manifest_path}")

    leaderboard_groups = leaderboard.get("groups", [])
    if not isinstance(leaderboard_groups, list) or not leaderboard_groups:
        raise ValueError(f"Invalid leaderboard.groups in {leaderboard_path}")

    lb_group_lookup: dict[str, dict[str, Any]] = {}
    for group in leaderboard_groups:
        if not isinstance(group, dict):
            continue
        gid = str(group.get("group_id", "")).strip()
        if gid:
            lb_group_lookup[gid] = group

    if baseline not in lb_group_lookup:
        raise ValueError(f"Baseline group '{baseline}' not found in leaderboard: {leaderboard_path}")

    wanted_candidates = candidate_groups or []
    if wanted_candidates:
        final_candidates = [gid for gid in wanted_candidates if gid in lb_group_lookup and gid != baseline]
        missing = sorted(set(wanted_candidates).difference(set(final_candidates)))
        if missing:
            warnings.append(f"[Warn] candidate groups not found in leaderboard and ignored: {missing}")
    else:
        final_candidates = [gid for gid in lb_group_lookup.keys() if gid != baseline]
    if not final_candidates:
        raise ValueError("No candidate groups to compare.")

    trace_index = _build_trace_index(trace_artifact_paths)
    failure_rows = _load_failure_samples(failure_sample_paths)

    baseline_lb_group = lb_group_lookup[baseline]
    baseline_manifest_group = manifest_group_lookup.get(baseline)
    baseline_paths = _build_group_paths(
        baseline,
        leaderboard_group=baseline_lb_group,
        manifest_group=baseline_manifest_group,
        leaderboard_base_dir=leaderboard_base_dir,
        manifest_base_dir=manifest_base_dir,
    )
    baseline_run_eval = _safe_load_optional_json(baseline_paths.get("run_eval"), warnings, f"run_eval[{baseline}]")
    baseline_judge = _safe_load_optional_json(baseline_paths.get("judge"), warnings, f"judge[{baseline}]")

    dataset_info = leaderboard.get("dataset", {})
    if not isinstance(dataset_info, dict):
        dataset_info = {}

    report_groups: list[dict[str, Any]] = []
    conclusions: list[dict[str, Any]] = []
    conclusion_counter = 1

    baseline_env = _safe_group_env(baseline_manifest_group or {})

    for candidate in final_candidates:
        candidate_lb_group = lb_group_lookup[candidate]
        candidate_manifest_group = manifest_group_lookup.get(candidate)
        candidate_paths = _build_group_paths(
            candidate,
            leaderboard_group=candidate_lb_group,
            manifest_group=candidate_manifest_group,
            leaderboard_base_dir=leaderboard_base_dir,
            manifest_base_dir=manifest_base_dir,
        )

        candidate_run_eval = _safe_load_optional_json(
            candidate_paths.get("run_eval"),
            warnings,
            f"run_eval[{candidate}]",
        )
        candidate_judge = _safe_load_optional_json(
            candidate_paths.get("judge"),
            warnings,
            f"judge[{candidate}]",
        )

        metric_rows = _collect_group_metrics(
            group_id=candidate,
            leaderboard_group=candidate_lb_group,
            run_eval_report=candidate_run_eval,
            baseline_run_eval_report=baseline_run_eval,
            judge_report=candidate_judge,
            baseline_judge_report=baseline_judge,
            leaderboard_path=leaderboard_path.resolve(),
            group_paths=candidate_paths,
            trace_index=trace_index,
            max_run_refs=max_run_refs,
        )

        candidate_env = _safe_group_env(candidate_manifest_group or {})
        layer_attribution = _build_layer_attribution(
            candidate_group=candidate,
            metric_rows=metric_rows,
            baseline_env=baseline_env,
            candidate_env=candidate_env,
            failure_rows=failure_rows,
            manifest_path=selected_manifest_path.resolve() if selected_manifest_path and selected_manifest_path.exists() else None,
            max_run_refs=max_run_refs,
        )

        sample_size = {
            "case_count": _to_int(_safe_get_nested(candidate_run_eval or {}, "summary.case_count")) or MISSING,
            "retrieval_case_count": _to_int(_safe_get_nested(candidate_run_eval or {}, "summary.retrieval_case_count")) or MISSING,
            "judge_case_count": _to_int((candidate_judge or {}).get("case_count")) or MISSING,
        }

        experiment_meta = {
            "dataset_version": str(dataset_info.get("version", MISSING)),
            "dataset_path": str(dataset_info.get("path", MISSING)),
            "baseline_group": baseline,
            "candidate_group": candidate,
            "sample_size": sample_size,
        }

        relative_metric = _pick_metric_for_relative(metric_rows)
        if relative_metric is not None:
            conclusions.append(
                _conclusion_from_metric(
                    conclusion_id=f"C{conclusion_counter}",
                    conclusion_type="relative_improvement",
                    baseline_group=baseline,
                    candidate_group=candidate,
                    metric_row=relative_metric,
                    experiment_meta=experiment_meta,
                )
            )
            conclusion_counter += 1

        absolute_metric = _pick_metric_for_absolute(metric_rows)
        if absolute_metric is not None:
            conclusions.append(
                _conclusion_from_metric(
                    conclusion_id=f"C{conclusion_counter}",
                    conclusion_type="absolute_improvement",
                    baseline_group=baseline,
                    candidate_group=candidate,
                    metric_row=absolute_metric,
                    experiment_meta=experiment_meta,
                )
            )
            conclusion_counter += 1

        stability_metric = _pick_metric_for_stability(metric_rows)
        if stability_metric is not None:
            conclusions.append(
                _conclusion_from_metric(
                    conclusion_id=f"C{conclusion_counter}",
                    conclusion_type="stability",
                    baseline_group=baseline,
                    candidate_group=candidate,
                    metric_row=stability_metric,
                    experiment_meta=experiment_meta,
                )
            )
            conclusion_counter += 1

        for layer_row in layer_attribution:
            conclusions.append(
                _conclusion_from_attribution(
                    conclusion_id=f"C{conclusion_counter}",
                    baseline_group=baseline,
                    candidate_group=candidate,
                    layer_row=layer_row,
                    experiment_meta=experiment_meta,
                )
            )
            conclusion_counter += 1

        candidate_run_refs, run_ref_field = _resolve_request_ids_to_run_refs(
            _collect_all_request_ids(candidate_run_eval or {}),
            trace_index=trace_index,
            max_count=max_run_refs,
        )
        repro_files = [str(leaderboard_path.resolve())]
        for key in ("run_eval", "judge"):
            path = candidate_paths.get(key)
            if path is not None:
                repro_files.append(str(path.resolve()))

        conclusions.append(
            _conclusion_from_reproducibility(
                conclusion_id=f"C{conclusion_counter}",
                baseline_group=baseline,
                candidate_group=candidate,
                dataset_info=dataset_info,
                sample_size=sample_size,
                source_files=repro_files,
                run_ids=candidate_run_refs,
                run_id_field=run_ref_field,
                matrix_manifest_path=selected_manifest_path.resolve() if selected_manifest_path and selected_manifest_path.exists() else None,
                experiment_meta=experiment_meta,
            )
        )
        conclusion_counter += 1

        report_groups.append(
            {
                "group_id": candidate,
                "description": str(candidate_lb_group.get("description", "")),
                "source_files": {
                    "leaderboard": str(leaderboard_path.resolve()),
                    "run_eval": str(candidate_paths["run_eval"].resolve()) if candidate_paths.get("run_eval") else MISSING,
                    "judge": str(candidate_paths["judge"].resolve()) if candidate_paths.get("judge") else MISSING,
                },
                "env_diff": _diff_env(baseline_env, candidate_env),
                "metrics": metric_rows,
                "layer_attribution": layer_attribution,
                "sample_size": sample_size,
            }
        )

    failure_by_layer: dict[str, int] = {}
    for row in failure_rows:
        layer = str(row.get("layer", "unknown"))
        failure_by_layer[layer] = failure_by_layer.get(layer, 0) + 1

    filtered_conclusions: list[dict[str, Any]] = []
    for item in conclusions:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        run_ids = evidence.get("run_ids", [])
        metric_files = evidence.get("metric_files", [])
        has_run_ids = isinstance(run_ids, list) and any(str(x).strip() for x in run_ids)
        has_files = isinstance(metric_files, list) and any(str(x).strip() for x in metric_files)
        if has_run_ids and has_files:
            filtered_conclusions.append(item)
        else:
            warnings.append(
                "[Warn] drop conclusion without full evidence: "
                f"id={item.get('id', MISSING)} type={item.get('type', MISSING)}"
            )

    report = {
        "generated_at_utc": _utc_now_iso(),
        "inputs": {
            "leaderboard": str(leaderboard_path.resolve()),
            "matrix_manifest": str(selected_manifest_path.resolve()) if selected_manifest_path and selected_manifest_path.exists() else MISSING,
            "failure_samples": [str(path.resolve()) for path in _collect_candidate_files(failure_sample_paths)],
            "trace_artifacts": [str(path.resolve()) for path in _collect_candidate_files(trace_artifact_paths)],
        },
        "experiment": {
            "dataset": {
                "version": str(dataset_info.get("version", MISSING)),
                "path": str(dataset_info.get("path", MISSING)),
                "suite": str(dataset_info.get("suite", MISSING)),
            },
            "baseline_group": baseline,
            "candidate_groups": final_candidates,
            "matrix_generated_at_utc": str((matrix_manifest or {}).get("generated_at_utc", MISSING)),
        },
        "groups": report_groups,
        "conclusions": filtered_conclusions,
        "failure_samples": {
            "total_count": len(failure_rows),
            "by_layer": failure_by_layer,
        },
        "trace_mapping": {
            "request_id_count": len(trace_index),
        },
        "warnings": warnings,
    }
    report["summary_markdown"] = _build_summary_markdown(report)
    return report


def _build_arg_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export resume/report-ready CV metrics from leaderboard + matrix artifacts.",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=eval_dir / "reports" / "leaderboard" / "latest.json",
        help="Leaderboard JSON path.",
    )
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        default=None,
        help="Matrix or pipeline manifest JSON path. Default reads leaderboard.manifest_path.",
    )
    parser.add_argument(
        "--failure-samples",
        type=Path,
        nargs="*",
        default=[],
        help="Failure sample JSON/JSONL files or directories.",
    )
    parser.add_argument(
        "--trace-artifacts",
        type=Path,
        nargs="*",
        default=[],
        help="LangSmith trace export JSON/JSONL files or directories (for request_id -> run_id mapping).",
    )
    parser.add_argument(
        "--baseline-group",
        type=str,
        default="",
        help="Override baseline group id.",
    )
    parser.add_argument(
        "--candidate-groups",
        type=str,
        default="",
        help="Comma-separated candidate groups. Default: all non-baseline groups in leaderboard.",
    )
    parser.add_argument(
        "--max-run-refs",
        type=int,
        default=20,
        help="Max run IDs attached to each conclusion.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=eval_dir / "reports" / "cv_metrics.json",
        help="Output path for cv_metrics JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=eval_dir / "reports" / "summary.md",
        help="Output path for summary markdown.",
    )
    return parser


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_arg_parser(eval_dir)
    args = parser.parse_args()

    leaderboard_path = args.leaderboard.resolve()
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard file not found: {leaderboard_path}")

    matrix_manifest_path = args.matrix_manifest.resolve() if args.matrix_manifest else None
    report = build_cv_metrics_report(
        leaderboard_path=leaderboard_path,
        matrix_manifest_path=matrix_manifest_path,
        failure_sample_paths=[path.resolve() for path in args.failure_samples],
        trace_artifact_paths=[path.resolve() for path in args.trace_artifacts],
        baseline_group=str(args.baseline_group).strip() or None,
        candidate_groups=_parse_csv(args.candidate_groups),
        max_run_refs=max(5, int(args.max_run_refs)),
    )

    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(str(report.get("summary_markdown", "")), encoding="utf-8")

    print(f"[CVMetrics] baseline={report['experiment']['baseline_group']} candidates={report['experiment']['candidate_groups']}")
    print(f"[CVMetrics] conclusions={len(report.get('conclusions', []))}")
    print(f"[CVMetrics] json={output_json}")
    print(f"[CVMetrics] md={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
