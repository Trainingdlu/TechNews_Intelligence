"""One-click orchestration for full eval pipeline (dataset -> matrix -> judge -> leaderboard).

This runner only orchestrates existing scripts:
- eval/run_matrix_eval.py (internally runs eval/run_eval.py)
- eval/run_judge_eval.py
- eval/build_leaderboard.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from common import (
        get_nested,
        parse_csv_tokens,
        read_json_object,
        run_subprocess_checked,
        safe_filename,
    )
    from dataset_loader import load_eval_cases
    from run_matrix_eval import load_matrix_groups
except ImportError:  # package-style import fallback
    from .common import (
        get_nested,
        parse_csv_tokens,
        read_json_object,
        run_subprocess_checked,
        safe_filename,
    )
    from .dataset_loader import load_eval_cases
    from .run_matrix_eval import load_matrix_groups


STATE_FILE_NAME = "pipeline_state.json"
PIPELINE_MANIFEST_FILE_NAME = "pipeline_manifest.json"
PIPELINE_SUMMARY_FILE_NAME = "pipeline_summary.json"
DATASET_PRIORITY = ("regression.jsonl", "smoke.jsonl", "challenge.jsonl", "default.jsonl")
DEFAULT_BASELINE_GROUP = "G0_baseline"


@dataclass(frozen=True)
class GroupRunArtifact:
    group_id: str
    run_eval_path: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_filename(token: str) -> str:
    return safe_filename(token)


def _parse_csv(value: str) -> list[str]:
    return parse_csv_tokens(value)


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(path, encoding="utf-8-sig")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    return get_nested(payload, path)


def _require_fields(payload: dict[str, Any], *, label: str, required_paths: list[str]) -> None:
    missing: list[str] = []
    for path in required_paths:
        if _get_nested(payload, path) is None:
            missing.append(path)
    if missing:
        raise ValueError(
            f"{label} schema missing required fields: {', '.join(missing)}"
        )


def _validate_dataset_contract(dataset_path: Path) -> dict[str, Any]:
    cases = load_eval_cases(
        dataset_path,
        strict_capability_check=False,
        include_disabled=True,
    )
    if not cases:
        raise ValueError(f"Dataset has no usable cases: {dataset_path}")

    missing: list[str] = []
    for idx, case in enumerate(cases, 1):
        case_id = str(case.get("id", "")).strip()
        question = str(case.get("question", "")).strip()
        capability = str(case.get("capability", "")).strip()
        if not case_id:
            missing.append(f"cases[{idx}].id")
        if not question:
            missing.append(f"cases[{idx}].question")
        if not capability:
            missing.append(f"cases[{idx}].capability")
    if missing:
        raise ValueError(
            f"Dataset contract missing required fields: {', '.join(missing[:20])}"
        )

    return {
        "dataset_path": str(dataset_path.resolve()),
        "case_count": len(cases),
        "with_retrieval_gold": sum(1 for c in cases if c.get("retrieval_gold_urls")),
    }


def _validate_run_eval_contract(payload: dict[str, Any], *, group_id: str) -> None:
    label = f"run_eval[{group_id}]"
    _require_fields(
        payload,
        label=label,
        required_paths=[
            "dataset",
            "summary.case_count",
            "summary.avg_error_rate",
            "summary.avg_recall_at_10",
            "summary.avg_mrr_at_10",
            "summary.avg_ndcg_at_10",
            "route_metrics.react_success_rate",
            "cases",
        ],
    )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{label} schema missing required fields: cases[]")

    missing: list[str] = []
    for idx, case in enumerate(cases):
        case_label = f"cases[{idx}]"
        if not isinstance(case, dict):
            missing.append(case_label)
            continue
        if not isinstance(case.get("metrics"), dict):
            missing.append(f"{case_label}.metrics")
        if not isinstance(case.get("runs"), list):
            missing.append(f"{case_label}.runs")
        outputs = case.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            missing.append(f"{case_label}.outputs")
    if missing:
        raise ValueError(
            f"{label} schema missing required fields: {', '.join(missing[:20])}"
        )


def _validate_judge_contract(payload: dict[str, Any], *, group_id: str) -> None:
    label = f"judge[{group_id}]"
    _require_fields(
        payload,
        label=label,
        required_paths=[
            "source_report",
            "case_count",
            "summary.avg_composite",
            "rows",
        ],
    )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} schema missing required fields: rows")

    required_scores = {
        "accuracy",
        "groundedness",
        "coherence",
        "completeness",
        "helpfulness",
        "composite",
    }
    missing: list[str] = []
    for idx, row in enumerate(rows):
        row_label = f"rows[{idx}]"
        if not isinstance(row, dict):
            missing.append(row_label)
            continue
        if not str(row.get("case_id", "")).strip():
            missing.append(f"{row_label}.case_id")
        if not isinstance(row.get("evidence"), dict):
            missing.append(f"{row_label}.evidence")
        if not str(row.get("verdict", "")).strip():
            missing.append(f"{row_label}.verdict")
        scores = row.get("scores")
        if not isinstance(scores, dict):
            missing.append(f"{row_label}.scores")
            continue
        for metric in required_scores:
            if metric not in scores:
                missing.append(f"{row_label}.scores.{metric}")
    if missing:
        raise ValueError(
            f"{label} schema missing required fields: {', '.join(missing[:20])}"
        )


def _validate_leaderboard_contract(payload: dict[str, Any], *, baseline_group: str) -> None:
    label = "leaderboard"
    _require_fields(
        payload,
        label=label,
        required_paths=[
            "baseline_group",
            "groups",
            "metric_order",
            "dataset.version",
            "dataset.path",
        ],
    )

    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError(
            "leaderboard schema missing required fields: groups must include baseline + candidate entries"
        )
    group_ids = {str(item.get("group_id", "")).strip() for item in groups if isinstance(item, dict)}
    if baseline_group not in group_ids:
        raise ValueError(
            f"leaderboard schema missing required fields: baseline group '{baseline_group}' not found"
        )

    metric_order = payload.get("metric_order")
    if not isinstance(metric_order, list) or not metric_order:
        raise ValueError("leaderboard schema missing required fields: metric_order[]")

    missing: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            missing.append("groups[].<invalid>")
            continue
        group_id = str(group.get("group_id", "")).strip() or "<unknown>"
        metrics = group.get("metrics")
        if not isinstance(metrics, dict):
            missing.append(f"{group_id}.metrics")
            continue
        for metric_name in metric_order:
            metric_key = str(metric_name)
            row = metrics.get(metric_key)
            if not isinstance(row, dict):
                missing.append(f"{group_id}.metrics.{metric_key}")
                continue
            for key in ("current", "baseline", "delta_abs", "n", "ci_95"):
                if key not in row:
                    missing.append(f"{group_id}.metrics.{metric_key}.{key}")
            ci_95 = row.get("ci_95")
            if not isinstance(ci_95, dict):
                missing.append(f"{group_id}.metrics.{metric_key}.ci_95")
            else:
                for key in ("lower", "upper", "method"):
                    if key not in ci_95:
                        missing.append(f"{group_id}.metrics.{metric_key}.ci_95.{key}")
    if missing:
        raise ValueError(
            f"leaderboard schema missing required fields: {', '.join(missing[:30])}"
        )


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"steps": {}}
    payload = _read_json(state_path)
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        payload["steps"] = {}
    return payload


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now_iso()
    _write_json(state_path, state)


def _mark_step_completed(state: dict[str, Any], *, step: str, payload: dict[str, Any]) -> None:
    steps = state.setdefault("steps", {})
    if not isinstance(steps, dict):
        state["steps"] = {}
        steps = state["steps"]
    steps[step] = {
        "status": "completed",
        "completed_at_utc": _utc_now_iso(),
        **payload,
    }


def _resolve_dataset_path(
    eval_dir: Path,
    dataset_version: str,
) -> tuple[str, Path]:
    versions_dir = (eval_dir / "datasets" / "versions").resolve()
    token = str(dataset_version or "").strip()
    if not token:
        raise ValueError("--dataset-version cannot be empty.")

    version_dir: Path
    resolved_version: str
    if token.lower() in {"latest", "current"}:
        candidates = sorted(p for p in versions_dir.glob("*") if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No dataset versions found under: {versions_dir}")
        version_dir = candidates[-1]
        resolved_version = version_dir.name
    else:
        version_dir = (versions_dir / token).resolve()
        resolved_version = token

    if not version_dir.exists() or not version_dir.is_dir():
        raise FileNotFoundError(f"找不到数据集版本目录: {version_dir}")

    for name in DATASET_PRIORITY:
        candidate = (version_dir / name).resolve()
        if candidate.exists():
            return resolved_version, candidate
    raise FileNotFoundError(
        f"No supported dataset file found in {version_dir}; "
        f"expected one of {list(DATASET_PRIORITY)}"
    )


def _resolve_path(raw: str, *, base_dir: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    run_subprocess_checked(
        command,
        cwd=cwd,
        env=env,
        log_prefix="[FullEval]",
        error_prefix="Full eval subprocess failed",
    )


def _latest_manifest_path(matrix_dir: Path) -> Path:
    manifests = sorted(matrix_dir.glob("*_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No matrix manifest found under: {matrix_dir}")
    return manifests[-1]


def _select_groups(
    *,
    matrix_path: Path,
    baseline_group: str,
    candidates_raw: str,
) -> list[str]:
    groups = load_matrix_groups(matrix_path)
    all_ids = [group.id for group in groups]
    if baseline_group not in all_ids:
        raise ValueError(
            f"基线组 '{baseline_group}' not found in matrix config: {matrix_path}"
        )

    candidate_ids = _parse_csv(candidates_raw)
    if not candidate_ids:
        return [baseline_group, *[gid for gid in all_ids if gid != baseline_group]]

    unknown = sorted(set(candidate_ids).difference(set(all_ids)))
    if unknown:
        raise ValueError(
            f"未知的候选组在 --candidates: {unknown}; known={all_ids}"
        )

    selected: list[str] = [baseline_group]
    selected.extend(gid for gid in all_ids if gid in set(candidate_ids) and gid != baseline_group)
    if len(selected) < 2:
        raise ValueError("--candidates must contain at least one non-baseline group.")
    return selected


def _validate_matrix_manifest(
    *,
    manifest_path: Path,
    selected_groups: list[str],
) -> list[GroupRunArtifact]:
    payload = _read_json(manifest_path)
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"matrix manifest missing required field: groups ({manifest_path})")

    by_id: dict[str, dict[str, Any]] = {}
    for item in groups:
        if isinstance(item, dict):
            gid = str(item.get("id", "")).strip()
            if gid:
                by_id[gid] = item

    missing_groups = [gid for gid in selected_groups if gid not in by_id]
    if missing_groups:
        raise ValueError(
            f"matrix manifest missing selected groups: {missing_groups} ({manifest_path})"
        )

    artifacts: list[GroupRunArtifact] = []
    manifest_dir = manifest_path.parent
    for gid in selected_groups:
        item = by_id[gid]
        status = str(item.get("status", "")).strip().lower()
        if status and status != "ok":
            raise ValueError(
                f"矩阵组 '{gid}' not successful; status={status} ({manifest_path})"
            )
        raw_output = str(item.get("output", "")).strip()
        if not raw_output:
            raise ValueError(f"矩阵组 '{gid}' 缺失输出路径 ({manifest_path})")
        run_eval_path = _resolve_path(raw_output, base_dir=manifest_dir)
        if not run_eval_path.exists():
            raise FileNotFoundError(
                f"矩阵组 '{gid}' run_eval output not found: {run_eval_path}"
            )
        payload = _read_json(run_eval_path)
        _validate_run_eval_contract(payload, group_id=gid)
        artifacts.append(GroupRunArtifact(group_id=gid, run_eval_path=run_eval_path))
    return artifacts


def _build_pipeline_manifest(
    *,
    matrix_manifest_path: Path,
    judge_outputs: dict[str, Path],
    run_id: str,
    output_path: Path,
) -> Path:
    manifest = _read_json(matrix_manifest_path)
    groups = manifest.get("groups")
    if not isinstance(groups, list):
        raise ValueError(
            f"matrix manifest missing required field: groups ({matrix_manifest_path})"
        )

    for item in groups:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("id", "")).strip()
        if not group_id:
            continue
        judge = judge_outputs.get(group_id)
        if judge is not None:
            item["judge_output"] = str(judge.resolve())

    manifest["pipeline"] = {
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
    }
    _write_json(output_path, manifest)
    return output_path


def _build_arg_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full one-click evaluation pipeline.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        required=True,
        help="Dataset version under eval/datasets/versions/ (or 'latest').",
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=eval_dir / "experiment_matrix.json",
        help="Matrix config path (JSON).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE_GROUP,
        help=f"基线组 id (default: {DEFAULT_BASELINE_GROUP}).",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="",
        help="Comma-separated candidate group ids; empty means all non-baseline groups.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Report directory name under eval/reports/. Default: UTC timestamp.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed steps via pipeline_state.json.",
    )
    return parser


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    project_root = eval_dir.parent
    parser = _build_arg_parser(eval_dir)
    args = parser.parse_args()

    matrix_config = args.matrix_config.resolve()
    if not matrix_config.exists():
        raise FileNotFoundError(f"Matrix config not found: {matrix_config}")

    run_id = str(args.run_id).strip() or _default_run_id()
    run_dir = (eval_dir / "reports" / run_id).resolve()
    matrix_dir = run_dir / "matrix"
    judge_dir = run_dir / "judge"
    leaderboard_dir = run_dir / "leaderboard"
    state_path = run_dir / STATE_FILE_NAME
    pipeline_manifest_path = run_dir / PIPELINE_MANIFEST_FILE_NAME
    summary_path = run_dir / PIPELINE_SUMMARY_FILE_NAME

    run_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    state["run_id"] = run_id
    state["resume"] = bool(args.resume)
    _save_state(state_path, state)

    selected_groups = _select_groups(
        matrix_path=matrix_config,
        baseline_group=str(args.baseline).strip() or DEFAULT_BASELINE_GROUP,
        candidates_raw=str(args.candidates),
    )
    baseline_group = selected_groups[0]

    print(f"[FullEval] run_id={run_id}")
    print(f"[FullEval] selected_groups={selected_groups}")

    dataset_step = (state.get("steps", {}) or {}).get("dataset", {})
    dataset_info: dict[str, Any]
    if (
        args.resume
        and isinstance(dataset_step, dict)
        and dataset_step.get("status") == "completed"
        and dataset_step.get("dataset_version") in {args.dataset_version, str(args.dataset_version).strip()}
    ):
        dataset_path = Path(str(dataset_step.get("dataset_path", ""))).resolve()
        if dataset_path.exists():
            dataset_info = _validate_dataset_contract(dataset_path)
            print(f"[FullEval] resume step=dataset dataset={dataset_path}")
        else:
            resolved_version, dataset_path = _resolve_dataset_path(eval_dir, str(args.dataset_version))
            dataset_info = _validate_dataset_contract(dataset_path)
            _mark_step_completed(
                state,
                step="dataset",
                payload={
                    "dataset_version": resolved_version,
                    **dataset_info,
                },
            )
            _save_state(state_path, state)
            print(f"[FullEval] step=dataset dataset={dataset_path}")
    else:
        resolved_version, dataset_path = _resolve_dataset_path(eval_dir, str(args.dataset_version))
        dataset_info = _validate_dataset_contract(dataset_path)
        _mark_step_completed(
            state,
            step="dataset",
            payload={
                "dataset_version": resolved_version,
                **dataset_info,
            },
        )
        _save_state(state_path, state)
        print(f"[FullEval] step=dataset dataset={dataset_path}")

    matrix_step = (state.get("steps", {}) or {}).get("matrix", {})
    group_artifacts: list[GroupRunArtifact]
    manifest_path: Path
    matrix_step_valid = False
    if args.resume and isinstance(matrix_step, dict) and matrix_step.get("status") == "completed":
        raw_manifest = str(matrix_step.get("manifest_path", "")).strip()
        if raw_manifest:
            candidate_manifest = Path(raw_manifest).resolve()
            if candidate_manifest.exists():
                try:
                    group_artifacts = _validate_matrix_manifest(
                        manifest_path=candidate_manifest,
                        selected_groups=selected_groups,
                    )
                    manifest_path = candidate_manifest
                    matrix_step_valid = True
                    print(f"[FullEval] resume step=matrix manifest={manifest_path}")
                except Exception:
                    matrix_step_valid = False
    if not matrix_step_valid:
        matrix_cmd = [
            sys.executable,
            str((eval_dir / "run_matrix_eval.py").resolve()),
            "--matrix",
            str(matrix_config),
            "--groups",
            ",".join(selected_groups),
            "--output-dir",
            str(matrix_dir),
            "--",
            "--dataset",
            str(dataset_path),
            "--include-outputs",
            "--include-trace-summary",
        ]
        _run_subprocess(matrix_cmd, cwd=project_root)
        manifest_path = _latest_manifest_path(matrix_dir)
        group_artifacts = _validate_matrix_manifest(
            manifest_path=manifest_path,
            selected_groups=selected_groups,
        )
        _mark_step_completed(
            state,
            step="matrix",
            payload={
                "manifest_path": str(manifest_path.resolve()),
                "groups": selected_groups,
            },
        )
        _save_state(state_path, state)
        print(f"[FullEval] step=matrix manifest={manifest_path}")

    judge_step = (state.get("steps", {}) or {}).get("judge", {})
    judge_outputs: dict[str, Path] = {}
    if isinstance(judge_step, dict):
        existing = judge_step.get("outputs", {})
        if isinstance(existing, dict):
            for gid, raw_path in existing.items():
                token = str(raw_path or "").strip()
                if token:
                    judge_outputs[str(gid)] = Path(token).resolve()

    for artifact in group_artifacts:
        gid = artifact.group_id
        out_path = judge_outputs.get(gid) or (judge_dir / f"{_safe_filename(gid)}.json").resolve()
        can_resume = args.resume and out_path.exists()
        if can_resume:
            payload = _read_json(out_path)
            _validate_judge_contract(payload, group_id=gid)
            judge_outputs[gid] = out_path
            print(f"[FullEval] resume step=judge group={gid} output={out_path}")
            continue

        judge_cmd = [
            sys.executable,
            str((eval_dir / "run_judge_eval.py").resolve()),
            "--report",
            str(artifact.run_eval_path),
            "--output",
            str(out_path),
        ]
        _run_subprocess(judge_cmd, cwd=project_root)
        payload = _read_json(out_path)
        _validate_judge_contract(payload, group_id=gid)
        judge_outputs[gid] = out_path
        print(f"[FullEval] step=judge group={gid} output={out_path}")

    _mark_step_completed(
        state,
        step="judge",
        payload={"outputs": {gid: str(path.resolve()) for gid, path in judge_outputs.items()}},
    )
    _save_state(state_path, state)

    pipeline_manifest = _build_pipeline_manifest(
        matrix_manifest_path=manifest_path,
        judge_outputs=judge_outputs,
        run_id=run_id,
        output_path=pipeline_manifest_path,
    )
    leaderboard_json = (leaderboard_dir / "latest.json").resolve()
    leaderboard_md = (leaderboard_dir / "latest.md").resolve()

    leaderboard_done = False
    leaderboard_step = (state.get("steps", {}) or {}).get("leaderboard", {})
    if args.resume and isinstance(leaderboard_step, dict) and leaderboard_step.get("status") == "completed":
        raw_json = str(leaderboard_step.get("output_json", "")).strip()
        if raw_json:
            candidate_json = Path(raw_json).resolve()
            if candidate_json.exists():
                payload = _read_json(candidate_json)
                _validate_leaderboard_contract(payload, baseline_group=baseline_group)
                leaderboard_json = candidate_json
                leaderboard_done = True
                print(f"[FullEval] resume step=leaderboard output={candidate_json}")

    if not leaderboard_done:
        leaderboard_cmd = [
            sys.executable,
            str((eval_dir / "build_leaderboard.py").resolve()),
            "--manifest",
            str(pipeline_manifest),
            "--baseline-group",
            baseline_group,
            "--output-json",
            str(leaderboard_json),
            "--output-md",
            str(leaderboard_md),
        ]
        _run_subprocess(leaderboard_cmd, cwd=project_root)
        payload = _read_json(leaderboard_json)
        _validate_leaderboard_contract(payload, baseline_group=baseline_group)
        _mark_step_completed(
            state,
            step="leaderboard",
            payload={
                "manifest_path": str(pipeline_manifest.resolve()),
                "output_json": str(leaderboard_json),
                "output_md": str(leaderboard_md),
            },
        )
        _save_state(state_path, state)
        print(f"[FullEval] step=leaderboard output={leaderboard_json}")

    summary_payload = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "dataset_version": state.get("steps", {}).get("dataset", {}).get("dataset_version", args.dataset_version),
        "dataset_path": str(dataset_path.resolve()),
        "baseline_group": baseline_group,
        "selected_groups": selected_groups,
        "matrix_manifest": str(manifest_path.resolve()),
        "pipeline_manifest": str(pipeline_manifest.resolve()),
        "judge_outputs": {gid: str(path.resolve()) for gid, path in judge_outputs.items()},
        "leaderboard_json": str(leaderboard_json.resolve()),
        "leaderboard_md": str(leaderboard_md.resolve()),
    }
    _write_json(summary_path, summary_payload)
    print(f"[FullEval] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

