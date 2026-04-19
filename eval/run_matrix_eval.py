"""Experiment-matrix runner for grouped eval comparisons.

This script orchestrates multiple eval runs under distinct environment variants.
Runner is configurable (for example `run_eval.py` or `run_task_eval_v1.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from common import parse_csv_set, read_json_object, run_subprocess, safe_filename
except ImportError:  # package-style import fallback
    from .common import parse_csv_set, read_json_object, run_subprocess, safe_filename


@dataclass(frozen=True)
class MatrixGroup:
    """One experiment group in the matrix."""

    id: str
    description: str
    env: dict[str, str]


@dataclass(frozen=True)
class MatrixConfig:
    """Top-level matrix config plus resolved groups."""

    groups: list[MatrixGroup]
    baseline_group: str
    frozen_dataset_version: str
    default_run_eval_args: list[str]
    runner: str
    runner_script: str


def _parse_csv(value: str) -> set[str]:
    return parse_csv_set(value)


def _safe_filename(token: str) -> str:
    return safe_filename(token)


def _parse_matrix_groups(payload: dict[str, Any]) -> list[MatrixGroup]:
    groups_raw = payload.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError("Matrix file must include non-empty 'groups' list.")

    groups: list[MatrixGroup] = []
    seen_ids: set[str] = set()
    for item in groups_raw:
        if not isinstance(item, dict):
            raise ValueError("Each matrix group must be a JSON object.")

        group_id = str(item.get("id", "")).strip()
        if not group_id:
            raise ValueError("Matrix group id cannot be empty.")
        if group_id in seen_ids:
            raise ValueError(f"Duplicate matrix group id: {group_id}")
        seen_ids.add(group_id)

        description = str(item.get("description", "")).strip()
        env_raw = item.get("env", {})
        if not isinstance(env_raw, dict):
            raise ValueError(f"Matrix group env must be object: {group_id}")

        env: dict[str, str] = {}
        for key, value in env_raw.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            env[normalized_key] = str(value).strip()

        groups.append(
            MatrixGroup(
                id=group_id,
                description=description,
                env=env,
            )
        )
    return groups


def load_matrix_config(matrix_path: Path) -> MatrixConfig:
    payload = read_json_object(matrix_path, encoding="utf-8")
    groups = _parse_matrix_groups(payload)

    baseline_group = str(payload.get("baseline_group", "")).strip() or groups[0].id
    if baseline_group not in {group.id for group in groups}:
        raise ValueError(
            f"Matrix baseline_group must match one group id: baseline_group={baseline_group}"
        )

    frozen_dataset_version = str(payload.get("frozen_dataset_version", "")).strip()

    runner = str(payload.get("runner", "run_eval")).strip().lower() or "run_eval"
    if runner not in {"run_eval", "task_eval_v1"}:
        raise ValueError("Matrix runner must be one of: run_eval, task_eval_v1.")

    default_script = "run_eval.py" if runner == "run_eval" else "run_task_eval_v1.py"
    runner_script = str(payload.get("runner_script", default_script)).strip() or default_script

    default_args_raw = payload.get("default_runner_args", payload.get("default_run_eval_args", []))
    if default_args_raw is None:
        default_args_raw = []
    if not isinstance(default_args_raw, list):
        raise ValueError("Matrix default_runner_args/default_run_eval_args must be a string list.")
    default_run_eval_args = [str(item).strip() for item in default_args_raw if str(item).strip()]

    return MatrixConfig(
        groups=groups,
        baseline_group=baseline_group,
        frozen_dataset_version=frozen_dataset_version,
        default_run_eval_args=default_run_eval_args,
        runner=runner,
        runner_script=runner_script,
    )


def load_matrix_groups(matrix_path: Path) -> list[MatrixGroup]:
    return load_matrix_config(matrix_path).groups


def select_groups(groups: list[MatrixGroup], selected_ids: set[str]) -> list[MatrixGroup]:
    if not selected_ids:
        return groups
    selected = [group for group in groups if group.id in selected_ids]
    missing = sorted(selected_ids.difference({group.id for group in selected}))
    if missing:
        raise ValueError(f"Unknown matrix group ids: {missing}")
    return selected


def _group_ids(groups: list[MatrixGroup]) -> set[str]:
    return {group.id for group in groups}


def _order_groups_with_baseline_first(groups: list[MatrixGroup], baseline_group: str) -> list[MatrixGroup]:
    if not groups:
        return []
    baseline = next((group for group in groups if group.id == baseline_group), None)
    if baseline is None:
        return groups
    others = [group for group in groups if group.id != baseline_group]
    return [baseline, *others]


def resolve_forwarded_run_eval_args(
    raw_args: list[str],
    *,
    default_run_eval_args: list[str] | None = None,
    runner: str = "run_eval",
) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    defaults = list(default_run_eval_args or [])
    if not args:
        if defaults:
            args = defaults
        elif str(runner).strip().lower() == "task_eval_v1":
            args = ["--runs-per-case", "1"]
        else:
            args = ["--suite", "default", "--runs-per-question", "3"]
    elif defaults:
        args = [*defaults, *args]

    normalized_runner = str(runner).strip().lower()
    if normalized_runner == "task_eval_v1":
        forbidden = {"--output"}
    else:
        forbidden = {"--output", "--experiment-group"}
    conflict = [item for item in args if item in forbidden]
    if conflict:
        if normalized_runner == "task_eval_v1":
            raise ValueError(
                "Do not pass --output via forwarded args; it is controlled by run_matrix_eval.py."
            )
        raise ValueError(
            "Do not pass --output/--experiment-group via forwarded args; "
            "they are controlled by run_matrix_eval.py."
        )
    return args


def _has_arg(args: list[str], key: str) -> bool:
    return any(str(item).strip() == key for item in args)


def _read_report_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json_object(path, encoding="utf-8")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_stage_d_delta(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    payload = report.get("stage_d_delta")
    if not isinstance(payload, dict):
        return None
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        return None
    items = comparison.get("items")
    if not isinstance(items, list):
        items = []
    return {
        "baseline_path": str(payload.get("baseline_path", "")).strip(),
        "improved_count": int(comparison.get("improved_count", 0) or 0),
        "regressed_count": int(comparison.get("regressed_count", 0) or 0),
        "unchanged_count": int(comparison.get("unchanged_count", 0) or 0),
        "missing_count": int(comparison.get("missing_count", 0) or 0),
        "items": [item for item in items if isinstance(item, dict)],
    }


def _extract_summary_metrics(report: dict[str, Any], *, runner: str) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return {}

    normalized_runner = str(runner).strip().lower()
    if normalized_runner == "task_eval_v1":
        intent = summary.get("intent", {})
        tool = summary.get("tool", {})
        retrieval = summary.get("retrieval", {})
        analysis = summary.get("analysis", {})
        system = summary.get("system", {})

        if not isinstance(intent, dict):
            intent = {}
        if not isinstance(tool, dict):
            tool = {}
        if not isinstance(retrieval, dict):
            retrieval = {}
        if not isinstance(analysis, dict):
            analysis = {}
        if not isinstance(system, dict):
            system = {}

        # Keep v1-native keys and compatibility aliases for downstream consumers.
        return {
            "intent_top1_accuracy": intent.get("top1_accuracy"),
            "intent_macro_f1": intent.get("macro_f1"),
            "intent_clarification_accuracy": intent.get("clarification_accuracy"),
            "tool_path_hit_rate": tool.get("acceptable_path_hit_rate"),
            "tool_forbidden_tool_rate": tool.get("forbidden_tool_rate"),
            "tool_param_accuracy": tool.get("param_accuracy"),
            "retrieval_case_count": retrieval.get("case_count"),
            "retrieval_recall_at_10": retrieval.get("recall_at_10"),
            "retrieval_mrr_at_10": retrieval.get("mrr_at_10"),
            "retrieval_ndcg_at_10": retrieval.get("ndcg_at_10"),
            "retrieval_gold_hit_rate": retrieval.get("gold_hit_rate"),
            "analysis_claim_support_rate": analysis.get("claim_support_rate"),
            "analysis_unsupported_claim_rate": analysis.get("unsupported_claim_rate"),
            "analysis_contradiction_rate": analysis.get("contradiction_rate"),
            "analysis_numeric_consistency": analysis.get("numeric_consistency"),
            "system_error_rate": system.get("error_rate"),
            "system_timeout_rate": system.get("timeout_rate"),
            "system_fallback_rate": system.get("fallback_rate"),
            "system_latency_p95_ms": system.get("latency_p95_ms"),
            "avg_recall_at_10": retrieval.get("recall_at_10"),
            "avg_mrr_at_10": retrieval.get("mrr_at_10"),
            "avg_ndcg_at_10": retrieval.get("ndcg_at_10"),
            "avg_error_rate": system.get("error_rate"),
        }

    return {
        "avg_recall_at_10": summary.get("avg_recall_at_10"),
        "avg_mrr_at_10": summary.get("avg_mrr_at_10"),
        "avg_ndcg_at_10": summary.get("avg_ndcg_at_10"),
        "avg_error_rate": summary.get("avg_error_rate"),
    }


def _build_manifest_delta_summary(
    group_records: list[dict[str, Any]],
    *,
    baseline_group: str,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for record in group_records:
        group_id = str(record.get("id", "")).strip()
        if not group_id or group_id == baseline_group:
            continue
        if str(record.get("status", "")).strip() != "ok":
            continue
        delta = record.get("stage_d_delta")
        if not isinstance(delta, dict):
            continue
        candidates.append(
            {
                "group_id": group_id,
                "baseline_path": str(delta.get("baseline_path", "")).strip(),
                "improved_count": int(delta.get("improved_count", 0) or 0),
                "regressed_count": int(delta.get("regressed_count", 0) or 0),
                "unchanged_count": int(delta.get("unchanged_count", 0) or 0),
                "missing_count": int(delta.get("missing_count", 0) or 0),
                "items": [item for item in delta.get("items", []) if isinstance(item, dict)],
            }
        )
    return {
        "baseline_group": baseline_group,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_runner_command(
    runner_script_path: Path,
    forwarded_args: list[str],
    *,
    output_path: Path,
    group_id: str,
    runner: str,
) -> list[str]:
    command = [
        sys.executable,
        str(runner_script_path),
        *forwarded_args,
        "--output",
        str(output_path),
    ]
    if str(runner).strip().lower() == "run_eval":
        command.extend(["--experiment-group", group_id])
    return command


def build_run_eval_command(
    run_eval_path: Path,
    forwarded_args: list[str],
    *,
    output_path: Path,
    group_id: str,
) -> list[str]:
    return build_runner_command(
        run_eval_path,
        forwarded_args,
        output_path=output_path,
        group_id=group_id,
        runner="run_eval",
    )


def _build_arg_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grouped eval experiment matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=eval_dir / "experiment_matrix.json",
        help="Path to experiment matrix JSON config.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help="Optional comma-separated group ids (default: run all groups).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=eval_dir / "reports" / "matrix",
        help="Directory for per-group report JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands/env without executing run_eval.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining groups when one group execution fails.",
    )
    parser.add_argument(
        "run_eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to eval/run_eval.py (prefix with `--`).",
    )
    return parser


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_arg_parser(eval_dir)
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    matrix_config = load_matrix_config(matrix_path)
    selected = select_groups(matrix_config.groups, _parse_csv(args.groups))
    selected = _order_groups_with_baseline_first(selected, matrix_config.baseline_group)
    forwarded_args = resolve_forwarded_run_eval_args(
        args.run_eval_args,
        default_run_eval_args=matrix_config.default_run_eval_args,
        runner=matrix_config.runner,
    )
    selected_ids = _group_ids(selected)
    if (
        len(selected) > 1
        and matrix_config.baseline_group not in selected_ids
        and not args.dry_run
    ):
        raise ValueError(
            "Selected groups must include baseline group for baseline->candidate->delta output: "
            f"baseline={matrix_config.baseline_group} selected={sorted(selected_ids)}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_script_token = str(matrix_config.runner_script).strip()
    runner_script_path = Path(runner_script_token)
    if not runner_script_path.is_absolute():
        runner_script_path = (eval_dir / runner_script_path).resolve()
    else:
        runner_script_path = runner_script_path.resolve()
    if not runner_script_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "dry_run": bool(args.dry_run),
        "baseline_group": matrix_config.baseline_group,
        "frozen_dataset_version": matrix_config.frozen_dataset_version,
        "runner": matrix_config.runner,
        "runner_script": str(runner_script_path),
        "forwarded_run_eval_args": list(forwarded_args),
        "groups": [],
    }

    project_root = eval_dir.parent
    fail_count = 0
    baseline_output_path: Path | None = None
    for group in selected:
        output_name = f"{timestamp}_{_safe_filename(group.id)}.json"
        output_path = output_dir / output_name
        group_args = list(forwarded_args)
        if (
            matrix_config.runner == "run_eval"
            and group.id != matrix_config.baseline_group
            and baseline_output_path is not None
            and not _has_arg(group_args, "--baseline")
        ):
            group_args.extend(["--baseline", str(baseline_output_path)])
        cmd = build_runner_command(
            runner_script_path,
            group_args,
            output_path=output_path,
            group_id=group.id,
            runner=matrix_config.runner,
        )
        if group.id == matrix_config.baseline_group:
            baseline_output_path = output_path
        env_overrides = {key: value for key, value in group.env.items() if value != ""}
        print(f"[Matrix] runner={matrix_config.runner} script={runner_script_path}")
        print(f"[Matrix] group={group.id} output={output_path}")
        print(f"[Matrix] env_overrides={env_overrides}")
        print(f"[Matrix] cmd={' '.join(cmd)}")

        record: dict[str, Any] = {
            "id": group.id,
            "description": group.description,
            "output": str(output_path),
            "env_overrides": env_overrides,
            "command": cmd,
            "status": "planned" if args.dry_run else "running",
        }

        if args.dry_run:
            manifest["groups"].append(record)
            continue

        run_env = os.environ.copy()
        run_env.update(env_overrides)
        completed = run_subprocess(
            cmd,
            cwd=project_root,
            env=run_env,
            log_prefix="[Matrix]",
        )

        record["exit_code"] = int(completed.returncode)
        if completed.returncode == 0:
            record["status"] = "ok"
            report = _read_report_if_exists(output_path)
            if report is not None:
                record["dataset"] = str(report.get("dataset", "")).strip()
                record["summary_metrics"] = _extract_summary_metrics(
                    report,
                    runner=matrix_config.runner,
                )
                system = report.get("system", {})
                if isinstance(system, dict):
                    record["system_metrics"] = {
                        "error_rate": system.get("error_rate"),
                        "citation_guard_block_rate": system.get("citation_guard_block_rate"),
                    }
                if matrix_config.runner == "run_eval":
                    stage_d_delta = _extract_stage_d_delta(report)
                    if stage_d_delta is not None:
                        record["stage_d_delta"] = stage_d_delta
        else:
            record["status"] = "failed"
            fail_count += 1
        manifest["groups"].append(record)

        if completed.returncode != 0 and not args.continue_on_error:
            break

    manifest["failed_groups"] = fail_count
    manifest["baseline_candidate_delta"] = _build_manifest_delta_summary(
        manifest.get("groups", []),
        baseline_group=matrix_config.baseline_group,
    )
    manifest_path = output_dir / f"{timestamp}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Matrix] manifest={manifest_path}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
