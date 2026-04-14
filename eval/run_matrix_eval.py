"""Experiment-matrix runner for grouped eval comparisons.

This script orchestrates multiple `eval/run_eval.py` runs under distinct
environment variants (for example retrieval vs agent optimization groups).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatrixGroup:
    """One experiment group in the matrix."""

    id: str
    description: str
    env: dict[str, str]


def _parse_csv(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _safe_filename(token: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in token)
    return cleaned or "group"


def load_matrix_groups(matrix_path: Path) -> list[MatrixGroup]:
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    groups_raw = payload.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError(f"Matrix file must include non-empty 'groups' list: {matrix_path}")

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


def select_groups(groups: list[MatrixGroup], selected_ids: set[str]) -> list[MatrixGroup]:
    if not selected_ids:
        return groups
    selected = [group for group in groups if group.id in selected_ids]
    missing = sorted(selected_ids.difference({group.id for group in selected}))
    if missing:
        raise ValueError(f"Unknown matrix group ids: {missing}")
    return selected


def resolve_forwarded_run_eval_args(raw_args: list[str]) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        args = ["--suite", "default", "--runs-per-question", "3"]

    forbidden = {"--output", "--experiment-group"}
    conflict = [item for item in args if item in forbidden]
    if conflict:
        raise ValueError(
            "Do not pass --output/--experiment-group via forwarded args; "
            "they are controlled by run_matrix_eval.py."
        )
    return args


def build_run_eval_command(
    run_eval_path: Path,
    forwarded_args: list[str],
    *,
    output_path: Path,
    group_id: str,
) -> list[str]:
    return [
        sys.executable,
        str(run_eval_path),
        *forwarded_args,
        "--experiment-group",
        group_id,
        "--output",
        str(output_path),
    ]


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

    groups = load_matrix_groups(matrix_path)
    selected = select_groups(groups, _parse_csv(args.groups))
    forwarded_args = resolve_forwarded_run_eval_args(args.run_eval_args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_eval_path = (eval_dir / "run_eval.py").resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "dry_run": bool(args.dry_run),
        "forwarded_run_eval_args": list(forwarded_args),
        "groups": [],
    }

    project_root = eval_dir.parent
    fail_count = 0
    for group in selected:
        output_name = f"{timestamp}_{_safe_filename(group.id)}.json"
        output_path = output_dir / output_name
        cmd = build_run_eval_command(
            run_eval_path,
            forwarded_args,
            output_path=output_path,
            group_id=group.id,
        )
        env_overrides = {key: value for key, value in group.env.items() if value != ""}
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
        completed = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=run_env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

        record["exit_code"] = int(completed.returncode)
        if completed.returncode == 0:
            record["status"] = "ok"
        else:
            record["status"] = "failed"
            fail_count += 1
        manifest["groups"].append(record)

        if completed.returncode != 0 and not args.continue_on_error:
            break

    manifest["failed_groups"] = fail_count
    manifest_path = output_dir / f"{timestamp}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Matrix] manifest={manifest_path}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

