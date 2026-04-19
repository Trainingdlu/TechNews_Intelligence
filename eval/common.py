"""Shared helper utilities for eval scripts.

Keep these helpers dependency-light so every eval entrypoint can reuse them
without creating import cycles.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def read_json_object(path: Path, *, encoding: str = "utf-8-sig") -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding=encoding))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def get_nested(payload: dict[str, Any] | None, path: str) -> Any:
    if not isinstance(payload, dict):
        return None
    cur: Any = payload
    for part in str(path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def safe_filename(token: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(token))
    return cleaned or "group"


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def parse_csv_tokens(value: str) -> list[str]:
    return dedupe_keep_order(str(value or "").split(","))


def parse_csv_set(value: str) -> set[str]:
    return set(parse_csv_tokens(value))


def to_float(
    value: Any,
    *,
    parse_string: bool = False,
    missing_token: str | None = None,
) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if parse_string and isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if missing_token is not None and raw.lower() == str(missing_token).lower():
            return None
        try:
            parsed = float(raw)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def to_int(
    value: Any,
    *,
    parse_string: bool = False,
    missing_token: str | None = None,
) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = int(value)
        return rounded if abs(value - rounded) <= 1e-9 else None
    if parse_string and isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if missing_token is not None and raw.lower() == str(missing_token).lower():
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_prefix: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if log_prefix:
        print(f"{log_prefix} cmd={' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def run_subprocess_checked(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_prefix: str | None = None,
    error_prefix: str = "Command failed",
) -> None:
    completed = run_subprocess(
        command,
        cwd=cwd,
        env=env,
        log_prefix=log_prefix,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{error_prefix} (exit_code={completed.returncode}): {' '.join(command)}"
        )
