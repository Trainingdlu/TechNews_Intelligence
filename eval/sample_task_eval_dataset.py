"""Stratified sampler for task_eval JSONL datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scenario(task_type: str) -> str:
    parts = [part.strip() for part in str(task_type or "").split(".") if part.strip()]
    if not parts:
        return "unknown"
    return parts[-1]


def _strata_key(case: dict[str, Any]) -> tuple[str, str]:
    skill = str(case.get("skill", "")).strip() or "unknown_skill"
    scenario = _scenario(str(case.get("task_type", "")).strip())
    return skill, scenario


def _build_quotas(
    *,
    counts: dict[tuple[str, str], int],
    ratio: float,
    total_target: int,
) -> dict[tuple[str, str], int]:
    quotas: dict[tuple[str, str], int] = {}
    fractional: list[tuple[float, tuple[str, str]]] = []
    for key, size in counts.items():
        ideal = float(size) * float(ratio)
        floor = int(ideal)
        quota = 1 if size > 0 and ideal > 0 else 0
        quota = max(quota, floor)
        quota = min(quota, size)
        quotas[key] = quota
        fractional.append((ideal - float(floor), key))

    current = sum(quotas.values())
    if current < total_target:
        remain = total_target - current
        for _, key in sorted(fractional, key=lambda item: item[0], reverse=True):
            if remain <= 0:
                break
            cap = counts[key]
            if quotas[key] >= cap:
                continue
            quotas[key] += 1
            remain -= 1
        if remain > 0:
            for key, size in sorted(counts.items(), key=lambda item: item[1], reverse=True):
                if remain <= 0:
                    break
                if quotas[key] >= size:
                    continue
                quotas[key] += 1
                remain -= 1
    elif current > total_target:
        overflow = current - total_target
        for key, _ in sorted(counts.items(), key=lambda item: quotas[item[0]], reverse=True):
            if overflow <= 0:
                break
            while quotas[key] > 1 and overflow > 0:
                quotas[key] -= 1
                overflow -= 1
        if overflow > 0:
            for key, _ in sorted(counts.items(), key=lambda item: quotas[item[0]], reverse=True):
                if overflow <= 0:
                    break
                while quotas[key] > 0 and overflow > 0:
                    quotas[key] -= 1
                    overflow -= 1
    return quotas


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    ratio: float,
    seed: int,
    max_cases: int = 0,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    ratio = max(0.0, min(1.0, float(ratio)))
    if ratio <= 0.0:
        return []
    if ratio >= 1.0:
        return list(rows)

    indexed_rows = list(enumerate(rows))
    strata: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in indexed_rows:
        key = _strata_key(row)
        strata.setdefault(key, []).append((idx, row))

    total_target = max(1, min(len(rows), int(round(len(rows) * ratio))))
    counts = {key: len(bucket) for key, bucket in strata.items()}
    quotas = _build_quotas(counts=counts, ratio=ratio, total_target=total_target)

    rng = random.Random(int(seed))
    selected: list[tuple[int, dict[str, Any]]] = []
    for key, bucket in strata.items():
        quota = max(0, min(len(bucket), int(quotas.get(key, 0))))
        if quota <= 0:
            continue
        chosen = rng.sample(bucket, quota) if quota < len(bucket) else list(bucket)
        selected.extend(chosen)

    selected.sort(key=lambda item: item[0])
    sampled = [row for _, row in selected]
    capped = max(0, int(max_cases))
    if capped <= 0 or len(sampled) <= capped:
        return sampled

    indexed_rows = list(enumerate(sampled))
    strata: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in indexed_rows:
        key = _strata_key(row)
        strata.setdefault(key, []).append((idx, row))

    counts = {key: len(bucket) for key, bucket in strata.items()}
    target = max(1, min(len(sampled), capped))
    cap_ratio = float(target) / float(len(sampled))
    quotas = _build_quotas(counts=counts, ratio=cap_ratio, total_target=target)

    rng2 = random.Random(int(seed) + 9973)
    capped_selected: list[tuple[int, dict[str, Any]]] = []
    for key, bucket in strata.items():
        quota = max(0, min(len(bucket), int(quotas.get(key, 0))))
        if quota <= 0:
            continue
        chosen = rng2.sample(bucket, quota) if quota < len(bucket) else list(bucket)
        capped_selected.extend(chosen)
    capped_selected.sort(key=lambda item: item[0])
    return [row for _, row in capped_selected]


def _parse_args() -> argparse.Namespace:
    eval_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build stratified sample dataset for task_eval.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Input task_eval JSONL dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output sampled JSONL path.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.25,
        help="Sample ratio in [0,1]. Default 0.25.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default 42.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If >0, cap sampled dataset size after stratified sampling.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=eval_dir / "reports" / "judge_sample_summary.json",
        help="Optional summary JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    rows = _read_jsonl(dataset)
    sampled = stratified_sample(
        rows,
        ratio=float(args.sample_ratio),
        seed=int(args.seed),
        max_cases=int(args.max_cases),
    )
    output = args.output.resolve()
    _write_jsonl(output, sampled)

    strata_count: dict[str, int] = {}
    for row in sampled:
        key = "|".join(_strata_key(row))
        strata_count[key] = strata_count.get(key, 0) + 1

    summary = {
        "dataset": str(dataset),
        "output": str(output),
        "input_case_count": len(rows),
        "sample_case_count": len(sampled),
        "sample_ratio": float(args.sample_ratio),
        "max_cases": int(args.max_cases),
        "seed": int(args.seed),
        "strata_count": dict(sorted(strata_count.items(), key=lambda item: item[0])),
    }
    summary_path = args.summary_output.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[TaskEvalSample] dataset={dataset}")
    print(f"[TaskEvalSample] input_cases={len(rows)} sample_cases={len(sampled)} ratio={float(args.sample_ratio):.3f}")
    print(f"[TaskEvalSample] output={output}")
    print(f"[TaskEvalSample] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
