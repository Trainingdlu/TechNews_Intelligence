"""Contract checks for formal eval dataset files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.dataset_loader import load_eval_cases


DATASET_DIR = Path(__file__).resolve().parents[2] / "eval" / "datasets"
FORMAL_DATASET_FILES = [
    "smoke.jsonl",
    "default.jsonl",
    "accuracy_snapshot.jsonl",
]


@pytest.mark.parametrize("filename", FORMAL_DATASET_FILES)
def test_formal_dataset_lines_are_valid_json(filename: str) -> None:
    path = DATASET_DIR / filename
    assert path.exists(), f"Missing dataset: {path}"

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        try:
            json.loads(item)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{filename}:{line_no} invalid JSON: {exc}") from exc


@pytest.mark.parametrize("filename", FORMAL_DATASET_FILES)
def test_formal_dataset_loads_with_strict_capability_check(filename: str) -> None:
    path = DATASET_DIR / filename
    cases = load_eval_cases(path, strict_capability_check=True)
    assert len(cases) > 0


@pytest.mark.parametrize("filename", FORMAL_DATASET_FILES)
def test_formal_dataset_case_ids_are_unique(filename: str) -> None:
    path = DATASET_DIR / filename
    cases = load_eval_cases(path, strict_capability_check=True)
    ids = [str(item.get("id")) for item in cases]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    assert not duplicates, f"{filename} contains duplicate ids: {duplicates}"
