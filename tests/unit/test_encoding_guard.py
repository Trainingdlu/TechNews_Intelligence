"""Unit tests for eval/encoding_guard.py."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

from eval import encoding_guard


def _write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


@contextmanager
def _case_dir():
    root = Path("tests/unit/.tmp_encoding_guard")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def test_scan_repository_detects_mojibake_and_replacement() -> None:
    with _case_dir() as root:
        sample = root / "eval" / "datasets" / "sample.jsonl"
        # Known mojibake-like token + replacement character.
        marker = encoding_guard.MOJIBAKE_MARKERS[0]
        _write_text(
            sample,
            f'{{"question":"{marker}?4","note":"bad\ufffdchar"}}\n',
        )

        report = encoding_guard.scan_repository(
            root=root,
            include_dirs=("eval",),
            extensions=(".jsonl",),
            exclude_parts=(),
            max_findings=50,
        )

        assert report["findings_count"] >= 2
        kinds = {item["kind"] for item in report["findings"]}
        assert "mojibake_marker" in kinds
        assert "replacement_char" in kinds


def test_scan_repository_detects_utf8_bom() -> None:
    with _case_dir() as root:
        bom_file = root / "eval" / "config" / "gates.yaml"
        bom_file.parent.mkdir(parents=True, exist_ok=True)
        bom_file.write_bytes(b"\xef\xbb\xbf" + b"rules: []\n")

        report = encoding_guard.scan_repository(
            root=root,
            include_dirs=("eval",),
            extensions=(".yaml",),
            exclude_parts=(),
            max_findings=50,
            check_bom=True,
        )

        kinds = {item["kind"] for item in report["findings"]}
        assert "utf8_bom" in kinds


def test_main_soft_mode_zero_strict_mode_nonzero() -> None:
    with _case_dir() as root:
        sample = root / "eval" / "datasets" / "sample.csv"
        marker = encoding_guard.MOJIBAKE_MARKERS[0]
        _write_text(sample, f"question\n{marker}?4\n")

        soft_report = root / "soft.json"
        strict_report = root / "strict.json"

        soft_exit = encoding_guard.main(
            [
                "--root",
                str(root),
                "--include-dirs",
                "eval",
                "--extensions",
                ".csv",
                "--report",
                str(soft_report),
            ]
        )
        strict_exit = encoding_guard.main(
            [
                "--root",
                str(root),
                "--include-dirs",
                "eval",
                "--extensions",
                ".csv",
                "--report",
                str(strict_report),
                "--strict",
            ]
        )

        assert soft_exit == 0
        assert strict_exit == 2

        payload = json.loads(soft_report.read_text(encoding="utf-8"))
        assert payload["findings_count"] >= 1
