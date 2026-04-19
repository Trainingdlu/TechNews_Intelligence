"""Soft encoding guard for CI/test pipelines.

This scanner detects common text-encoding anomalies (mojibake markers,
replacement characters, non-UTF-8 files, UTF-8 BOM) and emits warnings.
By default it never blocks execution. Use --strict to fail on findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INCLUDE_DIRS = (
    "agent",
    "app",
    "services",
    "eval",
    "tests",
    "docs",
    ".github",
)
DEFAULT_EXTENSIONS = (
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".ini",
    ".cfg",
    ".j2",
)
DEFAULT_EXCLUDE_PARTS = (
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".tmp",
    "reports",
    "node_modules",
    "dist",
    "build",
)

# Markers observed in UTF-8<->GBK mojibake incidents in this repository.
MOJIBAKE_MARKERS = (
    "鏈€杩",  # encoding-guard: ignore
    "灏忔椂",  # encoding-guard: ignore
    "濡傛灉",  # encoding-guard: ignore
    "杩囧幓30澶",  # encoding-guard: ignore
    "鏋勫缓",  # encoding-guard: ignore
    "瀵規瘮",  # encoding-guard: ignore
    "鍦?",  # encoding-guard: ignore
    "鍜?",  # encoding-guard: ignore
    "銆",  # encoding-guard: ignore
    "锛",  # encoding-guard: ignore
    "闂伙紵",  # encoding-guard: ignore
    "鏍煎眬",  # encoding-guard: ignore
    "娓╄繕",  # encoding-guard: ignore
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    message: str
    snippet: str


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in str(raw or "").split(","):
        token = str(item).strip()
        if not token or token in seen:
            continue
        out.append(token)
        seen.add(token)
    return tuple(out)


def _is_candidate_file(
    path: Path,
    *,
    root: Path,
    extensions: tuple[str, ...],
    exclude_parts: tuple[str, ...],
) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in extensions:
        return False
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        rel = path
    parts = {part.lower() for part in rel.parts}
    for part in exclude_parts:
        if part.lower() in parts:
            return False
    return True


def _iter_files(
    *,
    root: Path,
    include_dirs: tuple[str, ...],
    extensions: tuple[str, ...],
    exclude_parts: tuple[str, ...],
    explicit_paths: tuple[str, ...] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if explicit_paths is not None:
        for raw in explicit_paths:
            path = (root / raw).resolve()
            if not path.exists():
                continue
            if path.is_file():
                if _is_candidate_file(path, root=root, extensions=extensions, exclude_parts=exclude_parts):
                    candidates.append(path)
                continue
            for file_path in path.rglob("*"):
                if _is_candidate_file(file_path, root=root, extensions=extensions, exclude_parts=exclude_parts):
                    candidates.append(file_path.resolve())
        return sorted(set(candidates))

    includes = include_dirs or (".",)
    for raw in includes:
        start = root / raw
        if not start.exists():
            continue
        if start.is_file():
            if _is_candidate_file(start, root=root, extensions=extensions, exclude_parts=exclude_parts):
                candidates.append(start.resolve())
            continue
        for file_path in start.rglob("*"):
            if _is_candidate_file(file_path, root=root, extensions=extensions, exclude_parts=exclude_parts):
                candidates.append(file_path.resolve())
    return sorted(set(candidates))


def _relative_posix(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _trim_snippet(line: str, *, max_len: int = 220) -> str:
    text = line.replace("\t", "    ")
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def _scan_utf8_text(path: Path, *, root: Path, text: str, findings: list[Finding]) -> None:
    rel = _relative_posix(path, root)
    for idx, line in enumerate(text.splitlines(), 1):
        if "encoding-guard: ignore" in line:
            continue
        if "\ufffd" in line:
            findings.append(
                Finding(
                    path=rel,
                    line=idx,
                    kind="replacement_char",
                    message="Line contains Unicode replacement character (U+FFFD).",
                    snippet=_trim_snippet(line),
                )
            )
        marker_hits = [marker for marker in MOJIBAKE_MARKERS if marker in line]
        if marker_hits:
            findings.append(
                Finding(
                    path=rel,
                    line=idx,
                    kind="mojibake_marker",
                    message=f"Suspicious mojibake markers detected: {', '.join(marker_hits[:4])}",
                    snippet=_trim_snippet(line),
                )
            )


def scan_repository(
    *,
    root: Path,
    include_dirs: tuple[str, ...],
    extensions: tuple[str, ...],
    exclude_parts: tuple[str, ...],
    max_findings: int,
    explicit_paths: tuple[str, ...] | None = None,
    check_bom: bool = False,
) -> dict[str, Any]:
    findings: list[Finding] = []
    scanned = 0
    files = _iter_files(
        root=root,
        include_dirs=include_dirs,
        extensions=extensions,
        exclude_parts=exclude_parts,
        explicit_paths=explicit_paths,
    )
    for path in files:
        scanned += 1
        raw = path.read_bytes()
        rel = _relative_posix(path, root)
        if check_bom and raw.startswith(b"\xef\xbb\xbf"):
            findings.append(
                Finding(
                    path=rel,
                    line=1,
                    kind="utf8_bom",
                    message="File starts with UTF-8 BOM; prefer UTF-8 without BOM.",
                    snippet="",
                )
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            findings.append(
                Finding(
                    path=rel,
                    line=max(1, int(exc.start)),
                    kind="non_utf8",
                    message=f"File is not valid UTF-8: {exc}",
                    snippet="",
                )
            )
            if len(findings) >= max_findings:
                break
            continue

        _scan_utf8_text(path, root=root, text=text, findings=findings)
        if len(findings) >= max_findings:
            break

    count_by_kind: dict[str, int] = {}
    files_with_findings: set[str] = set()
    for item in findings:
        count_by_kind[item.kind] = count_by_kind.get(item.kind, 0) + 1
        files_with_findings.add(item.path)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "scanned_files": scanned,
        "findings_count": len(findings),
        "files_with_findings_count": len(files_with_findings),
        "counts_by_kind": count_by_kind,
        "findings": [asdict(item) for item in findings],
    }


def _gh_escape(text: str) -> str:
    return (
        str(text)
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def _emit_github_annotations(findings: list[dict[str, Any]]) -> None:
    for item in findings:
        path = _gh_escape(str(item.get("path", "")))
        line = int(item.get("line", 1) or 1)
        kind = _gh_escape(str(item.get("kind", "encoding_warning")))
        message = _gh_escape(str(item.get("message", "")))
        print(
            f"::warning file={path},line={line},title=EncodingGuard({kind})::{message}"
        )


def _print_summary(report: dict[str, Any], *, preview_limit: int = 20) -> None:
    _safe_print(
        "[EncodingGuard] scanned_files="
        f"{report.get('scanned_files', 0)} findings={report.get('findings_count', 0)} "
        f"files_with_findings={report.get('files_with_findings_count', 0)}"
    )
    counts = report.get("counts_by_kind", {})
    if isinstance(counts, dict) and counts:
        _safe_print(
            f"[EncodingGuard] counts_by_kind={json.dumps(counts, ensure_ascii=False)}"
        )
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        return
    for item in findings[:preview_limit]:
        path = item.get("path", "")
        line = item.get("line", 1)
        kind = item.get("kind", "")
        message = item.get("message", "")
        _safe_print(f"[EncodingGuard][{kind}] {path}:{line} {message}")


def _safe_print(text: str) -> None:
    output = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        encoded = output.encode(encoding, errors="replace")
        print(encoded.decode(encoding, errors="replace"))
    except Exception:
        print(output.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soft encoding guard scanner.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--include-dirs",
        type=str,
        default=",".join(DEFAULT_INCLUDE_DIRS),
        help="Comma-separated include roots relative to --root.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated file extensions to scan.",
    )
    parser.add_argument(
        "--exclude-parts",
        type=str,
        default=",".join(DEFAULT_EXCLUDE_PARTS),
        help="Comma-separated path parts to skip.",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=500,
        help="Maximum findings to collect before stopping.",
    )
    parser.add_argument(
        "--paths-file",
        type=Path,
        default=None,
        help="Optional file containing newline-delimited relative paths to scan.",
    )
    parser.add_argument(
        "--check-bom",
        action="store_true",
        help="Also detect UTF-8 BOM files (off by default to reduce CI noise).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("eval/reports/encoding_guard/latest.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--github-annotations",
        action="store_true",
        help="Emit GitHub Actions ::warning annotations.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when findings exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    root = args.root.resolve()
    include_dirs = _parse_csv_list(args.include_dirs)
    extensions = tuple(token.lower() for token in _parse_csv_list(args.extensions))
    exclude_parts = _parse_csv_list(args.exclude_parts)
    max_findings = max(1, int(args.max_findings))
    explicit_paths: tuple[str, ...] | None = None
    if args.paths_file is not None:
        paths_file = args.paths_file.resolve()
        if paths_file.exists():
            explicit_paths = tuple(
                line.strip()
                for line in paths_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            explicit_paths = ()

    report = scan_repository(
        root=root,
        include_dirs=include_dirs,
        extensions=extensions,
        exclude_parts=exclude_parts,
        max_findings=max_findings,
        explicit_paths=explicit_paths,
        check_bom=bool(args.check_bom),
    )
    _print_summary(report)

    if args.github_annotations:
        findings = report.get("findings", [])
        if isinstance(findings, list):
            _emit_github_annotations(findings)

    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _safe_print(f"[EncodingGuard] report={report_path}")

    findings_count = int(report.get("findings_count", 0) or 0)
    if args.strict and findings_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
