"""Dataset loading/validation helpers for eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from capabilities import CAPABILITY_CATALOG, resolve_capability, supported_capabilities
except ImportError:  # package-style import fallback
    from .capabilities import CAPABILITY_CATALOG, resolve_capability, supported_capabilities


DEFAULT_REQUIRED_TOOLS_BY_CAPABILITY: dict[str, list[str]] = {
    "compare_topics": ["compare_topics"],
    "timeline": ["build_timeline"],
    "landscape": ["analyze_landscape"],
    "trend_analysis": ["trend_analysis"],
    "compare_sources": ["compare_sources"],
    "query_news": ["query_news"],
    "fulltext_batch": ["fulltext_batch"],
    "general_qa": [],
}


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    elif isinstance(value, list):
        items = [str(x).strip() for x in value]
    else:
        items = [str(value).strip()]
    return [x for x in items if x]


def _to_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_eval_cases(
    dataset_path: Path,
    *,
    strict_capability_check: bool = True,
    include_disabled: bool = False,
) -> list[dict[str, Any]]:
    """Load and normalize JSONL eval cases."""
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    supported = supported_capabilities()

    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            question = str(item.get("question", "")).strip()
            if not question:
                raise ValueError(f"Missing question at line {line_no}")

            category = str(item.get("category", "general")).strip().lower()
            capability = resolve_capability(category, item.get("capability"))
            if strict_capability_check and capability not in supported:
                raise ValueError(
                    f"Unsupported capability '{capability}' at line {line_no}; "
                    f"supported={sorted(supported)}"
                )
            if capability not in supported:
                capability = "general_qa"

            case_id = str(item.get("id", f"case_{len(cases) + 1}")).strip()
            if not case_id:
                raise ValueError(f"Missing id at line {line_no}")
            if case_id in seen_ids:
                raise ValueError(f"Duplicate case id '{case_id}' at line {line_no}")
            seen_ids.add(case_id)

            enabled = _to_bool(item.get("enabled", True), default=True)
            if not enabled and not include_disabled:
                continue

            default_min_urls = int(CAPABILITY_CATALOG[capability].get("default_min_urls", 0))
            min_urls = max(0, _safe_int(item.get("min_urls", default_min_urls), default_min_urls))

            required_tools = _normalize_str_list(item.get("required_tools", []))
            if not required_tools:
                required_tools = list(DEFAULT_REQUIRED_TOOLS_BY_CAPABILITY.get(capability, []))

            case = {
                "id": case_id,
                "category": category,
                "capability": capability,
                "question": question,
                "min_urls": min_urls,
                "must_contain": _normalize_str_list(item.get("must_contain", [])),
                "expected_facts": _normalize_str_list(item.get("expected_facts", [])),
                "required_tools": required_tools,
                "must_not_contain": _normalize_str_list(item.get("must_not_contain", [])),
                "tags": _normalize_str_list(item.get("tags", [])),
                "enabled": enabled,
            }
            cases.append(case)

    return cases


def parse_csv_filter_arg(raw: str) -> set[str]:
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def filter_eval_cases(
    cases: list[dict[str, Any]],
    *,
    categories: set[str] | None = None,
    capabilities: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not cases:
        return []

    cat_filter = {x.lower() for x in (categories or set()) if x}
    cap_filter = {x.lower() for x in (capabilities or set()) if x}

    out: list[dict[str, Any]] = []
    for case in cases:
        if cat_filter and case.get("category", "").lower() not in cat_filter:
            continue
        if cap_filter and case.get("capability", "").lower() not in cap_filter:
            continue
        out.append(case)
    return out


def summarize_case_matrix(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    by_capability: dict[str, int] = {}
    for case in cases:
        c = str(case.get("category", "general")).lower()
        p = str(case.get("capability", "general_qa")).lower()
        by_category[c] = by_category.get(c, 0) + 1
        by_capability[p] = by_capability.get(p, 0) + 1

    return {
        "case_count": len(cases),
        "categories": by_category,
        "capabilities": by_capability,
    }
