"""Stateless helper utilities used by skill handlers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = minimum
    return max(minimum, min(maximum, n))


def _source_to_db_label(source: str) -> str | None:
    if not source:
        return None
    norm = source.strip().lower()
    if norm in {"hn", "hackernews", "hacker_news"}:
        return "HackerNews"
    if norm in {"tc", "techcrunch", "tech_crunch"}:
        return "TechCrunch"
    if norm in {"all", "*"}:
        return None
    return source.strip()


def _split_urls(urls: str) -> list[str]:
    if not urls:
        return []

    text = urls.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    parts = re.split(r"[\n,\s]+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_probable_url(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("http://") or t.startswith("https://")


def _extract_time_window_days(text: str, default: int = 14, maximum: int = 180) -> int:
    m = re.search(
        r"(?:最近|过去|last|recent|past)?\s*(\d{1,3})\s*(天|日|周|星期|月|day|days|week|weeks|month|months)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return _clamp_int(default, 1, maximum)
    value = int(m.group(1))
    unit = str(m.group(2)).lower()
    if unit in {"周", "星期", "week", "weeks"}:
        value *= 7
    elif unit in {"月", "month", "months"}:
        value *= 30
    return _clamp_int(value, 1, maximum)


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _to_utc_naive_datetime(value: Any) -> datetime | None:
    """Normalize timestamp-like values into UTC-naive datetime for safe comparison."""
    if value is None:
        return None

    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except Exception:
            return None
    else:
        return None

    if dt.tzinfo is not None:
        try:
            dt = dt.astimezone(timezone.utc)
        except Exception:
            pass
        dt = dt.replace(tzinfo=None)
    return dt


def _is_recent_timestamp(value: Any, cutoff: Any) -> bool:
    """Compare timestamps safely across naive/aware/string timestamp mixes."""
    lhs = _to_utc_naive_datetime(value)
    rhs = _to_utc_naive_datetime(cutoff)
    if lhs is None or rhs is None:
        return False
    try:
        return lhs >= rhs
    except Exception:
        return False


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _evidence_from_records(records: list[dict[str, Any]], max_items: int = 8) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for row in records:
        url = str(row.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = row.get("title_cn") or row.get("title")
        score_value = row.get("score")
        if score_value is None:
            score_value = row.get("points")
        evidence.append(
            {
                "url": url,
                "title": str(title).strip() if title else None,
                "source": str(row.get("source") or "").strip() or None,
                "created_at": str(row.get("created_at") or "").strip() or None,
                "score": _safe_float(score_value),
            }
        )
        if len(evidence) >= max_items:
            break
    return evidence


def _evidence_from_text_output(text: str, max_items: int = 8) -> list[dict[str, Any]]:
    """Extract evidence entries from structured text output containing URLs."""
    if not text:
        return []

    url_pattern = re.compile(r"https?://[^\s)\]]+")
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for line in text.splitlines():
        urls_in_line = url_pattern.findall(line)
        for url in urls_in_line:
            normalized = url.rstrip(".,;:!?")
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            title = None
            parts = line.split("|")
            if len(parts) >= 2:
                candidate = parts[0].strip()
                candidate = re.sub(
                    r"^\s*(?:\d+\.\s*)?(?:\[.*?\]\s*)*(?:#\d+\s*)?(?:\[.*?\]\s*)*",
                    "",
                    candidate,
                ).strip()
                if candidate and len(candidate) > 3:
                    title = candidate

            source = None
            source_match = re.search(r"\[(HackerNews|TechCrunch)\]", line)
            if source_match:
                source = source_match.group(1)

            evidence.append(
                {
                    "url": normalized,
                    "title": title,
                    "source": source,
                    "created_at": None,
                    "score": None,
                }
            )
            if len(evidence) >= max_items:
                return evidence
    return evidence
