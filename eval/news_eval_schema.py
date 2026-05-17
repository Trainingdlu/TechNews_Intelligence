"""Schemas and JSONL helpers for event-driven news-agent evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    from eval_core import normalize_url_for_retrieval
except ImportError:  # pragma: no cover
    from .eval_core import normalize_url_for_retrieval


EVENT_CARD_REQUIRED_FIELDS = (
    "event_id",
    "event_title",
    "entities",
    "time_window",
    "core_urls",
    "related_urls",
    "facts",
    "suitable_tasks",
)

RETRIEVAL_CASE_REQUIRED_FIELDS = (
    "case_id",
    "question",
    "query_type",
    "gold_event_id",
    "gold_urls",
)

GENERATION_CASE_REQUIRED_FIELDS = (
    "case_id",
    "question",
    "evidence",
    "required_claims",
    "forbidden_claims",
)

E2E_CASE_REQUIRED_FIELDS = (
    "case_id",
    "question",
    "gold_event_id",
    "gold_urls",
    "expected_behavior",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object.")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def _require_fields(row: dict[str, Any], fields: tuple[str, ...], *, label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")


def _as_non_empty_string(value: Any, *, field: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}.{field} must be non-empty.")
    return text


def _as_string_list(value: Any, *, field: str, label: str, min_items: int = 0) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label}.{field} must be a list.")
    items = [str(item).strip() for item in value if str(item).strip()]
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    if len(out) < min_items:
        raise ValueError(f"{label}.{field} must contain at least {min_items} item(s).")
    return out


def _as_url_list(value: Any, *, field: str, label: str, min_items: int = 0) -> list[str]:
    urls = _as_string_list(value, field=field, label=label, min_items=min_items)
    normalized_seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        normalized = normalize_url_for_retrieval(url)
        if not normalized:
            continue
        if normalized in normalized_seen:
            continue
        normalized_seen.add(normalized)
        out.append(url)
    if len(out) < min_items:
        raise ValueError(f"{label}.{field} must contain at least {min_items} valid URL(s).")
    return out


def validate_event_card(card: dict[str, Any]) -> dict[str, Any]:
    label = f"event_card[{card.get('event_id', '?')}]"
    _require_fields(card, EVENT_CARD_REQUIRED_FIELDS, label=label)

    event_id = _as_non_empty_string(card.get("event_id"), field="event_id", label=label)
    event_title = _as_non_empty_string(card.get("event_title"), field="event_title", label=label)
    entities = _as_string_list(card.get("entities", []), field="entities", label=label)
    core_urls = _as_url_list(card.get("core_urls", []), field="core_urls", label=label, min_items=1)
    related_urls = _as_url_list(card.get("related_urls", []), field="related_urls", label=label)
    suitable_tasks = _as_string_list(
        card.get("suitable_tasks", []),
        field="suitable_tasks",
        label=label,
        min_items=1,
    )

    time_window = card.get("time_window")
    if not isinstance(time_window, dict):
        raise ValueError(f"{label}.time_window must be an object.")
    start = str(time_window.get("start", "")).strip()
    end = str(time_window.get("end", "")).strip()
    if not start or not end:
        raise ValueError(f"{label}.time_window.start/end must be non-empty.")

    facts_raw = card.get("facts")
    if not isinstance(facts_raw, list) or not facts_raw:
        raise ValueError(f"{label}.facts must be a non-empty list.")
    facts: list[dict[str, str]] = []
    valid_urls = {normalize_url_for_retrieval(url) for url in core_urls + related_urls}
    for idx, fact in enumerate(facts_raw, 1):
        if not isinstance(fact, dict):
            raise ValueError(f"{label}.facts[{idx}] must be an object.")
        claim = _as_non_empty_string(fact.get("claim"), field="claim", label=f"{label}.facts[{idx}]")
        quote = _as_non_empty_string(fact.get("quote"), field="quote", label=f"{label}.facts[{idx}]")
        url = _as_non_empty_string(fact.get("url"), field="url", label=f"{label}.facts[{idx}]")
        normalized_url = normalize_url_for_retrieval(url)
        if not normalized_url:
            raise ValueError(f"{label}.facts[{idx}].url must be a valid URL.")
        if normalized_url not in valid_urls:
            related_urls.append(url)
            valid_urls.add(normalized_url)
        facts.append({"claim": claim, "quote": quote, "url": url})

    return {
        **card,
        "event_id": event_id,
        "event_title": event_title,
        "entities": entities,
        "time_window": {"start": start, "end": end},
        "core_urls": core_urls,
        "related_urls": related_urls,
        "facts": facts,
        "suitable_tasks": suitable_tasks,
    }


def validate_retrieval_case(case: dict[str, Any]) -> dict[str, Any]:
    label = f"retrieval_case[{case.get('case_id', '?')}]"
    _require_fields(case, RETRIEVAL_CASE_REQUIRED_FIELDS, label=label)
    gold_urls = _as_url_list(case.get("gold_urls", []), field="gold_urls", label=label, min_items=1)
    return {
        **case,
        "case_id": _as_non_empty_string(case.get("case_id"), field="case_id", label=label),
        "question": _as_non_empty_string(case.get("question"), field="question", label=label),
        "query_type": _as_non_empty_string(case.get("query_type"), field="query_type", label=label),
        "gold_event_id": _as_non_empty_string(case.get("gold_event_id"), field="gold_event_id", label=label),
        "gold_urls": gold_urls,
    }


def validate_generation_case(case: dict[str, Any]) -> dict[str, Any]:
    label = f"generation_case[{case.get('case_id', '?')}]"
    _require_fields(case, GENERATION_CASE_REQUIRED_FIELDS, label=label)
    evidence_raw = case.get("evidence")
    if not isinstance(evidence_raw, list) or not evidence_raw:
        raise ValueError(f"{label}.evidence must be a non-empty list.")
    evidence: list[dict[str, str]] = []
    for idx, item in enumerate(evidence_raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{label}.evidence[{idx}] must be an object.")
        title = str(item.get("title") or "").strip()
        quote = _as_non_empty_string(item.get("quote"), field="quote", label=f"{label}.evidence[{idx}]")
        url = _as_non_empty_string(item.get("url"), field="url", label=f"{label}.evidence[{idx}]")
        if not normalize_url_for_retrieval(url):
            raise ValueError(f"{label}.evidence[{idx}].url must be a valid URL.")
        evidence.append({"title": title, "quote": quote, "url": url})
    return {
        **case,
        "case_id": _as_non_empty_string(case.get("case_id"), field="case_id", label=label),
        "question": _as_non_empty_string(case.get("question"), field="question", label=label),
        "evidence": evidence,
        "required_claims": _as_string_list(
            case.get("required_claims", []),
            field="required_claims",
            label=label,
            min_items=1,
        ),
        "forbidden_claims": _as_string_list(case.get("forbidden_claims", []), field="forbidden_claims", label=label),
    }


def validate_e2e_case(case: dict[str, Any]) -> dict[str, Any]:
    label = f"e2e_case[{case.get('case_id', '?')}]"
    _require_fields(case, E2E_CASE_REQUIRED_FIELDS, label=label)
    gold_urls = _as_url_list(case.get("gold_urls", []), field="gold_urls", label=label, min_items=1)
    return {
        **case,
        "case_id": _as_non_empty_string(case.get("case_id"), field="case_id", label=label),
        "question": _as_non_empty_string(case.get("question"), field="question", label=label),
        "gold_event_id": _as_non_empty_string(case.get("gold_event_id"), field="gold_event_id", label=label),
        "gold_urls": gold_urls,
        "expected_behavior": _as_non_empty_string(case.get("expected_behavior"), field="expected_behavior", label=label),
    }


def load_event_cards(path: Path) -> list[dict[str, Any]]:
    return [validate_event_card(row) for row in read_jsonl(path)]


def load_retrieval_cases(path: Path) -> list[dict[str, Any]]:
    return [validate_retrieval_case(row) for row in read_jsonl(path)]


def load_generation_cases(path: Path) -> list[dict[str, Any]]:
    return [validate_generation_case(row) for row in read_jsonl(path)]


def load_e2e_cases(path: Path) -> list[dict[str, Any]]:
    return [validate_e2e_case(row) for row in read_jsonl(path)]

