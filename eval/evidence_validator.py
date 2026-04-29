"""Deterministic evidence validation for task-eval dataset generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz


ELLIPSIS_MARKERS = ("...", "\u2026")
PARTIAL_PASS_THRESHOLD = 95.0
PARTIAL_AUDIT_THRESHOLD = 90.0

LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
NUMBER_RE = re.compile(
    r"""
    (?P<prefix>[$\u20ac\u00a3\u00a5])?\s*
    (?P<num>\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)
    \s*
    (?P<suffix>%|percent|percentage|k|m|b|bn|t|million|billion|trillion|\u4e07|\u4ebf)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

WORD_NUMBER_RE = re.compile(
    r"\b(?P<num>one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<suffix>million|billion|trillion)\b",
    re.IGNORECASE,
)

WORD_NUMBERS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}

UNIT_MULTIPLIERS = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "\u4e07": 10_000.0,
    "\u4ebf": 100_000_000.0,
}


@dataclass
class EvidenceFeedback:
    failed_claim_index: int
    stage: str
    reason: str
    bad_quote: str = ""
    instruction: str = ""
    score: float | None = None
    doc_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "failed_claim_index": self.failed_claim_index,
            "stage": self.stage,
            "reason": self.reason,
            "instruction": self.instruction,
        }
        if self.bad_quote:
            out["bad_quote"] = self.bad_quote
        if self.score is not None:
            out["score"] = round(float(self.score), 4)
        if self.doc_id:
            out["doc_id"] = self.doc_id
        return out


@dataclass
class EvidenceMatch:
    claim_index: int
    doc_id: str
    quote: str
    status: str
    score: float
    best_window: str = ""
    numeric_status: str = "not_applicable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "doc_id": self.doc_id,
            "quote": self.quote,
            "status": self.status,
            "score": round(float(self.score), 4),
            "best_window": self.best_window,
            "numeric_status": self.numeric_status,
        }


@dataclass
class EvidenceValidationResult:
    accepted: bool
    audit_required: bool = False
    status: str = "not_checked"
    score: float = 0.0
    numeric_status: str = "not_applicable"
    matches: list[EvidenceMatch] = field(default_factory=list)
    feedback: list[EvidenceFeedback] = field(default_factory=list)

    def as_case_metadata(self) -> dict[str, Any]:
        status = self.status
        if self.accepted and self.audit_required and status == "borderline":
            status = "borderline_audit_required"
        return {
            "evidence_match_status": status,
            "evidence_match_score": round(float(self.score), 4),
            "numeric_equivalence_status": self.numeric_status,
            "audit_required": bool(self.audit_required),
            "evidence_matches": [match.as_dict() for match in self.matches],
        }


def normalize_compact(text: str) -> str:
    """Keep alphanumeric Unicode chars and remove whitespace/punctuation."""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def normalize_token_form(text: str) -> str:
    """Keep Latin word boundaries for fuzzy matching while preserving CJK runs."""
    raw = str(text or "").lower()
    parts: list[str] = []
    cursor = 0
    for match in LATIN_TOKEN_RE.finditer(raw):
        cjk = "".join(CJK_RE.findall(raw[cursor : match.start()]))
        if cjk:
            parts.append(cjk)
        parts.append(match.group(0))
        cursor = match.end()
    tail_cjk = "".join(CJK_RE.findall(raw[cursor:]))
    if tail_cjk:
        parts.append(tail_cjk)
    return " ".join(part for part in parts if part)


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(str(text or "")))


def _latin_token_count(text: str) -> int:
    return len(LATIN_TOKEN_RE.findall(str(text or "")))


def _quote_length_valid(quote: str, claim_type: str) -> tuple[bool, str]:
    compact = normalize_compact(quote)
    if str(claim_type or "").lower() == "number" and extract_quantities(quote):
        return (len(compact) >= 6, "number_quote_too_short")
    if _contains_cjk(quote):
        return (len(compact) >= 8, "cjk_quote_too_short")
    if len(compact) < 16 or _latin_token_count(quote) < 3:
        return False, "latin_quote_too_short"
    return True, ""


def _source_text(doc: dict[str, Any]) -> str:
    return "\n".join(
        str(doc.get(key, "") or "")
        for key in ("title", "summary", "evidence_text", "content", "full_text")
        if str(doc.get(key, "") or "").strip()
    )


def _best_window(source: str, quote: str, *, window_extra: int = 80) -> str:
    source_text = str(source or "")
    quote_text = str(quote or "")
    if not source_text:
        return ""
    compact_quote = normalize_compact(quote_text)
    if not compact_quote:
        return source_text[:240]
    needle = compact_quote[: min(24, len(compact_quote))]
    compact_source = normalize_compact(source_text)
    idx = compact_source.find(needle)
    if idx < 0:
        return source_text[:240]
    ratio = len(source_text) / max(1, len(compact_source))
    approx = int(idx * ratio)
    start = max(0, approx - window_extra)
    end = min(len(source_text), approx + len(quote_text) + window_extra)
    return source_text[start:end].strip()


def _quantity_value(raw_num: str, suffix: str | None) -> tuple[float, str]:
    number = float(str(raw_num).replace(",", ""))
    suffix_norm = str(suffix or "").strip().lower()
    if suffix_norm in {"%", "percent", "percentage"}:
        return number, "percent"
    multiplier = UNIT_MULTIPLIERS.get(suffix_norm, 1.0)
    return number * multiplier, "number"


def extract_quantities(text: str) -> list[tuple[float, str]]:
    values: list[tuple[float, str]] = []
    raw = str(text or "")
    for match in NUMBER_RE.finditer(raw):
        try:
            values.append(_quantity_value(match.group("num"), match.group("suffix")))
        except Exception:
            continue
    for match in WORD_NUMBER_RE.finditer(raw):
        number = WORD_NUMBERS.get(match.group("num").lower())
        multiplier = UNIT_MULTIPLIERS.get(match.group("suffix").lower())
        if number is not None and multiplier is not None:
            values.append((number * multiplier, "number"))
    return values


def _numbers_equivalent(left: tuple[float, str], right: tuple[float, str]) -> bool:
    lv, lu = left
    rv, ru = right
    if lu != ru:
        return False
    if not math.isfinite(lv) or not math.isfinite(rv):
        return False
    if lv == 0 or rv == 0:
        return abs(lv - rv) <= 1e-9
    return abs(lv - rv) / max(abs(lv), abs(rv)) <= 0.02


def numeric_equivalence_status(claim: str, quote: str) -> str:
    claim_values = extract_quantities(claim)
    quote_values = extract_quantities(quote)
    if not claim_values or not quote_values:
        return "uncertain"
    for claim_value in claim_values:
        if any(_numbers_equivalent(claim_value, quote_value) for quote_value in quote_values):
            return "equivalent"
    return "conflict"


def validate_case_evidence(case: dict[str, Any]) -> EvidenceValidationResult:
    if not bool(case.get("retrieval_evaluable")):
        return EvidenceValidationResult(
            accepted=True,
            audit_required=False,
            status="non_retrieval",
            score=100.0,
            numeric_status="not_applicable",
        )

    pool = case.get("input_news_pool", [])
    docs_by_id = {
        str(doc.get("doc_id", "")).strip(): doc
        for doc in pool
        if isinstance(doc, dict) and str(doc.get("doc_id", "")).strip()
    }
    claims = case.get("verifiable_claims", [])
    if not isinstance(claims, list) or not claims:
        return EvidenceValidationResult(
            accepted=False,
            status="failed",
            feedback=[
                EvidenceFeedback(
                    failed_claim_index=0,
                    stage="evidence_match",
                    reason="missing_verifiable_claims",
                    instruction="Add verifiable claims with evidence_quotes from the packed pool.",
                )
            ],
        )

    matches: list[EvidenceMatch] = []
    feedback: list[EvidenceFeedback] = []
    audit_required = False
    numeric_statuses: list[str] = []
    scores: list[float] = []
    statuses: list[str] = []

    for claim_idx, claim_row in enumerate(claims, 1):
        if not isinstance(claim_row, dict):
            feedback.append(
                EvidenceFeedback(
                    failed_claim_index=claim_idx,
                    stage="evidence_match",
                    reason="claim_not_object",
                    instruction="Return each verifiable_claim as an object.",
                )
            )
            continue
        claim_text = str(claim_row.get("claim", "") or "")
        claim_type = str(claim_row.get("claim_type", "fact") or "fact").lower()
        quotes = claim_row.get("evidence_quotes", [])
        if not isinstance(quotes, list) or not quotes:
            feedback.append(
                EvidenceFeedback(
                    failed_claim_index=claim_idx,
                    stage="evidence_match",
                    reason="missing_evidence_quotes",
                    instruction="Add at least one evidence_quote with a doc_id and a continuous exact excerpt.",
                )
            )
            continue

        claim_passed = False
        best_score = 0.0
        best_numeric = "not_applicable"
        for raw_quote in quotes:
            if isinstance(raw_quote, str):
                doc_ids = [str(item).strip() for item in claim_row.get("evidence_doc_ids", []) if str(item).strip()]
                quote_text = raw_quote
                doc_id = doc_ids[0] if doc_ids else ""
            elif isinstance(raw_quote, dict):
                quote_text = str(raw_quote.get("quote", "") or "")
                doc_id = str(raw_quote.get("doc_id", "") or "").strip()
            else:
                continue

            if any(marker in quote_text for marker in ELLIPSIS_MARKERS):
                feedback.append(
                    EvidenceFeedback(
                        failed_claim_index=claim_idx,
                        stage="evidence_match",
                        reason="quote_contains_ellipsis",
                        bad_quote=quote_text,
                        doc_id=doc_id,
                        instruction="Replace it with one single continuous exact excerpt from the same document.",
                    )
                )
                continue

            ok_len, len_reason = _quote_length_valid(quote_text, claim_type)
            if not ok_len:
                feedback.append(
                    EvidenceFeedback(
                        failed_claim_index=claim_idx,
                        stage="evidence_match",
                        reason=len_reason,
                        bad_quote=quote_text,
                        doc_id=doc_id,
                        instruction="Use a longer continuous excerpt that contains the full supporting fact.",
                    )
                )
                continue

            doc = docs_by_id.get(doc_id)
            if doc is None:
                feedback.append(
                    EvidenceFeedback(
                        failed_claim_index=claim_idx,
                        stage="evidence_match",
                        reason="quote_doc_not_in_pool",
                        bad_quote=quote_text,
                        doc_id=doc_id,
                        instruction="Use a doc_id from the packed input_news_pool.",
                    )
                )
                continue

            source = _source_text(doc)
            quote_compact = normalize_compact(quote_text)
            source_compact = normalize_compact(source)
            quote_token = normalize_token_form(quote_text)
            source_token = normalize_token_form(source)
            best_window = _best_window(source, quote_text)
            if quote_compact and quote_compact in source_compact:
                status = "exact"
                score = 100.0
            else:
                score = float(fuzz.partial_ratio(quote_token, source_token)) if quote_token and source_token else 0.0
                if score >= PARTIAL_PASS_THRESHOLD:
                    status = "partial_pass"
                elif score >= PARTIAL_AUDIT_THRESHOLD:
                    status = "borderline"
                    audit_required = True
                else:
                    status = "failed"

            numeric_status = "not_applicable"
            if claim_type == "number":
                numeric_status = numeric_equivalence_status(claim_text, quote_text)
                if numeric_status == "conflict":
                    status = "failed"
                elif numeric_status == "uncertain":
                    audit_required = True

            best_score = max(best_score, score)
            best_numeric = numeric_status
            if status != "failed":
                matches.append(
                    EvidenceMatch(
                        claim_index=claim_idx,
                        doc_id=doc_id,
                        quote=quote_text,
                        status=status,
                        score=score,
                        best_window=best_window,
                        numeric_status=numeric_status,
                    )
                )
                statuses.append(status)
                scores.append(score)
                numeric_statuses.append(numeric_status)
                claim_passed = True
                break

        if not claim_passed:
            feedback.append(
                EvidenceFeedback(
                    failed_claim_index=claim_idx,
                    stage="evidence_match",
                    reason="quote_not_found_or_below_threshold",
                    score=best_score,
                    instruction="Use a continuous exact excerpt from the same packed document; do not paraphrase.",
                )
            )
            if best_numeric == "conflict":
                feedback[-1].reason = "numeric_conflict"

    if feedback:
        return EvidenceValidationResult(
            accepted=False,
            audit_required=audit_required,
            status="failed",
            score=min(scores) if scores else 0.0,
            numeric_status=_merge_numeric_status(numeric_statuses),
            matches=matches,
            feedback=feedback,
        )

    status = "exact" if all(item == "exact" for item in statuses) else (
        "borderline" if any(item == "borderline" for item in statuses) else "partial_pass"
    )
    return EvidenceValidationResult(
        accepted=True,
        audit_required=audit_required,
        status=status,
        score=min(scores) if scores else 100.0,
        numeric_status=_merge_numeric_status(numeric_statuses),
        matches=matches,
        feedback=[],
    )


def _merge_numeric_status(statuses: list[str]) -> str:
    relevant = [status for status in statuses if status != "not_applicable"]
    if not relevant:
        return "not_applicable"
    if "conflict" in relevant:
        return "conflict"
    if "uncertain" in relevant:
        return "uncertain"
    return "equivalent"
