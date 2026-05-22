"""Offline entity alias candidate extraction and DeepSeek adjudication."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_ENTITY_MODEL = "deepseek-v4-flash"
DEEPSEEK_ENTITY_ESCALATION_MODEL = "deepseek-v4-pro"

REVIEW_REQUIRED_ALIASES = {
    "ai",
    "r1",
    "go",
    "o1",
    "apple",
    "meta",
    "gemini",
    "claude",
    "cursor",
}

_MODEL_PATTERN = re.compile(r"\b(?:GPT|Claude|Gemini|Llama|Mistral|Qwen|DeepSeek|Grok|Phi|o\d|R\d)[A-Za-z0-9.\- ]{0,24}\b")
_ORG_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:[ -][A-Z][A-Za-z0-9]+){0,4}\b")
_CHIP_PATTERN = re.compile(r"\b(?:H|B|A|MI)\d{2,4}[A-Z]?\b|\bBlackwell\b|\bCUDA\b")
_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,12}")


@dataclass(frozen=True)
class EntityDecision:
    canonical_name: str
    entity_type: str
    decision: str
    confidence: float
    reason: str
    aliases_to_add: list[str]
    status: str


def _clean_alias(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" \t\r\n,.;:()[]{}\"'"))


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        items = parsed if isinstance(parsed, list) else []
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def normalize_aliases(primary_alias: str, aliases_to_add: Any) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for alias in [primary_alias, *_json_list(aliases_to_add)]:
        normalized = _clean_alias(alias)
        if not normalized:
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            aliases.append(normalized)
    return aliases


def extract_alias_candidates_from_text(
    *,
    title: str = "",
    title_cn: str = "",
    summary: str = "",
    max_candidates: int = 40,
) -> list[str]:
    """Extract lightweight entity alias candidates from a news record."""
    text = "\n".join([title or "", title_cn or "", summary or ""])
    candidates: list[str] = []
    seen: set[str] = set()

    for pattern in (_MODEL_PATTERN, _CHIP_PATTERN, _ORG_PATTERN, _TICKER_PATTERN, _CJK_PATTERN):
        for match in pattern.finditer(text):
            alias = _clean_alias(match.group(0))
            if not alias:
                continue
            key = alias.lower()
            if key in seen:
                continue
            if len(alias) <= 1:
                continue
            seen.add(key)
            candidates.append(alias)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def requires_manual_review(alias: str, *, evidence_count: int, confidence: float) -> bool:
    """Return True for aliases that should not auto-approve."""
    normalized = _clean_alias(alias).lower()
    if normalized in REVIEW_REQUIRED_ALIASES:
        return True
    if evidence_count <= 1:
        return True
    if len(normalized) <= 2:
        return True
    return confidence < 0.82


def _deepseek_timeout() -> float:
    try:
        return max(5.0, min(float(os.getenv("DEEPSEEK_ENTITY_TIMEOUT_SEC", "30")), 120.0))
    except Exception:
        return 30.0


def _deepseek_base_url() -> str:
    return os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL).rstrip("/")


def _deepseek_model(*, escalate: bool = False) -> str:
    if escalate:
        return os.getenv("DEEPSEEK_ENTITY_ESCALATION_MODEL", DEEPSEEK_ENTITY_ESCALATION_MODEL)
    return os.getenv("DEEPSEEK_ENTITY_MODEL", DEEPSEEK_ENTITY_MODEL)


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek response is not a JSON object.")
    return payload


def _decision_status(*, alias: str, decision: str, confidence: float, evidence_count: int) -> str:
    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision in {"reject", "rejected", "no_match"}:
        return "rejected"
    if requires_manual_review(alias, evidence_count=evidence_count, confidence=confidence):
        return "pending"
    if normalized_decision in {"merge", "new_entity", "alias", "accept", "approved"}:
        return "auto_approved"
    return "pending"


def adjudicate_alias_with_deepseek(
    *,
    alias: str,
    contexts: Iterable[dict[str, Any]],
    canonical_candidates: Iterable[dict[str, Any]] = (),
    evidence_count: int = 1,
    escalate: bool = False,
) -> EntityDecision:
    """Ask DeepSeek to classify an alias candidate into a review status."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    alias_clean = _clean_alias(alias)
    if not api_key:
        return EntityDecision(
            canonical_name="",
            entity_type="unknown",
            decision="pending",
            confidence=0.0,
            reason="DEEPSEEK_API_KEY not set",
            aliases_to_add=[],
            status="pending",
        )

    prompt = {
        "task": "Resolve a technology-news entity alias candidate.",
        "alias": alias_clean,
        "contexts": list(contexts)[:8],
        "canonical_candidates": list(canonical_candidates)[:12],
        "instructions": [
            "Return only valid JSON.",
            "Do not invent facts not supported by contexts or candidate entities.",
            "Use decision one of: merge, new_entity, reject, pending.",
            "Use entity_type one of: company, product, model, person, topic, chip, source, unknown.",
        ],
        "schema": {
            "canonical_name": "string",
            "entity_type": "string",
            "decision": "merge|new_entity|reject|pending",
            "confidence": "number 0..1",
            "reason": "short string",
            "aliases_to_add": ["string"],
        },
    }

    resp = requests.post(
        f"{_deepseek_base_url()}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": _deepseek_model(escalate=escalate),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict entity-resolution reviewer for technology news search.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=_deepseek_timeout(),
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    payload = _parse_json_object(content)

    confidence = float(payload.get("confidence") or 0.0)
    confidence = max(0.0, min(confidence, 1.0))
    decision = str(payload.get("decision") or "pending").strip().lower()
    aliases_raw = payload.get("aliases_to_add")
    aliases = [_clean_alias(item) for item in aliases_raw] if isinstance(aliases_raw, list) else []
    aliases = [item for item in aliases if item]
    status = _decision_status(
        alias=alias_clean,
        decision=decision,
        confidence=confidence,
        evidence_count=evidence_count,
    )

    return EntityDecision(
        canonical_name=str(payload.get("canonical_name") or "").strip(),
        entity_type=str(payload.get("entity_type") or "unknown").strip().lower(),
        decision=decision,
        confidence=confidence,
        reason=str(payload.get("reason") or "").strip(),
        aliases_to_add=aliases,
        status=status,
    )


def decision_to_candidate_row(decision: EntityDecision, *, alias: str, source: str, evidence_urls: list[str]) -> dict[str, Any]:
    """Convert a DeepSeek decision into an entity_alias_candidate-style dict."""
    return {
        "alias": _clean_alias(alias),
        "canonical_name": decision.canonical_name,
        "entity_type": decision.entity_type,
        "source": source,
        "confidence": decision.confidence,
        "evidence_count": len(evidence_urls),
        "evidence_urls": evidence_urls,
        "status": decision.status,
        "reason": decision.reason,
        "aliases_to_add": decision.aliases_to_add,
    }
