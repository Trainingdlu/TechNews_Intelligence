"""Quality gates for task-eval news pools.

This module is only used by dataset generation. It does not affect the
production Agent retrieval path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}")

ANCHOR_STOPWORDS = {
    "a",
    "about",
    "an",
    "analyze",
    "and",
    "around",
    "build",
    "com",
    "compare",
    "day",
    "days",
    "english",
    "for",
    "from",
    "http",
    "https",
    "latest",
    "net",
    "news",
    "org",
    "recent",
    "related",
    "search",
    "the",
    "updates",
    "vs",
    "week",
    "with",
}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;|\s]+", value) if item.strip()]
    return []


def _params(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("parameter_template", {})
    return value if isinstance(value, dict) else {}


def _sampling(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("sampling", {})
    return value if isinstance(value, dict) else {}


def _tokenize(text: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def _dedupe(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        text = str(token or "").strip().lower().strip("_")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def anchor_tokens(text: Any) -> list[str]:
    tokens: list[str] = []
    raw = str(text or "").strip()
    if not raw:
        return []
    raw = re.sub(r"https?://", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[/_.?=&:#-]+", " ", raw)
    for token in _tokenize(raw):
        cleaned = token.strip("_").lower()
        if len(cleaned) < 2:
            continue
        if cleaned.isdigit():
            continue
        if cleaned in ANCHOR_STOPWORDS:
            continue
        if re.fullmatch(r"20\d{2}|\d+d?", cleaned):
            continue
        tokens.append(cleaned)
    return _dedupe(tokens)


def topic_anchor_texts(task: dict[str, Any]) -> list[str]:
    params = _params(task)
    sampling = _sampling(task)
    out: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)

    for key in ("query", "topic", "topic_a", "topic_b", "urls"):
        add(params.get(key))
    for entity in _as_list(params.get("entities")):
        add(entity)
    if not out:
        for keyword in _as_list(sampling.get("keywords", [])):
            add(keyword)
    return out


def topic_anchor_terms(task: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for text in topic_anchor_texts(task):
        terms.extend(anchor_tokens(text))
    return _dedupe(terms)


def quality_required(task: dict[str, Any]) -> bool:
    scenario = str(task.get("scenario", "")).strip().lower()
    return (
        str(task.get("retrieval_mode", "")).strip().lower() == "evaluable"
        and not bool(task.get("should_clarify", False))
        and scenario not in {"empty", "conflict"}
    )


def _doc_field_tokens(doc: dict[str, Any]) -> dict[str, set[str]]:
    fields = {}
    for key in ("title", "title_cn", "summary", "evidence_text", "url"):
        fields[key] = set(anchor_tokens(doc.get(key, "")))
    return fields


def required_anchor_hits(terms: list[str]) -> int:
    if not terms:
        return 0
    if len(terms) == 1:
        return 1
    if len(terms) <= 4:
        return 2
    return max(2, min(4, math.ceil(len(terms) * 0.35)))


def score_doc_topic_match(doc: dict[str, Any], terms: list[str]) -> dict[str, Any]:
    terms = _dedupe(terms)
    field_tokens = _doc_field_tokens(doc)
    matched: list[str] = []
    hit_fields: dict[str, list[str]] = {}
    for term in terms:
        term_fields = [field for field, tokens in field_tokens.items() if term in tokens]
        if not term_fields:
            continue
        matched.append(term)
        for field in term_fields:
            hit_fields.setdefault(field, []).append(term)

    hit_count = len(set(matched))
    coverage = (hit_count / len(terms)) if terms else 0.0
    seed_similarity = max(0.0, min(1.0, float(doc.get("seed_similarity", 0.0) or 0.0)))
    field_bonus = 0.0
    if hit_fields.get("title") or hit_fields.get("title_cn"):
        field_bonus += 0.12
    if hit_fields.get("url"):
        field_bonus += 0.06
    score = min(1.0, 0.70 * coverage + 0.24 * seed_similarity + field_bonus)
    required_hits = required_anchor_hits(terms)
    passed = bool(terms) and (
        hit_count >= required_hits
        or (coverage >= 0.5 and seed_similarity >= 0.34)
    )
    return {
        "topic_anchor_terms": terms,
        "matched_anchor_terms": _dedupe(matched),
        "anchor_hit_fields": {key: _dedupe(value) for key, value in hit_fields.items()},
        "topic_match_score": round(score, 4),
        "topic_match_passed": passed,
        "topic_match_hit_count": hit_count,
        "topic_match_required_hits": required_hits,
    }


def annotate_doc_topic_match(doc: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    metrics = score_doc_topic_match(doc, topic_anchor_terms(task))
    doc.update(metrics)
    return doc


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _centroid(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    size = len(vectors[0])
    values = [0.0] * size
    count = 0
    for vec in vectors:
        if len(vec) != size:
            continue
        count += 1
        for idx, item in enumerate(vec):
            values[idx] += float(item)
    if count <= 0:
        return None
    values = [value / count for value in values]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return None
    return [value / norm for value in values]


def embedding_coherence(docs: list[dict[str, Any]]) -> tuple[float | None, int]:
    vectors = [doc.get("embedding") for doc in docs if isinstance(doc.get("embedding"), list)]
    vectors = [vec for vec in vectors if vec]
    if len(vectors) < 3:
        return None, len(vectors)
    center = _centroid(vectors)
    if not center:
        return None, len(vectors)
    score = sum(max(0.0, _cosine(vec, center)) for vec in vectors) / len(vectors)
    return round(float(score), 4), len(vectors)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _time_span_days(docs: list[dict[str, Any]]) -> int | None:
    dates = [_parse_datetime(doc.get("published_at")) for doc in docs]
    dates = [dt for dt in dates if dt is not None]
    if len(dates) < 2:
        return None
    return int((max(dates) - min(dates)).days)


def _topic_side_for_doc(doc: dict[str, Any]) -> str:
    group = str(doc.get("topic_group", "")).strip().upper()
    if group in {"A", "B"}:
        return group
    labels = {str(item).strip().lower() for item in doc.get("query_labels", [])}
    if "topic_a" in labels and "topic_b" not in labels:
        return "A"
    if "topic_b" in labels and "topic_a" not in labels:
        return "B"
    return ""


def pool_quality_summary(docs: list[dict[str, Any]], task: dict[str, Any]) -> dict[str, Any]:
    terms = topic_anchor_terms(task)
    for doc in docs:
        if "topic_match_score" not in doc:
            annotate_doc_topic_match(doc, task)

    selected_count = len(docs)
    strong_count = sum(1 for doc in docs if bool(doc.get("topic_match_passed")))
    ratio = (strong_count / selected_count) if selected_count else 0.0
    avg_score = (
        sum(float(doc.get("topic_match_score", 0.0) or 0.0) for doc in docs) / selected_count
        if selected_count
        else 0.0
    )
    min_score = min((float(doc.get("topic_match_score", 0.0) or 0.0) for doc in docs), default=0.0)
    coherence, embedding_count = embedding_coherence(docs)
    source_counts = Counter(str(doc.get("source", "unknown")) or "unknown" for doc in docs)
    side_counts = Counter(_topic_side_for_doc(doc) for doc in docs)
    side_counts.pop("", None)

    reasons: list[str] = []
    required = quality_required(task)
    min_docs = min(3, max(1, selected_count))
    if required and selected_count < min_docs:
        reasons.append("too_few_selected_docs")
    if required and not terms:
        reasons.append("missing_topic_anchor")
    if required and ratio < 0.50:
        reasons.append("low_topic_match_ratio")
    if required and strong_count < min(3, max(1, selected_count)):
        reasons.append("too_few_topic_matched_docs")
    if required and coherence is not None and coherence < 0.35:
        reasons.append("low_embedding_coherence")

    tool = str(task.get("tool", "")).strip()
    if required and tool == "compare_topics":
        if int(side_counts.get("A", 0)) <= 0 or int(side_counts.get("B", 0)) <= 0:
            reasons.append("missing_compare_topic_side")
    if required and tool == "compare_sources":
        required_sources = [str(item).strip() for item in _sampling(task).get("sources", []) if str(item).strip()]
        missing_sources = [source for source in required_sources if source_counts.get(source, 0) <= 0]
        if missing_sources:
            reasons.append("missing_compare_source")

    return {
        "pool_quality_required": required,
        "pool_quality_passed": not reasons,
        "pool_quality_reasons": reasons,
        "topic_anchor_terms": terms,
        "topic_match_ratio": round(ratio, 4),
        "topic_matched_docs": strong_count,
        "selected_docs": selected_count,
        "avg_topic_match_score": round(avg_score, 4),
        "min_topic_match_score": round(min_score, 4),
        "embedding_coherence": coherence,
        "embedding_docs": embedding_count,
        "source_count": len(source_counts),
        "source_counts": dict(source_counts),
        "topic_side_counts": {key: int(value) for key, value in side_counts.items()},
        "time_span_days": _time_span_days(docs),
    }
