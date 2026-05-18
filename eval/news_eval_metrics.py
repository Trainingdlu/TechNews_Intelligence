"""Metrics for event-driven retrieval and end-to-end news evaluation."""

from __future__ import annotations

from typing import Any, Iterable

try:
    from eval_core import hit_rate_at_k, mrr_at_k, ndcg_at_k, normalize_url_for_retrieval, recall_at_k
except ImportError:  # pragma: no cover
    from .eval_core import hit_rate_at_k, mrr_at_k, ndcg_at_k, normalize_url_for_retrieval, recall_at_k


def build_url_event_index(event_cards: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    """Map every known event URL to the event ids it can satisfy."""

    index: dict[str, set[str]] = {}
    for card in event_cards:
        event_id = str(card.get("event_id", "")).strip()
        if not event_id:
            continue
        urls = list(card.get("core_urls", []) or []) + list(card.get("related_urls", []) or [])
        for url in urls:
            normalized = normalize_url_for_retrieval(str(url))
            if not normalized:
                continue
            index.setdefault(normalized, set()).add(event_id)
    return index


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _source_domains(urls: Iterable[str]) -> list[str]:
    domains: list[str] = []
    for url in urls:
        domain = normalize_url_for_retrieval(str(url), domain_only=True)
        if domain:
            domains.append(domain)
    return _dedupe(domains)


def build_event_metadata_index(event_cards: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build event-level metadata used by broad-topic coverage metrics."""

    index: dict[str, dict[str, Any]] = {}
    for card in event_cards:
        event_id = str(card.get("event_id", "")).strip()
        if not event_id:
            continue
        urls = list(card.get("core_urls", []) or []) + list(card.get("related_urls", []) or [])
        index[event_id] = {
            "event_id": event_id,
            "entities": _dedupe(str(item) for item in card.get("entities", []) or []),
            "event_type": str(card.get("event_type") or card.get("type") or "").strip(),
            "source_domains": _source_domains(str(url) for url in urls),
        }
    return index


def predicted_event_ids_at_k(
    pred_urls: list[str] | None,
    url_event_index: dict[str, set[str]],
    k: int,
) -> list[str]:
    """Return unique event ids represented by top-k predicted URLs."""

    if k <= 0:
        return []
    events: list[str] = []
    for url in (pred_urls or [])[:k]:
        normalized = normalize_url_for_retrieval(str(url))
        if not normalized:
            continue
        events.extend(sorted(url_event_index.get(normalized, set())))
    return _dedupe(events)


def event_hit_at_k(
    pred_urls: list[str] | None,
    gold_event_id: str,
    url_event_index: dict[str, set[str]],
    k: int,
) -> float | None:
    """Return 1 when any top-k predicted URL belongs to the gold event."""

    event_id = str(gold_event_id or "").strip()
    if not event_id:
        return None
    if k <= 0:
        return 0.0
    for url in (pred_urls or [])[:k]:
        normalized = normalize_url_for_retrieval(str(url))
        if normalized and event_id in url_event_index.get(normalized, set()):
            return 1.0
    return 0.0


def score_broad_topic_prediction(
    *,
    pred_urls: list[str],
    gold_event_ids: list[str],
    acceptable_event_ids: list[str] | None = None,
    url_event_index: dict[str, set[str]] | None = None,
    event_metadata_index: dict[str, dict[str, Any]] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Score a broad topic case where the target is an event set, not one URL."""

    event_index = url_event_index or {}
    metadata_index = event_metadata_index or {}
    gold = _dedupe(gold_event_ids)
    acceptable = _dedupe((acceptable_event_ids or []) + gold)
    pred_events = predicted_event_ids_at_k(pred_urls, event_index, int(k))
    hit_events = [event_id for event_id in pred_events if event_id in set(gold)]
    hit_set = set(hit_events)

    top_urls = (pred_urls or [])[: max(0, int(k))]
    irrelevant_count = 0
    acceptable_set = set(acceptable)
    for url in top_urls:
        normalized = normalize_url_for_retrieval(str(url))
        mapped = event_index.get(normalized, set()) if normalized else set()
        if not mapped or not mapped.intersection(acceptable_set):
            irrelevant_count += 1

    hit_metadata = [metadata_index.get(event_id, {}) for event_id in hit_events]
    entity_diversity = len(
        {
            str(entity).strip()
            for meta in hit_metadata
            for entity in (meta.get("entities", []) or [])
            if str(entity).strip()
        }
    )
    event_type_diversity = len(
        {
            str(meta.get("event_type", "")).strip()
            for meta in hit_metadata
            if str(meta.get("event_type", "")).strip()
        }
    )
    source_diversity = len(
        {
            str(domain).strip()
            for meta in hit_metadata
            for domain in (meta.get("source_domains", []) or [])
            if str(domain).strip()
        }
    )
    return {
        "gold_event_ids": gold,
        "acceptable_event_ids": acceptable,
        "pred_event_ids_at_k": pred_events,
        "hit_event_ids_at_k": _dedupe(hit_events),
        "event_set_recall_at_k": (len(hit_set) / len(gold)) if gold else None,
        "event_hit_count_at_k": len(hit_set),
        "event_diversity_at_k": len(hit_set),
        "entity_diversity_at_k": entity_diversity,
        "event_type_diversity_at_k": event_type_diversity,
        "source_diversity_at_k": source_diversity,
        "irrelevant_event_count_at_k": irrelevant_count,
        "irrelevant_event_ratio_at_k": (irrelevant_count / len(top_urls)) if top_urls else 0.0,
    }


def score_retrieval_prediction(
    *,
    pred_urls: list[str],
    gold_urls: list[str],
    gold_event_id: str = "",
    gold_event_ids: list[str] | None = None,
    acceptable_event_ids: list[str] | None = None,
    case_kind: str = "single_event",
    url_event_index: dict[str, set[str]] | None = None,
    event_metadata_index: dict[str, dict[str, Any]] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Score URL-level and event-level retrieval quality for one case."""

    event_index = url_event_index or {}
    normalized_kind = str(case_kind or "").strip().lower()
    if not normalized_kind:
        normalized_kind = "broad_topic" if gold_event_ids else "single_event"
    score = {
        "case_kind": normalized_kind,
        "pred_urls": pred_urls,
        "gold_urls": gold_urls,
        "k": int(k),
        "exact_recall_at_k": recall_at_k(pred_urls, gold_urls, int(k)),
        "exact_hit_at_k": hit_rate_at_k(pred_urls, gold_urls, int(k)),
        "mrr_at_k": mrr_at_k(pred_urls, gold_urls, int(k)),
        "ndcg_at_k": ndcg_at_k(pred_urls, gold_urls, int(k)),
        "event_hit_at_k": event_hit_at_k(pred_urls, gold_event_id, event_index, int(k)),
    }
    if normalized_kind == "broad_topic":
        score.update(
            score_broad_topic_prediction(
                pred_urls=pred_urls,
                gold_event_ids=gold_event_ids or [],
                acceptable_event_ids=acceptable_event_ids or [],
                url_event_index=event_index,
                event_metadata_index=event_metadata_index or {},
                k=int(k),
            )
        )
    return score


def mean_metric(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def summarize_retrieval_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    single_scores = [row for row in scores if str(row.get("case_kind") or "single_event") == "single_event"]
    broad_scores = [row for row in scores if str(row.get("case_kind") or "") == "broad_topic"]
    return {
        "case_count": len(scores),
        "single_event_case_count": len(single_scores),
        "broad_topic_case_count": len(broad_scores),
        "avg_exact_recall_at_k": mean_metric(scores, "exact_recall_at_k"),
        "avg_exact_hit_at_k": mean_metric(scores, "exact_hit_at_k"),
        "avg_mrr_at_k": mean_metric(scores, "mrr_at_k"),
        "avg_ndcg_at_k": mean_metric(scores, "ndcg_at_k"),
        "avg_event_hit_at_k": mean_metric(scores, "event_hit_at_k"),
        "avg_single_event_hit_at_k": mean_metric(single_scores, "event_hit_at_k"),
        "avg_single_mrr_at_k": mean_metric(single_scores, "mrr_at_k"),
        "avg_event_set_recall_at_k": mean_metric(broad_scores, "event_set_recall_at_k"),
        "avg_event_diversity_at_k": mean_metric(broad_scores, "event_diversity_at_k"),
        "avg_entity_diversity_at_k": mean_metric(broad_scores, "entity_diversity_at_k"),
        "avg_event_type_diversity_at_k": mean_metric(broad_scores, "event_type_diversity_at_k"),
        "avg_source_diversity_at_k": mean_metric(broad_scores, "source_diversity_at_k"),
        "avg_irrelevant_event_ratio_at_k": mean_metric(broad_scores, "irrelevant_event_ratio_at_k"),
    }
