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


def score_retrieval_prediction(
    *,
    pred_urls: list[str],
    gold_urls: list[str],
    gold_event_id: str = "",
    url_event_index: dict[str, set[str]] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Score URL-level and event-level retrieval quality for one case."""

    event_index = url_event_index or {}
    return {
        "pred_urls": pred_urls,
        "gold_urls": gold_urls,
        "k": int(k),
        "exact_recall_at_k": recall_at_k(pred_urls, gold_urls, int(k)),
        "exact_hit_at_k": hit_rate_at_k(pred_urls, gold_urls, int(k)),
        "mrr_at_k": mrr_at_k(pred_urls, gold_urls, int(k)),
        "ndcg_at_k": ndcg_at_k(pred_urls, gold_urls, int(k)),
        "event_hit_at_k": event_hit_at_k(pred_urls, gold_event_id, event_index, int(k)),
    }


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
    return {
        "case_count": len(scores),
        "avg_exact_recall_at_k": mean_metric(scores, "exact_recall_at_k"),
        "avg_exact_hit_at_k": mean_metric(scores, "exact_hit_at_k"),
        "avg_mrr_at_k": mean_metric(scores, "mrr_at_k"),
        "avg_ndcg_at_k": mean_metric(scores, "ndcg_at_k"),
        "avg_event_hit_at_k": mean_metric(scores, "event_hit_at_k"),
    }

