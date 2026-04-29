"""Independent corpus sampler for task-eval dataset generation.

This module intentionally does not call the production hybrid retrieval
pipeline. It shares the underlying tables and indexes, then builds eval pools
through channel union, clustering/fallback, and constraint-aware packing.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from services.db import get_conn, put_conn

try:
    from agent.skills.embeddings import get_query_embedding
except Exception:  # pragma: no cover
    get_query_embedding = None  # type: ignore[assignment]

try:
    from agent.skills.recall_profile import resolve_recall_profile
except Exception:  # pragma: no cover
    resolve_recall_profile = None  # type: ignore[assignment]


@dataclass
class SampledPool:
    pool_id: str
    docs: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusSampleResult:
    candidates: list[dict[str, Any]]
    pools: list[SampledPool]
    meta: dict[str, Any]


def doc_id_from_url(url: str) -> str:
    digest = hashlib.sha1(str(url).strip().encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def doc_language(title: str, title_cn: str) -> str:
    if str(title_cn or "").strip():
        return "zh"
    if any("\u4e00" <= ch <= "\u9fff" for ch in str(title or "")):
        return "zh"
    return "en"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _task_seed_queries(task: dict[str, Any]) -> list[dict[str, str]]:
    params = task.get("parameter_template", {})
    if not isinstance(params, dict):
        params = {}
    sampling = task.get("sampling", {})
    if not isinstance(sampling, dict):
        sampling = {}

    seeds: list[dict[str, str]] = []

    def add(label: str, value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in {row["query"] for row in seeds}:
            seeds.append({"label": label, "query": text})

    add("query", params.get("query"))
    urls_seed = params.get("urls")
    if isinstance(urls_seed, str) and not re.search(r"https?://", urls_seed, flags=re.IGNORECASE):
        add("urls", urls_seed)
    add("topic", params.get("topic"))
    add("topic_a", params.get("topic_a"))
    add("topic_b", params.get("topic_b"))
    entities = params.get("entities")
    if isinstance(entities, list):
        for idx, entity in enumerate(entities, 1):
            add(f"entity_{idx}", entity)
    elif isinstance(entities, str):
        for idx, entity in enumerate(re.split(r"[,，;；]", entities), 1):
            add(f"entity_{idx}", entity)

    if not seeds:
        keywords = [str(item).strip() for item in sampling.get("keywords", []) if str(item).strip()]
        if keywords:
            add("keywords", " ".join(keywords))
    if not seeds:
        add("example_question", task.get("example_question"))
    return seeds


def _vector_literal(vec: list[float] | None) -> str | None:
    if not vec:
        return None
    return "[" + ",".join(str(float(item)) for item in vec) + "]"


def _parse_vector(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [float(item) for item in raw]
        except Exception:
            return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    except Exception:
        return None


def _l2_normalize(vec: list[float] | None) -> list[float] | None:
    if not vec:
        return None
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0:
        return None
    return [float(x) / norm for x in vec]


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def _coerce_doc(row: tuple, *, channel: str, query_label: str, score: float = 0.0) -> dict[str, Any]:
    title_norm, title, summary, url, source_type, created_at, sentiment, points, title_cn = row[:9]
    url_text = str(url or "").strip()
    return {
        "doc_id": doc_id_from_url(url_text),
        "url": url_text,
        "title": str(title_norm or "").strip() or str(title or "").strip() or "(untitled)",
        "summary": str(summary or "").strip(),
        "evidence_text": str(summary or "").strip(),
        "published_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
        "source": str(source_type or "").strip() or "unknown",
        "sentiment": str(sentiment or "").strip(),
        "points": int(points or 0),
        "language": doc_language(str(title or ""), str(title_cn or "")),
        "channels": [channel],
        "channel_scores": {channel: float(score or 0.0)},
        "query_labels": [query_label] if query_label else [],
    }


def _merge_doc(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    for channel in incoming.get("channels", []):
        if channel not in existing.setdefault("channels", []):
            existing["channels"].append(channel)
    for label in incoming.get("query_labels", []):
        if label not in existing.setdefault("query_labels", []):
            existing["query_labels"].append(label)
    existing.setdefault("channel_scores", {}).update(incoming.get("channel_scores", {}))
    existing["score"] = max(float(existing.get("score", 0.0) or 0.0), float(incoming.get("score", 0.0) or 0.0))


def _select_base_fields_sql(extra_score: str = "0.0") -> str:
    return f"""
        SELECT
            COALESCE(v.title_cn, v.title) AS title_norm,
            v.title,
            COALESCE(v.summary, '') AS summary,
            v.url,
            v.source_type,
            v.created_at,
            COALESCE(v.sentiment, '') AS sentiment,
            COALESCE(v.points, 0) AS points,
            COALESCE(v.title_cn, '') AS title_cn,
            ({extra_score})::float AS channel_score
        FROM view_dashboard_news v
    """


def _fetch_channel_rows(
    *,
    conn: Any,
    sql: str,
    params: tuple[Any, ...] | dict[str, Any],
    channel: str,
    query_label: str,
) -> list[dict[str, Any]]:
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"[EvalSampler][Warn] channel={channel} failed: {exc}")
        return []

    docs = []
    for row in rows:
        url = str(row[3] or "").strip()
        if not url:
            continue
        docs.append(_coerce_doc(row, channel=channel, query_label=query_label, score=float(row[9] or 0.0)))
    return docs


def _fetch_semantic(
    *,
    conn: Any,
    query_vec: list[float] | None,
    days: int,
    limit: int,
    query_label: str,
) -> list[dict[str, Any]]:
    vec_literal = _vector_literal(query_vec)
    if not vec_literal:
        return []
    sql = (
        _select_base_fields_sql("1 - (e.embedding <=> %s::vector)")
        + """
        JOIN news_embeddings e ON e.url = v.url
        WHERE v.created_at >= NOW() - %s::interval
        ORDER BY e.embedding <=> %s::vector, v.created_at DESC
        LIMIT %s
        """
    )
    return _fetch_channel_rows(
        conn=conn,
        sql=sql,
        params=(vec_literal, f"{days} days", vec_literal, limit),
        channel="semantic",
        query_label=query_label,
    )


def _fetch_lexical(*, conn: Any, query: str, days: int, limit: int, query_label: str) -> list[dict[str, Any]]:
    sql = (
        _select_base_fields_sql("ts_rank_cd(si.search_tsv, q.tsq, 32)")
        + """
        JOIN news_search_index si ON si.url = v.url
        CROSS JOIN (
            SELECT (websearch_to_tsquery('english', %s) || websearch_to_tsquery('simple', %s)) AS tsq
        ) q
        WHERE v.created_at >= NOW() - %s::interval
          AND numnode(q.tsq) > 0
          AND si.search_tsv @@ q.tsq
        ORDER BY channel_score DESC, v.created_at DESC
        LIMIT %s
        """
    )
    return _fetch_channel_rows(
        conn=conn,
        sql=sql,
        params=(query, query, f"{days} days", limit),
        channel="lexical",
        query_label=query_label,
    )


def _fetch_alias_exact(*, conn: Any, query: str, days: int, limit: int, query_label: str) -> list[dict[str, Any]]:
    sql = (
        _select_base_fields_sql(
            """
            CASE
                WHEN v.title ILIKE ('%%' || t.term || '%%')
                  OR COALESCE(v.title_cn, '') ILIKE ('%%' || t.term || '%%') THEN 2.0 * t.term_weight
                WHEN similarity(LOWER(COALESCE(v.title, '')), LOWER(t.term)) >= 0.42
                  OR similarity(LOWER(COALESCE(v.title_cn, '')), LOWER(t.term)) >= 0.42 THEN 1.4 * t.term_weight
                WHEN COALESCE(v.summary, '') ILIKE ('%%' || t.term || '%%') THEN 0.8 * t.term_weight
                ELSE 0.0
            END
            """
        )
        + """
        JOIN (
            SELECT %s::text AS term, 1.0::float AS term_weight
            UNION
            SELECT ea.alias AS term, COALESCE(ea.weight, 1.0)::float AS term_weight
            FROM entity_alias ea
            JOIN entity_registry er ON er.entity_id = ea.entity_id
            WHERE ea.is_active = TRUE
              AND er.is_active = TRUE
              AND (
                    LOWER(ea.alias) = LOWER(%s)
                 OR LOWER(er.canonical_name) = LOWER(%s)
                 OR LOWER(%s) LIKE ('%%' || LOWER(ea.alias) || '%%')
                 OR LOWER(ea.alias) LIKE ('%%' || LOWER(%s) || '%%')
              )
            LIMIT 16
        ) t ON TRUE
        WHERE v.created_at >= NOW() - %s::interval
          AND (
                v.title ILIKE ('%%' || t.term || '%%')
             OR COALESCE(v.title_cn, '') ILIKE ('%%' || t.term || '%%')
             OR COALESCE(v.summary, '') ILIKE ('%%' || t.term || '%%')
             OR similarity(LOWER(COALESCE(v.title, '')), LOWER(t.term)) >= 0.42
             OR similarity(LOWER(COALESCE(v.title_cn, '')), LOWER(t.term)) >= 0.42
          )
        ORDER BY channel_score DESC, v.created_at DESC
        LIMIT %s
        """
    )
    return _fetch_channel_rows(
        conn=conn,
        sql=sql,
        params=(query, query, query, query, query, f"{days} days", limit),
        channel="alias",
        query_label=query_label,
    )


def _fetch_stratified(
    *,
    conn: Any,
    days: int,
    limit: int,
    sources: list[str],
    languages: set[str],
    rng_seed: int,
) -> list[dict[str, Any]]:
    where_parts = ["v.created_at >= NOW() - %s::interval"]
    params: list[Any] = [f"{days} days"]
    if sources:
        where_parts.append("COALESCE(v.source_type, '') = ANY(%s)")
        params.append(sources)
    sql = (
        _select_base_fields_sql("0.0")
        + f"""
        WHERE {' AND '.join(where_parts)}
        ORDER BY md5(v.url || %s), v.created_at DESC
        LIMIT %s
        """
    )
    params.extend([str(rng_seed), limit])
    docs = _fetch_channel_rows(
        conn=conn,
        sql=sql,
        params=tuple(params),
        channel="stratified",
        query_label="stratified",
    )
    if languages:
        docs = [doc for doc in docs if str(doc.get("language", "")).lower() in languages]
    return docs


def _load_embeddings(conn: Any, docs: list[dict[str, Any]]) -> None:
    urls = [str(doc.get("url", "")).strip() for doc in docs if str(doc.get("url", "")).strip()]
    if not urls:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT url, embedding
            FROM news_embeddings
            WHERE url = ANY(%s)
            """,
            (urls,),
        )
        rows = cur.fetchall()
        cur.close()
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"[EvalSampler][Warn] embedding load failed: {exc}")
        return
    by_url = {str(url): _l2_normalize(_parse_vector(embedding)) for url, embedding in rows}
    for doc in docs:
        emb = by_url.get(str(doc.get("url", "")))
        if emb:
            doc["embedding"] = emb
            doc["embedding_available"] = True
        else:
            doc["embedding_available"] = False


def sample_candidates(task: dict[str, Any], *, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampling = task.get("sampling", {})
    days = _safe_int(sampling.get("days", 30), 30)
    candidate_limit = _safe_int(sampling.get("candidate_limit", 300), 300)
    sources = [str(item).strip() for item in sampling.get("sources", []) if str(item).strip()]
    languages = {str(item).strip().lower() for item in sampling.get("languages", []) if str(item).strip()}
    queries = _task_seed_queries(task)
    channel_limit = max(24, min(300, candidate_limit))

    query_vectors: dict[str, list[float] | None] = {}
    if get_query_embedding is not None:
        for row in queries:
            query_vectors[row["label"]] = _l2_normalize(get_query_embedding(row["query"]))

    conn = get_conn()
    try:
        merged: dict[str, dict[str, Any]] = {}
        channel_counts: dict[str, int] = {}
        for row in queries:
            label = row["label"]
            query = row["query"]
            channel_docs = []
            channel_docs.extend(
                _fetch_semantic(
                    conn=conn,
                    query_vec=query_vectors.get(label),
                    days=days,
                    limit=channel_limit,
                    query_label=label,
                )
            )
            channel_docs.extend(_fetch_lexical(conn=conn, query=query, days=days, limit=channel_limit, query_label=label))
            channel_docs.extend(
                _fetch_alias_exact(conn=conn, query=query, days=days, limit=channel_limit, query_label=label)
            )
            for doc in channel_docs:
                if languages and str(doc.get("language", "")).lower() not in languages:
                    continue
                url = str(doc.get("url", "")).strip()
                if not url:
                    continue
                for channel in doc.get("channels", []):
                    channel_counts[channel] = channel_counts.get(channel, 0) + 1
                if url in merged:
                    _merge_doc(merged[url], doc)
                else:
                    merged[url] = doc

        stratified = _fetch_stratified(
            conn=conn,
            days=days,
            limit=max(24, min(candidate_limit, 120)),
            sources=sources,
            languages=languages,
            rng_seed=rng.randint(1, 1_000_000),
        )
        for doc in stratified:
            url = str(doc.get("url", "")).strip()
            if not url:
                continue
            channel_counts["stratified"] = channel_counts.get("stratified", 0) + 1
            if url in merged:
                _merge_doc(merged[url], doc)
            else:
                merged[url] = doc

        docs = list(merged.values())
        _load_embeddings(conn, docs)
    finally:
        put_conn(conn)

    seed_vectors = [vec for vec in query_vectors.values() if vec]
    for doc in docs:
        emb = doc.get("embedding")
        doc["seed_similarity"] = max((_cosine(emb, vec) for vec in seed_vectors), default=0.0)
        if "topic_a" in doc.get("query_labels", []):
            doc["topic_group"] = "A"
        elif "topic_b" in doc.get("query_labels", []):
            doc["topic_group"] = "B"

    docs.sort(
        key=lambda doc: (
            float(doc.get("seed_similarity", 0.0) or 0.0),
            len(doc.get("channels", [])),
            str(doc.get("published_at", "")),
        ),
        reverse=True,
    )
    if len(docs) > candidate_limit:
        docs = docs[:candidate_limit]

    return docs, {
        "candidate_source": "eval_corpus_sampler",
        "seed_queries": queries,
        "candidate_channel_counts": channel_counts,
        "candidate_docs": len(docs),
        "embedding_docs": sum(1 for doc in docs if doc.get("embedding_available")),
    }


def _recall_sim_floor() -> float:
    if resolve_recall_profile is None:
        return 0.20
    try:
        return float(resolve_recall_profile().sim_floor)
    except Exception:
        return 0.20


def _cluster_with_hdbscan(docs: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    emb_docs = [doc for doc in docs if doc.get("embedding")]
    if len(emb_docs) < 6:
        return {}, {"cluster_mode": "insufficient_embeddings", "fallback_reason": "embedding_docs_lt_6"}
    try:
        import hdbscan  # type: ignore[import-not-found]
        import numpy as np
    except Exception as exc:
        return {}, {"cluster_mode": "hdbscan_unavailable", "fallback_reason": type(exc).__name__}

    matrix = np.array([doc["embedding"] for doc in emb_docs], dtype=float)
    try:
        labels = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            cluster_selection_method="eom",
            metric="euclidean",
        ).fit_predict(matrix)
    except Exception as exc:
        return {}, {"cluster_mode": "hdbscan_failed", "fallback_reason": type(exc).__name__}

    clusters: dict[int, list[dict[str, Any]]] = {}
    for doc, label in zip(emb_docs, labels):
        label_int = int(label)
        doc["cluster_id"] = label_int
        if label_int < 0:
            continue
        clusters.setdefault(label_int, []).append(doc)
    clusters = {label: rows for label, rows in clusters.items() if len(rows) >= 3}
    return clusters, {
        "cluster_mode": "hdbscan",
        "cluster_count": len(clusters),
        "noise_count": sum(1 for label in labels if int(label) < 0),
    }


def _centroid(docs: list[dict[str, Any]]) -> list[float] | None:
    vectors = [doc.get("embedding") for doc in docs if doc.get("embedding")]
    if not vectors:
        return None
    size = len(vectors[0])
    values = [0.0] * size
    for vec in vectors:
        if len(vec) != size:
            continue
        for idx, item in enumerate(vec):
            values[idx] += float(item)
    return _l2_normalize([value / len(vectors) for value in values])


def _greedy_positive_groups(docs: list[dict[str, Any]], *, pool_size: int, sim_floor: float) -> list[list[dict[str, Any]]]:
    unused = [doc for doc in docs if doc.get("embedding")]
    unused.sort(key=lambda doc: float(doc.get("seed_similarity", 0.0) or 0.0), reverse=True)
    groups: list[list[dict[str, Any]]] = []
    while unused:
        anchor = unused.pop(0)
        group = [anchor]
        centroid = anchor.get("embedding")
        for threshold in (0.72, 0.66):
            changed = True
            while changed and len(group) < pool_size:
                changed = False
                best_idx = -1
                best_score = -1.0
                for idx, doc in enumerate(unused):
                    if float(doc.get("seed_similarity", 0.0) or 0.0) < sim_floor:
                        continue
                    score = _cosine(doc.get("embedding"), centroid)
                    if score >= threshold and score > best_score:
                        best_idx = idx
                        best_score = score
                if best_idx >= 0:
                    group.append(unused.pop(best_idx))
                    centroid = _centroid(group) or centroid
                    changed = True
            if len(group) >= min(3, pool_size):
                break
        groups.append(group)
    return [group for group in groups if len(group) >= min(3, pool_size)]


def _round_robin_by_source(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        groups.setdefault(str(doc.get("source", "unknown")) or "unknown", []).append(doc)
    for rows in groups.values():
        rows.sort(key=lambda doc: (float(doc.get("seed_similarity", 0.0) or 0.0), doc.get("published_at", "")), reverse=True)
    ordered: list[dict[str, Any]] = []
    while True:
        progressed = False
        for source in sorted(groups):
            if groups[source]:
                ordered.append(groups[source].pop(0))
                progressed = True
        if not progressed:
            return ordered


def _time_key(doc: dict[str, Any]) -> str:
    return str(doc.get("published_at", "") or "")


def _pick_time_strata(docs: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    rows = sorted(docs, key=_time_key)
    if not rows or count <= 0:
        return []
    picks: list[dict[str, Any]] = []
    for q in (0.1, 0.5, 0.9):
        idx = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * q))))
        doc = rows[idx]
        if doc not in picks:
            picks.append(doc)
        if len(picks) >= count:
            break
    return picks


def _append_unique(out: list[dict[str, Any]], docs: list[dict[str, Any]], *, limit: int) -> None:
    seen = {str(doc.get("url", "")) for doc in out}
    for doc in docs:
        url = str(doc.get("url", ""))
        if not url or url in seen:
            continue
        out.append(doc)
        seen.add(url)
        if len(out) >= limit:
            return


def _pack_compare_sources(task: dict[str, Any], cluster_docs: list[dict[str, Any]], pool_size: int) -> tuple[list[dict[str, Any]], bool]:
    sources = [str(item).strip() for item in task.get("sampling", {}).get("sources", []) if str(item).strip()]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for doc in cluster_docs:
        by_source.setdefault(str(doc.get("source", "unknown")) or "unknown", []).append(doc)
    for rows in by_source.values():
        rows.sort(key=lambda doc: float(doc.get("seed_similarity", 0.0) or 0.0), reverse=True)
    target_sources = sources or sorted(by_source)
    selected: list[dict[str, Any]] = []
    for source in target_sources:
        _append_unique(selected, by_source.get(source, [])[:2], limit=pool_size)
    if sources and any(not any(str(doc.get("source")) == source for doc in selected) for source in sources):
        return selected, False
    rest = sorted(cluster_docs, key=lambda doc: float(doc.get("seed_similarity", 0.0) or 0.0), reverse=True)
    _append_unique(selected, rest, limit=pool_size)
    return selected, len({doc.get("source") for doc in selected}) >= min(2, len(by_source))


def _pack_compare_topics(cluster_docs: list[dict[str, Any]], pool_size: int) -> tuple[list[dict[str, Any]], bool]:
    a_docs = [doc for doc in cluster_docs if doc.get("topic_group") == "A"]
    b_docs = [doc for doc in cluster_docs if doc.get("topic_group") == "B"]
    if not a_docs or not b_docs:
        return [], False
    a_docs.sort(key=lambda doc: float(doc.get("seed_similarity", 0.0) or 0.0), reverse=True)
    b_docs.sort(key=lambda doc: float(doc.get("seed_similarity", 0.0) or 0.0), reverse=True)
    half = max(1, pool_size // 2)
    selected = a_docs[:half] + b_docs[: pool_size - half]
    return selected, abs(len([doc for doc in selected if doc.get("topic_group") == "A"]) - len([doc for doc in selected if doc.get("topic_group") == "B"])) <= 1


def _pack_timeline(cluster_docs: list[dict[str, Any]], pool_size: int) -> tuple[list[dict[str, Any]], bool]:
    selected = _pick_time_strata(cluster_docs, min(3, pool_size))
    rest = sorted(cluster_docs, key=lambda doc: float(doc.get("seed_similarity", 0.0) or 0.0), reverse=True)
    _append_unique(selected, rest, limit=pool_size)
    return selected, len(selected) >= min(3, pool_size)


def _pack_default(cluster_docs: list[dict[str, Any]], pool_size: int) -> tuple[list[dict[str, Any]], bool]:
    selected: list[dict[str, Any]] = []
    _append_unique(selected, _round_robin_by_source(cluster_docs), limit=min(pool_size, max(2, pool_size // 2)))
    rest = sorted(cluster_docs, key=lambda doc: float(doc.get("seed_similarity", 0.0) or 0.0), reverse=True)
    _append_unique(selected, rest, limit=pool_size)
    source_count = len({doc.get("source") for doc in selected})
    return selected, bool(selected) and source_count >= min(2, len({doc.get("source") for doc in cluster_docs}))


def pack_cluster(task: dict[str, Any], cluster_docs: list[dict[str, Any]], *, pool_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skill = str(task.get("skill", "")).strip()
    scenario = str(task.get("scenario", "")).strip().lower()
    pool_size = max(1, min(12, max(8, int(pool_size))))
    strategy = "default_source_diverse"
    if scenario == "empty":
        selected, ok = _pack_default(cluster_docs, pool_size)
        strategy = "empty_negative_source_time"
    elif skill == "compare_sources":
        selected, ok = _pack_compare_sources(task, cluster_docs, pool_size)
        strategy = "source_stratified"
    elif skill == "compare_topics":
        selected, ok = _pack_compare_topics(cluster_docs, pool_size)
        strategy = "topic_balanced"
    elif skill in {"build_timeline", "trend_analysis"}:
        selected, ok = _pack_timeline(cluster_docs, pool_size)
        strategy = "time_stratified"
    else:
        selected, ok = _pack_default(cluster_docs, pool_size)

    for doc in selected:
        text = str(doc.get("evidence_text") or doc.get("summary") or "")
        if len(text) > 900:
            doc["evidence_text"] = text[:900]
    return selected, {
        "packing_strategy": strategy,
        "packing_constraints_passed": bool(ok),
        "selected_docs": len(selected),
        "cluster_docs": len(cluster_docs),
        "token_budget": 9000,
    }


def _cluster_score(docs: list[dict[str, Any]]) -> float:
    if not docs:
        return 0.0
    seed_similarity = sum(float(doc.get("seed_similarity", 0.0) or 0.0) for doc in docs) / len(docs)
    channels = {channel for doc in docs for channel in doc.get("channels", [])}
    sources = {str(doc.get("source", "")) for doc in docs if str(doc.get("source", ""))}
    channel_coverage = min(1.0, len(channels) / 4.0)
    source_diversity = min(1.0, len(sources) / 4.0)
    time_spread = 1.0 if len({_time_key(doc)[:10] for doc in docs if _time_key(doc)}) >= 3 else 0.4
    size_score = min(1.0, len(docs) / 12.0)
    return (
        0.40 * seed_similarity
        + 0.20 * channel_coverage
        + 0.15 * source_diversity
        + 0.15 * time_spread
        + 0.10 * size_score
    )


def build_pools(task: dict[str, Any], candidates: list[dict[str, Any]], *, pools_per_task: int, rng: random.Random) -> tuple[list[SampledPool], dict[str, Any]]:
    sampling = task.get("sampling", {})
    pool_size = _safe_int(sampling.get("pool_size", 12), 12)
    scenario = str(task.get("scenario", "")).strip().lower()
    sim_floor = _recall_sim_floor()
    meta: dict[str, Any] = {
        "cluster_fallback": False,
        "cluster_fallback_reason": "",
    }

    if scenario == "empty":
        ceiling = min(sim_floor, 0.20)
        low = [doc for doc in candidates if float(doc.get("seed_similarity", 0.0) or 0.0) <= ceiling]
        if not low:
            low = candidates[:]
        low.sort(key=lambda doc: (float(doc.get("seed_similarity", 0.0) or 0.0), doc.get("published_at", "")))
        groups = [low[idx : idx + pool_size] for idx in range(0, len(low), pool_size)]
        meta.update({"cluster_mode": "empty_negative", "empty_sim_ceiling": ceiling, "cluster_count": len(groups)})
    else:
        clusters, cluster_meta = _cluster_with_hdbscan(candidates)
        meta.update(cluster_meta)
        if clusters:
            groups = sorted(clusters.values(), key=_cluster_score, reverse=True)
        else:
            groups = _greedy_positive_groups(candidates, pool_size=pool_size, sim_floor=sim_floor)
            meta.update(
                {
                    "cluster_mode": "greedy_embedding" if groups else "round_robin",
                    "cluster_count": len(groups),
                    "cluster_fallback": True,
                    "cluster_fallback_reason": cluster_meta.get("fallback_reason", "hdbscan_no_valid_clusters"),
                }
            )
            if not groups:
                ordered = _round_robin_by_source(candidates)
                groups = [ordered[idx : idx + pool_size] for idx in range(0, len(ordered), pool_size)]

    pools: list[SampledPool] = []
    group_idx = 0
    attempts = 0
    while len(pools) < pools_per_task and groups and attempts < max(pools_per_task * 4, len(groups) * 2):
        cluster_docs = groups[group_idx % len(groups)]
        group_idx += 1
        attempts += 1
        if not cluster_docs:
            continue
        selected, packing_meta = pack_cluster(task, cluster_docs, pool_size=pool_size)
        if scenario != "empty" and not packing_meta.get("packing_constraints_passed"):
            continue
        if not selected:
            continue
        suffix = f"{len(pools) + 1:03d}"
        pool_id = f"{task['task_id']}.pool.{suffix}"
        for doc in selected:
            doc["pool_seed"] = pool_id
        pools.append(
            SampledPool(
                pool_id=pool_id,
                docs=selected,
                meta={
                    **packing_meta,
                    "cluster_score": round(_cluster_score(cluster_docs), 4),
                    "cluster_id": cluster_docs[0].get("cluster_id", f"group_{group_idx}"),
                },
            )
        )
        print(
            "[Packing] task=%s pool=%s strategy=%s selected=%s cluster_docs=%s constraints=%s"
            % (
                task["task_id"],
                pool_id,
                packing_meta.get("packing_strategy"),
                len(selected),
                len(cluster_docs),
                "pass" if packing_meta.get("packing_constraints_passed") else "fail",
            )
        )

    return pools, meta


def build_eval_sample(task: dict[str, Any], *, pools_per_task: int, rng: random.Random) -> CorpusSampleResult:
    candidates, sample_meta = sample_candidates(task, rng=rng)
    pools, cluster_meta = build_pools(task, candidates, pools_per_task=pools_per_task, rng=rng)
    return CorpusSampleResult(
        candidates=candidates,
        pools=pools,
        meta={
            **sample_meta,
            **cluster_meta,
            "pool_count": len(pools),
        },
    )
