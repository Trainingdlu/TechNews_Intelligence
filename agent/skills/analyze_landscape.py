"""Analyze-landscape skill implementation and structured adapter.

Refactored to use semantic vector pools for entity matching and
embedding-anchor classification for signal categorisation, replacing
the legacy hardcoded dictionaries and ILIKE keyword scanning.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_text_output, _is_recent_timestamp
from .retrieval import _get_query_embedding
from .schemas import AnalyzeLandscapeSkillInput
from .semantic_pool import fetch_semantic_url_pool

# ---------------------------------------------------------------------------
# Default entity list (kept as sensible defaults; users may override)
# ---------------------------------------------------------------------------

DEFAULT_LANDSCAPE_ENTITIES = [
    "OpenAI",
    "Anthropic",
    "Google",
    "Microsoft",
    "Meta",
    "Amazon",
    "Apple",
    "NVIDIA",
    "Tesla",
    "TSMC",
    "Intel",
    "AMD",
    "CrowdStrike",
    "Palo Alto Networks",
    "Cloudflare",
    "Cisco",
]

# ---------------------------------------------------------------------------
# Signal anchor texts for embedding-based classification (Method B)
# ---------------------------------------------------------------------------

SIGNAL_ANCHOR_TEXTS: dict[str, str] = {
    "compute_cost": (
        "GPU TPU chip semiconductor datacenter data center server compute "
        "training cost inference cost capex power consumption energy "
        "算力 芯片 电力 能耗 数据中心 服务器 计算成本"
    ),
    "algorithm_efficiency": (
        "model benchmark reasoning architecture transformer agent inference "
        "distillation fine-tuning parameter efficiency quantization "
        "算法 模型 推理 架构 蒸馏 微调 量化 参数效率"
    ),
    "data_moat": (
        "dataset data corpus licensing proprietary copyright privacy "
        "synthetic data curation annotation training data "
        "数据 语料 授权 版权 隐私 合成数据 标注"
    ),
    "go_to_market": (
        "enterprise customer pricing revenue subscription partnership "
        "adoption sales ARR growth market share commercialize "
        "商业化 客户 定价 收入 订阅 合作 落地 营收"
    ),
    "policy_security": (
        "regulation compliance antitrust lawsuit security breach "
        "vulnerability policy military export control sanctions "
        "监管 合规 诉讼 安全 漏洞 军方 出口管制 制裁"
    ),
}

LANDSCAPE_SIGNAL_LABELS: dict[str, str] = {
    "compute_cost": "Compute/Cost",
    "algorithm_efficiency": "Algorithm/Efficiency",
    "data_moat": "Data/Moat",
    "go_to_market": "Go-to-Market",
    "policy_security": "Policy/Security",
}

# Module-level lazy cache for signal anchor embeddings
_signal_anchor_cache: dict[str, list[float]] | None = None


def _get_signal_anchors() -> dict[str, list[float]]:
    """Compute and cache embeddings for each signal-category anchor text."""
    global _signal_anchor_cache
    if _signal_anchor_cache is not None:
        return _signal_anchor_cache

    anchors: dict[str, list[float]] = {}
    for key, text in SIGNAL_ANCHOR_TEXTS.items():
        vec = _get_query_embedding(text)
        if vec:
            anchors[key] = vec
    _signal_anchor_cache = anchors
    return anchors


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Simple cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _classify_signal_by_embedding(
    article_embeddings: dict[str, list[float]],
    anchors: dict[str, list[float]],
) -> dict[str, int]:
    """Classify articles into signal categories using embedding anchor similarity.

    For each article embedding, compute cosine similarity against each anchor
    and assign the article to the category with the highest similarity.

    Parameters
    ----------
    article_embeddings:
        Mapping from URL to embedding vector.
    anchors:
        Mapping from signal category key to anchor embedding.

    Returns
    -------
    dict[str, int]
        Count per signal category.
    """
    counts = {k: 0 for k in SIGNAL_ANCHOR_TEXTS.keys()}
    if not anchors:
        return counts

    anchor_keys = list(anchors.keys())
    anchor_vecs = [anchors[k] for k in anchor_keys]

    for _url, emb in article_embeddings.items():
        best_key = anchor_keys[0]
        best_sim = -1.0
        for key, anchor_vec in zip(anchor_keys, anchor_vecs):
            sim = _cosine_similarity(emb, anchor_vec)
            if sim > best_sim:
                best_sim = sim
                best_key = key
        counts[best_key] += 1

    return counts


def _normalize_landscape_entities(entities: str | list[str] | None) -> list[str]:
    """Parse user-supplied entity list, falling back to defaults.

    Unlike the old implementation, no alias translation is performed;
    the embedding model handles cross-language matching natively.
    """
    raw_items: list[str] = []
    if isinstance(entities, str):
        raw_items = [x.strip() for x in re.split(r"[,\n;/|，、]+", entities) if x.strip()]
    elif isinstance(entities, list):
        raw_items = [str(x).strip() for x in entities if str(x).strip()]

    if not raw_items:
        return list(DEFAULT_LANDSCAPE_ENTITIES)

    # Deduplicate while preserving order
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)

    return normalized[:12] if normalized else list(DEFAULT_LANDSCAPE_ENTITIES)


def _fetch_entity_url_pools(
    entity_list: list[str],
    *,
    topic: str,
    days: int,
    limit_per_entity: int = 50,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Build per-entity URL pools via semantic recall.

    Returns
    -------
    entity_url_map:
        Mapping from entity name to list of URLs.
    url_to_entity:
        Mapping from URL to entity with highest similarity (arbitration).
    """
    # Collect (url, sim, entity) tuples
    url_best_sim: dict[str, float] = {}
    url_best_entity: dict[str, str] = {}
    entity_raw_pools: dict[str, list[tuple[str, float]]] = {}

    for entity in entity_list:
        query = f"{entity} {topic}".strip() if topic else entity
        pool = fetch_semantic_url_pool(query, days=days, limit=limit_per_entity)
        entity_raw_pools[entity] = pool

        for url, sim in pool:
            if url not in url_best_sim or sim > url_best_sim[url]:
                url_best_sim[url] = sim
                url_best_entity[url] = entity

    # Build per-entity URL lists from arbitrated assignments
    entity_url_map: dict[str, list[str]] = {name: [] for name in entity_list}
    for url, entity in url_best_entity.items():
        entity_url_map[entity].append(url)

    return entity_url_map, url_best_entity


def analyze_landscape(topic: str = "", days: int = 30, entities: str = "", limit_per_entity: int = 3) -> str:
    """Analyze competitive landscape with entity-level metrics and evidence URLs."""
    topic = (topic or "").strip()
    topic_label = topic or "all"
    print(
        f"\n[Tool] analyze_landscape: topic={topic_label}, days={days}, "
        f"entities={entities or 'default'}, limit_per_entity={limit_per_entity}"
    )
    days = _clamp_int(days, 7, 180)
    limit_per_entity = _clamp_int(limit_per_entity, 1, 5)
    entity_list = _normalize_landscape_entities(entities)

    # --- Semantic entity matching ---
    entity_url_map, url_to_entity = _fetch_entity_url_pools(
        entity_list, topic=topic, days=days,
    )

    # Flatten all matched URLs
    all_matched_urls: list[str] = []
    seen_urls: set[str] = set()
    for urls in entity_url_map.values():
        for url in urls:
            if url not in seen_urls:
                seen_urls.add(url)
                all_matched_urls.append(url)

    if not all_matched_urls:
        return f"No landscape data in the last {days} days for entities: {', '.join(entity_list)}."

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT NOW()")
        db_now = cur.fetchone()[0]
        if db_now is not None and hasattr(db_now, "tzinfo") and db_now.tzinfo is not None:
            db_now = db_now.astimezone(timezone.utc).replace(tzinfo=None)
        if db_now is None or not isinstance(db_now, datetime):
            db_now = datetime.now(timezone.utc).replace(tzinfo=None)

        # --- Fetch all matched article metadata in one query ---
        cur.execute(
            """
            SELECT
                COALESCE(v.title_cn, v.title) AS headline,
                v.url,
                COALESCE(v.summary, '') AS summary,
                v.source_type,
                v.points,
                v.sentiment,
                v.created_at
            FROM view_dashboard_news v
            WHERE v.url = ANY(%s)
              AND v.created_at >= NOW() - %s::interval
            """,
            (all_matched_urls, f"{days} days"),
        )
        article_rows = cur.fetchall()

        # Build article lookup by URL
        article_by_url: dict[str, tuple] = {}
        for row in article_rows:
            headline, url, summary, source_type, points, sentiment, created_at = row
            article_by_url[url] = row

        total_cnt = len(article_by_url)
        active_entities = sum(1 for name in entity_list if any(u in article_by_url for u in entity_url_map.get(name, [])))

        # --- Compute per-entity stats ---
        split_days = max(3, days // 2)
        prev_days = max(1, days - split_days)
        cutoff_ts = db_now - timedelta(days=split_days)

        stat_map: dict[str, dict[str, Any]] = {}
        source_counts: dict[str, int] = {}
        entity_source_counts: dict[str, dict[str, int]] = {}
        momentum_map: dict[str, dict[str, int]] = {name: {"recent": 0, "previous": 0} for name in entity_list}

        for name in entity_list:
            entity_urls = entity_url_map.get(name, [])
            cnt = 0
            points_sum = 0.0
            pos_cnt = neu_cnt = neg_cnt = 0
            per_entity_source: dict[str, int] = {}

            for url in entity_urls:
                art = article_by_url.get(url)
                if art is None:
                    continue
                headline, _, summary, source_type, points, sentiment, created_at = art
                cnt += 1
                points_sum += int(points or 0)

                if sentiment == "Positive":
                    pos_cnt += 1
                elif sentiment == "Neutral":
                    neu_cnt += 1
                elif sentiment == "Negative":
                    neg_cnt += 1

                source_counts[source_type] = source_counts.get(source_type, 0) + 1
                per_entity_source[source_type] = per_entity_source.get(source_type, 0) + 1

                if _is_recent_timestamp(created_at, cutoff_ts):
                    momentum_map[name]["recent"] += 1
                else:
                    momentum_map[name]["previous"] += 1

            avg_points = round(points_sum / cnt, 1) if cnt > 0 else 0.0
            stat_map[name] = {
                "cnt": cnt,
                "avg_points": avg_points,
                "pos_cnt": pos_cnt,
                "neu_cnt": neu_cnt,
                "neg_cnt": neg_cnt,
            }
            entity_source_counts[name] = per_entity_source

        # --- Signal classification via embedding anchors ---
        anchors = _get_signal_anchors()
        signal_counts: dict[str, int] = {k: 0 for k in SIGNAL_ANCHOR_TEXTS.keys()}

        if anchors:
            # Fetch embeddings for matched articles
            cur.execute(
                """
                SELECT e.url, e.embedding
                FROM news_embeddings e
                WHERE e.url = ANY(%s)
                """,
                (all_matched_urls,),
            )
            emb_rows = cur.fetchall()
            if emb_rows:
                article_embeddings: dict[str, list[float]] = {}
                for emb_url, emb_vec in emb_rows:
                    # pgvector returns a string or list; normalise
                    if isinstance(emb_vec, str):
                        emb_vec = [float(x) for x in emb_vec.strip("[]").split(",")]
                    elif hasattr(emb_vec, "tolist"):
                        emb_vec = emb_vec.tolist()
                    article_embeddings[str(emb_url)] = emb_vec

                signal_counts = _classify_signal_by_embedding(article_embeddings, anchors)

        # --- Topic scope count ---
        topic_scope_sql = """
            SELECT COUNT(*) AS topic_articles
            FROM view_dashboard_news v
            WHERE v.created_at >= NOW() - %s::interval
        """
        topic_scope_params: list[Any] = [f"{days} days"]
        if topic:
            # Use semantic pool to estimate topic scope
            topic_pool = fetch_semantic_url_pool(topic, days=days, limit=500)
            topic_articles = len(topic_pool) if topic_pool else 0
        else:
            cur.execute(topic_scope_sql, tuple(topic_scope_params))
            topic_articles = int(cur.fetchone()[0] or 0)

        # --- Top evidence URLs per entity ---
        # Build from in-memory data: sort each entity's articles by points
        top_rows: list[tuple] = []
        for name in entity_list:
            entity_urls = entity_url_map.get(name, [])
            entity_articles = []
            for url in entity_urls:
                art = article_by_url.get(url)
                if art is not None:
                    headline, _, summary, source_type, points, sentiment, created_at = art
                    entity_articles.append((name, source_type, headline, url, points, created_at))
            # Sort by points desc, then created_at desc
            entity_articles.sort(key=lambda x: (-(x[4] or 0), x[5] if x[5] else datetime.min), reverse=False)
            entity_articles.sort(key=lambda x: -(x[4] or 0))
            for rank, art_tuple in enumerate(entity_articles[:limit_per_entity], 1):
                top_rows.append(art_tuple + (rank,))

        cur.close()

        # --- Format output ---
        lines = [
            f"Landscape snapshot: topic={topic_label} (last {days} days)",
            f"Entities requested: {', '.join(entity_list)}",
            (
                f"Coverage: topic_articles={topic_articles}, matched_entity_articles={total_cnt}, "
                f"active_entities={active_entities}/{len(entity_list)}"
            ),
            f"Time split: recent={split_days}d, previous={prev_days}d",
            "Source mix:",
        ]

        if source_counts:
            for source_type, source_cnt in sorted(source_counts.items(), key=lambda x: (-x[1], x[0])):
                source_share = (float(source_cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
                lines.append(f"  {source_type}: count={source_cnt}, share={source_share:.1f}%")
        else:
            lines.append("  no source records")

        lines.extend(["Entity stats:"])

        for name in entity_list:
            recent = momentum_map.get(name, {}).get("recent", 0)
            previous = momentum_map.get(name, {}).get("previous", 0)
            if previous == 0:
                delta_text = "+new" if recent > 0 else "0.0%"
            else:
                delta_text = f"{((float(recent) - float(previous)) / float(previous)) * 100:+.1f}%"

            stats = stat_map.get(name, {})
            cnt = stats.get("cnt", 0)
            if cnt == 0:
                lines.append(
                    f"  {name}: count=0, share=0.0%, avg_points=0, "
                    f"sentiment(P/N/Ng)=0/0/0, momentum_recent_vs_prev={recent}/{previous} ({delta_text})"
                )
                continue

            share = (float(cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
            avg_points_value = float(stats.get("avg_points", 0.0))
            pos_cnt = stats.get("pos_cnt", 0)
            neu_cnt = stats.get("neu_cnt", 0)
            neg_cnt = stats.get("neg_cnt", 0)

            top_source_note = ""
            per_entity_source = entity_source_counts.get(name, {})
            if per_entity_source:
                top_source, top_source_cnt = max(per_entity_source.items(), key=lambda x: (x[1], x[0]))
                top_source_share = (float(top_source_cnt) / float(cnt)) * 100 if cnt else 0.0
                top_source_note = f", top_source={top_source}({top_source_share:.1f}%)"
            lines.append(
                f"  {name}: count={cnt}, share={share:.1f}%, avg_points={avg_points_value:.1f}, "
                f"sentiment(P/N/Ng)={pos_cnt}/{neu_cnt}/{neg_cnt}, "
                f"momentum_recent_vs_prev={recent}/{previous} ({delta_text}){top_source_note}"
            )

        lines.append("Variable signals (embedding anchor classification):")
        for key, label in LANDSCAPE_SIGNAL_LABELS.items():
            signal_cnt = int(signal_counts.get(key, 0))
            signal_share = (float(signal_cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
            lines.append(f"  {label}: count={signal_cnt}, share={signal_share:.1f}%")
        lines.append("Signal note: classified by embedding similarity to category anchors.")

        lines.append("Evidence URLs:")
        for entity, source_type, headline, url, points, created_at, rank in top_rows:
            lines.append(
                f"  [{entity}] #{rank} [{source_type}] {headline} | points={points} | "
                f"{created_at.strftime('%Y-%m-%d %H:%M')} | {url}"
            )

        url_count = len(top_rows)
        coverage_ratio = (float(total_cnt) / float(topic_articles)) if topic_articles > 0 else 1.0
        if active_entities >= min(4, len(entity_list)) and total_cnt >= 15 and url_count >= 8 and coverage_ratio >= 0.4:
            confidence = "High"
        elif active_entities >= 2 and total_cnt >= 4 and url_count >= 2 and coverage_ratio >= 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"
        lines.append(f"Confidence: {confidence}")
        return "\n".join(lines)
    except Exception as exc:
        print(f"[Error] analyze_landscape failed: {exc}")
        return f"analyze_landscape failed: {exc}"
    finally:
        if conn is not None:
            put_conn(conn)


def analyze_ai_landscape(days: int = 30, entities: str = "", limit_per_entity: int = 3) -> str:
    """Analyze AI landscape. Alias for analyze_landscape(topic='AI')."""
    return analyze_landscape(topic="AI", days=days, entities=entities, limit_per_entity=limit_per_entity)


def analyze_landscape_skill(payload: AnalyzeLandscapeSkillInput) -> SkillEnvelope:
    request = payload.model_dump(mode="python")
    try:
        raw_output = analyze_landscape(
            topic=request.get("topic", ""),
            days=int(request.get("days", 30)),
            entities=request.get("entities", ""),
            limit_per_entity=int(request.get("limit_per_entity", 3)),
        )
    except Exception as exc:
        return build_error_envelope(
            tool="analyze_landscape",
            request=request,
            error="analyze_landscape_execution_failed",
            diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    if raw_output.startswith("No landscape data") or raw_output.startswith("analyze_landscape failed"):
        is_error = "failed" in raw_output
        return SkillEnvelope(
            tool="analyze_landscape",
            status="error" if is_error else "empty",
            request=request,
            data={"raw_output": raw_output},
            evidence=[],
            error=raw_output if is_error else None,
            diagnostics={"topic": request.get("topic", "")},
        )

    confidence = None
    for line in raw_output.splitlines():
        if line.strip().startswith("Confidence:"):
            confidence = line.strip().split(":", 1)[1].strip()

    evidence = _evidence_from_text_output(raw_output, max_items=12)
    return SkillEnvelope(
        tool="analyze_landscape",
        status="ok",
        request=request,
        data={"raw_output": raw_output, "confidence": confidence},
        evidence=evidence,
        diagnostics={"topic": request.get("topic", ""), "confidence": confidence},
    )
