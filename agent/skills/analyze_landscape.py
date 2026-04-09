"""Analyze-landscape skill implementation and structured adapter."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from services.db import get_conn, put_conn

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .helpers import _clamp_int, _evidence_from_text_output, _is_recent_timestamp
from .schemas import AnalyzeLandscapeSkillInput
from .sql_builders import _build_topic_where_clause

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

LANDSCAPE_ENTITY_ALIASES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "\u8c37\u6b4c": "Google",
    "microsoft": "Microsoft",
    "\u5fae\u8f6f": "Microsoft",
    "meta": "Meta",
    "amazon": "Amazon",
    "aws": "Amazon",
    "\u4e9a\u9a6c\u900a": "Amazon",
    "nvidia": "NVIDIA",
    "\u82f1\u4f1f\u8fbe": "NVIDIA",
    "apple": "Apple",
    "\u82f9\u679c": "Apple",
    "tesla": "Tesla",
    "\u7279\u65af\u62c9": "Tesla",
    "tsmc": "TSMC",
    "\u53f0\u79ef\u7535": "TSMC",
    "intel": "Intel",
    "\u82f1\u7279\u5c14": "Intel",
    "amd": "AMD",
    "crowdstrike": "CrowdStrike",
    "palo alto": "Palo Alto Networks",
    "palo alto networks": "Palo Alto Networks",
    "cloudflare": "Cloudflare",
    "cisco": "Cisco",
    "xai": "xAI",
    "x.ai": "xAI",
}

LANDSCAPE_SIGNAL_KEYWORDS = {
    "compute_cost": [
        "gpu",
        "tpu",
        "chip",
        "semiconductor",
        "datacenter",
        "data center",
        "server",
        "compute",
        "training cost",
        "inference cost",
        "capex",
        "\u7b97\u529b",
        "\u82af\u7247",
        "\u7535\u529b",
        "\u80fd\u8017",
    ],
    "algorithm_efficiency": [
        "model",
        "benchmark",
        "reasoning",
        "architecture",
        "transformer",
        "agent",
        "inference",
        "\u7b97\u6cd5",
        "\u6a21\u578b",
        "\u63a8\u7406",
        "\u67b6\u6784",
        "\u84b8\u998f",
        "\u5fae\u8c03",
    ],
    "data_moat": [
        "dataset",
        "data",
        "corpus",
        "licensing",
        "proprietary",
        "copyright",
        "privacy",
        "\u6570\u636e",
        "\u8bed\u6599",
        "\u6388\u6743",
        "\u7248\u6743",
        "\u9690\u79c1",
    ],
    "go_to_market": [
        "enterprise",
        "customer",
        "pricing",
        "revenue",
        "subscription",
        "partnership",
        "adoption",
        "sales",
        "\u5546\u4e1a\u5316",
        "\u5ba2\u6237",
        "\u5b9a\u4ef7",
        "\u6536\u5165",
        "\u8ba2\u9605",
        "\u5408\u4f5c",
        "\u843d\u5730",
    ],
    "policy_security": [
        "regulation",
        "compliance",
        "antitrust",
        "lawsuit",
        "security",
        "breach",
        "vulnerability",
        "policy",
        "military",
        "\u76d1\u7ba1",
        "\u5408\u89c4",
        "\u8bc9\u8bbc",
        "\u5b89\u5168",
        "\u6f0f\u6d1e",
        "\u519b\u65b9",
    ],
}

LANDSCAPE_SIGNAL_LABELS = {
    "compute_cost": "Compute/Cost",
    "algorithm_efficiency": "Algorithm/Efficiency",
    "data_moat": "Data/Moat",
    "go_to_market": "Go-to-Market",
    "policy_security": "Policy/Security",
}


def _normalize_landscape_entities(entities: str | list[str] | None) -> list[str]:
    raw_items: list[str] = []
    if isinstance(entities, str):
        raw_items = [x.strip() for x in re.split(r"[,\n;/|\uFF0C\u3001]+", entities) if x.strip()]
    elif isinstance(entities, list):
        raw_items = [str(x).strip() for x in entities if str(x).strip()]

    if not raw_items:
        return list(DEFAULT_LANDSCAPE_ENTITIES)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        alias_key = item.strip().lower()
        name = LANDSCAPE_ENTITY_ALIASES.get(alias_key, item.strip())
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)

    if not normalized:
        return list(DEFAULT_LANDSCAPE_ENTITIES)
    return normalized[:12]


def _landscape_signal_counts(rows: list[tuple]) -> dict[str, int]:
    """Keyword-proxy variable counts from (headline + summary) rows."""
    counts = {k: 0 for k in LANDSCAPE_SIGNAL_KEYWORDS.keys()}
    for _, _, headline, summary, _ in rows:
        text = f"{headline or ''} {summary or ''}".lower()
        for key, tokens in LANDSCAPE_SIGNAL_KEYWORDS.items():
            if any(token in text for token in tokens):
                counts[key] += 1
    return counts


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

    values_sql = ", ".join(["(%s, %s)"] * len(entity_list))
    params_entities: list[Any] = []
    for name in entity_list:
        params_entities.extend([name, f"%{name}%"])

    topic_where_sql = ""
    topic_params: list[Any] = []
    if topic:
        topic_clause_sql, topic_params = _build_topic_where_clause(topic, table_alias="v")
        topic_where_sql = f"AND {topic_clause_sql}"

    cte = f"""
        WITH entities(canonical, pattern) AS (
            VALUES {values_sql}
        ),
        matched AS (
            SELECT
                e.canonical AS entity,
                v.source_type,
                COALESCE(v.title_cn, v.title) AS headline,
                COALESCE(v.summary, '') AS summary,
                v.url,
                v.points,
                v.sentiment,
                v.created_at
            FROM view_dashboard_news v
            JOIN entities e
              ON (
                  v.title ILIKE e.pattern
                  OR COALESCE(v.title_cn, '') ILIKE e.pattern
                  OR COALESCE(v.summary, '') ILIKE e.pattern
              )
            WHERE v.created_at >= NOW() - %s::interval
              {topic_where_sql}
        )
    """

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

        cur.execute(
            cte
            + """
            SELECT
                entity,
                COUNT(*) AS cnt,
                ROUND(AVG(points)::numeric, 1) AS avg_points,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS pos_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neu_cnt,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_cnt
            FROM matched
            GROUP BY entity
            ORDER BY cnt DESC, entity ASC
            """,
            tuple(params_entities + [f"{days} days"] + topic_params),
        )
        stats_rows = cur.fetchall()

        cur.execute(
            cte
            + """
            SELECT COUNT(*) AS total_cnt, COUNT(DISTINCT entity) AS active_entities
            FROM matched
            """,
            tuple(params_entities + [f"{days} days"] + topic_params),
        )
        total_cnt, active_entities = cur.fetchone()

        topic_scope_sql = """
            SELECT COUNT(*) AS topic_articles
            FROM view_dashboard_news v
            WHERE v.created_at >= NOW() - %s::interval
        """
        topic_scope_params: list[Any] = [f"{days} days"]
        if topic:
            topic_clause_sql, topic_scope_topic_params = _build_topic_where_clause(topic, table_alias="v")
            topic_scope_sql += f" AND {topic_clause_sql}"
            topic_scope_params.extend(topic_scope_topic_params)
        cur.execute(topic_scope_sql, tuple(topic_scope_params))
        topic_articles = int(cur.fetchone()[0] or 0)

        cur.execute(
            cte
            + """
            , ranked AS (
                SELECT
                    entity,
                    source_type,
                    headline,
                    url,
                    points,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY entity
                        ORDER BY points DESC NULLS LAST, created_at DESC
                    ) AS rn
                FROM matched
            )
            SELECT entity, source_type, headline, url, points, created_at, rn
            FROM ranked
            WHERE rn <= %s
            ORDER BY entity, rn
            """,
            tuple(params_entities + [f"{days} days"] + topic_params + [limit_per_entity]),
        )
        top_rows = cur.fetchall()

        cur.execute(
            cte
            + """
            SELECT entity, source_type, headline, summary, created_at
            FROM matched
            """,
            tuple(params_entities + [f"{days} days"] + topic_params),
        )
        sample_rows = cur.fetchall()
        cur.close()

        if not total_cnt:
            if topic_articles > 0:
                return (
                    f"Topic '{topic_label}' has {topic_articles} articles in the last {days} days, "
                    "but no tracked entities matched. Try passing explicit entities."
                )
            return f"No landscape data in the last {days} days for entities: {', '.join(entity_list)}."

        stat_map: dict[str, tuple[int, Any, int, int, int]] = {}
        for entity, cnt, avg_points, pos_cnt, neu_cnt, neg_cnt in stats_rows:
            stat_map[entity] = (cnt, avg_points, pos_cnt, neu_cnt, neg_cnt)

        split_days = max(3, days // 2)
        prev_days = max(1, days - split_days)
        cutoff_ts = db_now - timedelta(days=split_days)

        source_counts: dict[str, int] = {}
        entity_source_counts: dict[str, dict[str, int]] = {}
        momentum_map: dict[str, dict[str, int]] = {name: {"recent": 0, "previous": 0} for name in entity_list}
        for entity, source_type, headline, summary, created_at in sample_rows:
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            per_entity = entity_source_counts.setdefault(entity, {})
            per_entity[source_type] = per_entity.get(source_type, 0) + 1
            if entity not in momentum_map:
                momentum_map[entity] = {"recent": 0, "previous": 0}
            if _is_recent_timestamp(created_at, cutoff_ts):
                momentum_map[entity]["recent"] += 1
            else:
                momentum_map[entity]["previous"] += 1

        signal_counts = _landscape_signal_counts(sample_rows)

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

            if name not in stat_map:
                lines.append(
                    f"  {name}: count=0, share=0.0%, avg_points=0, "
                    f"sentiment(P/N/Ng)=0/0/0, momentum_recent_vs_prev={recent}/{previous} ({delta_text})"
                )
                continue
            cnt, avg_points, pos_cnt, neu_cnt, neg_cnt = stat_map[name]
            share = (float(cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
            avg_points_value = float(avg_points) if avg_points is not None else 0.0
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

        lines.append("Variable signals (keyword proxy, headline+summary):")
        for key, label in LANDSCAPE_SIGNAL_LABELS.items():
            signal_cnt = int(signal_counts.get(key, 0))
            signal_share = (float(signal_cnt) / float(total_cnt)) * 100 if total_cnt else 0.0
            lines.append(f"  {label}: count={signal_cnt}, share={signal_share:.1f}%")
        lines.append("Signal note: keyword proxy for triage; verify with evidence URLs.")

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
