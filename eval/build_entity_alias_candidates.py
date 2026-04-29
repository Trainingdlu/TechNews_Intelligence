"""Build entity alias candidates from the local news corpus.

This is an offline/maintenance utility. It is intentionally not used on the
user-facing request path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.db import db_cursor, db_transaction  # noqa: E402
from agent.tools.entity_automation import (  # noqa: E402
    EntityDecision,
    adjudicate_alias_with_deepseek,
    decision_to_candidate_row,
    extract_alias_candidates_from_text,
)


def _apply_db_env_defaults() -> None:
    """Map deployment Postgres variables to the agent DB_* variables."""
    postgres_port = os.getenv("POSTGRES_PORT")
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    os.environ.setdefault("DB_HOST", "127.0.0.1")
    if postgres_port:
        os.environ.setdefault("DB_PORT", postgres_port)
    if postgres_db:
        os.environ.setdefault("DB_NAME", postgres_db)
    if postgres_user:
        os.environ.setdefault("DB_USER", postgres_user)
    if postgres_password:
        os.environ.setdefault("DB_PASS", postgres_password)


def _load_env_file(path: str | None) -> None:
    if path:
        env_path = Path(path)
        if not env_path.is_absolute():
            env_path = ROOT / env_path
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                else:
                    value = value.split("#", 1)[0].strip()
                os.environ[key] = value
    _apply_db_env_defaults()


def _load_news_rows(*, days: int, limit: int) -> list[dict[str, Any]]:
    with db_cursor() as (_conn, cur):
        cur.execute(
            """
            SELECT
                COALESCE(title, '') AS title,
                COALESCE(title_cn, '') AS title_cn,
                COALESCE(summary, '') AS summary,
                url,
                created_at
            FROM view_dashboard_news
            WHERE created_at >= NOW() - %s::interval
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (f"{days} days", limit),
        )
        rows = cur.fetchall()
    return [
        {
            "title": row[0],
            "title_cn": row[1],
            "summary": row[2],
            "url": row[3],
            "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4] or ""),
        }
        for row in rows
    ]


def _load_canonical_candidates(limit: int = 500) -> list[dict[str, Any]]:
    with db_cursor() as (_conn, cur):
        cur.execute(
            """
            SELECT entity_id, canonical_name, entity_type, wikidata_id
            FROM entity_registry
            WHERE is_active = TRUE
            ORDER BY confidence DESC, canonical_name ASC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "entity_id": row[0],
            "canonical_name": row[1],
            "entity_type": row[2],
            "wikidata_id": row[3],
        }
        for row in rows
    ]


def _collect_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    evidence_contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_urls: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        aliases = extract_alias_candidates_from_text(
            title=str(row.get("title") or ""),
            title_cn=str(row.get("title_cn") or ""),
            summary=str(row.get("summary") or ""),
        )
        for alias in aliases:
            key = alias.lower()
            if row["url"] not in evidence_urls[key]:
                evidence_urls[key].append(row["url"])
            if len(evidence_contexts[key]) < 8:
                evidence_contexts[key].append(
                    {
                        "title": row.get("title"),
                        "title_cn": row.get("title_cn"),
                        "summary": str(row.get("summary") or "")[:500],
                        "url": row.get("url"),
                    }
                )
            buckets[key] = {
                "alias": alias,
                "evidence_urls": evidence_urls[key],
                "contexts": evidence_contexts[key],
            }
    return buckets


def _pending_row(alias: str, evidence_urls: list[str]) -> dict[str, Any]:
    decision = EntityDecision(
        canonical_name="",
        entity_type="unknown",
        decision="pending",
        confidence=0.0,
        reason="DeepSeek adjudication not requested",
        aliases_to_add=[],
        status="pending",
    )
    return decision_to_candidate_row(
        decision,
        alias=alias,
        source="corpus",
        evidence_urls=evidence_urls,
    )


def _upsert_candidate(row: dict[str, Any]) -> None:
    with db_transaction() as (_conn, cur):
        cur.execute(
            """
            INSERT INTO entity_alias_candidate (
                alias,
                canonical_name,
                entity_type,
                source,
                confidence,
                evidence_count,
                evidence_urls,
                status,
                reason,
                aliases_to_add,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, NOW()
            )
            ON CONFLICT (source, LOWER(alias)) DO UPDATE
            SET canonical_name = CASE
                    WHEN entity_alias_candidate.status IN ('approved', 'rejected')
                      OR entity_alias_candidate.promoted_at IS NOT NULL
                    THEN entity_alias_candidate.canonical_name
                    ELSE EXCLUDED.canonical_name
                END,
                entity_type = CASE
                    WHEN entity_alias_candidate.status IN ('approved', 'rejected')
                      OR entity_alias_candidate.promoted_at IS NOT NULL
                    THEN entity_alias_candidate.entity_type
                    ELSE EXCLUDED.entity_type
                END,
                confidence = CASE
                    WHEN entity_alias_candidate.status IN ('approved', 'rejected')
                      OR entity_alias_candidate.promoted_at IS NOT NULL
                    THEN entity_alias_candidate.confidence
                    ELSE EXCLUDED.confidence
                END,
                evidence_count = EXCLUDED.evidence_count,
                evidence_urls = EXCLUDED.evidence_urls,
                status = CASE
                    WHEN entity_alias_candidate.status IN ('approved', 'rejected')
                      OR entity_alias_candidate.promoted_at IS NOT NULL
                    THEN entity_alias_candidate.status
                    ELSE EXCLUDED.status
                END,
                reason = CASE
                    WHEN entity_alias_candidate.status IN ('approved', 'rejected')
                      OR entity_alias_candidate.promoted_at IS NOT NULL
                    THEN entity_alias_candidate.reason
                    ELSE EXCLUDED.reason
                END,
                aliases_to_add = CASE
                    WHEN entity_alias_candidate.status IN ('approved', 'rejected')
                      OR entity_alias_candidate.promoted_at IS NOT NULL
                    THEN entity_alias_candidate.aliases_to_add
                    ELSE EXCLUDED.aliases_to_add
                END,
                updated_at = NOW()
            """,
            (
                row["alias"],
                row.get("canonical_name"),
                row.get("entity_type", "unknown"),
                row.get("source", "corpus"),
                float(row.get("confidence") or 0.0),
                int(row.get("evidence_count") or 0),
                json.dumps(row.get("evidence_urls") or [], ensure_ascii=False),
                row.get("status", "pending"),
                row.get("reason"),
                json.dumps(row.get("aliases_to_add") or [], ensure_ascii=False),
            ),
        )


def build_candidates(*, days: int, limit: int, use_deepseek: bool, dry_run: bool) -> list[dict[str, Any]]:
    rows = _load_news_rows(days=days, limit=limit)
    buckets = _collect_candidates(rows)
    canonical_candidates = _load_canonical_candidates() if use_deepseek else []

    output: list[dict[str, Any]] = []
    for item in buckets.values():
        evidence_urls = item["evidence_urls"]
        if use_deepseek:
            decision = adjudicate_alias_with_deepseek(
                alias=item["alias"],
                contexts=item["contexts"],
                canonical_candidates=canonical_candidates,
                evidence_count=len(evidence_urls),
            )
            row = decision_to_candidate_row(
                decision,
                alias=item["alias"],
                source="corpus",
                evidence_urls=evidence_urls,
            )
        else:
            row = _pending_row(item["alias"], evidence_urls)

        output.append(row)
        if not dry_run:
            _upsert_candidate(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build entity alias candidates from local news.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--use-deepseek", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-file", default="deployment/.env")
    args = parser.parse_args()

    _load_env_file(args.env_file)
    rows = build_candidates(
        days=max(1, min(args.days, 365)),
        limit=max(1, min(args.limit, 10000)),
        use_deepseek=bool(args.use_deepseek),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({"count": len(rows), "candidates": rows[:50]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
