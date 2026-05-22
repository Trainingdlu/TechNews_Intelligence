"""Promote reviewed entity alias candidates into the formal alias registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.db import db_cursor, db_transaction  # noqa: E402
from services.entity_resolution import normalize_aliases  # noqa: E402
from services.script_env import _load_env_file  # noqa: E402


def _detect_language(alias: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
        return "zh"
    if alias.isascii():
        return "en"
    return "unknown"


def _load_promotable_candidates(*, include_auto_approved: bool, limit: int) -> list[dict[str, Any]]:
    statuses = ["approved"]
    if include_auto_approved:
        statuses.append("auto_approved")

    with db_cursor() as (_conn, cur):
        cur.execute(
            """
            SELECT
                candidate_id,
                alias,
                canonical_name,
                entity_type,
                source,
                confidence,
                aliases_to_add,
                status
            FROM entity_alias_candidate
            WHERE status = ANY(%s)
              AND promoted_at IS NULL
              AND canonical_name IS NOT NULL
              AND canonical_name <> ''
            ORDER BY confidence DESC, evidence_count DESC, updated_at DESC
            LIMIT %s
            """,
            (statuses, limit),
        )
        rows = cur.fetchall()

    return [
        {
            "candidate_id": row[0],
            "alias": row[1],
            "canonical_name": row[2],
            "entity_type": row[3],
            "source": row[4],
            "confidence": float(row[5] or 0.0),
            "aliases_to_add": row[6],
            "status": row[7],
        }
        for row in rows
    ]


def _promote_candidate(row: dict[str, Any]) -> dict[str, Any]:
    aliases = normalize_aliases(row["alias"], row.get("aliases_to_add"))
    with db_transaction() as (_conn, cur):
        cur.execute(
            """
            INSERT INTO entity_registry (
                canonical_name,
                entity_type,
                source,
                confidence,
                is_active,
                updated_at
            )
            VALUES (%s, %s, %s, %s, TRUE, NOW())
            ON CONFLICT (canonical_name) DO UPDATE
            SET entity_type = CASE
                    WHEN entity_registry.entity_type = 'unknown'
                    THEN EXCLUDED.entity_type
                    ELSE entity_registry.entity_type
                END,
                confidence = GREATEST(entity_registry.confidence, EXCLUDED.confidence),
                is_active = TRUE,
                updated_at = NOW()
            RETURNING entity_id
            """,
            (
                row["canonical_name"],
                row.get("entity_type") or "unknown",
                row.get("source") or "candidate",
                float(row.get("confidence") or 0.0),
            ),
        )
        entity_id = cur.fetchone()[0]

        inserted_aliases: list[str] = []
        for alias in aliases:
            cur.execute(
                """
                INSERT INTO entity_alias (
                    entity_id,
                    alias,
                    language,
                    alias_type,
                    weight,
                    is_exact,
                    is_active,
                    updated_at
                )
                VALUES (%s, %s, %s, 'candidate', %s, TRUE, TRUE, NOW())
                ON CONFLICT (entity_id, LOWER(alias)) DO NOTHING
                RETURNING alias
                """,
                (
                    entity_id,
                    alias,
                    _detect_language(alias),
                    max(0.1, min(1.0, float(row.get("confidence") or 0.0))),
                ),
            )
            inserted = cur.fetchone()
            if inserted:
                inserted_aliases.append(inserted[0])

        cur.execute(
            """
            UPDATE entity_alias_candidate
            SET status = 'approved',
                reviewed_at = COALESCE(reviewed_at, NOW()),
                promoted_at = NOW(),
                updated_at = NOW()
            WHERE candidate_id = %s
            """,
            (row["candidate_id"],),
        )

    return {
        "candidate_id": row["candidate_id"],
        "canonical_name": row["canonical_name"],
        "entity_id": entity_id,
        "aliases": aliases,
        "inserted_aliases": inserted_aliases,
    }


def promote_candidates(*, include_auto_approved: bool, limit: int, dry_run: bool) -> list[dict[str, Any]]:
    candidates = _load_promotable_candidates(
        include_auto_approved=include_auto_approved,
        limit=limit,
    )
    if dry_run:
        return [
            {
                "candidate_id": row["candidate_id"],
                "canonical_name": row["canonical_name"],
                "aliases": normalize_aliases(row["alias"], row.get("aliases_to_add")),
                "status": row["status"],
            }
            for row in candidates
        ]
    return [_promote_candidate(row) for row in candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote reviewed entity alias candidates.")
    parser.add_argument("--include-auto-approved", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-file", default="deployment/.env")
    args = parser.parse_args()

    _load_env_file(args.env_file)
    promoted = promote_candidates(
        include_auto_approved=bool(args.include_auto_approved),
        limit=max(1, min(args.limit, 10000)),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({"count": len(promoted), "promoted": promoted}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
