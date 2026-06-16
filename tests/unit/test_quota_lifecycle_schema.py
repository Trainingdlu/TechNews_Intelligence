from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_access_tokens_schema_declares_tier_lifecycle_columns() -> None:
    ddl = (PROJECT_ROOT / "sql/infrastructure/schema/schema_ddl.sql").read_text(encoding="utf-8")

    assert "quota       INT          NOT NULL DEFAULT 10" in ddl
    assert "status      VARCHAR(20)  NOT NULL DEFAULT 'active',   -- active / pending / capped" in ddl
    assert "upgraded_at TIMESTAMPTZ," in ddl
    assert "tier        SMALLINT     NOT NULL DEFAULT 0" in ddl
    assert "unlimited   BOOLEAN      NOT NULL DEFAULT FALSE" in ddl


def test_quota_tier_migration_backfills_existing_access_tokens() -> None:
    migration = (
        PROJECT_ROOT / "sql/infrastructure/schema/migrate_access_tokens_quota_tiers.sql"
    ).read_text(encoding="utf-8")

    assert "ADD COLUMN IF NOT EXISTS tier SMALLINT NOT NULL DEFAULT 0" in migration
    assert "ADD COLUMN IF NOT EXISTS unlimited BOOLEAN NOT NULL DEFAULT FALSE" in migration
    assert "ALTER COLUMN quota SET DEFAULT 10" in migration
    assert "ARRAY[10, 50, 100, 200]" in migration
    assert "UPDATE public.access_tokens" in migration
    assert "trainingcqy@gmail.com" in migration
