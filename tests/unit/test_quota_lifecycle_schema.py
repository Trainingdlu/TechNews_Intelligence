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
