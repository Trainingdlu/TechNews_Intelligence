"""Static checks for entity seed SQL files."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEED_SQL = ROOT / "sql/infrastructure/seeds/seed_entity_core.sql"

EXPECTED_ENTITY_TERMS = (
    "Mistral AI",
    "Hugging Face",
    "ASML",
    "Google Cloud",
    "AWS Bedrock",
    "GitHub Copilot",
    "GB200",
    "AI infrastructure",
    "Semiconductor equipment",
    "甲骨文",
    "赛富时",
    "阿斯麦",
    "高通",
    "博通",
    "三星",
    "美光",
    "戴尔",
)
HIGH_AMBIGUITY_SHORT_ALIASES = ("'AI'", "'Go'", "'R1'", "'Arm'")


def _seed_sql() -> str:
    return SEED_SQL.read_text(encoding="utf-8")


def test_entity_core_seed_contains_expected_groups_and_utf8_aliases() -> None:
    text = _seed_sql()
    missing = [expected for expected in EXPECTED_ENTITY_TERMS if expected not in text]

    assert not missing, f"Missing expected seed terms: {missing}"


def test_entity_core_seed_avoids_high_ambiguity_short_aliases() -> None:
    text = _seed_sql()
    present = [alias for alias in HIGH_AMBIGUITY_SHORT_ALIASES if alias in text]

    assert not present, f"High-ambiguity aliases should stay out of the seed: {present}"
