"""Recall-profile resolution for retrieval-related skills.

Centralizes all recall knobs so query retrieval, semantic pools, and
macro-skill aggregation follow one consistent profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


PROFILE_BASE = "base"
PROFILE_WIDE = "wide"


@dataclass(frozen=True)
class RecallProfile:
    profile: str
    pgvector_probes: int
    sim_floor: float
    oversample_ratio: float
    query_cand_multiplier: int
    query_cand_max: int
    macro_pool_limit: int
    pre_rerank_limit: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "pgvector_probes": self.pgvector_probes,
            "sim_floor": self.sim_floor,
            "oversample_ratio": self.oversample_ratio,
            "query_cand_multiplier": self.query_cand_multiplier,
            "query_cand_max": self.query_cand_max,
            "macro_pool_limit": self.macro_pool_limit,
            "pre_rerank_limit": self.pre_rerank_limit,
        }


# Hardcoded fallback values if profile/env parsing fails.
_HARDCODED_DEFAULTS = RecallProfile(
    profile=PROFILE_BASE,
    pgvector_probes=10,
    sim_floor=0.20,
    oversample_ratio=2.0,
    query_cand_multiplier=6,
    query_cand_max=72,
    macro_pool_limit=200,
    pre_rerank_limit=30,
)


_PROFILE_TABLE: dict[str, RecallProfile] = {
    PROFILE_BASE: _HARDCODED_DEFAULTS,
    PROFILE_WIDE: RecallProfile(
        profile=PROFILE_WIDE,
        pgvector_probes=20,
        sim_floor=0.14,
        oversample_ratio=3.0,
        query_cand_multiplier=10,
        query_cand_max=120,
        macro_pool_limit=320,
        pre_rerank_limit=50,
    ),
}


def _to_int(value: str | None, *, min_value: int, max_value: int, fallback: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return int(fallback)
    return max(min_value, min(max_value, parsed))


def _to_float(value: str | None, *, min_value: float, max_value: float, fallback: float) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return float(fallback)
    return max(min_value, min(max_value, parsed))


def _read_env(*keys: str) -> str:
    for key in keys:
        raw = str(os.getenv(key, "")).strip()
        if raw:
            return raw
    return ""


def _resolve_profile_name() -> str:
    raw = str(os.getenv("EVAL_RECALL_PROFILE", PROFILE_BASE)).strip().lower()
    alias = {
        "": PROFILE_BASE,
        "default": PROFILE_BASE,
        PROFILE_BASE: PROFILE_BASE,
        PROFILE_WIDE: PROFILE_WIDE,
    }
    resolved = alias.get(raw, PROFILE_BASE)
    if raw and raw not in alias:
        print(f"[Warn] unknown EVAL_RECALL_PROFILE='{raw}', fallback='{PROFILE_BASE}'.")
    return resolved


def resolve_recall_profile() -> RecallProfile:
    """Resolve effective recall profile.

    Priority (high -> low):
    1. New fine-grained env (EVAL_RECALL_*)
    2. Legacy env keys
    3. Profile defaults (EVAL_RECALL_PROFILE=base|wide)
    4. Hardcoded defaults
    """
    profile_name = _resolve_profile_name()
    defaults = _PROFILE_TABLE.get(profile_name, _HARDCODED_DEFAULTS)

    pgvector_probes = _to_int(
        _read_env("EVAL_RECALL_PGVECTOR_PROBES", "PGVECTOR_PROBES"),
        min_value=1,
        max_value=200,
        fallback=defaults.pgvector_probes,
    )
    sim_floor = _to_float(
        _read_env("EVAL_RECALL_SIM_FLOOR", "SEMANTIC_POOL_SIM_FLOOR"),
        min_value=0.0,
        max_value=1.0,
        fallback=defaults.sim_floor,
    )
    oversample_ratio = _to_float(
        _read_env("EVAL_RECALL_OVERSAMPLE_RATIO", "SEMANTIC_POOL_OVERSAMPLE_RATIO"),
        min_value=1.0,
        max_value=10.0,
        fallback=defaults.oversample_ratio,
    )
    query_cand_multiplier = _to_int(
        _read_env("EVAL_RECALL_QUERY_CAND_MULTIPLIER", "SEARCH_NEWS_CANDIDATE_MULTIPLIER"),
        min_value=1,
        max_value=30,
        fallback=defaults.query_cand_multiplier,
    )
    query_cand_max = _to_int(
        _read_env("EVAL_RECALL_QUERY_CAND_MAX", "SEARCH_NEWS_CANDIDATE_MAX"),
        min_value=1,
        max_value=500,
        fallback=defaults.query_cand_max,
    )
    macro_pool_limit = _to_int(
        _read_env("EVAL_RECALL_MACRO_POOL_LIMIT", "MACRO_SKILL_POOL_LIMIT"),
        min_value=20,
        max_value=1000,
        fallback=defaults.macro_pool_limit,
    )
    pre_rerank_limit = _to_int(
        _read_env("EVAL_RECALL_PRE_RERANK_LIMIT", "MACRO_SKILL_PRE_RERANK_LIMIT"),
        min_value=5,
        max_value=300,
        fallback=defaults.pre_rerank_limit,
    )
    pre_rerank_limit = min(pre_rerank_limit, macro_pool_limit)

    return RecallProfile(
        profile=profile_name,
        pgvector_probes=pgvector_probes,
        sim_floor=sim_floor,
        oversample_ratio=oversample_ratio,
        query_cand_multiplier=query_cand_multiplier,
        query_cand_max=query_cand_max,
        macro_pool_limit=macro_pool_limit,
        pre_rerank_limit=pre_rerank_limit,
    )
