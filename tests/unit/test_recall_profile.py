from __future__ import annotations

from agent.skills.recall_profile import PROFILE_BASE, PROFILE_WIDE, resolve_recall_profile


def test_recall_profile_defaults_to_base(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_RECALL_PROFILE", raising=False)
    monkeypatch.delenv("EVAL_RECALL_PGVECTOR_PROBES", raising=False)
    monkeypatch.delenv("PGVECTOR_PROBES", raising=False)
    profile = resolve_recall_profile()
    assert profile.profile == PROFILE_BASE
    assert profile.pgvector_probes == 10
    assert profile.sim_floor == 0.20
    assert profile.oversample_ratio == 2.0
    assert profile.query_cand_multiplier == 6
    assert profile.query_cand_max == 72
    assert profile.macro_pool_limit == 200
    assert profile.pre_rerank_limit == 30


def test_recall_profile_wide_defaults(monkeypatch) -> None:
    monkeypatch.setenv("EVAL_RECALL_PROFILE", PROFILE_WIDE)
    monkeypatch.delenv("EVAL_RECALL_PGVECTOR_PROBES", raising=False)
    monkeypatch.delenv("PGVECTOR_PROBES", raising=False)
    profile = resolve_recall_profile()
    assert profile.profile == PROFILE_WIDE
    assert profile.pgvector_probes == 20
    assert profile.sim_floor == 0.14
    assert profile.oversample_ratio == 3.0
    assert profile.query_cand_multiplier == 10
    assert profile.query_cand_max == 120
    assert profile.macro_pool_limit == 320
    assert profile.pre_rerank_limit == 50


def test_recall_profile_prefers_new_env_over_legacy(monkeypatch) -> None:
    monkeypatch.setenv("EVAL_RECALL_PROFILE", PROFILE_BASE)
    monkeypatch.setenv("PGVECTOR_PROBES", "33")
    monkeypatch.setenv("EVAL_RECALL_PGVECTOR_PROBES", "18")
    monkeypatch.setenv("EVAL_RECALL_SIM_FLOOR", "0.17")
    monkeypatch.setenv("EVAL_RECALL_QUERY_CAND_MULTIPLIER", "9")
    monkeypatch.setenv("EVAL_RECALL_QUERY_CAND_MAX", "111")
    monkeypatch.setenv("EVAL_RECALL_MACRO_POOL_LIMIT", "260")
    monkeypatch.setenv("EVAL_RECALL_PRE_RERANK_LIMIT", "45")
    profile = resolve_recall_profile()
    assert profile.pgvector_probes == 18
    assert profile.sim_floor == 0.17
    assert profile.query_cand_multiplier == 9
    assert profile.query_cand_max == 111
    assert profile.macro_pool_limit == 260
    assert profile.pre_rerank_limit == 45


def test_recall_profile_unknown_value_falls_back_to_base(monkeypatch) -> None:
    monkeypatch.setenv("EVAL_RECALL_PROFILE", "unknown_profile")
    profile = resolve_recall_profile()
    assert profile.profile == PROFILE_BASE
