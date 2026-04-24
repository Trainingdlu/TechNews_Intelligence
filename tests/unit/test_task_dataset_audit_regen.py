from __future__ import annotations

from eval import build_task_dataset as mod


def _case(case_id: str, pool_id: str) -> dict:
    return {"case_id": case_id, "pool_id": pool_id}


def _pool(pool_id: str) -> mod.Pool:
    return mod.Pool(pool_id=pool_id, docs=[])


def test_failed_only_regen_reaudits_only_rejected(monkeypatch) -> None:
    generated_cases = [
        _case("case.a", "pool.a"),
        _case("case.b", "pool.b"),
        _case("case.c", "pool.c"),
    ]
    pools = [_pool("pool.a"), _pool("pool.b"), _pool("pool.c")]
    task = {"task_id": "task.demo"}

    audit_calls: list[list[str]] = []
    generate_calls: list[list[str]] = []

    def fake_audit_cases(_llm, _task, cases, **_kwargs):
        case_ids = [str(item["case_id"]) for item in cases]
        audit_calls.append(case_ids)
        if len(audit_calls) == 1:
            return {"case.b": "bad_path", "case.c": "bad_grounding"}
        if len(audit_calls) == 2:
            return {"case.c": "still_bad"}
        return {}

    def fake_generate_for_task(_llm, _task, regen_pools, **_kwargs):
        pool_ids = [pool.pool_id for pool in regen_pools]
        generate_calls.append(pool_ids)
        out = []
        for pool_id in pool_ids:
            suffix = pool_id.split(".")[-1]
            out.append(_case(f"case.{suffix}", pool_id))
        return out

    monkeypatch.setattr(mod, "_audit_cases", fake_audit_cases)
    monkeypatch.setattr(mod, "_generate_for_task", fake_generate_for_task)

    kept, rejected = mod._audit_regen_failed_only(
        llm=None,
        audit_llm=None,
        task=task,
        pools=pools,
        generated_cases=generated_cases,
        llm_max_retries=1,
        llm_backoff_sec=0.0,
        max_regen_rounds=3,
        initial_cases_per_audit_call=0,
        regen_cases_per_audit_call=1,
        pools_per_generation_call=1,
        regen_pools_per_generation_call=1,
        inter_llm_call_sleep_sec=0.0,
    )

    assert rejected == {}
    assert [row["case_id"] for row in kept] == ["case.a", "case.b", "case.c"]
    assert audit_calls == [
        ["case.a", "case.b", "case.c"],
        ["case.b", "case.c"],
        ["case.c"],
    ]
    assert generate_calls == [["pool.b", "pool.c"], ["pool.c"]]


def test_failed_only_regen_drops_persistent_rejects(monkeypatch) -> None:
    generated_cases = [
        _case("case.a", "pool.a"),
        _case("case.b", "pool.b"),
    ]
    pools = [_pool("pool.a"), _pool("pool.b")]
    task = {"task_id": "task.demo"}

    def fake_audit_cases(_llm, _task, cases, **_kwargs):
        case_ids = {str(item["case_id"]) for item in cases}
        if "case.b" in case_ids:
            return {"case.b": "always_bad"}
        return {}

    def fake_generate_for_task(_llm, _task, regen_pools, **_kwargs):
        out = []
        for pool in regen_pools:
            if pool.pool_id == "pool.b":
                out.append(_case("case.b", "pool.b"))
        return out

    monkeypatch.setattr(mod, "_audit_cases", fake_audit_cases)
    monkeypatch.setattr(mod, "_generate_for_task", fake_generate_for_task)

    kept, rejected = mod._audit_regen_failed_only(
        llm=None,
        audit_llm=None,
        task=task,
        pools=pools,
        generated_cases=generated_cases,
        llm_max_retries=1,
        llm_backoff_sec=0.0,
        max_regen_rounds=1,
        initial_cases_per_audit_call=0,
        regen_cases_per_audit_call=1,
        pools_per_generation_call=1,
        regen_pools_per_generation_call=1,
        inter_llm_call_sleep_sec=0.0,
    )

    assert "case.b" in rejected
    assert [row["case_id"] for row in kept] == ["case.a"]
