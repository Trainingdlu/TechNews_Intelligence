from __future__ import annotations

from eval import build_task_dataset as mod


def _pool(pool_id: str) -> mod.Pool:
    return mod.Pool(pool_id=pool_id, docs=[])


def test_generate_for_task_falls_back_to_single_pool_calls(monkeypatch) -> None:
    task = {"task_id": "search_news.normal"}
    pools = [_pool("search_news.normal.pool.1"), _pool("search_news.normal.pool.2")]

    invoke_prompts: list[str] = []

    def fake_generator_prompts(_task, pools_chunk, rejection_reasons=None):
        return "sys", ",".join(pool.pool_id for pool in pools_chunk)

    def fake_invoke_json(_llm, _system_prompt, user_prompt, **_kwargs):
        invoke_prompts.append(user_prompt)
        if "," in user_prompt:
            raise RuntimeError("chunk_generation_failed")
        return {"cases": [{"pool_id": user_prompt}]}

    def fake_repair(raw_case, _task, _pool_docs):
        return raw_case

    def fake_normalize_case(raw_case, **kwargs):
        return {
            "case_id": kwargs["case_id"],
            "pool_id": kwargs["pool_id"],
            "pool_from_llm": raw_case.get("pool_id"),
        }

    monkeypatch.setattr(mod, "_generator_prompts", fake_generator_prompts)
    monkeypatch.setattr(mod, "_invoke_json", fake_invoke_json)
    monkeypatch.setattr(mod, "_repair_generated_case", fake_repair)
    monkeypatch.setattr(mod, "normalize_case", fake_normalize_case)

    out = mod._generate_for_task(
        llm=None,
        task=task,
        pools=pools,
        llm_max_retries=1,
        llm_backoff_sec=0.0,
        pools_per_generation_call=2,
        inter_llm_call_sleep_sec=0.0,
    )

    assert invoke_prompts == [
        "search_news.normal.pool.1,search_news.normal.pool.2",
        "search_news.normal.pool.1",
        "search_news.normal.pool.2",
    ]
    assert [row["pool_id"] for row in out] == [
        "search_news.normal.pool.1",
        "search_news.normal.pool.2",
    ]
    assert [row["case_id"] for row in out] == [
        "search_news.normal.001",
        "search_news.normal.002",
    ]
