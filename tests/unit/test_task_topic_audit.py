from __future__ import annotations

from pathlib import Path

from eval.audit_task_topics import audit_task_config, audit_task_with_sample
from eval.task_eval_schema import load_task_types


def _task(
    *,
    skill: str = "search_news",
    scenario: str = "normal",
    retrieval_mode: str = "evaluable",
    should_clarify: bool = False,
    params: dict | None = None,
    keywords: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict:
    params = params or {"query": "Anthropic Claude model and product updates", "days": 21}
    return {
        "task_id": f"{skill}.{scenario}",
        "skill": skill,
        "intent_label": skill,
        "retrieval_mode": retrieval_mode,
        "scenario": scenario,
        "example_question": "q",
        "parameter_template": params,
        "acceptable_tool_paths": [[{"tool": skill, "args": dict(params)}]],
        "required_tools": [skill],
        "forbidden_tools": [],
        "should_clarify": should_clarify,
        "difficulty": "medium",
        "sampling": {
            "days": 30,
            "n_min": 3,
            "pool_size": 12,
            "candidate_limit": 300,
            "keywords": keywords or ["Anthropic", "Claude"],
            "sources": sources or [],
            "languages": [],
        },
        "tags": [scenario],
    }


def _doc(
    idx: int,
    *,
    source: str = "TechCrunch",
    day: int = 1,
    group: str = "",
    title: str = "Anthropic Claude update",
    sim: float = 0.2,
) -> dict:
    row = {
        "doc_id": f"doc_{idx}",
        "url": f"https://example.com/{idx}",
        "title": title,
        "summary": title,
        "evidence_text": title,
        "source": source,
        "published_at": f"2026-04-{day:02d}T00:00:00+00:00",
        "seed_similarity": sim,
        "embedding_available": True,
        "query_labels": [],
    }
    if group:
        row["topic_group"] = group
        row["query_labels"] = [f"topic_{group.lower()}"]
    return row


def _sample_meta(candidates: list[dict], **overrides) -> dict:
    meta = {
        "candidate_docs": len(candidates),
        "embedding_docs": len(candidates),
        "candidate_channel_counts": {"semantic": len(candidates)},
    }
    meta.update(overrides)
    return meta


def test_actual_task_topics_static_audit_passes() -> None:
    tasks = load_task_types(
        Path("eval/config/tasks_180.json"),
        strict_skill=False,
        enforce_coverage_policy=False,
    )

    assert len(tasks) == 28
    assert sum(task["sampling"]["n_min"] for task in tasks) == 180
    assert sum(task["sampling"]["n_min"] for task in tasks if task["scenario"] == "normal") == 84
    assert sum(task["sampling"]["n_min"] for task in tasks if task["scenario"] == "boundary") == 40
    assert sum(task["sampling"]["n_min"] for task in tasks if task["scenario"] == "conflict") == 28
    assert sum(task["sampling"]["n_min"] for task in tasks if task["scenario"] == "empty") == 28
    assert all(audit_task_config(task)["verdict"] == "pass" for task in tasks)


def test_config_audit_rejects_placeholder_evaluable_topic() -> None:
    task = _task(params={"query": "low frequency topic", "days": 14}, keywords=["low frequency topic"])

    result = audit_task_config(task)

    assert result["verdict"] == "fail"
    assert {issue["code"] for issue in result["issues"]} == {"topic_placeholder"}


def test_config_audit_rejects_path_topic_mismatch() -> None:
    task = _task()
    task["acceptable_tool_paths"] = [[{"tool": "search_news", "args": {"query": "OpenAI", "days": 21}}]]

    result = audit_task_config(task)

    assert result["verdict"] == "fail"
    assert any(issue["code"] == "path_topic_mismatch" for issue in result["issues"])


def test_config_audit_enforces_conflict_policy() -> None:
    task = _task(scenario="conflict", retrieval_mode="evaluable", should_clarify=True)

    result = audit_task_config(task)

    assert result["verdict"] == "fail"
    assert any(issue["code"] == "conflict_policy_mismatch" for issue in result["issues"])


def test_runtime_audit_rejects_candidate_shortfall() -> None:
    task = _task()
    candidates = [_doc(i) for i in range(1, 12)]

    result = audit_task_with_sample(task, candidates, _sample_meta(candidates), pools=[])

    assert result["verdict"] == "fail"
    assert any(issue["code"] == "candidate_coverage_insufficient" for issue in result["issues"])
    assert any(issue["code"] == "valid_pool_insufficient" for issue in result["issues"])


def test_runtime_audit_rejects_compare_sources_missing_source() -> None:
    task = _task(
        skill="compare_sources",
        params={"topic": "NVIDIA AI chip supply", "days": 21},
        keywords=["NVIDIA", "AI chip supply"],
        sources=["HackerNews", "TechCrunch"],
    )
    candidates = [_doc(i, source="HackerNews", day=(i % 5) + 1) for i in range(1, 80)]
    pools = [{"docs": candidates[:12]}, {"docs": candidates[12:24]}, {"docs": candidates[24:36]}]

    result = audit_task_with_sample(task, candidates, _sample_meta(candidates), pools=pools)

    assert result["verdict"] == "fail"
    assert any(issue["code"] == "source_coverage_insufficient" for issue in result["issues"])


def test_runtime_audit_rejects_compare_topics_missing_side() -> None:
    task = _task(
        skill="compare_topics",
        params={"topic_a": "OpenAI", "topic_b": "Anthropic", "days": 21},
        keywords=["OpenAI", "Anthropic"],
    )
    candidates = [_doc(i, day=(i % 5) + 1, group="A") for i in range(1, 80)]
    pools = [{"docs": candidates[:12]}, {"docs": candidates[12:24]}, {"docs": candidates[24:36]}]

    result = audit_task_with_sample(task, candidates, _sample_meta(candidates), pools=pools)

    assert result["verdict"] == "fail"
    assert any(issue["code"] == "topic_side_coverage_insufficient" for issue in result["issues"])


def test_runtime_audit_rejects_timeline_without_time_spread() -> None:
    task = _task(
        skill="build_timeline",
        params={"topic": "OpenAI model and product releases", "days": 30, "limit": 12},
        keywords=["OpenAI", "model releases"],
    )
    candidates = [_doc(i, day=1) for i in range(1, 80)]
    pools = [{"docs": candidates[:12]}, {"docs": candidates[12:24]}, {"docs": candidates[24:36]}]

    result = audit_task_with_sample(task, candidates, _sample_meta(candidates), pools=pools)

    assert result["verdict"] == "fail"
    assert any(issue["code"] == "time_spread_insufficient" for issue in result["issues"])


def test_runtime_audit_rejects_empty_positive_matches() -> None:
    task = _task(
        scenario="empty",
        retrieval_mode="non_retrieval",
        params={"query": "zzqv nonexistent Claude acquisition 2026", "days": 14},
        keywords=["zzqv", "nonexistent Claude acquisition 2026"],
    )
    candidates = [_doc(i, sim=0.1) for i in range(1, 12)]
    meta = _sample_meta(candidates, candidate_channel_counts={"semantic": 10, "lexical": 4, "alias": 0})

    result = audit_task_with_sample(task, candidates, meta, pools=[{"docs": candidates[:8]}])

    assert result["verdict"] == "fail"
    assert any(issue["code"] == "empty_positive_match_too_high" for issue in result["issues"])
