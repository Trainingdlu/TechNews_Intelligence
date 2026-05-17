from __future__ import annotations

import pytest

from eval.build_task_dataset import _validate_generated_case_alignment
from eval.corpus_sampler import pack_cluster
from eval.pool_quality import annotate_doc_topic_match, pool_quality_summary


def _task(query: str = "OpenAI Voice Engine") -> dict:
    return {
        "task_id": "search_news.semantic.normal",
        "tool": "search_news",
        "retrieval_mode": "evaluable",
        "scenario": "normal",
        "parameter_template": {"query": query, "days": 14},
        "sampling": {},
    }


def _case(*, question: str, answer: str, doc: dict, query: str = "OpenAI Voice Engine") -> dict:
    return {
        "case_id": "case-1",
        "tool": "search_news",
        "expected_question": question,
        "expected_answer": answer,
        "expected_tool_paths": [[{"tool": "search_news", "args": {"query": query, "days": 14}}]],
        "retrieval_evaluable": True,
        "retrieval_gold_doc_ids": [doc["doc_id"]],
        "retrieval_gold_urls": [doc["url"]],
        "input_news_pool": [doc],
        "verifiable_claims": [
            {
                "claim": answer,
                "evidence_doc_ids": [doc["doc_id"]],
                "evidence_quotes": [{"doc_id": doc["doc_id"], "quote": doc["summary"]}],
                "claim_type": "fact",
            }
        ],
    }


def test_alignment_rejects_question_topic_drift() -> None:
    doc = {
        "doc_id": "doc_1",
        "url": "https://example.com/openai-voice-engine",
        "title": "OpenAI Voice Engine expands voice cloning controls",
        "summary": "OpenAI Voice Engine adds new consent and safety controls.",
        "source": "TechCrunch",
    }
    case = _case(question="请总结 GPT-5.5 的最新信息", answer="OpenAI Voice Engine 增加了安全控制。", doc=doc)

    with pytest.raises(ValueError, match="expected_question is not aligned"):
        _validate_generated_case_alignment(case, _task())


def test_alignment_rejects_gold_doc_topic_drift() -> None:
    doc = {
        "doc_id": "doc_1",
        "url": "https://example.com/openai-lawsuit",
        "title": "OpenAI lawsuit reaches jury stage",
        "summary": "OpenAI and Elon Musk lawsuit documents were discussed in court.",
        "source": "TechCrunch",
    }
    case = _case(
        question="请检索 OpenAI Voice Engine 最近 14 天新闻",
        answer="OpenAI 诉讼进入陪审团阶段。",
        doc=doc,
    )

    with pytest.raises(ValueError, match="retrieval_gold_doc_ids do not match"):
        _validate_generated_case_alignment(case, _task())


def test_alignment_rejects_source_claim_mismatch() -> None:
    doc = {
        "doc_id": "doc_1",
        "url": "https://example.com/openai-voice-engine",
        "title": "OpenAI Voice Engine expands voice cloning controls",
        "summary": "OpenAI Voice Engine adds new consent and safety controls.",
        "source": "HackerNews",
    }
    case = _case(
        question="请检索 OpenAI Voice Engine 最近 14 天新闻",
        answer="TechCrunch 报道称 OpenAI Voice Engine 增加了安全控制。",
        doc=doc,
    )

    with pytest.raises(ValueError, match="expected_answer mentions sources"):
        _validate_generated_case_alignment(case, _task())


def _doc(doc_id: str, title: str, summary: str, *, source: str = "TechCrunch", seed: float = 0.8) -> dict:
    return {
        "doc_id": doc_id,
        "url": f"https://example.com/{doc_id}",
        "title": title,
        "summary": summary,
        "evidence_text": summary,
        "published_at": "2026-05-01T00:00:00+00:00",
        "source": source,
        "seed_similarity": seed,
        "channels": ["lexical"],
    }


def test_pool_quality_prefers_topic_matched_docs() -> None:
    task = _task("OpenAI Voice Engine")
    docs = [
        _doc("doc_voice_1", "OpenAI Voice Engine safety update", "OpenAI Voice Engine adds consent controls."),
        _doc("doc_voice_2", "OpenAI Voice Engine API notes", "Voice Engine receives new deployment rules."),
        _doc("doc_voice_3", "Voice Engine rollout", "OpenAI Voice Engine rollout adds safeguards."),
        _doc("doc_noise_1", "OpenAI lawsuit reaches jury", "Elon Musk and OpenAI lawsuit enters trial stage."),
        _doc("doc_noise_2", "Anthropic browser feature", "Claude browser feature receives updates."),
    ]
    for doc in docs:
        annotate_doc_topic_match(doc, task)

    selected, meta = pack_cluster(task, docs, pool_size=8)

    assert meta["pool_quality_passed"] is True
    assert {doc["doc_id"] for doc in selected[:2]} == {"doc_voice_1", "doc_voice_2"}
    assert meta["pool_quality"]["topic_matched_docs"] >= 2


def test_pool_quality_fails_when_topic_evidence_is_missing() -> None:
    task = _task("OpenAI Voice Engine")
    docs = [
        _doc("doc_noise_1", "OpenAI lawsuit reaches jury", "OpenAI lawsuit documents were discussed in court."),
        _doc("doc_noise_2", "OpenAI board dispute", "OpenAI board dispute continues."),
        _doc("doc_noise_3", "Anthropic product update", "Claude receives a coding update."),
    ]

    selected, meta = pack_cluster(task, docs, pool_size=8)

    assert selected
    assert meta["pool_quality_passed"] is False
    assert "low_topic_match_ratio" in meta["pool_quality_reasons"]


def test_compare_topics_pool_quality_requires_both_sides() -> None:
    task = {
        **_task("OpenAI Anthropic"),
        "tool": "compare_topics",
        "parameter_template": {"topic_a": "OpenAI Voice Engine", "topic_b": "Anthropic Claude Design"},
    }
    docs = [
        {**_doc("doc_a1", "OpenAI Voice Engine update", "OpenAI Voice Engine adds controls."), "topic_group": "A"},
        {**_doc("doc_a2", "OpenAI Voice Engine launch", "OpenAI Voice Engine expands."), "topic_group": "A"},
    ]

    summary = pool_quality_summary(docs, task)

    assert summary["pool_quality_passed"] is False
    assert "missing_compare_topic_side" in summary["pool_quality_reasons"]
