"""Unit tests for clarification reason detection and payload specificity."""

from __future__ import annotations

from agent.clarification import (
    CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
    CLARIFICATION_REASON_SOURCE_CONFLICT,
    build_clarification_payload,
    detect_scope_or_conflict_reason,
)


def test_build_ambiguous_scope_payload_is_specific() -> None:
    payload = build_clarification_payload(
        "帮我总结一下 AI 行业最近动态",
        reason=CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
        context={
            "source_labels": ["HackerNews", "TechCrunch"],
            "time_span_days": 98,
            "entity_candidates": ["OpenAI", "NVIDIA", "Google"],
        },
    ).to_dict()

    assert payload["reason"] == CLARIFICATION_REASON_AMBIGUOUS_SCOPE
    assert "最近 7 天还是 30 天" in payload["question"]
    assert "HackerNews" in payload["question"] or "TechCrunch" in payload["question"]
    assert any("时间范围" in hint for hint in payload["hints"])
    assert any("来源范围" in hint for hint in payload["hints"])
    assert any("分析维度" in hint for hint in payload["hints"])


def test_build_source_conflict_payload_is_specific() -> None:
    payload = build_clarification_payload(
        "OpenAI 现在前景怎么样？",
        reason=CLARIFICATION_REASON_SOURCE_CONFLICT,
        context={
            "source_labels": ["HackerNews", "TechCrunch"],
            "conflict_summary": "TechCrunch 偏正向，而 HackerNews 更偏谨慎",
            "entity_candidates": ["OpenAI"],
        },
    ).to_dict()

    assert payload["reason"] == CLARIFICATION_REASON_SOURCE_CONFLICT
    assert "冲突" in payload["question"] or "分歧" in payload["question"]
    assert any("来源范围" in hint for hint in payload["hints"])
    assert any("时间范围" in hint for hint in payload["hints"])
    assert any("分析维度" in hint for hint in payload["hints"])


def test_detect_reason_for_wide_query_is_ambiguous_scope() -> None:
    reason, context = detect_scope_or_conflict_reason(
        user_message="帮我做 AI 行业全景总结",
        candidate_answer=(
            "2025-01-01 到 2025-04-20 的多来源摘要覆盖 OpenAI、NVIDIA、Google、Microsoft、Meta、Anthropic。"
        ),
        valid_urls=[
            "https://news.ycombinator.com/item?id=1",
            "https://news.ycombinator.com/item?id=2",
            "https://techcrunch.com/2025/01/10/openai-update/",
            "https://techcrunch.com/2025/03/11/google-ai/",
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
            "https://example.com/d",
        ],
        tool_calls={"query_news", "compare_sources"},
    )

    assert reason == CLARIFICATION_REASON_AMBIGUOUS_SCOPE
    assert context["url_count"] >= 8


def test_detect_reason_for_conflicting_sources_is_source_conflict() -> None:
    reason, context = detect_scope_or_conflict_reason(
        user_message="OpenAI 现在前景怎么样？",
        candidate_answer=(
            "TechCrunch 对 OpenAI 商业化预期较乐观，增长强劲。\n"
            "HackerNews 对 OpenAI 商业化更谨慎，强调风险与成本压力。"
        ),
        valid_urls=[
            "https://techcrunch.com/2025/04/01/openai-growth/",
            "https://news.ycombinator.com/item?id=100",
        ],
        tool_calls={"compare_sources"},
    )

    assert reason == CLARIFICATION_REASON_SOURCE_CONFLICT
    assert "conflict_summary" in context

