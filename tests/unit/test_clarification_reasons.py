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


def test_build_ambiguous_scope_payload_avoids_unknown_source_names_and_bias() -> None:
    payload = build_clarification_payload(
        "最近发生了什么",
        reason=CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
        context={
            "source_labels": ["Newyorker", "Franzai", "Lemonade-server"],
            "time_span_days": None,
            "entity_candidates": ["OpenAI", "Google", "Agentic"],
        },
    ).to_dict()

    assert "Newyorker" not in payload["question"]
    assert "Franzai" not in payload["question"]
    assert "Lemonade-server" not in payload["question"]
    assert "建议指定 OpenAI" not in payload["question"]
    assert all("建议指定" not in hint for hint in payload["hints"])
    assert any("任选其一" in hint for hint in payload["hints"])
    assert any("单一来源" in hint or "多来源" in hint for hint in payload["hints"])


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


def test_detect_reason_source_labels_ignore_unknown_external_domains() -> None:
    reason, context = detect_scope_or_conflict_reason(
        user_message="帮我做 AI 行业全景总结",
        candidate_answer=(
            "2025-01-01 到 2025-04-20 的多来源摘要覆盖 OpenAI、NVIDIA、Google、Microsoft、Meta、Anthropic。"
        ),
        valid_urls=[
            "https://newyorker.com/a",
            "https://franzai.dev/b",
            "https://lemonade-server.net/c",
            "https://example.com/d",
            "https://example.com/e",
            "https://example.com/f",
            "https://example.com/g",
            "https://example.com/h",
        ],
        tool_calls={"query_news"},
    )

    assert reason in {None, CLARIFICATION_REASON_AMBIGUOUS_SCOPE}
    assert context["source_labels"] == []


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


def test_detect_reason_for_today_roundup_does_not_require_clarification() -> None:
    reason, _context = detect_scope_or_conflict_reason(
        user_message="今天发生了什么新闻，直接列出来就行了",
        candidate_answer=(
            "2026-04-12 今日快报：OpenAI、Google、Microsoft、Meta、Anthropic 的相关新闻如下。"
        ),
        valid_urls=[
            "https://news.ycombinator.com/item?id=101",
            "https://news.ycombinator.com/item?id=102",
            "https://techcrunch.com/2026/04/12/openai-news/",
            "https://techcrunch.com/2026/04/12/google-news/",
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
            "https://example.com/d",
        ],
        tool_calls={"query_news"},
    )

    assert reason is None


def test_detect_reason_for_today_roundup_skips_source_conflict() -> None:
    reason, _context = detect_scope_or_conflict_reason(
        user_message="今天发生了什么，列一下新闻",
        candidate_answer=(
            "TechCrunch 对 OpenAI 进展偏乐观，增长强劲。\n"
            "HackerNews 对同一主题更谨慎，强调风险压力。"
        ),
        valid_urls=[
            "https://techcrunch.com/2026/04/12/openai-growth/",
            "https://news.ycombinator.com/item?id=188",
        ],
        tool_calls={"query_news", "compare_sources"},
    )

    assert reason is None


def test_detect_reason_for_roundup_request_without_explicit_time_still_no_clarification() -> None:
    reason, _context = detect_scope_or_conflict_reason(
        user_message="最近有什么 AI 新闻，列一下就行",
        candidate_answer=(
            "TechCrunch 对 OpenAI 进展偏乐观，增长强劲。\n"
            "HackerNews 对同一主题更谨慎，强调风险压力。"
        ),
        valid_urls=[
            "https://news.ycombinator.com/item?id=201",
            "https://news.ycombinator.com/item?id=202",
            "https://techcrunch.com/2026/04/11/openai-news/",
            "https://techcrunch.com/2026/04/11/google-news/",
            "https://example.com/a",
            "https://example.com/b",
        ],
        tool_calls={"query_news"},
    )

    assert reason is None


def test_detect_reason_for_merged_followup_uses_user_intent_not_wrapper_text() -> None:
    merged_message = (
        "原问题：今天发生了什么\n"
        "用户补充澄清：直接给我今天发生的新闻，列出来就行了\n"
        "请基于原问题与补充范围重新检索，并给出有证据支撑的分析结论。"
    )
    reason, context = detect_scope_or_conflict_reason(
        user_message=merged_message,
        candidate_answer=(
            "2026-04-12 今日快报：OpenAI、Google、Microsoft、Meta、Anthropic 的相关新闻如下。"
        ),
        valid_urls=[
            "https://news.ycombinator.com/item?id=301",
            "https://news.ycombinator.com/item?id=302",
            "https://techcrunch.com/2026/04/12/openai-news/",
            "https://techcrunch.com/2026/04/12/google-news/",
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
            "https://example.com/d",
        ],
        tool_calls={"query_news", "compare_sources"},
    )

    assert reason is None
    assert "原问题" not in context["intent_text"]
    assert "用户补充澄清" not in context["intent_text"]


def test_detect_reason_for_analysis_heavy_broad_query_still_triggers_clarification() -> None:
    reason, context = detect_scope_or_conflict_reason(
        user_message="请做一个 AI 行业全景深度解读",
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
    assert int(context.get("ambiguous_scope_score", 0)) >= 5
