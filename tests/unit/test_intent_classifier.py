"""Tests for shared intent classifier utilities."""

from __future__ import annotations

from agent.core.intent import (
    classify_tool_profile,
    classify_user_intent,
    extract_user_intent_text,
    has_explicit_conflict_request,
    is_analysis_intent,
    is_roundup_listing_intent,
    is_smalltalk_or_capability_intent,
)


def test_extract_user_intent_text_strips_clarification_wrapper() -> None:
    merged = (
        "原问题：今天发生了什么\n"
        "用户补充澄清：直接给我今天发生的新闻，列出来就行了\n"
        "请基于原问题与补充范围重新检索，并给出有证据支撑的分析结论。"
    )
    text = extract_user_intent_text(merged)
    assert "原问题" not in text
    assert "用户补充澄清" not in text
    assert "今天发生了什么" in text
    assert "直接给我今天发生的新闻" in text


def test_classify_user_intent_for_smalltalk() -> None:
    assert is_smalltalk_or_capability_intent("你好") is True
    assert classify_user_intent("hello, what can you do") == "smalltalk_or_capability"


def test_classify_user_intent_for_roundup_listing() -> None:
    assert is_roundup_listing_intent("今天发生了什么新闻，列出来") is True
    assert classify_user_intent("最近有什么 AI 新闻，列一下就行") == "roundup_listing"


def test_classify_user_intent_for_conflict_resolution() -> None:
    assert classify_user_intent("OpenAI 现在前景怎么看？") == "conflict_resolution"
    assert has_explicit_conflict_request("不同来源判断有冲突吗？") is True


def test_classify_user_intent_for_analysis() -> None:
    assert is_analysis_intent("最近 30 天 AI 趋势对比") is True
    assert classify_user_intent("请做一个 AI 行业全景深度解读") == "analysis"


def test_classify_tool_profile() -> None:
    assert classify_tool_profile(set()) == "none"
    assert classify_tool_profile({"query_news", "search_news"}) == "retrieval_only"
    assert classify_tool_profile({"query_news", "compare_sources"}) == "analytical"
    assert classify_tool_profile({"custom_tool"}) == "mixed"
