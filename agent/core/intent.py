"""Shared intent detection utilities for agent routing and clarification."""

from __future__ import annotations

import re
from typing import Iterable

_SMALLTALK_INTENT_RE = re.compile(
    r"(?:^|\s)(?:hi|hello|hey|yo|thanks?|thank you|你好|您好|嗨|哈喽|在吗|早上好|下午好|晚上好|谢谢|辛苦了)(?:\s|$|[，。！？?!,])",
    re.IGNORECASE,
)
_CAPABILITY_INTENT_RE = re.compile(
    r"(?:你|您|机器人|助手|bot|assistant).*(?:能|可以|会|支持|can|able to).*(?:做什么|干什么|哪些功能|支持什么|怎么用|能做啥|what can you|how to use|capabilities|对比什么|分析什么)",
    re.IGNORECASE,
)
_CAPABILITY_HINT_RE = re.compile(
    r"(?:what can you|how can you help|your capabilities|你能做什么|你可以做什么|支持什么|怎么用|对比什么|分析什么)",
    re.IGNORECASE,
)
_ANALYSIS_INTENT_RE = re.compile(
    r"(?:最近|过去|近\d+天|近一周|近一个月|今天|今日|last\s*\d+\s*days?|recent|past|today|"
    r"analy[sz]e|analysis|compare|vs|timeline|trend|outlook|landscape|"
    r"分析|对比|比较|趋势|时间线|动态|格局|局势|复盘|汇总|梳理|盘点)",
    re.IGNORECASE,
)
_ROUNDUP_SUBJECT_RE = re.compile(
    r"新闻|快讯|要闻|动态|news|updates?|brief|roundup",
    re.IGNORECASE,
)
_ROUNDUP_ACTION_RE = re.compile(
    r"发生了什么|有什么|列出来|列出|列一下|盘点|汇总|梳理|直接给我|"
    r"what\s+happened|just\s+list|list\s+(?:them|it|out)?|show\s+me",
    re.IGNORECASE,
)
_ANALYSIS_HEAVY_RE = re.compile(
    r"深度|解读|洞察|研判|前景|判断|结论|预测|推演|影响|格局|复盘|归因|"
    r"analysis|deep\s+dive|insight|assessment|outlook|implication|forecast",
    re.IGNORECASE,
)
_CONFLICT_RESOLUTION_RE = re.compile(
    r"前景|怎么看|如何看|判断|结论|更好|更差|优劣|孰优|对比|比较|"
    r"是否冲突|是否分歧|冲突点|分歧点|"
    r"outlook|assessment|better|worse|compare|comparison|vs|versus|conflict|disagree",
    re.IGNORECASE,
)
_EXPLICIT_CONFLICT_REQUEST_RE = re.compile(
    r"分歧|冲突|矛盾|不一致|"
    r"conflict|contradict|disagree|diverg",
    re.IGNORECASE,
)

ANALYTICAL_TOOL_NAMES = frozenset({
    "compare_sources",
    "compare_topics",
    "trend_analysis",
    "analyze_landscape",
    "build_timeline",
})
RETRIEVAL_TOOL_NAMES = frozenset({
    "query_news",
    "search_news",
    "read_news_content",
    "fulltext_batch",
    "list_topics",
    "get_db_stats",
})


def extract_user_intent_text(user_message: str) -> str:
    """Strip wrapper scaffolding and keep only user-intent text."""
    text = str(user_message or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    extracted: list[str] = []
    for line in lines:
        if line.startswith("原问题："):
            payload = line.removeprefix("原问题：").strip()
            if payload:
                extracted.append(payload)
            continue
        if line.startswith("用户补充澄清："):
            payload = line.removeprefix("用户补充澄清：").strip()
            if payload:
                extracted.append(payload)
            continue
        lowered = line.lower()
        if lowered.startswith("original question:"):
            payload = line[len("original question:"):].strip()
            if payload:
                extracted.append(payload)
            continue
        if lowered.startswith("user clarification:"):
            payload = line[len("user clarification:"):].strip()
            if payload:
                extracted.append(payload)
            continue

    if extracted:
        return "\n".join(extracted)
    return text


def is_smalltalk_or_capability_intent(user_message: str) -> bool:
    text = extract_user_intent_text(user_message)
    if not text:
        return False
    return bool(
        _SMALLTALK_INTENT_RE.search(text)
        or _CAPABILITY_INTENT_RE.search(text)
        or _CAPABILITY_HINT_RE.search(text)
    )


def is_analysis_intent(user_message: str) -> bool:
    text = extract_user_intent_text(user_message)
    if not text:
        return False
    return bool(_ANALYSIS_INTENT_RE.search(text))


def is_roundup_listing_intent(user_message: str) -> bool:
    text = extract_user_intent_text(user_message)
    if not text:
        return False
    return bool(
        (_ROUNDUP_ACTION_RE.search(text) and is_analysis_intent(text))
        or (_ROUNDUP_SUBJECT_RE.search(text) and (_ROUNDUP_ACTION_RE.search(text) or is_analysis_intent(text)))
    )


def is_analysis_heavy_intent(user_message: str) -> bool:
    text = extract_user_intent_text(user_message)
    if not text:
        return False
    return bool(_ANALYSIS_HEAVY_RE.search(text))


def is_conflict_resolution_intent(user_message: str) -> bool:
    text = extract_user_intent_text(user_message)
    if not text:
        return False
    return bool(_CONFLICT_RESOLUTION_RE.search(text))


def has_explicit_conflict_request(user_message: str) -> bool:
    text = extract_user_intent_text(user_message)
    if not text:
        return False
    return bool(_EXPLICIT_CONFLICT_REQUEST_RE.search(text))


def classify_user_intent(user_message: str) -> str:
    """Classify user intent for risk gating.

    Returns one of:
    - smalltalk_or_capability
    - roundup_listing
    - conflict_resolution
    - analysis
    - generic
    """
    text = extract_user_intent_text(user_message)
    if not text:
        return "generic"
    if is_smalltalk_or_capability_intent(text):
        return "smalltalk_or_capability"
    if is_roundup_listing_intent(text):
        return "roundup_listing"
    if is_conflict_resolution_intent(text):
        return "conflict_resolution"
    if is_analysis_heavy_intent(text) or is_analysis_intent(text):
        return "analysis"
    return "generic"


def classify_tool_profile(tool_calls: Iterable[str] | None) -> str:
    """Classify tool-call pattern.

    Returns one of:
    - none
    - retrieval_only
    - analytical
    - mixed
    """
    normalized = {
        str(name or "").strip().lower()
        for name in (tool_calls or [])
        if str(name or "").strip()
    }
    if not normalized:
        return "none"

    has_analytical = bool(normalized & ANALYTICAL_TOOL_NAMES)
    has_retrieval = bool(normalized & RETRIEVAL_TOOL_NAMES)
    if has_analytical:
        return "analytical"
    if has_retrieval:
        return "retrieval_only"
    return "mixed"
