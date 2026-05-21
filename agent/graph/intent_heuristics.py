"""Intent and topic parsing heuristics for graph nodes."""

from __future__ import annotations

import re
from typing import Any

from agent.core.evidence import extract_urls
from agent.core.intent import classify_user_intent, extract_user_intent_text as _extract_user_intent_text

_KNOWN_ENTITY_ALIASES: tuple[tuple[str, str], ...] = (
    (r"(?<![a-z0-9])openai(?![a-z0-9])", "OpenAI"),
    (r"(?<![a-z0-9])google(?![a-z0-9])", "Google"),
    (r"谷歌", "Google"),
    (r"(?<![a-z0-9])gemini(?![a-z0-9])", "Gemini"),
    (r"(?<![a-z0-9])anthropic(?![a-z0-9])", "Anthropic"),
    (r"(?<![a-z0-9])claude(?![a-z0-9])", "Claude"),
    (r"(?<![a-z0-9])microsoft(?![a-z0-9])", "Microsoft"),
    (r"微软", "Microsoft"),
    (r"(?<![a-z0-9])meta(?![a-z0-9])", "Meta"),
    (r"(?<![a-z0-9])amazon(?![a-z0-9])", "Amazon"),
    (r"亚马逊", "Amazon"),
    (r"(?<![a-z0-9])apple(?![a-z0-9])", "Apple"),
    (r"苹果", "Apple"),
    (r"(?<![a-z0-9])nvidia(?![a-z0-9])", "NVIDIA"),
    (r"英伟达", "NVIDIA"),
    (r"(?<![a-z0-9])tesla(?![a-z0-9])", "Tesla"),
    (r"特斯拉", "Tesla"),
    (r"(?<![a-z0-9])xai(?![a-z0-9])", "xAI"),
    (r"(?<![a-z0-9])grok(?![a-z0-9])", "Grok"),
    (r"(?<![a-z0-9])deepseek(?![a-z0-9])", "DeepSeek"),
)
_COMPARE_DIMENSION_RE = re.compile(
    r"差异|区别|不同|战略|策略|商业化|企业市场|定价|开源|生态|布局|路线|侧重点|"
    r"strategy|pricing|enterprise|commerciali[sz]ation|ecosystem|difference|different",
    re.IGNORECASE,
)
_COMPARE_SIDE_TRAILING_RE = re.compile(
    r"(?:最近|近期|当前|目前|过去|近来|在|上|方面|的|战略|策略|商业化|企业市场|定价|开源|生态|布局|"
    r"差异|区别|不同|表现|动态|事件|新闻|产品|路线|方向|侧重点|"
    r"recent|latest|current|strategy|pricing|enterprise|commerciali[sz]ation|ecosystem|difference|different)",
    re.IGNORECASE,
)
_COMPARE_SIDE_PREFIX_RE = re.compile(
    r"^(?:帮我|请|看看|看一下|对比一下|比较一下|对比|比较|分析一下|分析|一下|"
    r"please|compare|analyze)\s*",
    re.IGNORECASE,
)
_GENERIC_COMPARE_SIDE_TERMS = {
    "ai",
    "人工智能",
    "企业市场",
    "商业化",
    "战略",
    "策略",
    "定价",
    "生态",
    "布局",
    "差异",
    "区别",
    "不同",
    "technology news",
}


# Canonical intent_type taxonomy. These are the exact strings _select_tools maps
# on; any LLM-produced intent_type outside this set is rejected so the heuristic
# classification is kept instead of silently falling through to generic tools.
VALID_INTENT_TYPES: frozenset[str] = frozenset(
    {
        "smalltalk_or_capability",
        "topic_comparison",
        "source_comparison",
        "trend",
        "timeline",
        "landscape",
        "article_read",
        "roundup_listing",
        "news_analysis",
    }
)


def _merge_intent(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    route = str(override.get("route") or "").strip()
    if route in {"direct_answer", "needs_clarification", "needs_tools"}:
        out["route"] = route
    # Only accept the LLM's intent_type when it matches the canonical taxonomy;
    # otherwise keep the heuristic's classification (which uses the same vocabulary).
    override_intent_type = str(override.get("intent_type") or "").strip()
    if override_intent_type in VALID_INTENT_TYPES:
        out["intent_type"] = override_intent_type
    for key in ("reason", "analysis_depth", "entities", "time_window", "risk_flags"):
        if key in override and override.get(key) not in (None, "", []):
            out[key] = override.get(key)
    try:
        out["confidence"] = float(override.get("confidence", out.get("confidence", 0.5)))
    except Exception:
        out["confidence"] = out.get("confidence", 0.5)
    out["requires_tools"] = out.get("route") == "needs_tools"
    return out


def _heuristic_intent(user_message: str) -> dict[str, Any]:
    text = _extract_user_intent_text(user_message).strip() or str(user_message or "").strip()
    lowered = text.lower()
    urls = extract_urls(text)
    rule_intent = classify_user_intent(text)
    if rule_intent == "smalltalk_or_capability":
        route = "direct_answer"
        intent_type = "smalltalk_or_capability"
    elif _looks_like_article_reference_without_url(text):
        route = "needs_clarification"
        intent_type = "article_read"
    else:
        route = "needs_tools"
        intent_type = "news_analysis"
    if urls:
        route = "needs_tools"
        intent_type = "article_read"
    if _compare_topics_hit(text):
        intent_type = "topic_comparison"
    elif "hackernews" in lowered or "techcrunch" in lowered or "source" in lowered or "来源" in text:
        if re.search(r"compare|comparison|vs|versus", lowered) or "对比" in text or "比较" in text:
            intent_type = "source_comparison"
    elif re.search(r"timeline|时间线|脉络|发生了什么", lowered):
        intent_type = "timeline"
    elif re.search(r"landscape|格局|全景|竞争|生态|排行", lowered):
        intent_type = "landscape"
    elif re.search(r"trend|趋势|动向|变化|增长|下降", lowered):
        intent_type = "trend"
    elif rule_intent == "roundup_listing":
        intent_type = "roundup_listing"

    return {
        "route": route,
        "intent_type": intent_type,
        "reason": "heuristic_fallback",
        "confidence": 0.55,
        "requires_tools": route == "needs_tools",
        "analysis_depth": "deep" if re.search(r"深度|深入|研判|判断|前景|risk|outlook", lowered) else "standard",
        "entities": _extract_entity_hints(text),
        "time_window": {"days": _extract_days(text)},
        "risk_flags": [],
    }


def _looks_like_article_reference_without_url(text: str) -> bool:
    lowered = str(text or "").lower()
    if extract_urls(text):
        return False
    return bool(
        re.search(r"这篇|这条|文章|链接|原文|报道", text)
        or re.search(r"\b(this|that)\s+(article|link|story|post)\b", lowered)
    )


def _compare_topics_hit(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    if re.search(r"\bvs\b|versus|compare|comparison", lowered) or "对比" in raw or "比较" in raw:
        return True
    if _COMPARE_DIMENSION_RE.search(raw) and len(_extract_entity_hints(raw)) >= 2:
        return True
    if _COMPARE_DIMENSION_RE.search(raw) and _has_two_specific_compare_sides(raw):
        return True
    return False


def _extract_days(text: str, default: int = 14) -> int:
    lowered = str(text or "").lower()
    match = re.search(r"(?:最近|过去|last|past)\s*(\d{1,3})\s*(?:天|day|days)", lowered)
    if match:
        try:
            return max(1, min(int(match.group(1)), 365))
        except Exception:
            return default
    if "今天" in text or "today" in lowered:
        return 1
    if "一周" in text or "week" in lowered:
        return 7
    if "一个月" in text or "month" in lowered:
        return 30
    return default


def _extract_entity_hints(text: str) -> list[str]:
    entities: list[str] = []
    raw = str(text or "")
    lowered = raw.lower()
    for pattern, label in _KNOWN_ENTITY_ALIASES:
        if re.search(pattern, lowered, flags=re.IGNORECASE) and label not in entities:
            entities.append(label)
    for token in re.findall(r"\b[A-Z][A-Za-z0-9+._-]{1,}\b", str(text or "")):
        if token.lower() in {"today", "latest", "recent", "news"}:
            continue
        if token not in entities:
            entities.append(token)
    return entities[:8]


def _split_compare_topics(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    entities = _extract_entity_hints(raw)
    if len(entities) >= 2:
        return entities[0], entities[1]
    for pattern in (r"(.+?)\s+(?:vs|versus)\s+(.+)",):
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return _clean_compare_topic(match.group(1)), _clean_compare_topic(match.group(2))
    for pattern in (
        r"(?:对比|比较)\s*(.+?)\s*(?:和|与|跟|同|及|以及)\s*(.+?)(?:的|在|上|方面|差异|区别|不同|$)",
        r"(.+?)\s*(?:和|与|跟|同|及|以及|、)\s*(.+?)(?:的)?(?:差异|区别|不同)",
        r"(.+?)(?:对比|比较)(.+)",
    ):
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return _clean_compare_topic(match.group(1)), _clean_compare_topic(match.group(2))
    return _topic_from_message(raw), "competitors"


def _has_two_specific_compare_sides(text: str) -> bool:
    left, right = _split_compare_topics(text)
    if right == "competitors":
        return False
    return _is_specific_compare_side(left) and _is_specific_compare_side(right)


def _is_specific_compare_side(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if len(normalized) < 2:
        return False
    if normalized in _GENERIC_COMPARE_SIDE_TERMS:
        return False
    if re.fullmatch(r"(?:最近|近期|当前|目前|过去|近来|recent|latest|current)", normalized):
        return False
    return True


def _topic_from_message(text: str) -> str:
    entities = _extract_entity_hints(text)
    if entities:
        return " ".join(entities[:3])
    cleaned = _clean_topic(text)
    return cleaned or "technology news"


def _clean_topic(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", "", str(text or "")).strip()
    cleaned = re.sub(r"(最近|过去|last|past)\s*\d{1,3}\s*(天|days?)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(分析|对比|比较|趋势|时间线|格局|新闻|动态|帮我|please|analyze|compare|trend|timeline|landscape)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。?？:：")
    return cleaned[:120] or "technology news"


def _clean_compare_topic(text: str) -> str:
    cleaned = _clean_topic(text)
    cleaned = _COMPARE_SIDE_PREFIX_RE.sub("", cleaned).strip(" ，,。?？:：")
    parts = _COMPARE_SIDE_TRAILING_RE.split(cleaned, maxsplit=1)
    if parts and parts[0].strip():
        cleaned = parts[0].strip(" ，,。?？:：")
    return cleaned[:120] or "technology news"
