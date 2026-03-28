"""Deterministic route and parameter extraction helpers."""

from __future__ import annotations

import re


def count_timeline_items(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"^\s*\d+\.\s", text, flags=re.MULTILINE))


def extract_days(text: str, default: int, maximum: int) -> int:
    m = re.search(r"(?:最近|过去|last)?\s*(\d{1,3})\s*(?:天|day|days)", text, flags=re.IGNORECASE)
    val = int(m.group(1)) if m else default
    return max(1, min(maximum, val))


def extract_limit(text: str, default: int, maximum: int) -> int:
    m = re.search(r"(?:最多|max|limit)\s*[:=]?\s*(\d{1,2})\s*(?:条|items?)?", text, flags=re.IGNORECASE)
    val = int(m.group(1)) if m else default
    return max(1, min(maximum, val))


def extract_source_label(text: str) -> str:
    lower = (text or "").lower()
    if any(k in lower for k in ["techcrunch", "tc"]):
        return "TechCrunch"
    if any(k in lower for k in ["hackernews", "hacker news", "hn"]):
        return "HackerNews"
    return "all"


def extract_query_request(user_message: str) -> tuple[str, str, int, str, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None
    lower = text.lower()
    query_markers = ["检索", "搜索", "查询", "query", "search", "查一下", "找一下", "look up"]
    content_markers = [
        "新闻",
        "报道",
        "文章",
        "资讯",
        "动态",
        "消息",
        "news",
        "article",
        "articles",
        "coverage",
        "headline",
        "headlines",
        "updates",
    ]
    filter_markers = [
        "来源",
        "source",
        "按热度",
        "按时间",
        "sort",
        "排序",
        "sentiment",
        "情绪",
        "category",
        "分类",
    ]
    has_recent_window = bool(
        re.search(r"(?:最近|过去|近|last|recent|past)\s*\d{0,3}\s*(?:天|day|days|小时|hour|hours)?", text, flags=re.IGNORECASE)
    )
    source = extract_source_label(text)
    has_explicit_query = any(k in text or k in lower for k in query_markers)
    has_content_intent = any(k in text or k in lower for k in content_markers)
    has_filter_intent = any(k in text or k in lower for k in filter_markers) or (source != "all")
    if not (has_explicit_query or has_filter_intent or (has_content_intent and has_recent_window)):
        return None
    if any(k in text or k in lower for k in ["全文", "fulltext", "批量读取", "批量读", "deep read"]):
        return None

    query = ""
    m = re.search(r"(?:检索|搜索|查询|query|search)\s*[:：]?\s*([^\n，,。]{1,40})", text, flags=re.IGNORECASE)
    if m:
        query = m.group(1).strip()
        query = re.split(r"[，,。]|(?:来源|source|最近|过去|sort|按)", query, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    if not query:
        m = re.search(
            r"(?:关于|有关|聊聊|look up|about)\s*([A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,24})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            query = m.group(1).strip()

    if not query:
        topic_pattern = r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{1,24})"
        stop = {
            "检索",
            "搜索",
            "查询",
            "来源",
            "按",
            "热度",
            "排序",
            "最近",
            "过去",
            "query",
            "search",
            "source",
            "sort",
            "heat",
            "time",
            "desc",
            "asc",
            "天",
            "小时",
            "hour",
            "hours",
            "day",
            "days",
            "最近",
            "过去",
            "近",
            "last",
            "recent",
            "past",
            "新闻",
            "报道",
            "文章",
            "资讯",
            "动态",
            "消息",
            "news",
            "article",
            "articles",
            "coverage",
            "headline",
            "headlines",
            "updates",
            "什么",
            "有什么",
            "哪些",
            "最新",
            "today",
        }
        for c in re.findall(topic_pattern, text):
            if c.lower() in stop:
                continue
            if re.fullmatch(r"\d{1,3}", c):
                continue
            query = c
            break

    if not query and not has_explicit_query:
        return None
    if not has_explicit_query and query.lower() in {"什么", "有什么", "哪些", "news", "update", "updates"}:
        return None
    days = extract_days(text, default=21, maximum=365)
    sort = "heat_desc" if any(k in text or k in lower for k in ["热度", "heat", "points"]) else "time_desc"
    limit = extract_limit(text, default=8, maximum=30)

    if not query:
        return None
    return query, source, days, sort, limit


def extract_trend_request(user_message: str) -> tuple[str, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None
    lower = text.lower()
    trend_markers = ["趋势", "升温", "降温", "动量", "trend", "momentum", "heating", "cooling"]
    if not any(k in text or k in lower for k in trend_markers):
        return None
    if any(k in text or k in lower for k in ["时间线", "timeline", "对比", "比较", "格局", "landscape"]):
        return None

    days = extract_days(text, default=7, maximum=60)
    topic = ""
    m = re.search(
        r"^\s*([A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,24})\s*(?:最近|过去|近|last|recent|past)\s*\d{0,3}\s*(?:天|day|days)?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        topic = m.group(1).strip()
    m = re.search(r"(?:趋势|trend|momentum)\s*(?:of|for|关于)?\s*([A-Za-z][A-Za-z0-9._&/-]{1,39})", lower)
    if m and not topic:
        topic = m.group(1).strip()
    if not topic:
        m = re.search(
            r"([A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,24})\s*(?:的)?\s*(?:趋势|升温|降温|trend|momentum)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            topic = m.group(1).strip()

    if not topic:
        return None
    return topic, days


def extract_source_compare_request(user_message: str) -> tuple[str, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None
    lower = text.lower()
    compare_markers = ["对比", "比较", "差异", "区别", "compare", "difference", "vs", "versus"]
    if not any(k in text or k in lower for k in compare_markers):
        return None

    has_hn = any(k in lower for k in ["hackernews", "hacker news", "hn"])
    has_tc = any(k in lower for k in ["techcrunch", "tc"])
    has_source_context = any(
        k in text or k in lower
        for k in [
            "来源",
            "source",
            "渠道",
            "平台",
            "媒体",
            "社区",
            "community",
            "media",
            "两种来源",
            "两个来源",
        ]
    )
    has_generic_pair = (
        ("社区" in text and "媒体" in text)
        or ("community" in lower and "media" in lower)
    )
    if not ((has_hn and has_tc) or has_source_context or has_generic_pair):
        return None

    pair = extract_compare_pair(text)
    if pair and not ((has_hn or has_tc) or has_source_context or has_generic_pair):
        return None

    days = extract_days(text, default=14, maximum=90)
    topic = ""
    m = re.search(r"(?:对比|比较)\s*([A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{1,24})\s*(?:在|于)", text)
    if m:
        topic = m.group(1).strip()
    if not topic:
        m = re.search(
            r"(?:compare)\s*([A-Za-z][A-Za-z0-9._&/-]{1,39})\s*(?:across|between|in)",
            lower,
        )
        if m:
            topic = m.group(1).strip()
    if not topic:
        m = re.search(
            r"(?:来源|source|社区|媒体)\s*(?:上|中的|for|for the)?\s*([A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{1,24})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            topic = m.group(1).strip()
    if not topic:
        topic = "AI"
    return topic, days


def extract_fulltext_request(user_message: str) -> tuple[str, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None
    lower = text.lower()
    fulltext_markers = ["全文", "fulltext", "批量读取", "批量读", "deep read", "深读"]
    if not any(k in text or k in lower for k in fulltext_markers):
        return None

    query = ""
    m = re.search(r"(?:批量读取|批量读|全文读取|全文|fulltext|deep read)\s*([^\n]{1,120})", text, flags=re.IGNORECASE)
    if m:
        query = m.group(1).strip()
    if not query:
        query = text

    query = re.sub(
        r"\s*(?:并|并且|并请|and)\s*(?:总结|分析|提炼|说明|解释|summarize|analy[sz]e|explain).*$",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    query = re.sub(r"\s*(?:相关)?\s*全文.*$", "", query, flags=re.IGNORECASE).strip()
    if not query:
        query = text

    max_chars = 4000
    m_chars = re.search(r"(?:max_chars|最大|最多)\s*[:=]?\s*(\d{3,5})", text, flags=re.IGNORECASE)
    if m_chars:
        try:
            max_chars = max(800, min(12000, int(m_chars.group(1))))
        except Exception:
            max_chars = 4000
    return query, max_chars


_LANDSCAPE_ENTITY_ALIASES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "microsoft": "Microsoft",
    "meta": "Meta",
    "amazon": "Amazon",
    "aws": "Amazon",
    "nvidia": "NVIDIA",
    "apple": "Apple",
    "tesla": "Tesla",
    "tsmc": "TSMC",
    "intel": "Intel",
    "amd": "AMD",
    "crowdstrike": "CrowdStrike",
    "palo alto": "Palo Alto Networks",
    "palo alto networks": "Palo Alto Networks",
    "cloudflare": "Cloudflare",
    "cisco": "Cisco",
    "xai": "xAI",
    "谷歌": "Google",
    "微软": "Microsoft",
    "亚马逊": "Amazon",
    "英伟达": "NVIDIA",
    "苹果": "Apple",
    "特斯拉": "Tesla",
    "台积电": "TSMC",
    "英特尔": "Intel",
}


_COMPARE_STOP_TOKENS = {
    "对比",
    "比较",
    "差异",
    "区别",
    "vs",
    "versus",
    "和",
    "与",
    "and",
    "the",
    "a",
    "an",
    "一下",
    "请",
    "请问",
    "我想",
    "想知道",
}


_LANDSCAPE_STOP_TOPICS = {
    "global",
    "world",
    "today",
    "current",
    "landscape",
    "ecosystem",
    "当今",
    "当前",
    "全球",
    "世界",
    "现在",
    "目前",
    "tech",
    "technology",
    "科技",
    "技术",
    "科技行业",
    "技术行业",
    "科技领域",
    "技术领域",
}


def normalize_landscape_topic_candidate(raw: str) -> str:
    candidate = re.sub(r"\s+", " ", (raw or "").strip()).strip("：:，,。. ")
    candidate = re.sub(r"^(?:当今|当前|目前|全球|世界|现在|如今)+", "", candidate).strip("的之 ")
    return candidate


def normalize_compare_entity(raw: str) -> str:
    token = re.sub(r"\s+", " ", (raw or "").strip()).strip("：:，,。.!?！？()[]{}\"'`")
    token = re.sub(r"^(?:请|请问|对比一下|比较一下|对比|比较)+", "", token).strip()
    return token


def is_valid_compare_entity(token: str) -> bool:
    if not token:
        return False
    low = token.lower()
    if low in _COMPARE_STOP_TOKENS:
        return False
    if re.fullmatch(r"\d{1,3}", token):
        return False
    if len(token) < 2:
        return False
    return True


def extract_compare_pair(text: str) -> tuple[str, str] | None:
    topic_pattern = r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9 ._&/-]{1,39})"
    patterns = [
        rf"(?:对比|比较|差异|区别)\s*(?:一下|下)?\s*(?P<a>{topic_pattern})\s*(?:和|与|vs|VS|Vs|versus|and|&)\s*(?P<b>{topic_pattern})",
        rf"(?P<a>{topic_pattern})\s*(?:和|与|vs|VS|Vs|versus|and|&)\s*(?P<b>{topic_pattern})",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        a = normalize_compare_entity(m.group("a"))
        b = normalize_compare_entity(m.group("b"))
        if not (is_valid_compare_entity(a) and is_valid_compare_entity(b)):
            continue
        if a.lower() == b.lower():
            continue
        return a, b
    return None


def extract_landscape_topic(text: str, lower: str) -> str:
    if bool(re.search(r"(?<![a-z])ai(?![a-z])", lower)) or ("人工智能" in text) or ("大模型" in text) or ("llm" in lower):
        return "AI"

    if any(k in lower for k in ["business", "market", "finance", "commercial"]) or any(
        k in text for k in ["商业", "金融", "市场", "财经"]
    ):
        return "business"

    if any(k in lower for k in ["security", "cybersecurity", "cyber"]) or any(
        k in text for k in ["安全", "网络安全", "攻防", "威胁"]
    ):
        return "security"

    m = re.search(r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9 _/-]{1,20})\s*(?:领域|行业|赛道)?\s*(?:格局|版图|生态)", text)
    if m:
        normalized = normalize_landscape_topic_candidate(m.group(1))
        candidate = normalized.lower()
        if candidate and candidate not in _LANDSCAPE_STOP_TOPICS:
            return normalized

    m = re.search(r"(?:landscape|ecosystem)\s*(?:of|for)?\s*([A-Za-z][A-Za-z0-9 _/-]{2,24})", lower)
    if m:
        candidate = normalize_landscape_topic_candidate(m.group(1)).lower()
        if candidate and candidate not in _LANDSCAPE_STOP_TOPICS:
            return candidate

    return ""


def extract_landscape_request(user_message: str) -> tuple[str, int, list[str]] | None:
    text = (user_message or "").strip()
    if not text:
        return None

    lower = text.lower()
    strong_landscape_keywords = [
        "格局",
        "局势",
        "态势",
        "版图",
        "全局",
        "全貌",
        "宏观",
        "生态位",
        "阵营",
        "角色",
        "玩家",
        "谁主导",
        "landscape",
        "power structure",
    ]
    weak_landscape_keywords = ["生态", "ecosystem"]
    has_strong = any((k in text) or (k in lower) for k in strong_landscape_keywords)
    has_weak = any((k in text) or (k in lower) for k in weak_landscape_keywords)
    if not (has_strong or has_weak):
        return None
    if has_weak and not has_strong:
        weak_context = ["格局", "版图", "角色", "玩家", "主导", "竞争", "地位", "who leads", "dominant"]
        if not any((k in text) or (k in lower) for k in weak_context):
            return None
    if any((k in text) or (k in lower) for k in ["时间线", "timeline", "里程碑"]):
        return None

    days = extract_days(text, default=30, maximum=180)
    topic = extract_landscape_topic(text, lower)

    entities: list[str] = []
    seen: set[str] = set()
    for alias, canonical in _LANDSCAPE_ENTITY_ALIASES.items():
        if re.fullmatch(r"[A-Za-z0-9. ]+", alias):
            if not re.search(rf"\b{re.escape(alias)}\b", lower):
                continue
        else:
            if alias not in text:
                continue
        key = canonical.lower()
        if key not in seen:
            seen.add(key)
            entities.append(canonical)

    return topic, days, entities


def extract_compare_request(user_message: str) -> tuple[str, str, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None
    lower = text.lower()

    pair = extract_compare_pair(text)
    if pair is None:
        return None

    has_explicit_marker = any(k in lower for k in ["对比", "比较", "差异", "区别", "vs", "versus"])
    has_comparative_question = any(
        k in lower
        for k in [
            "谁更",
            "哪个更",
            "哪家更",
            "谁强",
            "高于",
            "低于",
            "more than",
            "less than",
            "better",
            "hotter",
            "stronger",
        ]
    )
    score = 1
    if has_explicit_marker:
        score += 2
    if has_comparative_question:
        score += 2
    if re.search(r"\b(?:vs|versus)\b", lower):
        score += 1

    if score < 3:
        return None

    topic_a, topic_b = pair
    days = extract_days(text, default=14, maximum=90)
    return topic_a, topic_b, days


def extract_timeline_request(user_message: str) -> tuple[str, int, int] | None:
    text = (user_message or "").strip()
    if not text:
        return None

    lower = text.lower()
    explicit_timeline_markers = ["timeline", "时间线", "里程碑", "大事记", "发展历程"]
    action_markers = [
        "动作",
        "动态",
        "动向",
        "进展",
        "更新",
        "事件",
        "发生了什么",
        "都做了什么",
        "moves",
        "actions",
        "updates",
        "developments",
    ]
    has_explicit_marker = any(k in lower for k in explicit_timeline_markers)
    has_recent_window = bool(re.search(r"(最近|过去|近|last|recent|past)\s*\d{0,3}\s*(天|day|days)?", lower))
    has_action_intent = any(k in lower for k in action_markers)

    if not (has_explicit_marker or (has_recent_window and has_action_intent)):
        return None

    days = extract_days(text, default=30, maximum=180)
    limit = extract_limit(text, default=12, maximum=40)

    topic_pattern = r"(?:[A-Za-z][A-Za-z0-9._&/-]{1,39}|[\u4e00-\u9fffA-Za-z0-9]{2,24})"
    patterns = [
        rf"(?:构建|生成|给我|做|列出|整理|build|make|create|show)\s+(?P<t>{topic_pattern})",
        rf"(?:最近|过去|近|last|recent|past)\s*\d{{0,3}}\s*(?:天|day|days)?\s*(?P<t>{topic_pattern})\s*(?:的)?\s*(?:动作|动态|动向|进展|更新|事件|moves?|actions?|updates?|developments?)",
        rf"(?P<t>{topic_pattern})\s*(?:最近|过去|近|last|recent|past)\s*\d{{0,3}}\s*(?:天|day|days)?\s*(?:的)?\s*(?:动作|动态|动向|进展|更新|事件|moves?|actions?|updates?|developments?)",
        rf"(?:最近|过去|近|last|recent|past)\s*(?P<t>{topic_pattern})\s*(?:的)?\s*(?:动作|动态|动向|进展|更新|事件|moves?|actions?|updates?|developments?)",
        rf"(?P<t>{topic_pattern})\s*(?:过去|最近|last)?\s*\d{{0,3}}\s*(?:天|day|days)?\s*(?:时间线|timeline)",
        rf"(?:时间线|timeline)\s*(?:关于|for)?\s*(?P<t>{topic_pattern})",
    ]

    topic = ""
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            topic = m.group("t").strip()
            break

    if not topic:
        stop = {
            "构建",
            "生成",
            "给我",
            "做",
            "列出",
            "整理",
            "时间线",
            "里程碑",
            "演进",
            "timeline",
            "build",
            "create",
            "show",
            "recent",
            "last",
            "最近",
            "过去",
            "天",
            "day",
            "days",
        }
        candidates = re.findall(topic_pattern, text)
        for c in candidates:
            lc = c.lower()
            if lc in stop:
                continue
            if re.fullmatch(r"\d{1,3}", c):
                continue
            topic = c
            break

    if not topic:
        return None
    topic = re.sub(r"(?:的|之)$", "", topic).strip()
    topic = re.sub(r"(?:'s)$", "", topic, flags=re.IGNORECASE).strip()
    return topic, days, limit
