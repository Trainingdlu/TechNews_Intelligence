"""Deterministic route and parameter extraction helpers."""

from __future__ import annotations

import re


_EN_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


_TOPIC_EN_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._#+&/-]{0,39}"
_TOPIC_MIXED_PATTERN = r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9 ._#+&/-]{0,39}"
_TOPIC_PATTERN = rf"(?:{_TOPIC_EN_PATTERN}|{_TOPIC_MIXED_PATTERN})"

_DURATION_NUMBER_PATTERN = (
    r"(?:\d{1,3}|[一二两三四五六七八九十百]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
)
_DURATION_UNIT_PATTERN = r"(?:天|日|周|星期|个月|月|day|days|week|weeks|month|months)"
_DURATION_PATTERN = rf"{_DURATION_NUMBER_PATTERN}\s*{_DURATION_UNIT_PATTERN}"


def _parse_cn_number(token: str) -> int | None:
    value = (token or "").strip()
    if not value:
        return None
    if value in _CN_DIGITS:
        return _CN_DIGITS[value]
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        if left:
            left_num = _CN_DIGITS.get(left)
            if left_num is None:
                return None
        else:
            left_num = 1
        if right:
            right_num = _CN_DIGITS.get(right)
            if right_num is None:
                return None
        else:
            right_num = 0
        return left_num * 10 + right_num
    if "百" in value:
        left, right = value.split("百", 1)
        left_num = _CN_DIGITS.get(left) if left else 1
        if left_num is None:
            return None
        if not right:
            return left_num * 100
        tail = _parse_cn_number(right)
        if tail is None:
            return None
        return left_num * 100 + tail
    return None


def _parse_number_token(token: str) -> int | None:
    raw = (token or "").strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw in _EN_NUMBER_WORDS:
        return _EN_NUMBER_WORDS[raw]
    return _parse_cn_number(raw)


def _days_from_unit(num: int, unit: str) -> int:
    u = (unit or "").lower()
    if u in {"day", "days", "天", "日"}:
        return num
    if u in {"week", "weeks", "周", "星期"}:
        return num * 7
    if u in {"month", "months", "月", "个月"}:
        return num * 30
    if u in {"year", "years", "年"}:
        return num * 365
    if u in {"hour", "hours", "小时"}:
        return max(1, (num + 23) // 24)
    return num


def count_timeline_items(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"^\s*\d+\.\s", text, flags=re.MULTILINE))


def extract_days(text: str, default: int, maximum: int) -> int:
    m = re.search(
        r"(?:最近|过去|近|last|recent|past)?\s*"
        r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|[一二两三四五六七八九十百]+)\s*"
        r"(天|日|周|星期|个月|月|年|小时|day|days|week|weeks|month|months|year|years|hour|hours)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return max(1, min(maximum, default))

    num = _parse_number_token(m.group(1))
    if num is None:
        return max(1, min(maximum, default))
    val = _days_from_unit(num, m.group(2))
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


_PLACEHOLDER_TOPIC_TOKENS = {
    "什么",
    "有什么",
    "有哪些",
    "啥",
    "有啥",
    "哪家",
    "哪个",
    "哪些",
    "最近",
    "过去",
    "现在",
    "动态",
    "动作",
    "新闻",
    "消息",
    "进展",
    "更新",
    "事件",
    "what",
    "which",
    "who",
    "how",
    "why",
    "latest",
    "recent",
    "updates",
    "news",
    "did",
    "do",
    "does",
    "has",
    "have",
    "been",
    "情况",
    "最近情况",
    "怎么样",
    "怎么了",
    "在忙什么",
    "都在做什么",
    "做什么",
    "忙什么",
}


def _clean_topic_candidate(token: str) -> str:
    cleaned = re.sub(r"\s+", " ", (token or "")).strip("：:，,。.!?！？()[]{}\"'` ")
    if not cleaned:
        return ""

    lead_patterns = [
        (
            r"^(?:那么|那|请问|请|给我|帮我|我想|我想看|我想知道|想看|想知道|"
            r"总结下|总结一下|分析下|分析一下|说说|聊聊|看下|看一下|看|查下|查一下|查|搜下|搜一下|搜|问下|问一下|问)+"
        ),
        r"^(?:what|how|did|do|does|has|have|been)\b\s*",
    ]
    for _ in range(4):
        prev = cleaned
        for pattern in lead_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == prev or not cleaned:
            break

    if not cleaned:
        return ""

    suffix_patterns = [
        rf"(?:最近|过去|近|last|recent|past)\s*(?:{_DURATION_NUMBER_PATTERN})?\s*(?:{_DURATION_UNIT_PATTERN})?\s*"
        r"(?:都|有|有什么|有哪些|有啥|what|which|any)?\s*(?:的)?\s*(?:最新)?\s*"
        r"(?:(?:大|重大|主要|关键|最新)?(?:情况|新闻|报道|文章|资讯|动态|消息|进展|更新|动作|动向|事件)|"
        r"在忙什么|都在做什么|在做什么|忙什么|怎么样|怎么了|"
        r"news|updates?|coverage|headline|headlines|moves?|actions?)?$",
        r"(?:(?:大|重大|主要|关键|最新)?(?:情况|新闻|报道|文章|资讯|动态|消息|进展|更新|动作|动向|事件)|"
        r"在忙什么|都在做什么|在做什么|忙什么|怎么样|怎么了|"
        r"news|updates?|coverage|headline|headlines|moves?|actions?)$",
        r"(?:有)?什么(?:(?:大|重大|主要|关键|最新)?(?:情况|新闻|报道|文章|资讯|动态|消息|进展|更新|动作|动向|事件)?)?$",
        r"(?:有)?哪些(?:(?:大|重大|主要|关键|最新)?(?:情况|新闻|报道|文章|资讯|动态|消息|进展|更新|动作|动向|事件)?)?$",
        r"(?:的)?最新$",
        rf"(?:最近|过去|近|last|recent|past)\s*(?:{_DURATION_NUMBER_PATTERN})?\s*(?:{_DURATION_UNIT_PATTERN})?$",
        r"(?:的)$",
    ]
    for _ in range(8):
        prev = cleaned
        for pattern in suffix_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip("：:，,。.!?！？")
        if cleaned == prev or not cleaned:
            break

    return cleaned.strip()


def _is_placeholder_topic(token: str) -> bool:
    normalized = re.sub(r"\s+", "", _clean_topic_candidate(token)).strip("：:，,。.!?！？")
    if len(normalized) < 2:
        return not bool(re.fullmatch(r"[A-Za-z0-9#+]", normalized))
    low = normalized.lower()
    if re.fullmatch(
        r"(?:\d{1,3}|[一二两三四五六七八九十百]+)"
        r"(?:天|日|周|星期|个月|月|年|小时|day|days|week|weeks|month|months|year|years|hour|hours)",
        low,
    ):
        return True
    if low in _PLACEHOLDER_TOPIC_TOKENS:
        return True
    if normalized in {"发生了什么", "都做了什么"}:
        return True
    if re.fullmatch(r"(?:有)?什么(?:(?:大|重大|主要|关键|最新)?(?:动态|动作|新闻|消息|进展|更新|事件)?)?", normalized):
        return True
    if re.fullmatch(r"(?:有)?哪些(?:动态|动作|新闻|消息|进展|更新|事件)?", normalized):
        return True
    return False


def _extract_recent_subject(text: str) -> str:
    recent_marker = r"(?:最近|过去|last|recent|past|(?<!最)近)"
    source_marker = r"(?:techcrunch|tc|hackernews|hacker\s+news|hn)"
    patterns = [
        rf"(?P<t>{_TOPIC_PATTERN})\s*(?:在|on|in)\s*{source_marker}\s*{recent_marker}\s*(?:{_DURATION_PATTERN})?\s*(?:都|有|有什么|有哪些|what|which|any)?\s*(?:的)?\s*(?:新闻|报道|文章|资讯|动态|消息|news|article|articles|coverage|headline|headlines|updates?)",
        rf"(?P<t>{_TOPIC_PATTERN})\s*{recent_marker}\s*(?:{_DURATION_PATTERN})?\s*(?:都|有|有什么|有哪些|what|which|any)?\s*(?:的)?\s*(?:新闻|报道|文章|资讯|动态|消息|news|article|articles|coverage|headline|headlines|updates?)",
        rf"{recent_marker}\s*(?:{_DURATION_PATTERN})?\s*(?P<t>{_TOPIC_PATTERN})\s*(?:都|有|有什么|有哪些|what|which|any)?\s*(?:的)?\s*(?:新闻|报道|文章|资讯|动态|消息|news|article|articles|coverage|headline|headlines|updates?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = _clean_topic_candidate(m.group("t"))
        if _is_placeholder_topic(candidate):
            continue
        return candidate
    return ""


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
        "情况",
        "怎么样",
        "怎么了",
        "在忙什么",
        "都在做什么",
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
    has_subject_content = False
    m_subject_content = re.search(
        rf"(?P<t>{_TOPIC_PATTERN})\s*(?:的)?\s*(?:新闻|报道|文章|资讯|动态|消息|进展|更新|news|article|articles|coverage|headline|headlines|updates?)",
        text,
        flags=re.IGNORECASE,
    )
    if m_subject_content:
        candidate = _clean_topic_candidate(m_subject_content.group("t"))
        has_subject_content = bool(candidate) and not _is_placeholder_topic(candidate)
    source = extract_source_label(text)
    has_explicit_query = any(k in text or k in lower for k in query_markers)
    has_content_intent = any(k in text or k in lower for k in content_markers)
    has_filter_intent = any(k in text or k in lower for k in filter_markers) or (source != "all")
    if not (
        has_explicit_query
        or has_filter_intent
        or (has_content_intent and (has_recent_window or has_subject_content))
    ):
        return None
    if any(k in text or k in lower for k in ["全文", "fulltext", "批量读取", "批量读", "deep read"]):
        return None

    query = ""
    m = re.search(r"(?:检索|搜索|查询|query|search)\s*[:：]?\s*([^\n，,。]{1,40})", text, flags=re.IGNORECASE)
    if m:
        query = m.group(1).strip()
        query = re.split(r"[，,。]|(?:来源|source|最近|过去|sort|按)", query, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        query = _clean_topic_candidate(query)

    if not query:
        m = re.search(rf"(?:关于|有关|聊聊|look up|about)\s*({_TOPIC_PATTERN})", text, flags=re.IGNORECASE)
        if m:
            query = _clean_topic_candidate(m.group(1))

    if not query:
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
        for c in re.findall(_TOPIC_PATTERN, text):
            if c.lower() in stop:
                continue
            if _is_placeholder_topic(c):
                continue
            if re.fullmatch(r"\d{1,3}", c):
                continue
            query = _clean_topic_candidate(c)
            break

    # Prefer a concrete subject in "X 最近有什么新闻/动态" style questions.
    if not has_explicit_query and (
        not query
        or _is_placeholder_topic(query)
        or bool(re.search(r"(最近|过去|近|新闻|报道|文章|资讯|动态|消息|news|updates?)", query, flags=re.IGNORECASE))
    ):
        recent_subject = _extract_recent_subject(text)
        if recent_subject:
            query = recent_subject

    if not query and not has_explicit_query:
        return None
    query = _clean_topic_candidate(query)
    if _is_placeholder_topic(query):
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
        rf"^\s*({_TOPIC_PATTERN})\s*(?:最近|过去|近|last|recent|past)\s*\d{{0,3}}\s*(?:天|day|days)?",
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
            rf"({_TOPIC_PATTERN})\s*(?:的)?\s*(?:趋势|升温|降温|trend|momentum)",
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
    m = re.search(rf"(?:对比|比较)\s*({_TOPIC_PATTERN})\s*(?:在|于)", text)
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
            rf"(?:来源|source|社区|媒体)\s*(?:上|中的|for|for the)?\s*({_TOPIC_PATTERN})",
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
    # Prefer subject-before-marker forms: "OpenAI全文并总结", "帮我看OpenAI全文".
    m = re.search(
        r"([^\n，,。!?！？]{1,120}?)(?:相关)?\s*(?:全文(?:读取)?|fulltext|deep read|深读)(?:\s|$|并|并且|并请|and)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        query = m.group(1).strip()
    if not query:
        m = re.search(r"(?:批量读取|批量读|全文读取|全文|fulltext|deep read|深读)\s*([^\n]{1,120})", text, flags=re.IGNORECASE)
        if m:
            query = m.group(1).strip()
    if not query:
        query = text
    original_query = query

    query = re.sub(
        r"^(?:请|请问|帮我|给我|我想|想|麻烦|请你|请帮我)?\s*"
        r"(?:批量读取|批量读|读取|读|看下|看一下|看|查看|搜下|搜一下|搜索|query|search|read|show|fetch)\s*",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    query = re.sub(
        r"\s*(?:并|并且|并请|and)\s*(?:总结|分析|提炼|说明|解释|summarize|analy[sz]e|explain).*$",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    tail_cleanup_patterns = [
        r"\s*(?:相关)?\s*全文(?:读取)?\s*(?:并|并且|并请|and)?\s*(?:总结|分析|提炼|说明|解释|summarize|analy[sz]e|explain)?\s*$",
        r"\s*(?:相关)?\s*fulltext\s*(?:and\s*)?(?:summarize|analy[sz]e|explain)?\s*$",
        r"\s*(?:并|并且|并请|and)\s*(?:总结|分析|提炼|说明|解释|summarize|analy[sz]e|explain).*$",
        r"\s*相关\s*$",
    ]
    for _ in range(3):
        prev = query
        for pattern in tail_cleanup_patterns:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE).strip()
        if query == prev or not query:
            break
    if not query:
        if str(original_query).strip().lower() in {"全文", "fulltext"}:
            query = str(original_query).strip()
        else:
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
    patterns = [
        rf"(?:对比|比较|差异|区别)\s*(?:一下|下)?\s*(?P<a>{_TOPIC_PATTERN})\s*(?:和|与|vs|VS|Vs|versus|and|&)\s*(?P<b>{_TOPIC_PATTERN})",
        rf"(?P<a>{_TOPIC_PATTERN})\s*(?:和|与|vs|VS|Vs|versus|and|&)\s*(?P<b>{_TOPIC_PATTERN})",
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
        "干了什么",
        "做了什么",
        "发生了什么",
        "发生什么",
        "都做了什么",
        "moves",
        "actions",
        "updates",
        "developments",
        "what happened",
        "what did",
        "在忙什么",
        "忙什么",
        "都在做什么",
        "在做什么",
        "怎么样",
        "怎么了",
        "情况",
    ]
    has_explicit_marker = any(k in lower for k in explicit_timeline_markers)
    has_recent_window = bool(
        re.search(
            r"(最近|过去|近|last|recent|past)\s*"
            r"(?:\d{0,3}|[一二两三四五六七八九十百]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)?\s*"
            r"(?:天|日|周|星期|个月|月|day|days|week|weeks|month|months)?",
            lower,
        )
    )
    has_action_intent = any(k in lower for k in action_markers)

    if not (has_explicit_marker or (has_recent_window and has_action_intent)):
        return None

    days = extract_days(text, default=30, maximum=180)
    limit = extract_limit(text, default=12, maximum=40)

    topic_pattern = _TOPIC_PATTERN
    duration_required_pattern = _DURATION_PATTERN
    recent_marker = r"(?:最近|过去|last|recent|past|(?<!最)近)"
    action_tail_pattern = (
        r"(?:(?:大|重大|主要|关键|最新)?(?:动作|动态|动向|进展|更新|事件)|"
        r"干了什么|做了什么|发生了什么|发生什么|在忙什么|忙什么|都在做什么|在做什么|怎么样|怎么了|情况|"
        r"moves?|actions?|updates?|developments?|what happened|what did)"
    )
    patterns = [
        rf"(?:构建|生成|给我|做|列出|整理|build|make|create|show)\s+(?P<t>{topic_pattern})",
        rf"{recent_marker}\s*{duration_required_pattern}\s*(?P<t>{topic_pattern})\s*(?:都|有|有什么|有哪些|what|which|any)?\s*(?:的)?\s*{action_tail_pattern}",
        rf"{recent_marker}\s*(?P<t>{topic_pattern})\s*(?:都|有|有什么|有哪些|what|which|any)?\s*(?:的)?\s*{action_tail_pattern}",
        rf"(?P<t>{topic_pattern})\s*{recent_marker}\s*(?:{duration_required_pattern})?\s*(?:都|有|有什么|有哪些|what|which|any)?\s*(?:的)?\s*{action_tail_pattern}",
        rf"(?P<t>{topic_pattern})\s*(?:过去|最近|last)?\s*(?:{duration_required_pattern})?\s*(?:时间线|timeline)",
        rf"(?:时间线|timeline)\s*(?:关于|for)?\s*(?P<t>{topic_pattern})",
    ]

    topic = ""
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            candidate = _clean_topic_candidate(m.group("t"))
            if _is_placeholder_topic(candidate):
                continue
            topic = candidate
            break

    if not topic:
        m = re.search(
            r"(?:最近|过去|近|last|recent|past)?\s*"
            rf"(?P<t>{_TOPIC_PATTERN})\s*"
            r"(?:领域|行业|赛道)\s*(?:的)?\s*(?:重大|重要|关键)?\s*(?:产品|事件|动态)?\s*"
            r"(?:时间线|timeline)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            topic = m.group("t").strip()

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
            "重大",
            "重要",
            "关键",
            "产品",
            "事件",
            "动态",
            "什么",
            "是什么",
        }
        candidates = re.findall(topic_pattern, text)
        for c in candidates:
            lc = c.lower()
            if lc in stop:
                continue
            if _is_placeholder_topic(c):
                continue
            if re.fullmatch(r"\d{1,3}", c):
                continue
            topic = _clean_topic_candidate(c)
            break

    topic = _clean_topic_candidate(topic)

    if not topic:
        if bool(re.search(r"(?<![a-z])ai(?![a-z])", lower)) or ("人工智能" in text) or ("大模型" in text) or ("llm" in lower):
            topic = "AI"
        else:
            return None
    low_topic = topic.lower()
    if (
        bool(re.search(r"(?<![a-z])ai(?![a-z])", low_topic))
        or topic in {"人工智能", "大模型", "模型", "LLM", "llm"}
    ):
        topic = "AI"
    if _is_placeholder_topic(topic) or topic.lower() in {"重大", "重要", "关键", "产品", "事件", "动态", "什么", "是什么"}:
        return None
    topic = re.sub(r"(?:的|之)$", "", topic).strip()
    topic = re.sub(r"(?:'s)$", "", topic, flags=re.IGNORECASE).strip()
    return topic, days, limit


def extract_timeline_request_with_confidence(user_message: str) -> tuple[str, int, int, float] | None:
    req = extract_timeline_request(user_message)
    if req is None:
        return None

    topic, days, limit = req
    text = (user_message or "").strip()
    lower = text.lower()

    explicit_markers = ["timeline", "时间线", "里程碑", "大事记", "发展历程"]
    strong_action_markers = [
        "动作",
        "动态",
        "动向",
        "进展",
        "更新",
        "事件",
        "干了什么",
        "做了什么",
        "发生了什么",
        "发生什么",
        "moves",
        "actions",
        "updates",
        "developments",
        "what did",
        "what happened",
    ]
    weak_action_markers = ["在忙什么", "忙什么", "都在做什么", "在做什么", "怎么样", "怎么了", "情况"]

    has_explicit = any(k in text or k in lower for k in explicit_markers)
    has_recent_window = bool(
        re.search(
            r"(最近|过去|近|last|recent|past)\s*"
            r"(?:\d{0,3}|[一二两三四五六七八九十百]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)?\s*"
            r"(?:天|日|周|星期|个月|月|day|days|week|weeks|month|months)?",
            text,
            flags=re.IGNORECASE,
        )
    )
    has_strong_action = any(k in text or k in lower for k in strong_action_markers)
    has_weak_action = any(k in text or k in lower for k in weak_action_markers)

    confidence = 0.55
    if has_explicit:
        confidence = 0.95
    elif has_recent_window and has_strong_action:
        confidence = 0.85
    elif has_recent_window and has_weak_action:
        confidence = 0.68
    elif has_strong_action:
        confidence = 0.62

    if _is_placeholder_topic(topic):
        confidence = min(confidence, 0.35)
    if len(topic.strip()) > 30:
        confidence = min(confidence, 0.55)
    if re.search(r"(?:what|which|how|did|do|does|has|have|最近|过去|情况|怎么样|怎么了)", topic, flags=re.IGNORECASE):
        confidence = min(confidence, 0.5)

    return topic, days, limit, max(0.0, min(1.0, confidence))
