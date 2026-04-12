"""Clarification helpers for evidence-insufficient HITL flow."""

from __future__ import annotations

import re
from datetime import date
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

CLARIFICATION_KIND = "clarification_required"
CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CLARIFICATION_REASON_AMBIGUOUS_SCOPE = "ambiguous_scope"
CLARIFICATION_REASON_SOURCE_CONFLICT = "source_conflict"

_TIME_RANGE_RE = re.compile(
    r"(?:近|最近|过去)\s*\d+\s*天|近一周|最近一周|近一个月|最近一个月|"
    r"last\s*\d+\s*days?|past\s*(?:week|month)|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
    re.IGNORECASE,
)
_SOURCE_RANGE_RE = re.compile(
    r"hackernews|hacker\s*news|techcrunch|tech\s*crunch|"
    r"\bhn\b|\btc\b|全部来源|全来源|all\s+sources?",
    re.IGNORECASE,
)
_ANALYSIS_DIM_RE = re.compile(
    r"趋势|对比|比较|时间线|格局|"
    r"trend|compare|comparison|timeline|landscape|outlook",
    re.IGNORECASE,
)
_MULTI_SOURCE_SCOPE_HINT_RE = re.compile(
    r"多来源|跨来源|不同来源|来源范围|source\s+scope|source\s+mix|cross-?source",
    re.IGNORECASE,
)
_GENERIC_TOPIC_RE = re.compile(
    r"科技新闻|科技动态|行业动态|最近有什么|发生了什么|总结一下|分析一下|"
    r"latest\s+news|what\s+happened|summari[sz]e",
    re.IGNORECASE,
)
_WIDE_QUERY_RE = re.compile(
    r"行业|全局|总体|宏观|盘点|综述|总结|全景|格局|"
    r"industry|market|overall|overview|landscape|big\s+picture|broad",
    re.IGNORECASE,
)
_GENERIC_TOPIC_TOKENS = {
    "ai",
    "news",
    "trend",
    "analysis",
    "recent",
    "latest",
    "update",
    "updates",
    "tech",
    "technology",
}
_GENERIC_ENTITY_TOKENS = {
    "today",
    "latest",
    "recent",
    "analysis",
    "trend",
    "market",
    "industry",
    "source",
    "sources",
    "news",
    "summary",
}
_SOURCE_ALIAS_MAP = {
    "hackernews": "HackerNews",
    "hacker news": "HackerNews",
    "hn": "HackerNews",
    "techcrunch": "TechCrunch",
    "tech crunch": "TechCrunch",
    "tc": "TechCrunch",
}
_POSITIVE_WORDS_RE = re.compile(
    r"增长|上升|强劲|乐观|改善|领先|看好|利好|突破|正向|"
    r"grow(?:th|ing)?|surge|bullish|positive|strong|improv(?:e|ed|ing)|leading|optimistic",
    re.IGNORECASE,
)
_NEGATIVE_WORDS_RE = re.compile(
    r"下滑|下降|放缓|承压|担忧|风险|悲观|负面|不确定|分歧|谨慎|保守|"
    r"declin(?:e|ing)|slowdown|bearish|negative|risk|concern|cautious|uncertain",
    re.IGNORECASE,
)
_CONFLICT_WORDS_RE = re.compile(
    r"分歧|相反|冲突|不一致|矛盾|mixed\s+signals?|conflict|contradict|disagree|diverg",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b")
_CJK_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_KNOWN_ENTITIES = (
    "OpenAI",
    "Anthropic",
    "Google",
    "Microsoft",
    "Meta",
    "Amazon",
    "Apple",
    "NVIDIA",
    "Tesla",
    "xAI",
    "DeepSeek",
    "字节跳动",
    "阿里巴巴",
    "腾讯",
    "百度",
    "英伟达",
    "谷歌",
    "微软",
    "苹果",
    "特斯拉",
)


@dataclass
class ClarificationPayload:
    """Structured clarification payload returned to transport layers."""

    question: str
    hints: list[str] = field(default_factory=list)
    reason: str = CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE
    kind: str = CLARIFICATION_KIND
    original_question: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "question": self.question,
            "hints": list(self.hints),
            "original_question": self.original_question,
        }


class ClarificationRequiredError(Exception):
    """Control-flow exception for HITL clarification step."""

    def __init__(self, clarification: ClarificationPayload):
        super().__init__(clarification.question)
        self.clarification = clarification
        self.code = "clarification_required"


def infer_clarification_reason(user_message: str) -> str:
    text = str(user_message or "").strip()
    if _MULTI_SOURCE_SCOPE_HINT_RE.search(text) and not _SOURCE_RANGE_RE.search(text):
        return CLARIFICATION_REASON_AMBIGUOUS_SCOPE
    return CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE


def build_clarification_payload(
    user_message: str,
    *,
    reason: str | None = None,
    context: dict[str, Any] | None = None,
) -> ClarificationPayload:
    text = str(user_message or "").strip()
    ctx = context if isinstance(context, dict) else {}
    reason_value = str(reason or infer_clarification_reason(text)).strip().lower()
    if reason_value not in {
        CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE,
        CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
        CLARIFICATION_REASON_SOURCE_CONFLICT,
    }:
        reason_value = CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE
    scope = _query_scope_signals(text)

    missing_hints: list[str] = []
    if not _TIME_RANGE_RE.search(text):
        missing_hints.append("时间范围：例如最近 7 天或最近 30 天")
    if not _SOURCE_RANGE_RE.search(text):
        missing_hints.append("来源范围：例如 HackerNews、TechCrunch 或全部来源")
    if not _has_specific_topic(text):
        missing_hints.append("具体实体/主题：例如 OpenAI、Google、AI 芯片")
    if not _ANALYSIS_DIM_RE.search(text):
        missing_hints.append("分析维度：趋势 / 对比 / 时间线 / 格局")
    if not missing_hints:
        missing_hints.append("请补充时间范围、来源范围、主题或分析维度中的任意一项")

    source_labels = _normalize_source_labels(ctx.get("source_labels", []))
    source_clause = "、".join(source_labels[:3]) if source_labels else "多个来源"
    time_span_days = _coerce_int(ctx.get("time_span_days"))
    entities = _normalize_entity_candidates(ctx.get("entity_candidates", []))
    focus_entity = entities[0] if entities else "一个具体实体（如 OpenAI）"
    conflict_summary = str(ctx.get("conflict_summary", "")).strip()

    if reason_value == CLARIFICATION_REASON_SOURCE_CONFLICT:
        if conflict_summary:
            question = (
                f"当前多来源结论存在冲突：{conflict_summary}。"
                "为了降低综合偏差，你希望我如何收敛范围？"
                "例如“仅 TechCrunch + 最近 14 天 + 做趋势”或“保留双来源但只做时间线对比”。"
            )
        else:
            question = (
                f"当前 {source_clause} 对同一主题结论存在分歧。"
                "请先确认分析范围：是缩小来源、缩短时间窗口，还是指定分析维度（趋势/对比/时间线/格局）？"
            )
        hints = [
            "来源范围：只看 HackerNews / 只看 TechCrunch / 保留双来源对比",
            "时间范围：例如最近 7 天、14 天或 30 天",
            f"主题实体：建议指定 {focus_entity}",
            "分析维度：趋势 / 对比 / 时间线 / 格局",
        ]
    elif reason_value == CLARIFICATION_REASON_AMBIGUOUS_SCOPE:
        span_hint = f"时间跨度约 {time_span_days} 天" if time_span_days is not None and time_span_days > 0 else "时间范围尚未收敛"
        entity_hint = f"主题涉及较广（如 {', '.join(entities[:3])}）" if entities else "主题跨度较大"
        question = (
            f"当前候选证据覆盖 {source_clause}，{span_hint}，且{entity_hint}。"
            "为避免范围过宽导致误综合，请确认一个更具体的分析范围。"
            "你更希望看最近 7 天还是 30 天？聚焦哪个来源和实体？"
        )
        hints = [
            "时间范围：例如最近 7 天 / 最近 30 天",
            "来源范围：例如仅 HackerNews、仅 TechCrunch 或双来源对比",
            f"具体主题：建议指定 {focus_entity}",
            "分析维度：趋势 / 对比 / 时间线 / 格局",
        ]
    else:
        question = (
            "目前证据不足，先补充一个更具体的分析范围后我再继续。"
            "你可以补充时间范围、来源范围、具体主题，或想看的分析维度（趋势/对比/时间线/格局）。"
        )
        hints = missing_hints

    return ClarificationPayload(
        reason=reason_value,
        question=question,
        hints=hints,
        original_question=text,
    )


def resolve_user_message_with_history_clarification(
    history: list[dict] | None,
    user_message: str,
) -> tuple[str, ClarificationPayload | None]:
    """Merge original question + clarification follow-up when pending exists."""
    pending = get_pending_clarification(history)
    if pending is None:
        return user_message, None

    original = pending.original_question.strip() or _find_previous_user_message(history)
    addition = str(user_message or "").strip()
    if not original or not addition:
        return user_message, pending

    merged = (
        f"原问题：{original}\n"
        f"用户补充澄清：{addition}\n"
        "请基于原问题与补充范围重新检索，并给出有证据支撑的分析结论。"
    )
    return merged, pending


def build_clarification_history_item(clarification: dict[str, Any] | ClarificationPayload) -> dict[str, Any]:
    payload = (
        clarification.to_dict()
        if isinstance(clarification, ClarificationPayload)
        else _coerce_clarification_payload(clarification).to_dict()
    )
    return {
        "role": "model",
        "kind": CLARIFICATION_KIND,
        "parts": [{"text": payload.get("question", "")}],
        "clarification": payload,
    }


def get_pending_clarification(history: list[dict] | None) -> ClarificationPayload | None:
    if not history:
        return None
    for idx in range(len(history) - 1, -1, -1):
        item = history[idx]
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role != "model":
            continue
        payload = None
        clarification = item.get("clarification")
        if isinstance(clarification, dict):
            payload = _coerce_clarification_payload(clarification)
        elif str(item.get("kind", "")).strip().lower() == CLARIFICATION_KIND:
            payload = ClarificationPayload(
                reason=CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE,
                question=_extract_message_text(item),
                hints=[],
                original_question=_find_previous_user_message(history[:idx]),
            )

        if payload is None:
            return None
        if not payload.original_question:
            payload.original_question = _find_previous_user_message(history[:idx])
        return payload
    return None


def _coerce_clarification_payload(payload: dict[str, Any]) -> ClarificationPayload:
    question = str(payload.get("question", "")).strip()
    hints_raw = payload.get("hints", [])
    hints: list[str] = []
    if isinstance(hints_raw, list):
        hints = [str(item).strip() for item in hints_raw if str(item).strip()]
    reason = str(payload.get("reason", CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE)).strip().lower()
    if reason not in {
        CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE,
        CLARIFICATION_REASON_AMBIGUOUS_SCOPE,
        CLARIFICATION_REASON_SOURCE_CONFLICT,
    }:
        reason = CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE
    original_question = str(payload.get("original_question", "")).strip()
    if not question:
        question = "请补充分析范围后我再继续。"
    return ClarificationPayload(
        kind=CLARIFICATION_KIND,
        reason=reason,
        question=question,
        hints=hints,
        original_question=original_question,
    )


def _extract_message_text(item: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for part in item.get("parts", []) if isinstance(item, dict) else []:
        if isinstance(part, dict):
            txt = str(part.get("text", "")).strip()
            if txt:
                text_parts.append(txt)
    return "\n".join(text_parts).strip()


def _find_previous_user_message(history: list[dict] | None) -> str:
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role != "user":
            continue
        text = _extract_message_text(item)
        if text:
            return text
    return ""


def _has_specific_topic(user_message: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    if _GENERIC_TOPIC_RE.search(text):
        return False

    english_tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,}", text)]
    if any(token not in _GENERIC_TOPIC_TOKENS for token in english_tokens):
        return True

    cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    generic_chunks = {"最近", "新闻", "动态", "分析", "趋势", "对比", "时间线", "格局", "科技", "人工智能"}
    for chunk in cjk_chunks:
        if chunk not in generic_chunks:
            return True
    return False


def detect_scope_or_conflict_reason(
    *,
    user_message: str,
    candidate_answer: str,
    valid_urls: list[str] | set[str],
    tool_calls: set[str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Detect whether evidence is too broad/conflicting to answer directly."""
    urls = _normalize_urls(valid_urls)
    if not urls:
        return None, {}

    scope = _query_scope_signals(user_message)
    source_labels = _source_labels_from_urls(urls)
    time_span_days = _extract_time_span_days(candidate_answer)
    entity_candidates = _extract_entity_candidates(candidate_answer)
    context: dict[str, Any] = {
        "source_labels": source_labels,
        "source_count": len(source_labels),
        "url_count": len(urls),
        "time_span_days": time_span_days,
        "entity_candidates": entity_candidates,
        "tool_calls": sorted(tool_calls or set()),
    }

    conflict_hit, conflict_summary = _detect_source_conflict(
        user_message=user_message,
        candidate_answer=candidate_answer,
        source_labels=source_labels,
        scope=scope,
    )
    if conflict_hit:
        if conflict_summary:
            context["conflict_summary"] = conflict_summary
        return CLARIFICATION_REASON_SOURCE_CONFLICT, context

    if _detect_ambiguous_scope(scope=scope, source_labels=source_labels, url_count=len(urls), time_span_days=time_span_days, entity_span=len(entity_candidates), user_message=user_message):
        return CLARIFICATION_REASON_AMBIGUOUS_SCOPE, context

    return None, context


def _normalize_urls(valid_urls: list[str] | set[str]) -> list[str]:
    dedup: list[str] = []
    seen: set[str] = set()
    iterable = valid_urls if isinstance(valid_urls, list) else list(valid_urls)
    for raw in iterable:
        url = str(raw or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        dedup.append(url)
    return dedup


def _normalize_source_labels(labels: list[str] | tuple[str, ...] | Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(labels, (list, tuple)):
        return out
    for label in labels:
        value = str(label or "").strip()
        if not value:
            continue
        key = value.lower()
        canonical = _SOURCE_ALIAS_MAP.get(key, value)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _normalize_entity_candidates(entities: list[str] | tuple[str, ...] | Any) -> list[str]:
    if not isinstance(entities, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in entities:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _source_labels_from_urls(urls: list[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for url in urls:
        label = _source_label_from_url(url)
        if not label:
            continue
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _source_label_from_url(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower().strip()
    except Exception:
        host = ""
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]

    if "techcrunch.com" in host:
        return "TechCrunch"
    if "news.ycombinator.com" in host or host.endswith("ycombinator.com"):
        return "HackerNews"

    parts = [part for part in host.split(".") if part]
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net"}:
        root = parts[-3]
    elif len(parts) >= 2:
        root = parts[-2]
    else:
        root = parts[0]
    root = root.strip("-_")
    if not root:
        return host
    return root[:1].upper() + root[1:]


def _query_scope_signals(user_message: str) -> dict[str, Any]:
    text = str(user_message or "").strip()
    has_time = bool(_TIME_RANGE_RE.search(text))
    has_source = bool(_SOURCE_RANGE_RE.search(text))
    has_topic = _has_specific_topic(text)
    has_dimension = bool(_ANALYSIS_DIM_RE.search(text))
    specified_count = sum(1 for hit in (has_time, has_source, has_topic, has_dimension) if hit)
    broad_query = bool(_GENERIC_TOPIC_RE.search(text) or _WIDE_QUERY_RE.search(text))
    if not broad_query and specified_count <= 1:
        broad_query = len(text) <= 28
    return {
        "has_time": has_time,
        "has_source": has_source,
        "has_topic": has_topic,
        "has_dimension": has_dimension,
        "specified_count": specified_count,
        "broad_query": broad_query,
    }


def _detect_ambiguous_scope(
    *,
    scope: dict[str, Any],
    source_labels: list[str],
    url_count: int,
    time_span_days: int | None,
    entity_span: int,
    user_message: str,
) -> bool:
    specified = int(scope.get("specified_count", 0) or 0)
    if specified >= 3:
        return False

    dispersion_score = 0
    if len(source_labels) >= 2 and not bool(scope.get("has_source")):
        dispersion_score += 1
    if url_count >= 8:
        dispersion_score += 1
    if time_span_days is not None and time_span_days >= 45 and not bool(scope.get("has_time")):
        dispersion_score += 1
    if entity_span >= 5 and not bool(scope.get("has_topic")):
        dispersion_score += 1

    broad_query = bool(scope.get("broad_query"))
    if broad_query and dispersion_score >= 2 and specified <= 2:
        return True
    if dispersion_score >= 3 and specified <= 2:
        return True

    # Extra guard: broad analysis requests without clear scope, multi-source + many urls.
    if (
        len(source_labels) >= 2
        and url_count >= 10
        and not bool(scope.get("has_time"))
        and not bool(scope.get("has_source"))
        and _WIDE_QUERY_RE.search(str(user_message or ""))
    ):
        return True
    return False


def _detect_source_conflict(
    *,
    user_message: str,
    candidate_answer: str,
    source_labels: list[str],
    scope: dict[str, Any],
) -> tuple[bool, str]:
    if len(source_labels) < 2:
        return False, ""

    text = str(candidate_answer or "")
    if not text.strip():
        return False, ""

    polarity = _source_polarity_scores(text, source_labels)
    positive_sources = [src for src, score in polarity.items() if score >= 1]
    negative_sources = [src for src, score in polarity.items() if score <= -1]

    divergence = bool(positive_sources and negative_sources)
    conflict_words_hit = bool(_CONFLICT_WORDS_RE.search(text))
    explicit_compare = bool(re.search(r"对比|比较|compare|comparison|vs", str(user_message or ""), flags=re.IGNORECASE))

    if not divergence and not conflict_words_hit:
        return False, ""

    # If user already constrained very tightly, keep normal answer path.
    if (
        bool(scope.get("has_source"))
        and bool(scope.get("has_time"))
        and bool(scope.get("has_topic"))
        and (bool(scope.get("has_dimension")) or explicit_compare)
    ):
        return False, ""

    if divergence:
        summary = f"{positive_sources[0]} 偏正向，而 {negative_sources[0]} 更偏谨慎"
        return True, summary
    return True, "不同来源对同一主题的判断存在明显分歧"


def _source_polarity_scores(text: str, source_labels: list[str]) -> dict[str, int]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    score_map = {source: 0 for source in source_labels}

    for line in lines:
        mentioned = _sources_mentioned_in_line(line, source_labels)
        if not mentioned:
            continue
        pos_hits = len(_POSITIVE_WORDS_RE.findall(line))
        neg_hits = len(_NEGATIVE_WORDS_RE.findall(line))
        line_score = pos_hits - neg_hits
        if line_score == 0 and _CONFLICT_WORDS_RE.search(line):
            line_score = -1
        if line_score == 0:
            continue
        for source in mentioned:
            score_map[source] += line_score
    return score_map


def _sources_mentioned_in_line(line: str, source_labels: list[str]) -> list[str]:
    lowered = line.lower()
    matched: list[str] = []
    for label in source_labels:
        canonical = str(label or "").strip()
        if not canonical:
            continue
        aliases = _source_aliases(canonical)
        if any(alias in lowered for alias in aliases):
            matched.append(canonical)
    return matched


def _source_aliases(source_label: str) -> list[str]:
    label = str(source_label or "").strip().lower()
    if label == "hackernews":
        return ["hackernews", "hacker news", "hn", "news.ycombinator.com"]
    if label == "techcrunch":
        return ["techcrunch", "tech crunch", "tc"]
    return [label]


def _extract_time_span_days(text: str) -> int | None:
    dates = _extract_dates(text)
    if len(dates) < 2:
        return None
    minimum = min(dates)
    maximum = max(dates)
    return max(0, (maximum - minimum).days)


def _extract_dates(text: str) -> list[date]:
    out: list[date] = []
    for regex in (_ISO_DATE_RE, _CJK_DATE_RE):
        for match in regex.finditer(str(text or "")):
            dt = _safe_parse_date(match.group(1), match.group(2), match.group(3))
            if dt is not None:
                out.append(dt)
    return out


def _safe_parse_date(y: str, m: str, d: str) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def _extract_entity_candidates(text: str) -> list[str]:
    message = str(text or "")
    found: list[str] = []
    seen: set[str] = set()

    for entity in _KNOWN_ENTITIES:
        if re.search(re.escape(entity), message, flags=re.IGNORECASE):
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                found.append(entity)

    for token in re.findall(r"\b[A-Z][A-Za-z0-9+._-]{2,}\b", message):
        key = token.lower()
        if key in _GENERIC_ENTITY_TOKENS:
            continue
        if key in seen:
            continue
        seen.add(key)
        found.append(token)
        if len(found) >= 12:
            return found
    return found


def _coerce_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed
