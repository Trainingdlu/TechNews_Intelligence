"""Clarification helpers for evidence-insufficient HITL flow."""

from __future__ import annotations

import re
from datetime import date
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
from .core.intent import (
    classify_tool_profile,
    classify_user_intent,
    extract_user_intent_text as _extract_user_intent_text,
    has_explicit_conflict_request,
    is_analysis_heavy_intent,
    is_conflict_resolution_intent,
    is_roundup_listing_intent,
)

CLARIFICATION_KIND = "clarification_required"
CLARIFICATION_REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CLARIFICATION_REASON_AMBIGUOUS_SCOPE = "ambiguous_scope"
CLARIFICATION_REASON_SOURCE_CONFLICT = "source_conflict"

_TIME_RANGE_RE = re.compile(
    r"(?:近|最近|过去)\s*\d+\s*天|近一周|最近一周|近一个月|最近一个月|"
    r"今天|今日|当天|本日|近?24\s*小时|最近24小时|过去24小时|"
    r"today|last\s*24\s*hours?|past\s*24\s*hours?|"
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
_GENERIC_CJK_CHUNK_RE = re.compile(
    r"(?:帮我|请|给我|原问题|用户补充澄清|分析|总结|新闻|动态|最近|今天|今日|发生|什么|"
    r"全景|行业|格局|时间线|对比|比较|趋势|列出|列一下|直接|快讯|要闻|汇总|梳理|"
    r"重新检索|有证据支撑|分析结论)",
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
_DISPLAY_SOURCE_LABELS = frozenset({"HackerNews", "TechCrunch"})
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
    "Elon Musk",
    "Musk",
    "马斯克",
    "xAI",
    "X Chat",
    "xchat",
    "X",
    "SpaceX",
    "Neuralink",
    "Boring Company",
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

_FOLLOWUP_REFERENCE_RE = re.compile(
    r"(上条|上面那条|上一条|刚才|之前|前一条|没提|补充一下|展开一下|"
    r"that one|previous|earlier|last answer|follow[- ]?up|why.*mention|didn'?t.*mention)",
    re.IGNORECASE,
)


def _tokenize_for_overlap(text: str) -> set[str]:
    source = str(text or "").lower()
    if not source:
        return set()
    tokens = re.findall(r"[a-z0-9][a-z0-9+._-]{1,}|\d+|[\u4e00-\u9fff]{2,}", source)
    out: set[str] = set()
    for token in tokens:
        normalized = token.strip()
        if not normalized:
            continue
        out.add(normalized)
    return out


def _jaccard_similarity(a: str, b: str) -> float:
    ta = _tokenize_for_overlap(a)
    tb = _tokenize_for_overlap(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union <= 0:
        return 0.0
    return inter / union


def _latest_user_and_model_text(history: list[dict] | None) -> tuple[str, str]:
    prev_user = ""
    prev_model = ""
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        text = _extract_message_text(item)
        if not text:
            continue
        if not prev_model and role == "model":
            prev_model = text
            continue
        if not prev_user and role == "user":
            prev_user = text
            if prev_model:
                break
    return prev_user, prev_model


def _standalone_inverse_score(user_message: str) -> float:
    scope = _query_scope_signals(user_message)
    specified = int(scope.get("specified_count", 0) or 0)
    if specified >= 3:
        return 0.1
    if specified == 2:
        return 0.35
    if specified == 1:
        return 0.7
    return 0.9


def _entity_continuity_score(user_message: str, prev_user: str, prev_model: str) -> float:
    cur_entities = _extract_entity_candidates(user_message)
    prev_entities = _extract_entity_candidates("\n".join([prev_user, prev_model]))
    if not cur_entities and not prev_entities:
        return 0.0
    if not cur_entities:
        if _FOLLOWUP_REFERENCE_RE.search(user_message):
            return 0.4
        return 0.0
    prev_set = {x.lower() for x in prev_entities}
    if not prev_set:
        return 0.0
    hit = sum(1 for entity in cur_entities if entity.lower() in prev_set)
    return min(1.0, hit / max(1, len(cur_entities)))


def evaluate_followup_confidence(
    history: list[dict] | None,
    user_message: str,
) -> dict[str, Any]:
    """Evaluate whether the current message is context-dependent follow-up.

    Returns:
    - score: float in [0, 1]
    - decision: followup_strong | followup_dual_path | fresh
    - features: scoring feature breakdown
    """
    text = str(user_message or "").strip()
    prev_user, prev_model = _latest_user_and_model_text(history)
    if not text or (not prev_user and not prev_model):
        return {
            "score": 0.0,
            "decision": "fresh",
            "features": {
                "semantic_continuity": 0.0,
                "entity_continuity": 0.0,
                "reference_dependency": 0.0,
                "standalone_inverse": 0.0,
            },
            "previous_user": prev_user,
            "previous_model": prev_model,
            "augmented": False,
        }

    semantic_continuity = max(
        _jaccard_similarity(text, prev_user),
        _jaccard_similarity(text, prev_model),
    )
    entity_continuity = _entity_continuity_score(text, prev_user, prev_model)
    reference_dependency = 1.0 if _FOLLOWUP_REFERENCE_RE.search(text) else 0.0
    standalone_inverse = _standalone_inverse_score(text)

    score = (
        0.25 * semantic_continuity
        + 0.20 * entity_continuity
        + 0.35 * reference_dependency
        + 0.20 * standalone_inverse
    )
    if reference_dependency >= 1.0 and (prev_user or prev_model):
        score = max(score, 0.62)
        scope = _query_scope_signals(text)
        if not bool(scope.get("has_time")) and not bool(scope.get("has_source")):
            score = max(score, 0.74)
    score = max(0.0, min(1.0, score))

    if score >= 0.72:
        decision = "followup_strong"
    elif score >= 0.52:
        decision = "followup_dual_path"
    else:
        decision = "fresh"

    return {
        "score": round(score, 4),
        "decision": decision,
        "features": {
            "semantic_continuity": round(semantic_continuity, 4),
            "entity_continuity": round(entity_continuity, 4),
            "reference_dependency": round(reference_dependency, 4),
            "standalone_inverse": round(standalone_inverse, 4),
        },
        "previous_user": prev_user,
        "previous_model": prev_model,
        "augmented": False,
    }


def resolve_user_message_with_followup_context(
    history: list[dict] | None,
    user_message: str,
) -> tuple[str, dict[str, Any]]:
    """Augment message when follow-up confidence indicates context dependence."""
    text = str(user_message or "").strip()
    profile = evaluate_followup_confidence(history, text)
    decision = str(profile.get("decision", "fresh"))
    if decision == "fresh":
        return text, profile

    previous_user = str(profile.get("previous_user", "")).strip()
    previous_model = str(profile.get("previous_model", "")).strip()
    previous_model_short = previous_model[:480].strip()

    if decision == "followup_strong":
        augmented = (
            f"Current user follow-up question: {text}\n"
            f"Previous user question: {previous_user}\n"
            f"Previous assistant answer: {previous_model_short}\n"
            "Instruction: resolve references in the follow-up using prior context, "
            "then answer with concrete entities and evidence."
        )
    else:
        augmented = (
            f"User question: {text}\n"
            f"Related previous context: {previous_user}\n"
            "Instruction: if this question depends on previous context, resolve it before retrieval."
        )

    profile["augmented"] = True
    profile["effective_message_preview"] = augmented[:220]
    return augmented, profile


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


def _topic_hint_text(entities: list[str]) -> str:
    if entities:
        picks = " / ".join(entities[:3])
        return f"具体主题：例如 {picks}（任选其一）"
    return "具体主题：例如某家公司、产品或事件（任选其一）"


def _display_source_labels(labels: list[str] | tuple[str, ...] | Any) -> list[str]:
    normalized = _normalize_source_labels(labels)
    return [label for label in normalized if label in _DISPLAY_SOURCE_LABELS]


def _source_hint_text(source_labels: list[str]) -> str:
    labels = _display_source_labels(source_labels)
    if len(labels) >= 2:
        return f"来源范围：例如仅 {labels[0]}、仅 {labels[1]} 或保留多来源对比"
    if len(labels) == 1:
        return f"来源范围：例如仅 {labels[0]} 或保留多来源"
    return "来源范围：例如仅单一来源或保留多来源对比"


def _build_missing_hints(
    *,
    scope: dict[str, Any],
    source_labels: list[str],
    entities: list[str],
) -> list[str]:
    hints: list[str] = []
    if not bool(scope.get("has_time")):
        hints.append("时间范围：例如最近 7 天或最近 30 天")
    if not bool(scope.get("has_source")):
        hints.append(_source_hint_text(source_labels))
    if not bool(scope.get("has_topic")):
        hints.append(_topic_hint_text(entities))
    if not bool(scope.get("has_dimension")):
        hints.append("分析维度：趋势 / 对比 / 时间线 / 格局")
    if not hints:
        hints.append("请补充时间范围、来源范围、主题或分析维度中的任意一项")
    return hints


def _reason_text_from_risk_tags(tags: list[str]) -> str:
    mapping = {
        "few_constraints": "约束条件较少",
        "partial_constraints": "约束信息不完整",
        "multi_source_unscoped": "涉及多个来源且未限定来源范围",
        "large_url_set": "候选证据量较大",
        "medium_url_set": "候选证据较多",
        "long_time_span": "时间跨度较长",
        "moderate_time_span": "时间窗口偏宽",
        "many_entities_unscoped": "涉及实体较多且未聚焦",
        "entity_dispersion": "主题实体分散",
        "analytical_tool_chain": "分析链路较复杂",
    }
    parts = [mapping[tag] for tag in tags if tag in mapping]
    if not parts:
        return ""
    return "；".join(parts[:2])


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

    source_labels = _display_source_labels(ctx.get("source_labels", []))
    source_clause = "、".join(source_labels[:3]) if source_labels else "多个来源"
    time_span_days = _coerce_int(ctx.get("time_span_days"))
    entities = _normalize_entity_candidates(ctx.get("entity_candidates", []))
    conflict_summary = str(ctx.get("conflict_summary", "")).strip()
    risk_tags = [str(item).strip() for item in (ctx.get("ambiguous_scope_reasons") or []) if str(item).strip()]
    missing_hints = _build_missing_hints(scope=scope, source_labels=source_labels, entities=entities)
    source_hint = _source_hint_text(source_labels)
    topic_hint = _topic_hint_text(entities)

    if reason_value == CLARIFICATION_REASON_SOURCE_CONFLICT:
        ask_parts: list[str] = []
        if not bool(scope.get("has_source")):
            ask_parts.append("来源范围")
        if not bool(scope.get("has_time")):
            ask_parts.append("时间窗口")
        if not bool(scope.get("has_dimension")):
            ask_parts.append("分析维度")
        ask_clause = "、".join(ask_parts[:3]) if ask_parts else "分析维度"
        if conflict_summary:
            question = (
                f"当前多来源结论存在冲突：{conflict_summary}。"
                f"为了降低综合偏差，请先明确 {ask_clause} 后我再继续。"
            )
        else:
            question = (
                f"当前 {source_clause} 对同一主题结论存在分歧。"
                f"请先确认 {ask_clause}，我再给出可追溯的结论。"
            )
        hints = [
            source_hint,
            "时间范围：例如最近 7 天、14 天或 30 天",
            topic_hint,
            "分析维度：趋势 / 对比 / 时间线 / 格局",
        ]
    elif reason_value == CLARIFICATION_REASON_AMBIGUOUS_SCOPE:
        span_hint = f"时间跨度约 {time_span_days} 天" if time_span_days is not None and time_span_days > 0 else "时间范围尚未收敛"
        entity_hint = f"主题涉及较广（如 {', '.join(entities[:3])}）" if entities else "主题跨度较大"
        risk_reason_text = _reason_text_from_risk_tags(risk_tags)
        if risk_reason_text:
            question = (
                f"当前候选证据覆盖 {source_clause}，{span_hint}，且{entity_hint}。"
                f"主要原因：{risk_reason_text}。"
                "请确认更具体的分析范围后我再继续。"
            )
        else:
            question = (
                f"当前候选证据覆盖 {source_clause}，{span_hint}，且{entity_hint}。"
                "为避免范围过宽导致误综合，请确认更具体的分析范围。"
            )
        if not bool(scope.get("has_time")):
            question += "你更希望看最近 7 天还是 30 天？"
        hints = missing_hints
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
    text = _extract_user_intent_text(user_message)
    if not text:
        return False
    if _GENERIC_TOPIC_RE.search(text):
        return False

    english_tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,}", text)]
    if any(token not in _GENERIC_TOPIC_TOKENS for token in english_tokens):
        return True

    cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for chunk in cjk_chunks:
        if len(chunk) <= 2:
            continue
        if _GENERIC_CJK_CHUNK_RE.search(chunk):
            continue
        if chunk in {"科技", "人工智能"}:
            continue
        if re.fullmatch(r"[一二三四五六七八九十百千万两几多]+", chunk):
            continue
        if re.fullmatch(r"(最近|今天|今日|过去|近来)", chunk):
            continue
        if re.fullmatch(r"(新闻|动态|快讯|要闻|总结|分析)", chunk):
            continue
        else:
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

    intent_text = _extract_user_intent_text(user_message)
    user_intent = classify_user_intent(intent_text)
    normalized_tool_calls = _normalize_tool_calls(tool_calls)
    tool_profile = classify_tool_profile(normalized_tool_calls)
    scope = _query_scope_signals(intent_text)
    source_labels = _source_labels_from_urls(urls)
    source_labels = _augment_source_labels_with_context(
        source_labels=source_labels,
        user_message=intent_text,
        tool_calls=normalized_tool_calls,
    )
    time_span_days = _extract_time_span_days(candidate_answer)
    entity_candidates = _extract_entity_candidates(candidate_answer)
    context: dict[str, Any] = {
        "intent_text": intent_text,
        "user_intent": user_intent,
        "tool_profile": tool_profile,
        "source_labels": source_labels,
        "source_count": len(source_labels),
        "url_count": len(urls),
        "time_span_days": time_span_days,
        "entity_candidates": entity_candidates,
        "tool_calls": sorted(normalized_tool_calls),
    }
    if user_intent == "smalltalk_or_capability":
        return None, context

    conflict_hit, conflict_summary = _detect_source_conflict(
        user_message=intent_text,
        candidate_answer=candidate_answer,
        source_labels=source_labels,
        scope=scope,
        intent_label=user_intent,
        tool_profile=tool_profile,
    )
    if conflict_hit:
        if conflict_summary:
            context["conflict_summary"] = conflict_summary
        return CLARIFICATION_REASON_SOURCE_CONFLICT, context

    ambiguous_hit, ambiguous_score, ambiguous_reasons = _detect_ambiguous_scope(
        scope=scope,
        source_labels=source_labels,
        url_count=len(urls),
        time_span_days=time_span_days,
        entity_span=len(entity_candidates),
        user_message=intent_text,
        intent_label=user_intent,
        tool_profile=tool_profile,
    )
    context["ambiguous_scope_score"] = ambiguous_score
    context["ambiguous_scope_reasons"] = ambiguous_reasons
    if ambiguous_hit:
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


def _normalize_tool_calls(tool_calls: set[str] | None) -> set[str]:
    normalized: set[str] = set()
    for raw in (tool_calls or set()):
        name = str(raw or "").strip().lower()
        if name:
            normalized.add(name)
    return normalized


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


def _extract_requested_source_labels(user_message: str) -> list[str]:
    text = _extract_user_intent_text(user_message)
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for alias, canonical in _SOURCE_ALIAS_MAP.items():
        if re.search(re.escape(alias), text, flags=re.IGNORECASE):
            if canonical not in seen:
                seen.add(canonical)
                out.append(canonical)
    return out


def _augment_source_labels_with_context(
    *,
    source_labels: list[str],
    user_message: str,
    tool_calls: set[str],
) -> list[str]:
    labels = _normalize_source_labels(source_labels)
    requested = _extract_requested_source_labels(user_message)
    if requested:
        labels = _normalize_source_labels([*labels, *requested])

    if "compare_sources" in tool_calls:
        labels = _normalize_source_labels([*labels, "HackerNews", "TechCrunch"])
    return labels


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
    return ""


def _query_scope_signals(user_message: str) -> dict[str, Any]:
    text = _extract_user_intent_text(user_message)
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


def _is_roundup_listing_request(user_message: str, scope: dict[str, Any]) -> bool:
    text = _extract_user_intent_text(user_message)
    if not text:
        return False
    if bool(scope.get("has_topic")):
        return False
    if bool(scope.get("has_dimension")):
        return False
    return is_roundup_listing_intent(text)


def _is_analysis_heavy_request(user_message: str, scope: dict[str, Any]) -> bool:
    text = _extract_user_intent_text(user_message)
    if not text:
        return False
    if _is_roundup_listing_request(text, scope):
        return False
    if bool(scope.get("has_dimension")):
        return True
    if is_analysis_heavy_intent(text):
        return True
    return bool(_WIDE_QUERY_RE.search(text))


def _scope_is_tight(scope: dict[str, Any]) -> bool:
    specified = int(scope.get("specified_count", 0) or 0)
    if (
        bool(scope.get("has_source"))
        and bool(scope.get("has_time"))
        and bool(scope.get("has_topic"))
        and (bool(scope.get("has_dimension")) or specified >= 3)
    ):
        return True
    return specified >= 3


def _ambiguous_scope_risk_score(
    *,
    scope: dict[str, Any],
    source_labels: list[str],
    url_count: int,
    time_span_days: int | None,
    entity_span: int,
    user_message: str,
    intent_label: str,
    tool_profile: str,
) -> tuple[int, list[str]]:
    text = _extract_user_intent_text(user_message)
    score = 0
    reasons: list[str] = []
    specified = int(scope.get("specified_count", 0) or 0)

    if bool(scope.get("broad_query")):
        score += 2
        reasons.append("broad_query")
    if specified <= 1:
        score += 2
        reasons.append("few_constraints")
    elif specified == 2:
        score += 1
        reasons.append("partial_constraints")

    if len(source_labels) >= 2 and not bool(scope.get("has_source")):
        score += 2
        reasons.append("multi_source_unscoped")

    if url_count >= 10:
        score += 2
        reasons.append("large_url_set")
    elif url_count >= 6:
        score += 1
        reasons.append("medium_url_set")

    if time_span_days is not None and not bool(scope.get("has_time")):
        if time_span_days >= 60:
            score += 2
            reasons.append("long_time_span")
        elif time_span_days >= 30:
            score += 1
            reasons.append("moderate_time_span")

    if entity_span >= 8 and not bool(scope.get("has_topic")):
        score += 2
        reasons.append("many_entities_unscoped")
    elif entity_span >= 5 and not bool(scope.get("has_topic")):
        score += 1
        reasons.append("entity_dispersion")

    # Penalize overly aggressive clarification when user already gave useful constraints.
    if bool(scope.get("has_time")):
        score = max(0, score - 1)
    if bool(scope.get("has_source")):
        score = max(0, score - 1)
    if bool(scope.get("has_topic")):
        score = max(0, score - 1)

    if len(source_labels) >= 2 and url_count >= 10 and _WIDE_QUERY_RE.search(text):
        score += 1
        reasons.append("extra_wide_multi_source_guard")

    if tool_profile == "analytical":
        score += 1
        reasons.append("analytical_tool_chain")
    elif tool_profile == "retrieval_only":
        score = max(0, score - 1)
        reasons.append("retrieval_only_tool_chain")

    if intent_label == "roundup_listing":
        score = max(0, score - 2)
        reasons.append("roundup_intent_discount")

    return score, reasons


def _detect_ambiguous_scope(
    *,
    scope: dict[str, Any],
    source_labels: list[str],
    url_count: int,
    time_span_days: int | None,
    entity_span: int,
    user_message: str,
    intent_label: str,
    tool_profile: str,
) -> tuple[bool, int, list[str]]:
    if intent_label in {"smalltalk_or_capability", "roundup_listing"}:
        return False, 0, [f"intent_{intent_label}"]
    if _scope_is_tight(scope):
        return False, 0, ["scope_tight"]

    score, reasons = _ambiguous_scope_risk_score(
        scope=scope,
        source_labels=source_labels,
        url_count=url_count,
        time_span_days=time_span_days,
        entity_span=entity_span,
        user_message=user_message,
        intent_label=intent_label,
        tool_profile=tool_profile,
    )

    threshold = 7
    if intent_label == "analysis":
        threshold = 6
    elif intent_label == "conflict_resolution":
        threshold = 5

    analysis_heavy = _is_analysis_heavy_request(user_message, scope)
    if analysis_heavy and tool_profile == "analytical":
        threshold = max(4, threshold - 1)

    if score >= threshold:
        return True, score, reasons
    return False, score, reasons


def _detect_source_conflict(
    *,
    user_message: str,
    candidate_answer: str,
    source_labels: list[str],
    scope: dict[str, Any],
    intent_label: str,
    tool_profile: str,
) -> tuple[bool, str]:
    if intent_label in {"smalltalk_or_capability", "roundup_listing"}:
        return False, ""
    if not _needs_conflict_resolution(
        user_message,
        scope,
        intent_label=intent_label,
        tool_profile=tool_profile,
    ):
        return False, ""

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
    explicit_conflict_request = has_explicit_conflict_request(user_message)

    if not divergence:
        if conflict_words_hit and explicit_conflict_request and not _scope_is_tight(scope):
            return True, "不同来源对同一主题的判断存在明显分歧"
        return False, ""

    # If user already constrained very tightly, keep normal answer path.
    if _scope_is_tight(scope) and not explicit_compare and not explicit_conflict_request:
        return False, ""

    if divergence:
        summary = f"{positive_sources[0]} 偏正向，而 {negative_sources[0]} 更偏谨慎"
        return True, summary
    return True, "不同来源对同一主题的判断存在明显分歧"


def _needs_conflict_resolution(
    user_message: str,
    scope: dict[str, Any],
    *,
    intent_label: str,
    tool_profile: str,
) -> bool:
    if intent_label in {"smalltalk_or_capability", "roundup_listing"}:
        return False
    if intent_label == "conflict_resolution":
        return True
    if intent_label == "analysis" and tool_profile == "analytical":
        return True

    text = _extract_user_intent_text(user_message)
    if not text:
        return False
    if is_conflict_resolution_intent(text):
        return True
    if bool(scope.get("has_dimension")) and tool_profile in {"analytical", "mixed"}:
        return True
    return False


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
