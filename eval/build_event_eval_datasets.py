"""Build retrieval, generation, and end-to-end datasets from event cards."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from services.llm_provider import DEFAULT_DEEPSEEK_MODEL, build_chat_model, resolve_model_config

try:
    from news_eval_schema import (
        load_event_cards,
        validate_e2e_case,
        validate_generation_case,
        validate_retrieval_case,
        write_jsonl,
    )
except ImportError:  # pragma: no cover
    from .news_eval_schema import (
        load_event_cards,
        validate_e2e_case,
        validate_generation_case,
        validate_retrieval_case,
        write_jsonl,
    )


ENTITY_ALIASES: tuple[tuple[str, str], ...] = (
    ("OpenAI", "OpenAI"),
    ("Anthropic", "Anthropic"),
    ("Claude", "Claude"),
    ("ChatGPT", "ChatGPT"),
    ("DeepSeek", "DeepSeek"),
    ("Google", "Google"),
    ("谷歌", "谷歌"),
    ("Chrome", "Chrome"),
    ("Apple", "Apple"),
    ("苹果", "苹果"),
    ("Microsoft", "Microsoft"),
    ("微软", "微软"),
    ("GitHub", "GitHub"),
    ("Ghostty", "Ghostty"),
    ("Zed", "Zed"),
    ("Valve", "Valve"),
    ("Framework", "Framework"),
    ("Bambu", "Bambu Lab"),
    ("VS Code", "VS Code"),
)

GENERIC_ENTITIES = {
    "api",
    "issue",
    "issues",
    "actions",
    "pro",
    "flash",
    "stp",
    "core ultra",
    "co-authored",
    "ai",
    "model",
    "作者",
    "本文",
    "用户",
    "开发者",
    "公司",
    "项目",
    "平台",
    "工具",
    "模型",
}

QUESTION_SYSTEM_PROMPT = (
    "你负责为科技新闻智能体评测生成真实用户问题。"
    "你只能基于给定的公司、产品和事件类型生成自然问题，不能照抄新闻标题或事实句，不能输出 URL。"
    "问题要像用户实际会问的中文问题，不要出现“核心议题”“本文记录”“分析显示”“这条和”等内部摘要词。"
    "返回 JSON，不要 markdown。"
)

PRODUCT_PATTERNS: tuple[str, ...] = (
    r"GPT[-\s]?\d(?:\.\d+)?(?:\s*(?:Instant|Pro|Cyber|Codex))?",
    r"Claude(?:\s+(?:Code|Opus|Sonnet|Design|Platform|Cowork))?(?:\s+\d(?:\.\d+)?)?",
    r"DeepSeek\s*(?:V?\d(?:\.\d+)?|v\d)?(?:\s*(?:Pro|Flash))?",
    r"ChatGPT(?:\s+[A-Za-z0-9-]+)?",
    r"Chrome",
    r"Ghostty",
    r"Zed",
    r"VS Code",
    r"Steam(?:\s*手柄|\s*Controller)?",
    r"Framework\s+Laptop\s+\d+\s*Pro",
    r"Bambu\s+Lab",
    r"OrcaSlicer",
    r"reCAPTCHA",
    r"Android",
    r"GitHub(?:\s+Actions)?",
    r"Copilot",
    r"Plaid",
)

DOMAIN_HINTS: tuple[tuple[str, str], ...] = (
    (r"个人理财|金融机构|银行账户|Plaid", "个人理财功能"),
    (r"AI\s*模型|本地AI|静默安装|4GB", "AI 模型安装"),
    (r"开发者.*实名|实名.*开发者|侧载", "开发者验证和侧载限制"),
    (r"reCAPTCHA", "reCAPTCHA 验证"),
    (r"CEO|接任|交接|执行董事长", "管理层交接"),
    (r"计费|成本|Token|额度|加价|价格", "计费和成本"),
    (r"CAD|图纸|\.STP|STP", "CAD 图纸"),
    (r"GitHub.*迁|迁.*GitHub|迁出|迁移", "迁出 GitHub"),
    (r"故障|宕机|中断|异常|修复", "服务故障"),
    (r"隐私|环保|未.*授权|未经.*同意", "隐私争议"),
    (r"起诉|诉讼|法院|庭审|法律", "法律争议"),
    (r"模型|版本|API|发布|推出|上线|预览版|正式版", "产品发布"),
)

EVENT_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("migration", ("迁出", "迁移")),
    ("legal", ("起诉", "诉讼", "法院", "庭审", "法律", "威胁起诉")),
    ("leadership", ("CEO", "接任", "交接", "执行董事长")),
    ("incident", ("故障", "宕机", "中断", "异常", "拒答", "耗尽", "修复", "下滑")),
    ("policy", ("政策", "限制", "实名", "验证", "侧载", "隐私", "绑定", "未授权", "未经同意")),
    ("release", ("发布", "推出", "上线", "正式版", "预览版", "开源", "接入", "集成")),
    ("business", ("营收", "融资", "IPO", "交易", "成本", "价格", "计费", "商业化")),
    ("opinion", ("指出", "认为", "呼吁", "建议", "批评")),
    ("controversy", ("争议", "反对", "质疑", "担忧", "滥用", "垄断")),
)


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        elif ch.isspace() or ch in "-_./":
            chars.append("_")
    return re.sub(r"_+", "_", "".join(chars)).strip("_")[:64] or "case"


def _clean_event_title(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*[\[【][^\]】]{1,12}[\]】]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _card_haystack(card: dict[str, Any]) -> str:
    facts = " ".join(
        str(item.get("claim") or item.get("quote") or "")
        for item in card.get("facts", []) or []
        if isinstance(item, dict)
    )
    return f"{card.get('event_title', '')}\n{facts}"


def _primary_entity(card: dict[str, Any]) -> str:
    title = _clean_event_title(str(card.get("event_title", "")))
    haystack = _card_haystack(card)
    title_matches: list[tuple[int, str]] = []
    for pattern, label in ENTITY_ALIASES:
        match = re.search(re.escape(pattern), title, flags=re.IGNORECASE)
        if match:
            title_matches.append((match.start(), label))
    if title_matches:
        return sorted(title_matches, key=lambda item: item[0])[0][1]

    for pattern, label in ENTITY_ALIASES:
        if re.search(re.escape(pattern), haystack, flags=re.IGNORECASE):
            return label

    entities = [str(item).strip() for item in card.get("entities", []) if str(item).strip()]
    for entity in entities:
        if entity.lower() not in GENERIC_ENTITIES:
            return entity

    return ""


def _first_fact_claim(card: dict[str, Any]) -> str:
    for fact in card.get("facts", []) or []:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("claim") or fact.get("quote") or "").strip()
        if text:
            return text
    return ""


def _remove_leading_entity(text: str, entity: str) -> str:
    out = str(text or "").strip()
    entity_text = str(entity or "").strip()
    if entity_text:
        patterns = [
            rf"^{re.escape(entity_text)}(?:项目|公司|平台|团队|模型|工具|代码编辑器|浏览器)?",
            rf"^关于{re.escape(entity_text)}(?:的)?",
        ]
        for pattern in patterns:
            out = re.sub(pattern, "", out, flags=re.IGNORECASE).strip()
    out = re.sub(r"^(项目|公司|平台|团队)?(宣布|发布|推出|正式发布|指出|确认|计划|拟|将|默认|威胁|回应|披露)", "", out).strip()
    return out


def _topic_phrase(card: dict[str, Any], entity: str) -> str:
    title = _clean_event_title(str(card.get("event_title", "")))
    claim = _first_fact_claim(card)
    claim_clean = _clean_event_title(claim)
    if re.match(r"^(该|此|这|上述|本文|分析显示|核心议题)", claim_clean):
        base = title
    else:
        base = _clean_event_title(claim_clean or title)
    base = _remove_leading_entity(base, entity)
    base = re.sub(r"^(核心议题|分析显示|本文记录|近日|最近)(：|:)?", "", base).strip()
    base = re.sub(r"^(：|:|，|,|。|\s)+", "", base).strip()
    if not base:
        base = title
    base = _compact_topic(base)
    return base or "这条消息"


def _event_type(card: dict[str, Any]) -> str:
    haystack = _card_haystack(card)
    for event_type, keywords in EVENT_TYPE_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return event_type
    return "generic"


def _find_product(card: dict[str, Any], entity: str) -> str:
    title = _clean_event_title(str(card.get("event_title", "")))
    for pattern in PRODUCT_PATTERNS:
        match = re.search(pattern, title, flags=re.IGNORECASE)
        if not match:
            continue
        product = re.sub(r"\s+", " ", match.group(0)).strip()
        if product and product.lower() != str(entity or "").lower():
            return product

    # Facts often mention adjacent tools or competitors. If the title already
    # identifies the main entity, do not promote a fact-only mention to product.
    if entity and re.search(re.escape(entity), title, flags=re.IGNORECASE):
        return ""

    for item in card.get("entities", []) or []:
        candidate = str(item or "").strip()
        if not candidate or candidate.lower() in GENERIC_ENTITIES:
            continue
        if entity and candidate.lower() == entity.lower():
            continue
        if len(candidate) <= 32:
            return candidate
    return ""


def _domain_hint(card: dict[str, Any]) -> str:
    haystack = _card_haystack(card)
    for pattern, label in DOMAIN_HINTS:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return label
    return ""


def _query_profile(card: dict[str, Any]) -> dict[str, str] | None:
    entity = _primary_entity(card)
    product = _find_product(card, entity)
    if not entity and product:
        entity = product
    if not entity:
        return None
    sources = [str(item).strip() for item in card.get("sources", []) if str(item).strip()]
    return {
        "entity": entity,
        "product": product,
        "subject": product or entity,
        "domain": _domain_hint(card),
        "event_type": _event_type(card),
        "source": sources[0] if sources else "",
    }


def _clean_question(text: str) -> str:
    question = re.sub(r"\s+", " ", str(text or "")).strip()
    question = question.strip("。；;，, ")
    return f"{question}？" if question and not question.endswith(("?", "？")) else question


def _focus_phrase(focus: str) -> str:
    text = str(focus or "").strip()
    if not text:
        return ""
    if re.search(r"[A-Za-z]", text):
        return f"在 {text}上"
    return f"在{text}上"


def _add_question(rows: list[tuple[str, str]], query_type: str, question: str, card: dict[str, Any]) -> None:
    clean = _clean_question(question)
    if not _question_is_usable(clean, card):
        return
    if clean in {item[1] for item in rows}:
        return
    rows.append((query_type, clean))


def _compact_topic(value: str, *, max_chars: int = 24) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    for sep in ("，", ",", "；", ";", "。", "：", ":"):
        if sep in text and text.index(sep) >= 6:
            text = text.split(sep, 1)[0]
            break
    if len(text) <= max_chars:
        return text.strip("，,。；;：: ")
    cut = text[:max_chars]
    cut = re.sub(r"[A-Za-z0-9.+-]+$", "", cut).strip()
    return (cut or text[:max_chars]).strip("，,。；;：: ")


def _time_window_days(card: dict[str, Any]) -> str:
    start = str((card.get("time_window") or {}).get("start", "")).strip()
    end = str((card.get("time_window") or {}).get("end", "")).strip()
    if start and end and start != end:
        return f"{start} 至 {end}"
    return end or start or "最近"


def _retrieval_questions(card: dict[str, Any]) -> list[tuple[str, str]]:
    profile = _query_profile(card)
    if profile is None:
        return []

    entity = profile["entity"]
    subject = profile["subject"]
    product = profile["product"]
    domain = profile["domain"]
    event_type = profile["event_type"]
    source = profile["source"]
    if event_type == "migration" and product.lower().startswith("github"):
        subject = entity
    rows: list[tuple[str, str]] = []

    if event_type == "migration":
        _add_question(rows, "single_event", f"{subject} 为什么要迁出 GitHub", card)
        _add_question(rows, "latest_update", f"帮我查一下 {subject} 最近迁移代码托管平台的消息", card)
        _add_question(rows, "deep_reading", f"整理一下 {subject} 迁出 GitHub 的背景和影响", card)
    elif event_type == "leadership":
        _add_question(rows, "single_event", f"{entity} 最近管理层有什么变化", card)
        _add_question(rows, "latest_update", f"{entity} 最近有哪些高层变动消息", card)
        _add_question(rows, "deep_reading", f"帮我整理 {entity} 管理层交接的时间线和影响", card)
    elif event_type == "incident":
        _add_question(rows, "single_event", f"{subject} 最近是不是出了故障或异常", card)
        _add_question(rows, "latest_update", f"{entity} 最近有什么服务异常或修复进展", card)
        _add_question(rows, "deep_reading", f"帮我整理 {subject} 这次异常的原因、影响和来源", card)
    elif event_type == "policy":
        focus = domain or "政策和限制"
        focus_event = f"这次 {focus}争议" if re.search(r"[A-Za-z]", focus) else f"这次{focus}争议"
        _add_question(rows, "single_event", f"{entity} 最近{_focus_phrase(focus)}有什么新动作", card)
        _add_question(rows, "latest_update", f"帮我查一下 {subject} 最近的限制或政策变化", card)
        _add_question(rows, "deep_reading", f"整理一下 {entity} {focus_event}的背景和影响", card)
    elif event_type == "legal":
        _add_question(rows, "single_event", f"{entity} 最近卷入了什么法律争议", card)
        _add_question(rows, "latest_update", f"帮我查一下 {entity} 最近的诉讼相关进展", card)
        _add_question(rows, "deep_reading", f"整理一下 {entity} 这起法律事件的关键事实和来源", card)
    elif event_type == "release":
        if product and product.lower() != entity.lower():
            if domain and domain != "产品发布":
                _add_question(rows, "single_event", f"{entity} 最近在 {product} 的{domain}上有什么新动作", card)
                _add_question(rows, "latest_update", f"帮我查一下 {product} 最近和{domain}有关的发布消息", card)
                _add_question(rows, "deep_reading", f"整理一下 {product} 这次{domain}更新的关键信息和来源", card)
            else:
                _add_question(rows, "single_event", f"{entity} 最近在 {product} 上推出了什么新功能", card)
                _add_question(rows, "latest_update", f"帮我查一下 {product} 最近的发布消息", card)
                _add_question(rows, "deep_reading", f"整理一下 {product} 这次更新的关键信息和来源", card)
        else:
            _add_question(rows, "single_event", f"{entity} 最近发布了什么新产品或新版本", card)
            _add_question(rows, "latest_update", f"{entity} 最近有哪些产品更新", card)
            _add_question(rows, "deep_reading", f"帮我整理 {entity} 这次产品发布的关键信息和来源", card)
    elif event_type == "business":
        _add_question(rows, "single_event", f"{entity} 最近在商业化或成本上有什么变化", card)
        _add_question(rows, "latest_update", f"帮我查一下 {subject} 最近的商业进展", card)
        _add_question(rows, "deep_reading", f"整理一下 {entity} 这次商业变化的背景、影响和来源", card)
    elif event_type == "opinion":
        _add_question(rows, "single_event", f"{entity} 最近关于 AI 开发有什么观点", card)
        _add_question(rows, "latest_update", f"帮我查一下 {entity} 最近发表的技术观点", card)
        _add_question(rows, "deep_reading", f"整理一下 {entity} 这次观点的核心论据和来源", card)
    elif event_type == "controversy":
        _add_question(rows, "single_event", f"{subject} 最近有什么争议", card)
        _add_question(rows, "latest_update", f"帮我查一下 {entity} 最近的争议事件", card)
        _add_question(rows, "deep_reading", f"整理一下 {subject} 争议的背景、影响和来源", card)
    else:
        if domain:
            _add_question(rows, "single_event", f"{entity} 最近在{domain}上有什么动态", card)
        _add_question(rows, "latest_update", f"{entity} 最近有什么值得关注的科技新闻", card)
        _add_question(rows, "deep_reading", f"帮我整理 {entity} 最近这次事件的背景和影响", card)

    if source and len(rows) < 4:
        _add_question(rows, "source_limited", f"从 {source} 的报道看，{subject} 最近有什么值得关注的消息", card)
    return rows


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return json.loads(match.group(1))
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found in question-generation response.")
    depth = 0
    for idx, ch in enumerate(raw[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : idx + 1])
    raise ValueError("Incomplete JSON object in question-generation response.")


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", item)))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content or "")


def _question_is_usable(question: str, card: dict[str, Any]) -> bool:
    text = str(question or "").strip()
    if len(text) < 8 or len(text) > 72:
        return False
    if "http://" in text.lower() or "https://" in text.lower():
        return False
    blocked = (
        "核心议题",
        "本文记录",
        "分析显示",
        "该功能",
        "该模型",
        "该公司",
        "这条和",
        "重点看",
        "有关的消息是什么情况",
    )
    if any(item in text for item in blocked):
        return False
    if "「" in text or "」" in text:
        return False
    title = _clean_event_title(str(card.get("event_title", "")))
    if len(title) >= 16 and title in text:
        return False
    return True


def _parse_llm_questions(payload: dict[str, Any], card: dict[str, Any]) -> list[tuple[str, str]]:
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        return []
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        query_type = str(item.get("query_type") or "").strip() or "single_event"
        question = str(item.get("question") or "").strip()
        if not _question_is_usable(question, card):
            continue
        if question in seen:
            continue
        seen.add(question)
        rows.append((query_type, question))
    return rows


def _llm_question_prompt(card: dict[str, Any], *, questions_per_event: int) -> str:
    profile = _query_profile(card) or {}
    facts = [
        str(item.get("claim") or item.get("quote") or "").strip()
        for item in card.get("facts", []) or []
        if isinstance(item, dict) and str(item.get("claim") or item.get("quote") or "").strip()
    ][:2]
    sources = [str(item).strip() for item in card.get("sources", []) if str(item).strip()]
    return json.dumps(
        {
            "task": f"生成 {max(1, questions_per_event)} 个真实用户可能提出的科技新闻检索问题。",
            "constraints": [
                "不要照抄完整新闻标题。",
                "不要把 facts 里的整句或半句放进问题。",
                "不要出现 URL。",
                "不要使用引号包住长主题。",
                "不要出现“核心议题”“本文记录”“分析显示”“这条和”“重点看”等内部摘要词。",
                "问题可以包含公司名、产品名、事件类型，但要像真实用户自然提问。",
                "query_type 只能从 single_event/latest_update/deep_reading/source_limited 中选择。",
            ],
            "query_profile": {
                "entity": profile.get("entity", ""),
                "product": profile.get("product", ""),
                "domain": profile.get("domain", ""),
                "event_type": profile.get("event_type", ""),
                "source": profile.get("source", ""),
            },
            "event_reference_do_not_copy": {
                "title": _clean_event_title(str(card.get("event_title", ""))),
                "facts": facts,
                "sources": sources,
            },
            "output_schema": {
                "questions": [
                    {"query_type": "single_event", "question": "自然中文问题"}
                ]
            },
        },
        ensure_ascii=False,
    )


def _build_question_model(provider: str | None, model: str | None) -> Any:
    resolved = resolve_model_config(
        provider=provider or os.getenv("TASK_EVAL_PROVIDER", "deepseek"),
        model_name=model or os.getenv("TASK_EVAL_MODEL", "") or None,
        default_provider="deepseek",
        default_model=DEFAULT_DEEPSEEK_MODEL,
    )
    return build_chat_model(
        provider=resolved.provider,
        model_name=resolved.model,
        temperature=0.2,
        default_provider="deepseek",
        default_model=DEFAULT_DEEPSEEK_MODEL,
    )


def _llm_retrieval_questions(
    model: Any,
    card: dict[str, Any],
    *,
    questions_per_event: int,
) -> list[tuple[str, str]]:
    raw = model.invoke(
        [
            SystemMessage(content=QUESTION_SYSTEM_PROMPT),
            HumanMessage(content=_llm_question_prompt(card, questions_per_event=questions_per_event)),
        ]
    )
    payload = _extract_json_object(_coerce_text_content(getattr(raw, "content", raw)))
    return _parse_llm_questions(payload, card)


def _generation_question(card: dict[str, Any]) -> str:
    profile = _query_profile(card)
    if profile is None:
        return "基于给定证据，总结这起科技新闻事件的关键事实和影响。"
    subject = profile["subject"]
    domain = profile["domain"]
    if domain:
        return f"基于给定证据，总结 {subject} {_focus_phrase(domain)}的关键事实和影响。"
    return f"基于给定证据，总结 {subject} 相关消息的关键事实和影响。"


def _evidence_from_card(card: dict[str, Any]) -> list[dict[str, str]]:
    title = _clean_event_title(str(card.get("event_title", "")))
    evidence: list[dict[str, str]] = []
    for fact in card.get("facts", []) or []:
        if not isinstance(fact, dict):
            continue
        evidence.append(
            {
                "title": title,
                "quote": str(fact.get("quote") or "").strip(),
                "url": str(fact.get("url") or "").strip(),
            }
        )
    return [item for item in evidence if item["quote"] and item["url"]]


def build_datasets(
    event_cards: list[dict[str, Any]],
    *,
    max_events: int,
    questions_per_event: int,
    question_model: Any | None = None,
    question_mode: str = "archetype",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval_cases: list[dict[str, Any]] = []
    generation_cases: list[dict[str, Any]] = []
    e2e_cases: list[dict[str, Any]] = []

    for card in event_cards[: max(1, max_events)]:
        event_id = str(card.get("event_id", "")).strip()
        event_slug = _slug(event_id)
        gold_urls = [str(item).strip() for item in card.get("core_urls", []) if str(item).strip()]
        if not gold_urls:
            continue
        question_rows: list[tuple[str, str]] = []
        normalized_mode = str(question_mode or "archetype").strip().lower()
        if normalized_mode == "llm" and question_model is not None:
            try:
                question_rows = _llm_retrieval_questions(
                    question_model,
                    card,
                    questions_per_event=max(1, questions_per_event),
                )
            except Exception as exc:
                print(f"[EventEvalDatasets][Warn] LLM question generation failed for {event_id}: {exc}")
        if len(question_rows) < max(1, questions_per_event):
            fallback = _retrieval_questions(card)
            existing = {question for _, question in question_rows}
            for row in fallback:
                if row[1] not in existing:
                    question_rows.append(row)
                    existing.add(row[1])
                if len(question_rows) >= max(1, questions_per_event):
                    break
        question_rows = question_rows[: max(1, questions_per_event)]
        for idx, (query_type, question) in enumerate(question_rows, 1):
            retrieval_cases.append(
                validate_retrieval_case(
                    {
                        "case_id": f"retrieval.{event_slug}.{idx:03d}",
                        "question": question,
                        "query_type": query_type,
                        "gold_event_id": event_id,
                        "gold_urls": gold_urls,
                        "time_window": _time_window_days(card),
                    }
                )
            )

        evidence = _evidence_from_card(card)
        required_claims = [
            str(fact.get("claim") or "").strip()
            for fact in card.get("facts", []) or []
            if isinstance(fact, dict) and str(fact.get("claim") or "").strip()
        ]
        if evidence and required_claims:
            generation_cases.append(
                validate_generation_case(
                    {
                        "case_id": f"generation.{event_slug}.001",
                        "question": _generation_question(card),
                        "event_id": event_id,
                        "evidence": evidence,
                        "required_claims": required_claims,
                        "forbidden_claims": ["证据中没有出现的发布时间、价格、地区或数字"],
                    }
                )
            )

        if question_rows:
            first_question = question_rows[0][1]
            e2e_cases.append(
                validate_e2e_case(
                    {
                        "case_id": f"e2e.{event_slug}.001",
                        "question": first_question,
                        "gold_event_id": event_id,
                        "gold_urls": gold_urls,
                        "expected_behavior": "retrieve_then_answer",
                        "time_window": _time_window_days(card),
                    }
                )
            )

    return retrieval_cases, generation_cases, e2e_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-driven eval datasets from event cards.")
    parser.add_argument("--events", type=Path, default=Path("eval/datasets/event_cards.jsonl"))
    parser.add_argument("--retrieval-output", type=Path, default=Path("eval/datasets/retrieval_cases.jsonl"))
    parser.add_argument("--generation-output", type=Path, default=Path("eval/datasets/generation_cases.jsonl"))
    parser.add_argument("--e2e-output", type=Path, default=Path("eval/datasets/e2e_cases.jsonl"))
    parser.add_argument("--manifest-output", type=Path, default=Path("eval/datasets/event_eval_manifest.json"))
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--questions-per-event", type=int, default=3)
    parser.add_argument("--question-mode", choices=["archetype", "llm", "template"], default="archetype")
    parser.add_argument("--question-provider", type=str, default=None)
    parser.add_argument("--question-model", type=str, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    return parser.parse_args()


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
    if env_file:
        load_dotenv(dotenv_path=env_file.resolve(), override=True)


def main() -> None:
    args = parse_args()
    _load_eval_env(args.env_file)
    event_cards = load_event_cards(args.events)
    question_model = None
    if str(args.question_mode).strip().lower() == "llm":
        question_model = _build_question_model(args.question_provider, args.question_model)
    retrieval_cases, generation_cases, e2e_cases = build_datasets(
        event_cards,
        max_events=max(1, int(args.max_events)),
        questions_per_event=max(1, int(args.questions_per_event)),
        question_model=question_model,
        question_mode=str(args.question_mode),
    )
    write_jsonl(args.retrieval_output, retrieval_cases)
    write_jsonl(args.generation_output, generation_cases)
    write_jsonl(args.e2e_output, e2e_cases)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": str(args.events),
        "event_count": len(event_cards),
        "retrieval_cases": len(retrieval_cases),
        "generation_cases": len(generation_cases),
        "e2e_cases": len(e2e_cases),
        "question_mode": str(args.question_mode),
        "question_provider": str(args.question_provider or os.getenv("TASK_EVAL_PROVIDER", "deepseek")),
        "question_model": str(args.question_model or os.getenv("TASK_EVAL_MODEL", DEFAULT_DEEPSEEK_MODEL)),
        "retrieval_output": str(args.retrieval_output),
        "generation_output": str(args.generation_output),
        "e2e_output": str(args.e2e_output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[EventEvalDatasets] retrieval={len(retrieval_cases)} generation={len(generation_cases)} e2e={len(e2e_cases)}")
    print(f"[EventEvalDatasets] manifest={args.manifest_output}")


if __name__ == "__main__":
    main()
