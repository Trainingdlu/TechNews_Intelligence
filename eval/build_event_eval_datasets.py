"""Build retrieval, generation, and end-to-end datasets from event cards."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
}


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

    lead_match = re.match(
        r"^([A-Za-z][A-Za-z0-9 .+-]{1,36}|[\u4e00-\u9fffA-Za-z0-9·]{2,24}?)(?:宣布|发布|推出|正式|将|拟|因|指出|默认|威胁|未经|通过|识别|绑定|成为)",
        title,
    )
    if lead_match:
        return lead_match.group(1).strip()
    return title[:16] if title else "这家公司"


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
    if re.match(r"^(该|此|这|上述)", claim_clean):
        base = title
    else:
        base = _clean_event_title(claim_clean or title)
    base = _remove_leading_entity(base, entity)
    base = re.sub(r"^(：|:|，|,|。|\s)+", "", base).strip()
    if not base:
        base = title
    if len(base) > 34:
        base = base[:34].rstrip("，,。；;：:")
    return base or "这条消息"


def _time_window_days(card: dict[str, Any]) -> str:
    start = str((card.get("time_window") or {}).get("start", "")).strip()
    end = str((card.get("time_window") or {}).get("end", "")).strip()
    if start and end and start != end:
        return f"{start} 至 {end}"
    return end or start or "最近"


def _retrieval_questions(card: dict[str, Any]) -> list[tuple[str, str]]:
    entity = _primary_entity(card)
    topic = _topic_phrase(card, entity)
    source = ""
    sources = [str(item).strip() for item in card.get("sources", []) if str(item).strip()]
    if sources:
        source = sources[0]
    questions = [
        ("single_event", f"{entity} 最近这条和「{topic}」有关的消息是什么情况？"),
        ("latest_update", f"{entity} 最近有什么值得关注的动态？重点看「{topic}」相关的消息。"),
        ("deep_reading", f"帮我检索最近关于 {entity} 的相关新闻，整理「{topic}」的原因、影响和来源。"),
    ]
    if source:
        questions.append(("source_limited", f"请从 {source} 相关报道里找一下 {entity} 和「{topic}」有关的消息。"))
    return questions


def _generation_question(card: dict[str, Any]) -> str:
    entity = _primary_entity(card)
    topic = _topic_phrase(card, entity)
    return f"基于给定证据，总结 {entity} 和「{topic}」相关消息的关键事实和影响。"


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
        question_rows = _retrieval_questions(card)[: max(1, questions_per_event)]
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

        first_question = question_rows[0][1] if question_rows else f"请检索 {card.get('event_title')} 相关新闻。"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_cards = load_event_cards(args.events)
    retrieval_cases, generation_cases, e2e_cases = build_datasets(
        event_cards,
        max_events=max(1, int(args.max_events)),
        questions_per_event=max(1, int(args.questions_per_event)),
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
