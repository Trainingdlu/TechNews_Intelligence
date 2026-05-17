"""Build retrieval, generation, and end-to-end datasets from event cards."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from news_eval_schema import load_event_cards, validate_e2e_case, validate_generation_case, validate_retrieval_case, write_jsonl
except ImportError:  # pragma: no cover
    from .news_eval_schema import (
        load_event_cards,
        validate_e2e_case,
        validate_generation_case,
        validate_retrieval_case,
        write_jsonl,
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


def _primary_entity(card: dict[str, Any]) -> str:
    entities = [str(item).strip() for item in card.get("entities", []) if str(item).strip()]
    if entities:
        return entities[0]
    title = str(card.get("event_title", "")).strip()
    return title[:24] if title else "这家公司"


def _time_window_days(card: dict[str, Any]) -> str:
    start = str((card.get("time_window") or {}).get("start", "")).strip()
    end = str((card.get("time_window") or {}).get("end", "")).strip()
    if start and end and start != end:
        return f"{start} 至 {end}"
    return end or start or "最近"


def _retrieval_questions(card: dict[str, Any]) -> list[tuple[str, str]]:
    title = str(card.get("event_title", "")).strip()
    entity = _primary_entity(card)
    source = ""
    sources = [str(item).strip() for item in card.get("sources", []) if str(item).strip()]
    if sources:
        source = sources[0]
    questions = [
        ("single_event", f"请帮我查一下「{title}」这件事的关键信息。"),
        ("latest_update", f"最近关于「{entity}」有什么和「{title}」相关的最新动态？"),
        ("deep_reading", f"围绕「{title}」，检索相关新闻并总结重点。"),
    ]
    if source:
        questions.append(("source_limited", f"请从 {source} 相关报道里找出「{title}」这件事。"))
    return questions


def _generation_question(card: dict[str, Any]) -> str:
    return f"基于给定证据，总结「{card.get('event_title')}」的关键事实和影响。"


def _evidence_from_card(card: dict[str, Any]) -> list[dict[str, str]]:
    title = str(card.get("event_title", "")).strip()
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

        first_question = question_rows[0][1] if question_rows else f"请检索「{card.get('event_title')}」相关新闻。"
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

