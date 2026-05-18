"""Run end-to-end Agent evaluation with event-driven cases and failure attribution."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from eval_core import extract_urls, normalize_url_for_retrieval
    from news_eval_metrics import (
        build_event_metadata_index,
        build_url_event_index,
        score_retrieval_prediction,
        summarize_retrieval_scores,
    )
    from news_eval_schema import load_e2e_cases, load_event_cards
except ImportError:  # pragma: no cover
    from .eval_core import extract_urls, normalize_url_for_retrieval
    from .news_eval_metrics import (
        build_event_metadata_index,
        build_url_event_index,
        score_retrieval_prediction,
        summarize_retrieval_scores,
    )
    from .news_eval_schema import load_e2e_cases, load_event_cards


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
    if env_file:
        load_dotenv(dotenv_path=env_file.resolve(), override=True)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _bootstrap_agent() -> Any:
    from agent import generate_response_eval_payload  # pylint: disable=import-outside-toplevel

    return generate_response_eval_payload


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _tool_calls_from_trace(summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    calls: list[str] = []
    for span in summary.get("spans", []) or []:
        if isinstance(span, dict) and span.get("span_type") == "tool_call":
            name = str(span.get("name", "")).strip()
            if name:
                calls.append(name)
    return calls


def _retrieved_urls_from_trace(summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    urls: list[str] = []
    for span in summary.get("spans", []) or []:
        if not isinstance(span, dict) or span.get("span_type") != "tool_call":
            continue
        out_summary = span.get("output_summary", {})
        if not isinstance(out_summary, dict):
            continue
        urls.extend(str(item).strip() for item in out_summary.get("evidence_urls", []) or [] if str(item).strip())
    return _dedupe(urls)


def _collect_urls(payload: dict[str, Any], trace_summary: dict[str, Any] | None) -> list[str]:
    payload_urls = [str(item).strip() for item in payload.get("valid_urls", []) or [] if str(item).strip()]
    return _dedupe(payload_urls + _retrieved_urls_from_trace(trace_summary))


def _unsupported_answer_urls(answer: str, valid_urls: list[str]) -> list[str]:
    valid = {normalize_url_for_retrieval(url) for url in valid_urls if normalize_url_for_retrieval(url)}
    out: list[str] = []
    for url in extract_urls(answer):
        normalized = normalize_url_for_retrieval(url)
        if normalized and normalized not in valid:
            out.append(url)
    return out


def _attribution(
    *,
    payload: dict[str, Any],
    trace_summary: dict[str, Any] | None,
    retrieval_score: dict[str, Any],
    answer: str,
    retrieved_urls: list[str],
) -> str:
    if payload.get("error") or str(payload.get("status", "")).lower() == "error":
        return "SYSTEM_FAIL"
    tool_calls = _tool_calls_from_trace(trace_summary)
    if not tool_calls:
        return "TOOL_PATH_FAIL"
    if str(retrieval_score.get("case_kind") or "") == "broad_topic":
        if float(retrieval_score.get("event_set_recall_at_k") or 0.0) <= 0.0:
            return "RETRIEVAL_FAIL"
    elif float(retrieval_score.get("event_hit_at_k") or 0.0) <= 0.0 and float(retrieval_score.get("exact_hit_at_k") or 0.0) <= 0.0:
        return "RETRIEVAL_FAIL"
    if not answer.strip():
        return "GENERATION_FAIL"
    if _unsupported_answer_urls(answer, retrieved_urls):
        return "GUARD_OR_CITATION_FAIL"
    answer_urls = {normalize_url_for_retrieval(url) for url in extract_urls(answer)}
    if answer_urls and not answer_urls.intersection({normalize_url_for_retrieval(url) for url in retrieved_urls}):
        return "EVIDENCE_USE_FAIL"
    return "OK"


def _invoke_agent(generate_response_eval_payload: Any, question: str, *, request_id: str, case_id: str) -> dict[str, Any]:
    try:
        return generate_response_eval_payload(
            [],
            question,
            request_id=request_id,
            case_id=case_id,
            experiment_group="event_e2e_eval",
            include_trace_summary=True,
        )
    except TypeError:
        return generate_response_eval_payload([], question, request_id=request_id, case_id=case_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end event eval.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/datasets/e2e_cases.jsonl"))
    parser.add_argument("--events", type=Path, default=Path("eval/datasets/event_cards.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/e2e_event_eval_latest.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--env-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_eval_env(args.env_file)
    cases = load_e2e_cases(args.dataset)
    if args.max_cases > 0:
        cases = cases[: int(args.max_cases)]
    event_cards = load_event_cards(args.events)
    url_event_index = build_url_event_index(event_cards)
    event_metadata_index = build_event_metadata_index(event_cards)
    generate_response_eval_payload = _bootstrap_agent()

    results: list[dict[str, Any]] = []
    for case in cases:
        request_id = f"event-e2e-{uuid.uuid4().hex[:12]}"
        try:
            payload = _invoke_agent(generate_response_eval_payload, str(case["question"]), request_id=request_id, case_id=str(case["case_id"]))
            if not isinstance(payload, dict):
                payload = {"text": str(payload)}
            error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            payload = {"text": "", "error": str(exc), "status": "error"}
            error = str(exc)
        trace_summary = payload.get("trace_summary") if isinstance(payload.get("trace_summary"), dict) else None
        answer = str(payload.get("text") or payload.get("answer") or "")
        retrieved_urls = _collect_urls(payload, trace_summary)
        retrieval_score = score_retrieval_prediction(
            pred_urls=retrieved_urls,
            gold_urls=case.get("gold_urls", []),
            gold_event_id=str(case.get("gold_event_id", "")),
            gold_event_ids=case.get("gold_event_ids", []),
            acceptable_event_ids=case.get("acceptable_event_ids", []),
            case_kind=str(case.get("case_kind", "single_event")),
            url_event_index=url_event_index,
            event_metadata_index=event_metadata_index,
            k=int(args.k),
        )
        attribution = _attribution(
            payload=payload,
            trace_summary=trace_summary,
            retrieval_score=retrieval_score,
            answer=answer,
            retrieved_urls=retrieved_urls,
        )
        results.append(
            {
                "case_id": case["case_id"],
                "request_id": request_id,
                "question": case["question"],
                "case_kind": case.get("case_kind", "single_event"),
                "gold_event_id": case.get("gold_event_id", ""),
                "gold_event_ids": case.get("gold_event_ids", []),
                "acceptable_event_ids": case.get("acceptable_event_ids", []),
                "topic": case.get("topic", ""),
                "expected_entities": case.get("expected_entities", []),
                "expected_event_types": case.get("expected_event_types", []),
                "answer": answer,
                "retrieved_urls": retrieved_urls,
                "tool_calls": _tool_calls_from_trace(trace_summary),
                "retrieval_score": retrieval_score,
                "attribution": attribution,
                "error": error or payload.get("error"),
            }
        )

    retrieval_scores = [row["retrieval_score"] for row in results]
    attribution_counts: dict[str, int] = {}
    for row in results:
        attribution_counts[row["attribution"]] = attribution_counts.get(row["attribution"], 0) + 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "events": str(args.events),
        "k": int(args.k),
        "summary": {
            **summarize_retrieval_scores(retrieval_scores),
            "attribution_counts": attribution_counts,
        },
        "results": results,
        "env": {
            "AGENT_MODEL_PROVIDER": os.getenv("AGENT_MODEL_PROVIDER", ""),
            "EVAL_RECALL_PROFILE": os.getenv("EVAL_RECALL_PROFILE", ""),
            "NEWS_RERANK_MODE": os.getenv("NEWS_RERANK_MODE", ""),
            "JINA_API_KEY_PRESENT": bool(os.getenv("JINA_API_KEY")),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E2EEventEval] cases={len(results)} output={args.output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
