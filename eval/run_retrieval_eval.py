"""Run retrieval-only evaluation against event-driven retrieval cases."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from news_eval_metrics import build_url_event_index, score_retrieval_prediction, summarize_retrieval_scores
    from news_eval_schema import load_event_cards, load_retrieval_cases
except ImportError:  # pragma: no cover
    from .news_eval_metrics import build_url_event_index, score_retrieval_prediction, summarize_retrieval_scores
    from .news_eval_schema import load_event_cards, load_retrieval_cases


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
    if env_file:
        load_dotenv(dotenv_path=env_file.resolve(), override=True)


def _days_from_case(case: dict[str, Any], default: int) -> int:
    raw = str(case.get("time_window", "") or "")
    numbers = [int(item) for item in re.findall(r"\d+", raw)]
    if not numbers:
        return int(default)
    if any(1 <= item <= 365 for item in numbers):
        return max(1, min(365, numbers[-1]))
    return int(default)


def _run_search_news(question: str, *, days: int) -> dict[str, Any]:
    from agent.tools.schemas import SearchNewsToolInput  # pylint: disable=import-outside-toplevel
    from agent.tools.search_news import search_news_tool  # pylint: disable=import-outside-toplevel

    envelope = search_news_tool(SearchNewsToolInput(query=question, days=days))
    urls = [str(item.url).strip() for item in envelope.evidence if str(item.url).strip()]
    return {
        "status": envelope.status,
        "pred_urls": urls,
        "diagnostics": envelope.diagnostics,
        "error_code": envelope.error_code,
        "error": envelope.error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only event eval.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/datasets/retrieval_cases.jsonl"))
    parser.add_argument("--events", type=Path, default=Path("eval/datasets/event_cards.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/retrieval_eval_latest.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--default-days", type=int, default=30)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--env-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_eval_env(args.env_file)
    cases = load_retrieval_cases(args.dataset)
    if args.max_cases > 0:
        cases = cases[: int(args.max_cases)]
    event_cards = load_event_cards(args.events)
    url_event_index = build_url_event_index(event_cards)

    results: list[dict[str, Any]] = []
    for case in cases:
        days = _days_from_case(case, int(args.default_days))
        started = datetime.now(timezone.utc)
        try:
            run = _run_search_news(str(case["question"]), days=days)
            error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            run = {"status": "error", "pred_urls": [], "diagnostics": {}, "error_code": "retrieval_runner_error", "error": str(exc)}
            error = str(exc)
        scores = score_retrieval_prediction(
            pred_urls=run.get("pred_urls", []),
            gold_urls=case.get("gold_urls", []),
            gold_event_id=str(case.get("gold_event_id", "")),
            url_event_index=url_event_index,
            k=int(args.k),
        )
        results.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "query_type": case["query_type"],
                "gold_event_id": case["gold_event_id"],
                "status": run.get("status"),
                "started_at": started.isoformat(),
                "days": days,
                "scores": scores,
                "diagnostics": run.get("diagnostics", {}),
                "error": error or run.get("error"),
                "error_code": run.get("error_code"),
            }
        )

    flat_scores = [dict(row["scores"]) for row in results]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "events": str(args.events),
        "runner": "search_news_tool",
        "k": int(args.k),
        "summary": summarize_retrieval_scores(flat_scores),
        "results": results,
        "env": {
            "AGENT_RETRIEVAL_RERANK_MODE": os.getenv("AGENT_RETRIEVAL_RERANK_MODE", ""),
            "AGENT_RECALL_PROFILE": os.getenv("AGENT_RECALL_PROFILE", ""),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RetrievalEval] cases={len(results)} output={args.output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

