"""G2 Phase B — LLM relevance judging over the pooled retrieval results.

Reads runs/retrieval_results.jsonl and, for each query, asks an LLM to rate the
relevance of every pooled document (title + summary) on a 0/1/2 scale:
  0 = not relevant
  1 = partially relevant (mentions the topic but it is not the focus)
  2 = highly relevant (directly about the query)

These judgments are the shared gold labels used to score every ablation config.
One LLM call per query (all pooled docs judged together) to keep cost low.
Resume support: queries already judged (status=success) are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from eval.eval_retry import call_with_retry  # noqa: E402

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict relevance judge for a tech-news retrieval system. "
    "Given a search query and a numbered list of candidate articles (title + summary), "
    "rate EACH candidate's relevance to the query on this scale:\n"
    "  2 = highly relevant: the article is directly about what the query asks\n"
    "  1 = partially relevant: mentions the topic/entity but it is not the article's focus\n"
    "  0 = not relevant: different topic, only incidental or no connection\n"
    "Judge topical relevance to the QUERY, not article quality or recency. "
    'Return JSON only: {"judgments":[{"index":1,"relevance":2,"reason":"..."}, ...]} '
    "with one entry per candidate index."
)


def _load_env() -> None:
    env_path = PROJECT_ROOT / "agent" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            out.append(json.loads(text))
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def _read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for rec in _read_jsonl(path):
        if rec.get("status") == "success":
            done.add(str(rec.get("case_id") or ""))
    return done


def _build_candidate_block(pool: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, doc in enumerate(pool, start=1):
        title = str(doc.get("title") or "").strip()
        summary = str(doc.get("summary") or "").strip()
        if len(summary) > 500:
            summary = summary[:500] + "..."
        lines.append(f"[{i}] Title: {title}\n    Summary: {summary}")
    return "\n".join(lines)


def _judge_query(
    query: str,
    pool: list[dict[str, Any]],
    *,
    client: Any,
    coerce_text_fn: Any,
    extract_json_fn: Any,
) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433

    block = _build_candidate_block(pool)
    messages = [
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Query: {query}\n\n"
                f"Candidates ({len(pool)}):\n{block}\n\n"
                "Return JSON only."
            )
        ),
    ]
    raw = client.invoke(messages)
    text = coerce_text_fn(getattr(raw, "content", raw))
    parsed = extract_json_fn(text or "")
    by_index: dict[int, dict[str, Any]] = {}
    if isinstance(parsed, dict):
        for item in parsed.get("judgments") or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            try:
                rel = int(item.get("relevance"))
            except (TypeError, ValueError):
                rel = 0
            rel = max(0, min(2, rel))
            by_index[idx] = {"relevance": rel, "reason": str(item.get("reason") or "")}

    judgments: list[dict[str, Any]] = []
    for i, doc in enumerate(pool, start=1):
        verdict = by_index.get(i, {"relevance": 0, "reason": "(no judgment returned)"})
        judgments.append(
            {
                "url": str(doc.get("url") or ""),
                "title": str(doc.get("title") or ""),
                "relevance": verdict["relevance"],
                "reason": verdict["reason"],
            }
        )
    return judgments


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G2 relevance judge.")
    parser.add_argument("--results", type=Path, default=here / "runs" / "retrieval_results.jsonl")
    parser.add_argument("--output", type=Path, default=here / "runs" / "relevance_judgments.jsonl")
    parser.add_argument("--only-case-id", type=str, default=None)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Fixed delay between judged queries to stay under the model RPM quota.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_env()

    results_path = args.results.resolve()
    if not results_path.exists():
        print(f"Retrieval results not found: {results_path}")
        return 1
    raw_cases = [r for r in _read_jsonl(results_path) if r.get("status") == "success"]
    # dedupe by case_id (retrieval output may have resume duplicates), keep last
    by_id: dict[str, dict[str, Any]] = {}
    for r in raw_cases:
        by_id[str(r.get("case_id") or "")] = r
    cases = list(by_id.values())
    if args.only_case_id:
        cases = [c for c in cases if str(c.get("case_id")) == args.only_case_id]
        if not cases:
            print(f"No case matching case_id={args.only_case_id}")
            return 1

    output_path = args.output.resolve()
    done = _read_done(output_path)
    if done:
        print(f"Resume: {len(done)} query(ies) already judged; skipping them.")

    from agent.graph.model_io import _coerce_to_text, _extract_json_object  # noqa: E402
    from services.llm_provider import build_chat_model, resolve_agent_model_config  # noqa: E402

    config = resolve_agent_model_config()
    client = build_chat_model(
        provider=config.provider,
        model_name=config.model,
        temperature=0.0,
        default_provider=config.provider,
        default_model=config.model,
    )
    print(f"Judge model: provider={config.provider} model={config.model}")

    pending = [c for c in cases if str(c.get("case_id")) not in done]
    print(f"Total queries: {len(cases)}; pending: {len(pending)}")

    success = 0
    error = 0
    for idx, case in enumerate(pending, start=1):
        case_id = str(case.get("case_id") or "")
        query = str(case.get("query") or "")
        pool = list(case.get("pool") or [])
        print(f"[{idx}/{len(pending)}] {case_id}: judging {len(pool)} docs ...", flush=True)
        try:
            judgments = call_with_retry(
                lambda: _judge_query(
                    query,
                    pool,
                    client=client,
                    coerce_text_fn=_coerce_to_text,
                    extract_json_fn=_extract_json_object,
                ),
                label=f"{case_id} judge",
            )
            dist = {0: 0, 1: 0, 2: 0}
            for j in judgments:
                dist[j["relevance"]] = dist.get(j["relevance"], 0) + 1
            record = {
                "case_id": case_id,
                "query": query,
                "judgments": judgments,
                "status": "success",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  -> relevance dist 0/1/2 = {dist[0]}/{dist[1]}/{dist[2]}")
            success += 1
        except Exception as exc:  # noqa: BLE001
            record = {
                "case_id": case_id,
                "query": query,
                "status": "error",
                "error_message": f"{type(exc).__name__}: {exc}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  -> ERROR: {record['error_message']}")
            error += 1
        _append_jsonl(output_path, record)
        if args.sleep_seconds > 0 and idx < len(pending):
            time.sleep(args.sleep_seconds)

    print("=" * 60)
    print(f"Done. success={success}, error={error}, output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
