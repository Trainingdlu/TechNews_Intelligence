"""G2 retrieval runner — collect ranked results under 3 ablation configs + pool.

For each query, runs the production retrieval (lookup_candidates_by_query) under
three configs and records the top-10 ranking of each, plus the pooled union of
URLs (with title/summary) for later relevance judging.

Ablation configs (reusing the existing matrix.json semantics):
  G0: base recall profile  + no rerank   (baseline)
  G1: wide recall profile  + no rerank
  G2: wide recall profile  + Jina llm_rerank

Prerequisite: n8n ingestion paused so the DB is a stable snapshot across configs.
"""

from __future__ import annotations

import argparse
import json
import os
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

TOP_K = 10

CONFIGS: dict[str, dict[str, str]] = {
    "G0": {"recall_profile": "base", "rerank_mode": "none"},
    "G1": {"recall_profile": "wide", "rerank_mode": "none"},
    "G2": {"recall_profile": "wide", "rerank_mode": "llm_rerank"},
}


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


def _confirm_n8n_paused() -> bool:
    print("=" * 60)
    print("PREREQUISITE: n8n ingestion must be paused (stable DB snapshot).")
    print("=" * 60)
    return input("Confirm n8n workflows are paused? [y/N] ").strip().lower() == "y"


def _run_one_query(
    query: str,
    days: int,
    *,
    lookup_fn: Any,
) -> dict[str, Any]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    pool: dict[str, dict[str, Any]] = {}

    for cfg_name, cfg in CONFIGS.items():
        os.environ["EVAL_RECALL_PROFILE"] = cfg["recall_profile"]
        candidates, _meta = lookup_fn(
            query,
            days=days,
            limit=TOP_K,
            rerank_mode=cfg["rerank_mode"],
        )
        ranked: list[dict[str, Any]] = []
        for rank, cand in enumerate(candidates[:TOP_K], start=1):
            url = str(cand.get("url") or "").strip()
            if not url:
                continue
            ranked.append(
                {
                    "rank": rank,
                    "url": url,
                    "score": float(cand.get("score") or 0.0),
                }
            )
            if url not in pool:
                pool[url] = {
                    "url": url,
                    "title": str(cand.get("title") or ""),
                    "summary": str(cand.get("summary") or ""),
                }
        rankings[cfg_name] = ranked

    return {"rankings": rankings, "pool": list(pool.values())}


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G2 retrieval runner (3-config ablation + pooling).")
    parser.add_argument("--queries", type=Path, default=here / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=here / "runs" / "retrieval_results.jsonl")
    parser.add_argument("--only-case-id", type=str, default=None)
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Fixed delay between queries to stay under retrieval/rerank rate limits.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_env()

    if not args.skip_confirm and not _confirm_n8n_paused():
        print("Aborted: n8n not confirmed paused.")
        return 1

    queries_path = args.queries.resolve()
    if not queries_path.exists():
        print(f"Queries not found: {queries_path}")
        return 1
    cases = _read_jsonl(queries_path)
    if args.only_case_id:
        cases = [c for c in cases if str(c.get("case_id")) == args.only_case_id]
        if not cases:
            print(f"No case matching case_id={args.only_case_id}")
            return 1

    output_path = args.output.resolve()
    done = _read_done(output_path)
    if done:
        print(f"Resume: {len(done)} query(ies) already done; skipping them.")

    from agent.tools.retrieval import lookup_candidates_by_query  # noqa: E402

    pending = [c for c in cases if str(c.get("case_id")) not in done]
    print(f"Total queries: {len(cases)}; pending: {len(pending)}")

    success = 0
    error = 0
    for idx, case in enumerate(pending, start=1):
        case_id = str(case.get("case_id") or "")
        query = str(case.get("query") or "")
        days = int(case.get("days") or 30)
        print(f"[{idx}/{len(pending)}] {case_id}: {query} ...", flush=True)
        try:
            result = call_with_retry(
                lambda: _run_one_query(query, days, lookup_fn=lookup_candidates_by_query),
                label=f"{case_id} retrieval",
            )
            record = {
                "case_id": case_id,
                "query": query,
                "days": days,
                "rankings": result["rankings"],
                "pool": result["pool"],
                "status": "success",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            pool_n = len(result["pool"])
            sizes = {k: len(v) for k, v in result["rankings"].items()}
            print(f"  -> pool={pool_n} sizes={sizes}")
            success += 1
        except Exception as exc:  # noqa: BLE001
            record = {
                "case_id": case_id,
                "query": query,
                "days": days,
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
