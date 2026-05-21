"""G2 Phase D (1/2) — build a BLIND human-calibration sheet.

Randomly samples N (query, doc) pairs from the judged pool and writes a markdown
sheet for the user to label relevance 0/1/2 INDEPENDENTLY. The LLM judge's score
is deliberately NOT shown (blind labeling) so the resulting Cohen's kappa measures
genuine LLM-vs-human agreement instead of anchoring the human to the LLM.

A hidden answer-key (LLM scores) is written to a separate file keyed by pair id;
compute_kappa.py joins the human labels back to it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            out.append(json.loads(text))
    return out


def _dedupe_by_case(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("status") and r.get("status") != "success":
            continue
        by_id[str(r.get("case_id") or "")] = r
    return by_id


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build blind calibration sheet.")
    parser.add_argument("--judgments", type=Path, default=here / "runs" / "relevance_judgments.jsonl")
    parser.add_argument("--results", type=Path, default=here / "runs" / "retrieval_results.jsonl")
    parser.add_argument("--sheet", type=Path, default=here / "calibration_sheet.md")
    parser.add_argument("--key", type=Path, default=here / "runs" / "calibration_key.jsonl")
    parser.add_argument("--n", type=int, default=100, help="Number of pairs to sample.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    judgments = _dedupe_by_case(_read_jsonl(args.judgments.resolve()))
    results = _dedupe_by_case(_read_jsonl(args.results.resolve()))

    # query text + per-doc summary come from retrieval pool; relevance from judgments
    pool_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for cid, rrec in results.items():
        pool_by_case[cid] = {str(d.get("url") or ""): d for d in (rrec.get("pool") or [])}

    pairs: list[dict[str, Any]] = []
    for cid, jrec in judgments.items():
        query = str(jrec.get("query") or "")
        pool = pool_by_case.get(cid, {})
        for j in jrec.get("judgments") or []:
            url = str(j.get("url") or "")
            doc = pool.get(url, {})
            pairs.append(
                {
                    "pair_id": f"{cid}::{url}",
                    "case_id": cid,
                    "query": query,
                    "url": url,
                    "title": str(j.get("title") or doc.get("title") or ""),
                    "summary": str(doc.get("summary") or ""),
                    "llm_relevance": int(j.get("relevance") or 0),
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    sample = pairs[: max(1, int(args.n))]
    # stable order in the sheet (by case then url) for easier reading
    sample.sort(key=lambda p: p["pair_id"])

    # write hidden key (llm scores)
    key_path = args.key.resolve()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with key_path.open("w", encoding="utf-8") as fh:
        for p in sample:
            fh.write(json.dumps({"pair_id": p["pair_id"], "llm_relevance": p["llm_relevance"]}, ensure_ascii=False) + "\n")

    # write blind sheet (NO llm score shown)
    lines: list[str] = []
    lines.append("# G2 Relevance Calibration Sheet (BLIND)")
    lines.append("")
    lines.append(f"Total pairs to label: **{len(sample)}**")
    lines.append("")
    lines.append("For each pair, judge how relevant the article is to the QUERY and fill `[ ]` with 0, 1, or 2:")
    lines.append("- **2** = highly relevant (article is directly about what the query asks)")
    lines.append("- **1** = partially relevant (mentions the topic/entity but not the focus)")
    lines.append("- **0** = not relevant")
    lines.append("")
    lines.append("Do NOT change the `pair_id` lines. Only fill the `Your score:` brackets.")
    lines.append("")
    lines.append("---")
    lines.append("")
    for i, p in enumerate(sample, start=1):
        summary = p["summary"].strip()
        if len(summary) > 600:
            summary = summary[:600] + "..."
        lines.append(f"## {i}. `{p['pair_id']}`")
        lines.append("")
        lines.append(f"**Query:** {p['query']}")
        lines.append("")
        lines.append(f"**Title:** {p['title']}")
        lines.append("")
        lines.append(f"**Summary:** {summary}")
        lines.append("")
        lines.append("**Your score:** `[ ]`")
        lines.append("")
        lines.append("---")
        lines.append("")

    sheet_path = args.sheet.resolve()
    sheet_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote blind calibration sheet: {sheet_path} ({len(sample)} pairs)")
    print(f"Wrote hidden answer key: {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
