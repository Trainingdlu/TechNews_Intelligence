"""G2 Phase C — compute IR metrics per ablation config from rankings + judgments.

Reads:
  runs/retrieval_results.jsonl   (per query: rankings per config G0/G1/G2)
  runs/relevance_judgments.jsonl (per query: url -> relevance 0/1/2, shared gold)

Computes per config, averaged over queries:
  Hit@5        - fraction of queries with >=1 relevant (rel>=1) doc in top-5
  Precision@5  - mean fraction of top-5 that are relevant (rel>=1)
  MRR@10       - mean reciprocal rank of the first relevant (rel>=1) doc in top-10
  nDCG@10      - mean normalized DCG with exponential gain (2^rel - 1)

Writes report.md with the 3-config ablation table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CONFIGS = ["G0", "G1", "G2"]
CONFIG_LABELS = {
    "G0": "基础召回 + 不重排",
    "G1": "宽召回 + 不重排",
    "G2": "宽召回 + Jina 重排",
}


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


def _hit_at_k(rels: list[int], k: int) -> float:
    return 1.0 if any(r >= 1 for r in rels[:k]) else 0.0


def _precision_at_k(rels: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for r in rels[:k] if r >= 1) / float(k)


def _mrr_at_k(rels: list[int], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r >= 1:
            return 1.0 / i
    return 0.0


def _dcg(rels: list[int], k: int) -> float:
    total = 0.0
    for i, r in enumerate(rels[:k], start=1):
        total += ((2 ** r) - 1) / math.log2(i + 1)
    return total


def _ndcg_at_k(ranked_rels: list[int], all_rels: list[int], k: int) -> float:
    idcg = _dcg(sorted(all_rels, reverse=True), k)
    if idcg <= 0:
        return 0.0
    return _dcg(ranked_rels, k) / idcg


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G2 compute IR metrics.")
    parser.add_argument("--results", type=Path, default=here / "runs" / "retrieval_results.jsonl")
    parser.add_argument("--judgments", type=Path, default=here / "runs" / "relevance_judgments.jsonl")
    parser.add_argument("--report", type=Path, default=here / "report.md")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results_path = args.results.resolve()
    judgments_path = args.judgments.resolve()
    if not results_path.exists():
        print(f"Retrieval results not found: {results_path}")
        return 1
    if not judgments_path.exists():
        print(f"Judgments not found: {judgments_path}")
        return 1

    results = _dedupe_by_case(_read_jsonl(results_path))
    judgments = _dedupe_by_case(_read_jsonl(judgments_path))

    rel_by_query: dict[str, dict[str, int]] = {}
    all_rels_by_query: dict[str, list[int]] = {}
    for case_id, jrec in judgments.items():
        url_map: dict[str, int] = {}
        rels: list[int] = []
        for j in jrec.get("judgments") or []:
            url = str(j.get("url") or "")
            rel = int(j.get("relevance") or 0)
            if url:
                url_map[url] = rel
                rels.append(rel)
        rel_by_query[case_id] = url_map
        all_rels_by_query[case_id] = rels

    scored_ids = [cid for cid in results if cid in rel_by_query]
    missing = [cid for cid in results if cid not in rel_by_query]

    per_config: dict[str, dict[str, float]] = {
        c: {"hit5": 0.0, "p5": 0.0, "mrr10": 0.0, "ndcg10": 0.0} for c in CONFIGS
    }
    n = len(scored_ids)

    for cid in scored_ids:
        url_rel = rel_by_query[cid]
        all_rels = all_rels_by_query[cid]
        rankings = results[cid].get("rankings") or {}
        for cfg in CONFIGS:
            ranked = rankings.get(cfg) or []
            ranked_urls = [
                str(item.get("url") or "")
                for item in sorted(ranked, key=lambda x: int(x.get("rank") or 0))
            ]
            ranked_rels = [int(url_rel.get(u, 0)) for u in ranked_urls]
            per_config[cfg]["hit5"] += _hit_at_k(ranked_rels, 5)
            per_config[cfg]["p5"] += _precision_at_k(ranked_rels, 5)
            per_config[cfg]["mrr10"] += _mrr_at_k(ranked_rels, 10)
            per_config[cfg]["ndcg10"] += _ndcg_at_k(ranked_rels, all_rels, 10)

    for cfg in CONFIGS:
        for key in per_config[cfg]:
            per_config[cfg][key] = per_config[cfg][key] / n if n else 0.0

    lines: list[str] = []
    lines.append("# G2 检索质量评测报告")
    lines.append("")
    lines.append(f"已评分查询数：**{n}**" + (f"（缺少判分：{len(missing)}）" if missing else ""))
    lines.append("")
    lines.append("相关性标注：对 3 组配置 pooled top-10 结果进行 LLM 0/1/2 判分，并在各配置间共享标签。")
    lines.append("")
    lines.append("## 消融对比表")
    lines.append("")
    lines.append("| 配置 | 检索策略 | Hit@5 | Precision@5 | MRR@10 | nDCG@10 |")
    lines.append("|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        m = per_config[cfg]
        lines.append(
            f"| {cfg} | {CONFIG_LABELS[cfg]} | {m['hit5']*100:.1f}% | "
            f"{m['p5']*100:.1f}% | {m['mrr10']:.3f} | {m['ndcg10']:.3f} |"
        )
    lines.append("")

    base = per_config["G0"]
    lines.append("## 相对 G0（基线）的变化")
    lines.append("")
    lines.append("| 配置 | Hit@5 | Precision@5 | MRR@10 | nDCG@10 |")
    lines.append("|---|---|---|---|---|")
    for cfg in ("G1", "G2"):
        m = per_config[cfg]
        lines.append(
            f"| {cfg} | {(m['hit5']-base['hit5'])*100:+.1f}pt | "
            f"{(m['p5']-base['p5'])*100:+.1f}pt | "
            f"{m['mrr10']-base['mrr10']:+.3f} | "
            f"{m['ndcg10']-base['ndcg10']:+.3f} |"
        )
    lines.append("")

    report_path = args.report.resolve()
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    for cfg in CONFIGS:
        m = per_config[cfg]
        print(
            f"  {cfg}: Hit@5={m['hit5']*100:.1f}% P@5={m['p5']*100:.1f}% "
            f"MRR@10={m['mrr10']:.3f} nDCG@10={m['ndcg10']:.3f}"
        )
    if missing:
        print(f"  Missing judgments for: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
