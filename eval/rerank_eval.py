"""Offline benchmark for comparing recall order vs rerank order."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / "agent" / ".env")

try:
    from agent.skills.rerank import rerank_candidates
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from agent.skills.rerank import rerank_candidates  # type: ignore


def _dcg(labels: list[int]) -> float:
    import math

    score = 0.0
    for idx, rel in enumerate(labels, 1):
        gain = (2 ** max(int(rel), 0)) - 1
        score += gain / math.log2(idx + 1)
    return score


def _ndcg(labels: list[int]) -> float:
    if not labels:
        return 0.0
    ideal = sorted(labels, reverse=True)
    best = _dcg(ideal)
    if best <= 0:
        return 0.0
    return _dcg(labels) / best


def _mrr(labels: list[int]) -> float:
    for idx, rel in enumerate(labels, 1):
        if int(rel) > 0:
            return 1.0 / idx
    return 0.0


def _load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


def _labels_for_order(candidates: list[dict[str, Any]], top_k: int) -> list[int]:
    labels: list[int] = []
    for item in candidates[:top_k]:
        labels.append(int(item.get("label", 0) or 0))
    return labels


def evaluate_dataset(cases: list[dict[str, Any]], *, mode: str, top_k: int) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    baseline_ndcg: list[float] = []
    baseline_mrr: list[float] = []
    rerank_ndcg: list[float] = []
    rerank_mrr: list[float] = []
    fallback_count = 0

    for case in cases:
        query = str(case.get("query") or "").strip()
        case_id = str(case.get("id") or "").strip() or query
        raw_candidates = case.get("candidates")
        if not query or not isinstance(raw_candidates, list) or not raw_candidates:
            continue

        candidates = [dict(item) for item in raw_candidates if isinstance(item, dict)]
        if not candidates:
            continue

        k = max(1, min(int(top_k), len(candidates)))
        baseline_labels = _labels_for_order(candidates, k)
        baseline_ndcg_val = _ndcg(baseline_labels)
        baseline_mrr_val = _mrr(baseline_labels)

        reranked, rerank_meta = rerank_candidates(query, candidates, mode=mode, top_k=k)
        reranked_labels = _labels_for_order(reranked, k)
        rerank_ndcg_val = _ndcg(reranked_labels)
        rerank_mrr_val = _mrr(reranked_labels)
        if bool(rerank_meta.get("fallback")):
            fallback_count += 1

        baseline_ndcg.append(baseline_ndcg_val)
        baseline_mrr.append(baseline_mrr_val)
        rerank_ndcg.append(rerank_ndcg_val)
        rerank_mrr.append(rerank_mrr_val)
        reports.append(
            {
                "id": case_id,
                "query": query,
                "baseline_labels": baseline_labels,
                "reranked_labels": reranked_labels,
                "baseline_ndcg": baseline_ndcg_val,
                "reranked_ndcg": rerank_ndcg_val,
                "baseline_mrr": baseline_mrr_val,
                "reranked_mrr": rerank_mrr_val,
                "rerank": rerank_meta,
            }
        )

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "case_count": len(reports),
        "mode": mode,
        "top_k": int(top_k),
        "fallback_count": int(fallback_count),
        "baseline_avg_ndcg": _avg(baseline_ndcg),
        "reranked_avg_ndcg": _avg(rerank_ndcg),
        "delta_avg_ndcg": _avg(rerank_ndcg) - _avg(baseline_ndcg),
        "baseline_avg_mrr": _avg(baseline_mrr),
        "reranked_avg_mrr": _avg(rerank_mrr),
        "delta_avg_mrr": _avg(rerank_mrr) - _avg(baseline_mrr),
    }
    return {"summary": summary, "cases": reports}


def _build_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate rerank quality on a small offline benchmark.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=eval_dir / "datasets" / "rerank_mini.json",
        help="Path to rerank benchmark dataset JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="llm_rerank",
        help="Rerank mode: none | llm_rerank.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Cutoff K for ranking metrics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=eval_dir / "reports" / "rerank_eval_latest.json",
        help="Path for JSON report output.",
    )
    return parser


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    args = _build_parser(eval_dir).parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cases = _load_cases(dataset_path)
    report = evaluate_dataset(cases, mode=str(args.mode), top_k=max(1, int(args.top_k)))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        **report,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    print(
        "[RerankEval] "
        f"cases={summary['case_count']} "
        f"mode={summary['mode']} top_k={summary['top_k']} "
        f"fallback={summary['fallback_count']} "
        f"baseline_ndcg={summary['baseline_avg_ndcg']:.4f} "
        f"reranked_ndcg={summary['reranked_avg_ndcg']:.4f} "
        f"delta_ndcg={summary['delta_avg_ndcg']:+.4f} "
        f"baseline_mrr={summary['baseline_avg_mrr']:.4f} "
        f"reranked_mrr={summary['reranked_avg_mrr']:.4f} "
        f"delta_mrr={summary['delta_avg_mrr']:+.4f}"
    )
    print(f"[RerankEval] report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
