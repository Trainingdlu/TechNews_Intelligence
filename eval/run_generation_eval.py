"""Run generation-only evaluation with fixed evidence."""

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

try:
    from eval_core import extract_urls, normalize_url_for_retrieval
    from news_eval_schema import load_generation_cases
except ImportError:  # pragma: no cover
    from .eval_core import extract_urls, normalize_url_for_retrieval
    from .news_eval_schema import load_generation_cases


SYSTEM_PROMPT = (
    "你是科技新闻分析助手。只能基于用户提供的证据回答，不要补充证据外事实。"
    "如果证据不足，要明确说明不足。回答末尾列出使用到的来源 URL。"
)


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
    if env_file:
        load_dotenv(dotenv_path=env_file.resolve(), override=True)


def _build_model(provider: str | None, model: str | None) -> Any:
    from services.llm_provider import DEFAULT_VERTEX_MODEL, build_chat_model  # pylint: disable=import-outside-toplevel

    return build_chat_model(
        provider=str(provider or os.getenv("TASK_EVAL_PROVIDER", "vertex")),
        model_name=str(model or os.getenv("TASK_EVAL_MODEL", DEFAULT_VERTEX_MODEL)),
        temperature=0.0,
        default_provider="vertex",
        default_model=DEFAULT_VERTEX_MODEL,
    )


def _evidence_text(evidence: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(evidence, 1):
        title = str(item.get("title") or "").strip()
        quote = str(item.get("quote") or "").strip()
        url = str(item.get("url") or "").strip()
        lines.append(f"[{idx}] 标题：{title}\n证据：{quote}\nURL：{url}")
    return "\n\n".join(lines)


def _invoke_generation(model: Any, case: dict[str, Any]) -> str:
    user = f"问题：{case['question']}\n\n证据：\n{_evidence_text(case['evidence'])}"
    raw = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)])
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item) if isinstance(item, dict) else item) for item in content)
    return str(content or "")


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}", str(value or ""))}


def _claim_covered(answer: str, claim: str) -> bool:
    answer_norm = str(answer or "").lower()
    claim_norm = str(claim or "").lower()
    if claim_norm and claim_norm in answer_norm:
        return True
    claim_tokens = _tokens(claim_norm)
    if not claim_tokens:
        return False
    answer_tokens = _tokens(answer_norm)
    overlap = len(claim_tokens.intersection(answer_tokens)) / max(1, len(claim_tokens))
    return overlap >= 0.45


def _score_generation(case: dict[str, Any], answer: str) -> dict[str, Any]:
    required = [str(item).strip() for item in case.get("required_claims", []) if str(item).strip()]
    covered = [claim for claim in required if _claim_covered(answer, claim)]
    evidence_urls = {
        normalize_url_for_retrieval(str(item.get("url", "")))
        for item in case.get("evidence", [])
        if normalize_url_for_retrieval(str(item.get("url", "")))
    }
    answer_urls = [normalize_url_for_retrieval(url) for url in extract_urls(answer)]
    answer_urls = [url for url in answer_urls if url]
    unsupported_urls = [url for url in answer_urls if url not in evidence_urls]
    forbidden_hits = [
        item
        for item in case.get("forbidden_claims", [])
        if str(item).strip() and str(item).strip().lower() in str(answer).lower()
    ]
    return {
        "required_claim_count": len(required),
        "covered_claim_count": len(covered),
        "claim_coverage": (len(covered) / len(required)) if required else None,
        "evidence_url_count": len(evidence_urls),
        "answer_url_count": len(answer_urls),
        "unsupported_url_count": len(unsupported_urls),
        "unsupported_urls": unsupported_urls,
        "forbidden_hit_count": len(forbidden_hits),
        "forbidden_hits": forbidden_hits,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return (sum(vals) / len(vals)) if vals else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation-only event eval.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/datasets/generation_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/generation_eval_latest.json"))
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_eval_env(args.env_file)
    cases = load_generation_cases(args.dataset)
    if args.max_cases > 0:
        cases = cases[: int(args.max_cases)]
    model = _build_model(args.provider, args.model)
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            answer = _invoke_generation(model, case)
            error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            answer = ""
            error = str(exc)
        scores = _score_generation(case, answer)
        results.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "answer": answer,
                "scores": scores,
                "error": error,
            }
        )
    score_rows = [row["scores"] for row in results]
    summary = {
        "case_count": len(results),
        "avg_claim_coverage": _mean(score_rows, "claim_coverage"),
        "avg_unsupported_url_count": _mean(score_rows, "unsupported_url_count"),
        "avg_forbidden_hit_count": _mean(score_rows, "forbidden_hit_count"),
        "error_count": sum(1 for row in results if row.get("error")),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "provider": args.provider or os.getenv("TASK_EVAL_PROVIDER", "vertex"),
        "model": args.model or os.getenv("TASK_EVAL_MODEL", ""),
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GenerationEval] cases={len(results)} output={args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

