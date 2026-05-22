"""G3 Phase B — judge generation faithfulness + relevancy, compute url-leak.

Reads runs/generation_enriched.jsonl and for each (question, evidence, answer):
  - LLM judge: faithfulness (1-5) and answer_relevancy (1-5)
  - deterministic: url_leak = URLs appearing in the answer that are NOT in the
    evidence set the answer was supposed to be grounded in

Writes runs/faithfulness_judgments.jsonl and report.md.
Resume support: cases already judged (status=success) are skipped.
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
    "You are a strict faithfulness judge for a tech-news analysis agent. "
    "You are given a user QUESTION, the EVIDENCE the answer was supposed to be grounded in "
    "(a list of article title + summary), and the agent's ANSWER. Rate two things:\n"
    "faithfulness (1-5): are the ANSWER's factual claims supported by the EVIDENCE?\n"
    "  5 = every claim is supported by the evidence\n"
    "  4 = almost all supported; at most a trivial unsupported detail\n"
    "  3 = some claims are not supported by the evidence\n"
    "  2 = several unsupported / fabricated claims\n"
    "  1 = largely ungrounded or fabricated\n"
    "answer_relevancy (1-5): does the ANSWER actually address the QUESTION? "
    "(5 = fully on point, 1 = off topic)\n"
    "Judge ONLY against the provided evidence; do not use outside knowledge. "
    'Return JSON only: {"faithfulness":N,"answer_relevancy":N,'
    '"unsupported_claims":["..."],"reason":"..."}'
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for rec in _read_jsonl(path):
        if rec.get("status") == "success":
            done.add(str(rec.get("case_id") or ""))
    return done


def _answer_url_leaks(answer: str, evidence_urls: list[str]) -> list[str]:
    from agent.core.evidence import extract_urls, normalize_url_for_match  # noqa: E402

    normalized_evidence = {
        normalized
        for url in evidence_urls
        if (normalized := normalize_url_for_match(str(url).strip()))
    }
    leaks: list[str] = []
    for url in sorted(set(extract_urls(answer))):
        normalized = normalize_url_for_match(url)
        if normalized and normalized not in normalized_evidence:
            leaks.append(url)
    return leaks


def _refresh_url_leaks(
    judgments: dict[str, dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
) -> bool:
    changed = False
    for case_id, judgment in judgments.items():
        if judgment.get("status") != "success":
            continue
        case = cases_by_id.get(case_id)
        if not case:
            continue
        leaks = _answer_url_leaks(
            str(case.get("answer") or ""),
            [str(u) for u in (case.get("evidence_urls") or [])],
        )
        if judgment.get("url_leak_count") != len(leaks) or judgment.get("leaked_urls") != leaks:
            judgment["url_leak_count"] = len(leaks)
            judgment["leaked_urls"] = leaks
            changed = True
    return changed


def _evidence_block(evidence: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, doc in enumerate(evidence, start=1):
        title = str(doc.get("title") or doc.get("title_cn") or "").strip()
        summary = str(doc.get("summary") or "").strip()
        if len(summary) > 600:
            summary = summary[:600] + "..."
        lines.append(f"[{i}] {title}\n    {summary}\n    URL: {doc.get('url','')}")
    return "\n".join(lines)


def _clamp_score(value: Any) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 0
    return max(1, min(5, v))


def _judge_one(
    question: str,
    evidence_text: str,
    answer: str,
    *,
    client: Any,
    coerce_text_fn: Any,
    extract_json_fn: Any,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433

    messages = [
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"QUESTION:\n{question}\n\n"
                f"EVIDENCE (exact context the agent generated from):\n{evidence_text}\n\n"
                f"ANSWER:\n{answer}\n\n"
                "Return JSON only."
            )
        ),
    ]
    raw = client.invoke(messages)
    text = coerce_text_fn(getattr(raw, "content", raw))
    parsed = extract_json_fn(text or "")
    if not isinstance(parsed, dict):
        return {"faithfulness": 0, "answer_relevancy": 0, "unsupported_claims": [], "reason": "parse_failed"}
    return {
        "faithfulness": _clamp_score(parsed.get("faithfulness")),
        "answer_relevancy": _clamp_score(parsed.get("answer_relevancy")),
        "unsupported_claims": list(parsed.get("unsupported_claims") or []),
        "reason": str(parsed.get("reason") or ""),
    }


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G3 faithfulness judge.")
    parser.add_argument("--results", type=Path, default=here / "runs" / "generation_enriched.jsonl")
    parser.add_argument("--output", type=Path, default=here / "runs" / "faithfulness_judgments.jsonl")
    parser.add_argument("--report", type=Path, default=here / "report.md")
    parser.add_argument("--only-case-id", type=str, default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Fixed delay between judged cases to stay under the model RPM quota.",
    )
    return parser.parse_args()


def _build_report(judgments: list[dict[str, Any]]) -> str:
    ok = [j for j in judgments if j.get("status") == "success"]
    n = len(ok)
    if n == 0:
        return "# G3 Faithfulness Report\n\nNo judged cases.\n"

    avg_faith = sum(j["faithfulness"] for j in ok) / n
    avg_rel = sum(j["answer_relevancy"] for j in ok) / n
    halluc = sum(1 for j in ok if j["faithfulness"] <= 2) / n
    grounded = sum(1 for j in ok if j["faithfulness"] >= 4) / n
    leak_any = sum(1 for j in ok if j.get("url_leak_count", 0) > 0) / n
    total_leaks = sum(int(j.get("url_leak_count", 0)) for j in ok)

    lines: list[str] = []
    lines.append("# G3 Generation Faithfulness Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"Cases judged: **{n}**")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Faithfulness (avg, 1-5) | **{avg_faith:.2f}** |")
    lines.append(f"| Answer relevancy (avg, 1-5) | **{avg_rel:.2f}** |")
    lines.append(f"| Well-grounded rate (faithfulness>=4) | {grounded*100:.1f}% |")
    lines.append(f"| Hallucination rate (faithfulness<=2) | {halluc*100:.1f}% |")
    lines.append(f"| URL-leak rate (>=1 url outside evidence) | {leak_any*100:.1f}% |")
    lines.append(f"| Total leaked URLs | {total_leaks} |")
    lines.append("")
    lines.append("## Lowest-faithfulness cases")
    lines.append("")
    worst = sorted(ok, key=lambda j: j["faithfulness"])[:8]
    for j in worst:
        lines.append(
            f"- `{j['case_id']}` faithfulness={j['faithfulness']} relevancy={j['answer_relevancy']} "
            f"leaks={j.get('url_leak_count',0)} — {str(j.get('reason') or '')[:140]}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    _load_env()

    results_path = args.results.resolve()
    if not results_path.exists():
        print(f"Generation results not found: {results_path}")
        return 1
    raw = [r for r in _read_jsonl(results_path) if r.get("status") == "success"]
    by_id = {str(r.get("case_id") or ""): r for r in raw}
    cases = list(by_id.values())
    if args.only_case_id:
        cases = [c for c in cases if str(c.get("case_id")) == args.only_case_id]

    output_path = args.output.resolve()
    existing = {str(r.get("case_id") or ""): r for r in _read_jsonl(output_path)} if output_path.exists() else {}

    if not args.report_only:
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

        done = {cid for cid, r in existing.items() if r.get("status") == "success"}
        pending = [c for c in cases if str(c.get("case_id")) not in done]
        print(f"Total: {len(cases)}; pending: {len(pending)}")

        for idx, case in enumerate(pending, start=1):
            case_id = str(case.get("case_id") or "")
            question = str(case.get("question") or "")
            answer = str(case.get("answer") or "")
            evidence_urls = set(str(u).strip() for u in (case.get("evidence_urls") or []))
            # Prefer the exact synthesizer context (tool stats + full text); fall back to summaries.
            evidence_text = str(case.get("synthesizer_evidence") or "").strip()
            if not evidence_text:
                evidence_text = _evidence_block(list(case.get("evidence") or []))
            print(f"[{idx}/{len(pending)}] {case_id} ...", flush=True)
            try:
                verdict = call_with_retry(
                    lambda: _judge_one(
                        question, evidence_text, answer,
                        client=client,
                        coerce_text_fn=_coerce_to_text,
                        extract_json_fn=_extract_json_object,
                    ),
                    label=f"{case_id} judge",
                )
                leaks = _answer_url_leaks(answer, list(evidence_urls))
                record = {
                    "case_id": case_id,
                    "question": question,
                    "faithfulness": verdict["faithfulness"],
                    "answer_relevancy": verdict["answer_relevancy"],
                    "unsupported_claims": verdict["unsupported_claims"],
                    "reason": verdict["reason"],
                    "url_leak_count": len(leaks),
                    "leaked_urls": leaks,
                    "status": "success",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                print(
                    f"  -> faith={verdict['faithfulness']} rel={verdict['answer_relevancy']} leaks={len(leaks)}"
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "case_id": case_id,
                    "question": question,
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                print(f"  -> ERROR: {record['error_message']}")
            _append_jsonl(output_path, record)
            existing[case_id] = record
            if args.sleep_seconds > 0 and idx < len(pending):
                time.sleep(args.sleep_seconds)

    if _refresh_url_leaks(existing, by_id):
        _write_jsonl(output_path, list(existing.values()))
        print(f"Refreshed deterministic URL-leak fields in {output_path}")

    report = _build_report(list(existing.values()))
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
