"""G4 intent classification eval runner (option A: direct intent_router call).

Reads eval/intent_eval/queries.jsonl, replicates the intent_router pipeline
(heuristic intent -> intent_router LLM -> merge) without running any downstream
tools, records predictions, and writes an auto-generated report.

Resume support: if predictions.jsonl already contains rows with status=success,
those case_ids are skipped on re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from eval.eval_retry import call_with_retry  # noqa: E402


def _load_env() -> None:
    env_path = PROJECT_ROOT / "agent" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    if not str(os.getenv("AGENT_MODEL_PROVIDER", "")).strip():
        os.environ["AGENT_MODEL_PROVIDER"] = "vertex"


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


def _classify_one(
    question: str,
    *,
    handle: Any,
    heuristic_fn: Any,
    merge_fn: Any,
    extract_json_fn: Any,
    coerce_text_fn: Any,
    system_prompt: str,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433

    heuristic = heuristic_fn(question)
    if handle.client is None:
        merged = dict(heuristic)
        merged["_source"] = "heuristic_only_no_client"
        merged["_raw_llm_text"] = ""
        return merged

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                "Recent conversation context:\n(none)\n\n"
                f"Current user message:\n{question}\n\n"
                "Return JSON only."
            )
        ),
    ]
    raw_result = handle.client.invoke(messages)
    text = coerce_text_fn(getattr(raw_result, "content", raw_result))
    model_intent = extract_json_fn(text or "")
    if isinstance(model_intent, dict):
        merged = merge_fn(heuristic, model_intent)
        merged["_source"] = "heuristic+llm"
    else:
        merged = dict(heuristic)
        merged["_source"] = "heuristic_only_llm_unparsed"
    merged["_raw_llm_text"] = text or ""
    return merged


def _is_correct(record: dict[str, Any]) -> bool:
    expected = str(record.get("expected_intent_type") or "")
    if expected == "needs_clarification":
        return str(record.get("predicted_route") or "") == "needs_clarification"
    return str(record.get("predicted_intent_type") or "") == expected


def _build_report(
    cases: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> str:
    total = len(cases)
    correct = 0
    per_bucket_total: Counter = Counter()
    per_bucket_correct: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    missing_cases: list[str] = []

    for case in cases:
        case_id = str(case.get("case_id") or "")
        expected = str(case.get("expected_intent_type") or "")
        per_bucket_total[expected] += 1
        rec = predictions.get(case_id)
        if not rec or rec.get("status") != "success":
            missing_cases.append(case_id)
            confusion[expected]["(missing)"] += 1
            continue
        if _is_correct(rec):
            correct += 1
            per_bucket_correct[expected] += 1
        if expected == "needs_clarification":
            confusion[expected][str(rec.get("predicted_route") or "(empty)")] += 1
        else:
            confusion[expected][str(rec.get("predicted_intent_type") or "(empty)")] += 1

    accuracy = (correct / total * 100.0) if total else 0.0

    lines: list[str] = []
    lines.append("# G4 Intent Classification Eval Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"## Headline: intent_type_accuracy = **{accuracy:.1f}%**  ({correct}/{total})")
    lines.append("")
    if accuracy >= 90.0:
        lines.append("**Status: stable** (>=90%). Per the agreed thresholds, no deeper investigation needed.")
    elif accuracy < 85.0:
        lines.append("**Status: investigate** (<85%). Review confusion matrix below to identify failing buckets.")
    else:
        lines.append("**Status: marginal** (85-90%).")
    lines.append("")

    if missing_cases:
        lines.append(f"> Warning: {len(missing_cases)} case(s) missing prediction (status != success).")
        lines.append("")

    lines.append("## Per-bucket accuracy")
    lines.append("")
    lines.append("| Bucket (expected_intent_type) | Accuracy | Correct | Total |")
    lines.append("|---|---|---|---|")
    for bucket in sorted(per_bucket_total.keys()):
        bt = per_bucket_total[bucket]
        bc = per_bucket_correct[bucket]
        acc = (bc / bt * 100.0) if bt else 0.0
        lines.append(f"| `{bucket}` | {acc:.1f}% | {bc} | {bt} |")
    lines.append("")

    lines.append("## Confusion matrix")
    lines.append("")
    lines.append(
        "Rows = expected; columns = predicted "
        "(predicted intent_type, except for `needs_clarification` row where the column is predicted route)."
    )
    lines.append("")
    all_predicted = sorted({p for row in confusion.values() for p in row.keys()})
    if all_predicted:
        header = "| Expected \\ Predicted | " + " | ".join(f"`{p}`" for p in all_predicted) + " |"
        sep = "|---|" + "|".join(["---"] * len(all_predicted)) + "|"
        lines.append(header)
        lines.append(sep)
        for bucket in sorted(confusion.keys()):
            row = [f"| `{bucket}`"]
            for p in all_predicted:
                row.append(str(confusion[bucket].get(p, 0)))
            lines.append(" | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G4 intent classification eval runner.")
    parser.add_argument("--queries", type=Path, default=here / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=here / "runs" / "g4_predictions.jsonl")
    parser.add_argument("--report", type=Path, default=here / "report.md")
    parser.add_argument("--only-case-id", type=str, default=None)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip running cases; only rebuild the report from existing predictions.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Fixed delay between cases to stay under the model RPM quota.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_env()

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
    predictions: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for rec in _read_jsonl(output_path):
            cid = str(rec.get("case_id") or "")
            if cid:
                predictions[cid] = rec

    if not args.report_only:
        done_ids = {cid for cid, rec in predictions.items() if rec.get("status") == "success"}
        if done_ids:
            print(f"Resume: {len(done_ids)} case(s) already succeeded; skipping them.")

        from agent.graph.intent_heuristics import (  # noqa: E402
            _heuristic_intent,
            _merge_intent,
        )
        from agent.graph.model_io import _coerce_to_text, _extract_json_object  # noqa: E402
        from agent.graph.models import build_graph_models  # noqa: E402
        from agent.graph.prompts import _INTENT_ROUTER_SYSTEM_PROMPT  # noqa: E402

        models = build_graph_models()
        handle = models.intent_router
        print(
            f"Intent router model: provider={handle.provider} model={handle.model} "
            f"client={'ready' if handle.client else 'MISSING'}"
        )

        pending = [c for c in cases if str(c.get("case_id")) not in done_ids]
        print(f"Total cases: {len(cases)}; pending: {len(pending)}")

        for idx, case in enumerate(pending, start=1):
            case_id = str(case.get("case_id") or "")
            question = str(case.get("question") or "")
            print(f"[{idx}/{len(pending)}] {case_id} ...", flush=True)
            try:
                predicted = call_with_retry(
                    lambda: _classify_one(
                        question,
                        handle=handle,
                        heuristic_fn=_heuristic_intent,
                        merge_fn=_merge_intent,
                        extract_json_fn=_extract_json_object,
                        coerce_text_fn=_coerce_to_text,
                        system_prompt=_INTENT_ROUTER_SYSTEM_PROMPT,
                    ),
                    label=f"{case_id} intent",
                )
                record = {
                    "case_id": case_id,
                    "expected_intent_type": str(case.get("expected_intent_type") or ""),
                    "expected_route": str(case.get("expected_route") or ""),
                    "question": question,
                    "predicted_route": str(predicted.get("route") or ""),
                    "predicted_intent_type": str(predicted.get("intent_type") or ""),
                    "predicted_source": str(predicted.get("_source") or ""),
                    "raw_llm_text": str(predicted.get("_raw_llm_text") or ""),
                    "status": "success",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                ok = _is_correct(record)
                print(
                    "  -> "
                    + ("OK" if ok else "WRONG")
                    + f" predicted intent_type={record['predicted_intent_type']} "
                    + f"route={record['predicted_route']}"
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "case_id": case_id,
                    "expected_intent_type": str(case.get("expected_intent_type") or ""),
                    "expected_route": str(case.get("expected_route") or ""),
                    "question": question,
                    "predicted_route": "",
                    "predicted_intent_type": "",
                    "predicted_source": "",
                    "raw_llm_text": "",
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                print(f"  -> ERROR: {record['error_message']}")
            _append_jsonl(output_path, record)
            predictions[case_id] = record
            if args.sleep_seconds > 0 and idx < len(pending):
                time.sleep(args.sleep_seconds)

    report = _build_report(cases, predictions)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
