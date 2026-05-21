"""G5 tool-selection eval runner.

Replicates the agent's tool-planning path WITHOUT executing tools or synthesis:
  intent_router (heuristic + LLM merge) -> _select_tools -> tool_worker (LLM)
and records which tool the worker picked.

Captures enough to attribute failures:
  - intent_type / candidates / expected_in_candidates
    -> if expected tool is not even a candidate, the root cause is intent (Layer 1)
  - llm_tools / final_tools / selection_source
    -> if expected tool IS a candidate but not chosen, the root cause is the
       tool_worker prompt (what prompt optimization can fix)

--tool-worker-prompt-file overrides the production tool_worker system prompt so a
new prompt can be measured for a before/after delta without touching production.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402


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


def _classify_intent(
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
        return heuristic
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
    raw = handle.client.invoke(messages)
    text = coerce_text_fn(getattr(raw, "content", raw))
    model_intent = extract_json_fn(text or "")
    if isinstance(model_intent, dict):
        return merge_fn(heuristic, model_intent)
    return heuristic


def _plan_tools(
    question: str,
    intent: dict[str, Any],
    *,
    handle: Any,
    select_tools_fn: Any,
    schema_brief_fn: Any,
    normalize_calls_fn: Any,
    heuristic_calls_fn: Any,
    extract_json_fn: Any,
    coerce_text_fn: Any,
    tool_worker_prompt: str,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433

    candidates = select_tools_fn(intent)
    if not candidates:
        return {
            "candidates": [],
            "llm_tools": [],
            "final_tools": [],
            "selection_source": "no_candidates",
        }

    schema_brief = schema_brief_fn(candidates)
    llm_calls: list[dict[str, Any]] = []
    if handle.client is not None:
        messages = [
            SystemMessage(content=tool_worker_prompt),
            HumanMessage(
                content=(
                    "Recent conversation context:\n(none)\n\n"
                    f"Current user message:\n{question}\n\n"
                    f"Selected tools:\n{schema_brief}\n\n"
                    "Already executed tools: []\n"
                    "Return JSON only."
                )
            ),
        ]
        raw = handle.client.invoke(messages)
        text = coerce_text_fn(getattr(raw, "content", raw))
        model_plan = extract_json_fn(text or "")
        llm_calls = normalize_calls_fn(model_plan)

    if llm_calls:
        final_calls = llm_calls
        source = "llm"
    else:
        final_calls = heuristic_calls_fn(
            user_message=question,
            intent=intent,
            selected_tools=candidates,
            tool_results=[],
        )
        source = "heuristic_fallback" if final_calls else "empty"

    return {
        "candidates": candidates,
        "llm_tools": [str(c.get("name") or "") for c in llm_calls],
        "final_tools": [str(c.get("name") or "") for c in final_calls],
        "selection_source": source,
    }


def _build_report(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> str:
    total = len(cases)
    correct = 0
    worker_caused = 0  # expected in candidates but not chosen
    intent_caused = 0  # expected not even a candidate
    per_tool_total: Counter = Counter()
    per_tool_correct: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    source_counter: Counter = Counter()
    missing = 0

    for case in cases:
        case_id = str(case.get("case_id") or "")
        expected = str(case.get("expected_primary_tool") or "")
        per_tool_total[expected] += 1
        rec = predictions.get(case_id)
        if not rec or rec.get("status") != "success":
            missing += 1
            continue
        predicted = str(rec.get("predicted_primary") or "(none)")
        source_counter[str(rec.get("selection_source") or "")] += 1
        confusion[expected][predicted] += 1
        if rec.get("correct"):
            correct += 1
            per_tool_correct[expected] += 1
        else:
            if rec.get("expected_in_candidates"):
                worker_caused += 1
            else:
                intent_caused += 1

    acc = (correct / total * 100.0) if total else 0.0
    wrong = total - correct - missing

    lines: list[str] = []
    lines.append("# G5 Tool-Selection Eval Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"## Headline: tool_selection_accuracy = **{acc:.1f}%**  ({correct}/{total})")
    lines.append("")
    if missing:
        lines.append(f"> Warning: {missing} case(s) missing prediction (status != success).")
        lines.append("")

    lines.append("## Failure attribution")
    lines.append("")
    lines.append("| Cause | Count | Meaning |")
    lines.append("|---|---|---|")
    lines.append(f"| worker-caused | {worker_caused} | expected tool WAS a candidate but tool_worker did not pick it (prompt-fixable) |")
    lines.append(f"| intent-caused | {intent_caused} | expected tool was NOT even a candidate (intent_router misclassified) |")
    lines.append("")

    lines.append("## Selection source")
    lines.append("")
    lines.append("| Source | Count |")
    lines.append("|---|---|")
    for src, n in source_counter.most_common():
        lines.append(f"| `{src}` | {n} |")
    lines.append("")

    lines.append("## Per-tool accuracy")
    lines.append("")
    lines.append("| Expected tool | Accuracy | Correct | Total |")
    lines.append("|---|---|---|---|")
    for tool in sorted(per_tool_total.keys()):
        tt = per_tool_total[tool]
        tc = per_tool_correct[tool]
        a = (tc / tt * 100.0) if tt else 0.0
        lines.append(f"| `{tool}` | {a:.1f}% | {tc} | {tt} |")
    lines.append("")

    lines.append("## Confusion matrix (expected -> predicted primary)")
    lines.append("")
    all_pred = sorted({p for row in confusion.values() for p in row.keys()})
    if all_pred:
        header = "| Expected \\ Predicted | " + " | ".join(f"`{p}`" for p in all_pred) + " |"
        sep = "|---|" + "|".join(["---"] * len(all_pred)) + "|"
        lines.append(header)
        lines.append(sep)
        for tool in sorted(confusion.keys()):
            row = [f"| `{tool}`"]
            for p in all_pred:
                row.append(str(confusion[tool].get(p, 0)))
            lines.append(" | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G5 tool-selection eval runner.")
    parser.add_argument("--queries", type=Path, default=here / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=here / "runs" / "g5_predictions.jsonl")
    parser.add_argument("--report", type=Path, default=here / "report.md")
    parser.add_argument("--only-case-id", type=str, default=None)
    parser.add_argument(
        "--tool-worker-prompt-file",
        type=Path,
        default=None,
        help="Override the tool_worker system prompt with this file's contents (for A/B testing).",
    )
    parser.add_argument(
        "--intent-router-prompt-file",
        type=Path,
        default=None,
        help="Override the intent_router system prompt with this file's contents (for A/B testing).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip running; only rebuild the report from existing predictions.",
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
        from agent.graph.intent_heuristics import _heuristic_intent, _merge_intent  # noqa: E402
        from agent.graph.model_io import _coerce_to_text, _extract_json_object  # noqa: E402
        from agent.graph.models import build_graph_models  # noqa: E402
        from agent.graph.prompts import (  # noqa: E402
            _INTENT_ROUTER_SYSTEM_PROMPT,
            _TOOL_WORKER_SYSTEM_PROMPT,
        )
        from agent.graph.tool_planning import (  # noqa: E402
            _heuristic_tool_calls,
            _normalize_tool_calls,
            _select_tools,
            _tool_schema_brief,
        )

        tool_worker_prompt = _TOOL_WORKER_SYSTEM_PROMPT
        if args.tool_worker_prompt_file:
            prompt_path = args.tool_worker_prompt_file.resolve()
            if not prompt_path.exists():
                print(f"Prompt file not found: {prompt_path}")
                return 1
            tool_worker_prompt = prompt_path.read_text(encoding="utf-8").strip()
            print(f"Using OVERRIDE tool_worker prompt from {prompt_path}")

        intent_router_prompt = _INTENT_ROUTER_SYSTEM_PROMPT
        if args.intent_router_prompt_file:
            intent_prompt_path = args.intent_router_prompt_file.resolve()
            if not intent_prompt_path.exists():
                print(f"Intent prompt file not found: {intent_prompt_path}")
                return 1
            intent_router_prompt = intent_prompt_path.read_text(encoding="utf-8").strip()
            print(f"Using OVERRIDE intent_router prompt from {intent_prompt_path}")

        models = build_graph_models()
        intent_handle = models.intent_router
        worker_handle = models.tool_worker
        print(
            f"intent_router: provider={intent_handle.provider} model={intent_handle.model} "
            f"client={'ready' if intent_handle.client else 'MISSING'}"
        )
        print(
            f"tool_worker: provider={worker_handle.provider} model={worker_handle.model} "
            f"client={'ready' if worker_handle.client else 'MISSING'}"
        )

        done_ids = {cid for cid, rec in predictions.items() if rec.get("status") == "success"}
        if done_ids:
            print(f"Resume: {len(done_ids)} case(s) already succeeded; skipping them.")
        pending = [c for c in cases if str(c.get("case_id")) not in done_ids]
        print(f"Total cases: {len(cases)}; pending: {len(pending)}")

        for idx, case in enumerate(pending, start=1):
            case_id = str(case.get("case_id") or "")
            question = str(case.get("question") or "")
            expected = str(case.get("expected_primary_tool") or "")
            acceptable = list(case.get("acceptable_tools") or [expected])
            print(f"[{idx}/{len(pending)}] {case_id} ...", flush=True)
            try:
                intent = _classify_intent(
                    question,
                    handle=intent_handle,
                    heuristic_fn=_heuristic_intent,
                    merge_fn=_merge_intent,
                    extract_json_fn=_extract_json_object,
                    coerce_text_fn=_coerce_to_text,
                    system_prompt=intent_router_prompt,
                )
                plan = _plan_tools(
                    question,
                    intent,
                    handle=worker_handle,
                    select_tools_fn=_select_tools,
                    schema_brief_fn=_tool_schema_brief,
                    normalize_calls_fn=_normalize_tool_calls,
                    heuristic_calls_fn=_heuristic_tool_calls,
                    extract_json_fn=_extract_json_object,
                    coerce_text_fn=_coerce_to_text,
                    tool_worker_prompt=tool_worker_prompt,
                )
                final_tools = plan["final_tools"]
                predicted_primary = final_tools[0] if final_tools else ""
                expected_in_candidates = expected in plan["candidates"]
                correct = predicted_primary in acceptable
                record = {
                    "case_id": case_id,
                    "question": question,
                    "expected_primary_tool": expected,
                    "acceptable_tools": acceptable,
                    "intent_route": str(intent.get("route") or ""),
                    "intent_type": str(intent.get("intent_type") or ""),
                    "candidates": plan["candidates"],
                    "expected_in_candidates": expected_in_candidates,
                    "llm_tools": plan["llm_tools"],
                    "final_tools": final_tools,
                    "selection_source": plan["selection_source"],
                    "predicted_primary": predicted_primary,
                    "correct": correct,
                    "status": "success",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                print(
                    "  -> "
                    + ("OK" if correct else "WRONG")
                    + f" intent_type={record['intent_type']} predicted={predicted_primary} "
                    + f"(expected {expected}, in_candidates={expected_in_candidates}, src={plan['selection_source']})"
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "case_id": case_id,
                    "question": question,
                    "expected_primary_tool": expected,
                    "acceptable_tools": acceptable,
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                print(f"  -> ERROR: {record['error_message']}")
            _append_jsonl(output_path, record)
            predictions[case_id] = record

    report = _build_report(cases, predictions)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
