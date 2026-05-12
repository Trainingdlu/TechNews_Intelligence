"""Run task-driven eval dataset with layered scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from eval_core import extract_urls
    from task_eval_schema import load_cases_jsonl
    from task_eval_scoring import aggregate_layer_summary, score_case
except ImportError:  # package-style import fallback
    from .eval_core import extract_urls
    from .task_eval_schema import load_cases_jsonl
    from .task_eval_scoring import aggregate_layer_summary, score_case


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
    if env_file:
        candidate = env_file.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Env file not found: {candidate}")
        load_dotenv(dotenv_path=candidate, override=True)

    # Default to Vertex for eval-time model calls unless explicitly configured.
    if not str(os.getenv("AGENT_MODEL_PROVIDER", "")).strip():
        os.environ["AGENT_MODEL_PROVIDER"] = "vertex"


def _bootstrap_agent() -> Any:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from agent import generate_response_eval_payload  # pylint: disable=import-outside-toplevel

    return generate_response_eval_payload


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _retrieved_urls_from_trace(summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    urls: list[str] = []
    for span in summary.get("spans", []) or []:
        if not isinstance(span, dict) or span.get("span_type") != "tool_call":
            continue
        out_summary = span.get("output_summary", {})
        if not isinstance(out_summary, dict):
            continue
        for url in out_summary.get("evidence_urls", []) or []:
            text = str(url).strip()
            if text:
                urls.append(text)
        context_docs = out_summary.get("context_docs", [])
        if isinstance(context_docs, list):
            for doc in context_docs:
                if not isinstance(doc, dict):
                    continue
                url = str(doc.get("url", "")).strip()
                if url:
                    urls.append(url)
    return _dedupe(urls)


def _tool_calls_detailed_from_trace(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    out: list[dict[str, Any]] = []
    spans = [
        item
        for item in summary.get("spans", []) or []
        if isinstance(item, dict) and item.get("span_type") == "tool_call"
    ]
    for span in sorted(spans, key=lambda item: int(item.get("started_at_ms") or 0)):
        tool = str(span.get("name", "")).strip()
        if not tool:
            continue
        input_summary = span.get("input_summary", {})
        args: dict[str, Any] = {}
        if isinstance(input_summary, dict):
            nested_args = input_summary.get("args")
            if isinstance(nested_args, dict):
                args = nested_args
            else:
                args = {key: value for key, value in input_summary.items() if key not in {"tool"}}
        out.append({"tool": tool, "args": args})
    return out


def _model_calls_from_trace(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    model_io_by_span = {
        str(item.get("span_id", "")): item
        for item in summary.get("model_io", []) or []
        if isinstance(item, dict)
    }
    out: list[dict[str, Any]] = []
    spans = [
        item
        for item in summary.get("spans", []) or []
        if isinstance(item, dict) and item.get("span_type") == "model_call"
    ]
    for span in sorted(spans, key=lambda item: int(item.get("started_at_ms") or 0)):
        span_id = str(span.get("span_id", ""))
        model_io = model_io_by_span.get(span_id, {})
        out.append(
            {
                "span_id": span_id,
                "node": str(model_io.get("node") or span.get("name") or ""),
                "provider": str(model_io.get("provider") or (span.get("metadata") or {}).get("provider") or ""),
                "model": str(model_io.get("model") or (span.get("metadata") or {}).get("model") or ""),
                "status": str(span.get("status", "")),
                "latency_ms": span.get("latency_ms"),
                "token_usage": model_io.get("token_usage") or (span.get("output_summary") or {}).get("token_usage"),
                "has_full_io": bool(model_io),
            }
        )
    return out


def _guard_events_from_trace(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    out: list[dict[str, Any]] = []
    for span in summary.get("spans", []) or []:
        if not isinstance(span, dict) or span.get("span_type") not in {"guard", "postprocess"}:
            continue
        out.append(
            {
                "span_id": span.get("span_id"),
                "type": span.get("span_type"),
                "name": span.get("name"),
                "status": span.get("status"),
                "output_summary": span.get("output_summary") or {},
            }
        )
    return out


def _collect_retrieved_urls(
    payload: dict[str, Any],
    trace_summary: dict[str, Any] | None,
) -> list[str]:
    payload_urls = [str(item).strip() for item in payload.get("valid_urls", []) if str(item).strip()]
    trace_urls = _retrieved_urls_from_trace(trace_summary)
    return _dedupe(payload_urls + trace_urls)


def _invoke_eval_payload(
    generate_response_eval_payload: Any,
    question: str,
    *,
    request_id: str,
    case_id: str,
) -> dict[str, Any]:
    try:
        payload = generate_response_eval_payload(
            [],
            question,
            request_id=request_id,
            case_id=case_id,
            experiment_group="task_eval",
            include_trace_summary=True,
        )
    except TypeError:
        try:
            payload = generate_response_eval_payload(
                [],
                question,
                request_id=request_id,
                case_id=case_id,
            )
        except TypeError:
            payload = generate_response_eval_payload([], question, request_id=request_id)
    if not isinstance(payload, dict):
        return {"text": str(payload), "tool_calls": []}
    return payload


def _parse_args() -> argparse.Namespace:
    eval_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run task-driven eval (layered).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=eval_dir / "datasets" / "task_eval_cases.jsonl",
        help="Case dataset JSONL generated by build_task_dataset.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=eval_dir / "reports" / "task_eval_latest.json",
        help="Output report path.",
    )
    parser.add_argument(
        "--runs-per-case",
        type=int,
        default=1,
        help="Runs per case. Use 3 for stability-focused subsets.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between repeated runs of the same case.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, evaluate only first N cases.",
    )
    parser.add_argument(
        "--strict-tool-check",
        action="store_true",
        default=False,
        help="Validate case tool names against live catalog while loading dataset.",
    )
    parser.add_argument(
        "--include-trace-summary",
        action="store_true",
        help="Include raw trace_summary in each run payload.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file loaded after agent/.env.",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-a-Judge for generation quality (faithfulness + relevancy).",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help="LLM provider for judge (defaults to eval provider).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="LLM model for judge (defaults to eval model).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.runs_per_case) < 1:
        raise ValueError("--runs-per-case must be >= 1.")

    _load_eval_env(args.env_file)
    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cases = load_cases_jsonl(dataset_path, strict_tool=bool(args.strict_tool_check))
    if args.max_cases and int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]
    if not cases:
        raise ValueError("No cases selected.")

    generate_response_eval_payload = _bootstrap_agent()
    started = time.time()
    case_rows: list[dict[str, Any]] = []

    # Optional LLM-as-a-Judge for generation quality
    judge = None
    if args.enable_llm_judge:
        try:
            from llm_judge import LLMJudge
        except ImportError:
            from .llm_judge import LLMJudge
        judge = LLMJudge(
            provider=args.judge_provider,
            model=args.judge_model,
            temperature=0.0,
        )
        print("[TaskEval] LLM-as-a-Judge enabled.")

    for case in cases:
        question = str(case.get("expected_question", "")).strip()
        if not question:
            raise ValueError(f"{case.get('case_id')}: expected_question is empty.")
        run_rows: list[dict[str, Any]] = []

        for run_index in range(1, int(args.runs_per_case) + 1):
            request_id = f"task-eval-{case['case_id']}-{run_index}-{uuid.uuid4().hex[:8]}"
            payload: dict[str, Any] = {}
            error_text = ""
            try:
                payload = _invoke_eval_payload(
                    generate_response_eval_payload,
                    question,
                    request_id=request_id,
                    case_id=str(case.get("case_id", "")),
                )
                final_answer = str(payload.get("text", "")).strip()
                tool_calls_raw = payload.get("tool_calls", [])
                tool_calls = [str(item).strip() for item in tool_calls_raw if str(item).strip()]
                trace_summary = payload.get("trace_summary")
                if not isinstance(trace_summary, dict):
                    trace_summary = None
            except Exception as exc:  # noqa: BLE001
                final_answer = f"[EVAL_ERROR] {type(exc).__name__}: {exc}"
                tool_calls = []
                trace_summary = None
                error_text = f"{type(exc).__name__}: {exc}"

            if not error_text and final_answer.startswith("[EVAL_ERROR]"):
                error_text = final_answer

            retrieved_urls = _collect_retrieved_urls(payload, trace_summary)
            citations = _dedupe(extract_urls(final_answer))

            latency_ms = 0.0
            final_status = ""
            if isinstance(trace_summary, dict):
                latency_ms = float(trace_summary.get("latency_ms", 0.0) or 0.0)
                final_status = str(trace_summary.get("final_status", "")).strip().lower()

            tool_calls_detailed = _tool_calls_detailed_from_trace(trace_summary)
            model_calls = _model_calls_from_trace(trace_summary)
            guard_events = _guard_events_from_trace(trace_summary)
            predicted_intent_label = tool_calls[0] if tool_calls else "none"
            clarification_triggered = final_status == "clarification_required"

            run_record: dict[str, Any] = {
                "run_index": run_index,
                "request_id": str(payload.get("request_id", request_id)),
                "final_answer": final_answer,
                "tool_calls": tool_calls,
                "tool_calls_detailed": tool_calls_detailed,
                "model_calls": model_calls,
                "guard_events": guard_events,
                "predicted_intent_label": predicted_intent_label,
                "retrieved_urls": retrieved_urls,
                "citations": citations,
                "latency_ms": latency_ms,
                "final_status": final_status,
                "clarification_triggered": clarification_triggered,
                "error": error_text,
            }
            if args.include_trace_summary:
                run_record["trace_summary"] = trace_summary

            # LLM-as-a-Judge: inject generation quality scores
            if judge and final_answer and not error_text:
                try:
                    # Collect tool output context for faithfulness check
                    tool_context = ""
                    if isinstance(trace_summary, dict):
                        for span in trace_summary.get("spans", []) or []:
                            if not isinstance(span, dict) or span.get("span_type") != "tool_call":
                                continue
                            out_text = str(span.get("output_summary", ""))
                            if out_text:
                                tool_context += out_text + "\n"
                    if tool_context.strip():
                        judge_result = judge.judge_both(
                            question=question,
                            context=tool_context,
                            answer=final_answer,
                        )
                        run_record["faithfulness_score"] = judge_result["faithfulness"]["score"]
                        run_record["answer_relevancy_score"] = judge_result["relevancy"]["score"]
                        run_record["judge_details"] = judge_result
                except Exception as judge_exc:
                    print(f"[TaskEval][Warn] Judge failed for {case.get('case_id')}: {judge_exc}")

            run_rows.append(run_record)

            if float(args.sleep_seconds) > 0 and run_index < int(args.runs_per_case):
                time.sleep(float(args.sleep_seconds))

        layers = score_case(case, run_rows)
        case_rows.append(
            {
                "case": {
                    "case_id": case.get("case_id"),
                    "task_type": case.get("task_type"),
                    "tool": case.get("tool"),
                    "intent_label": case.get("intent_label"),
                    "retrieval_evaluable": case.get("retrieval_evaluable"),
                    "tags": case.get("tags", []),
                },
                "layers": {
                    "intent": layers["intent"],
                    "tool": layers["tool"],
                    "retrieval": layers["retrieval"],
                    "analysis": layers["analysis"],
                    "generation": layers.get("generation", {}),
                    "system": layers["system"],
                },
                "attribution": layers["attribution"],
                "runs": run_rows,
            }
        )
        print(
            "[TaskEval] case=%s tool_path_hit=%.3f retrieval_gold_hit=%s error_rate=%.3f"
            % (
                case.get("case_id"),
                float(layers["tool"]["acceptable_path_hit_rate"]),
                layers["retrieval"].get("gold_hit_rate"),
                float(layers["system"]["error_rate"]),
            )
        )

    summary = aggregate_layer_summary(case_rows)
    elapsed = round(time.time() - started, 3)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "runs_per_case": int(args.runs_per_case),
        "case_count": len(case_rows),
        "elapsed_seconds": elapsed,
        "summary": summary,
        "cases": case_rows,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[TaskEval] cases=%s elapsed=%.3fs intent_top1=%.3f path_hit=%.3f retrieval_recall10=%s "
        "analysis_support=%s system_error=%.3f output=%s"
        % (
            len(case_rows),
            elapsed,
            float(summary["intent"]["top1_accuracy"]),
            float(summary["tool"]["acceptable_path_hit_rate"]),
            summary["retrieval"]["recall_at_10"],
            summary["analysis"]["claim_support_rate"],
            float(summary["system"]["error_rate"]),
            output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

