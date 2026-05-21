"""G1 error analysis batch runner.

Reads eval/error_analysis/queries.jsonl, runs each case against the real agent
pipeline via agent.generate_response_eval_payload, and appends per-case results
to runs/g1_run.jsonl with resume support.

Three-tier isolation per case:
  - history=[] at first turn (accumulated only within multi-turn case)
  - unique thread_id = eval_g1_<case_id>_<run_token> (fresh thread memory)
  - unique request_id per call
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
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


def _read_queries(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            cases.append(json.loads(text))
    return cases


def _read_done_case_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in ("success", "clarification"):
                done.add(str(rec.get("case_id") or ""))
    return done


def _append_record(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def _confirm_n8n_paused() -> bool:
    print("=" * 60)
    print("PREREQUISITE: please pause n8n ingestion workflows manually")
    print("              to keep the news database stable during the run.")
    print("=" * 60)
    answer = input("Confirm n8n workflows are paused? [y/N] ").strip().lower()
    return answer == "y"


def _run_single_case(
    case: dict[str, Any],
    *,
    generate_fn: Any,
    run_token: str,
    clarification_exc_cls: type[BaseException],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    category = str(case.get("category") or "")
    multi_turn = bool(case.get("multi_turn") or False)
    probe = str(case.get("probe") or "")
    turns_spec = list(case.get("turns") or [])

    thread_id = f"eval_g1_{case_id}_{run_token}"

    history: list[dict[str, Any]] = []
    turn_payloads: list[dict[str, Any]] = []
    error_message = ""

    clarification_hit = False
    for idx, turn in enumerate(turns_spec, start=1):
        user_text = str(turn.get("text") or "")
        request_id = f"eval_g1_{case_id}_t{idx}_{uuid.uuid4().hex[:8]}"
        try:
            payload = generate_fn(
                history,
                user_text,
                request_id=request_id,
                thread_id=thread_id,
                case_id=case_id,
                experiment_group="g1_error_analysis",
                include_trace_summary=True,
            )
        except clarification_exc_cls as exc:
            # Agent properly chose to ask for clarification — this is a valid outcome,
            # not a runner failure. Record the clarification question as the turn's
            # agent_text and stop further turns (downstream turns assumed a non-clarifying
            # turn 1).
            clarification_text = str(exc)
            clarification_obj = getattr(exc, "clarification", None)
            turn_payloads.append(
                {
                    "turn_index": idx,
                    "request_id": request_id,
                    "user_text": user_text,
                    "agent_text": clarification_text,
                    "valid_urls": [],
                    "tool_calls": [],
                    "trace_summary": None,
                    "clarification": {
                        "reason": str(getattr(clarification_obj, "reason", "") or ""),
                        "question": str(
                            getattr(clarification_obj, "question", clarification_text) or clarification_text
                        ),
                    },
                }
            )
            clarification_hit = True
            break
        except Exception as exc:  # noqa: BLE001
            error_message = f"turn {idx}: {type(exc).__name__}: {exc}"
            turn_payloads.append(
                {
                    "turn_index": idx,
                    "request_id": request_id,
                    "user_text": user_text,
                    "agent_text": "",
                    "valid_urls": [],
                    "tool_calls": [],
                    "trace_summary": None,
                    "error": error_message,
                }
            )
            break

        if not isinstance(payload, dict):
            payload = {"text": str(payload), "valid_urls": [], "tool_calls": []}

        agent_text = str(payload.get("text") or "")
        valid_urls = list(payload.get("valid_urls") or [])
        tool_calls = list(payload.get("tool_calls") or [])
        trace_summary = payload.get("trace_summary")
        turn_payloads.append(
            {
                "turn_index": idx,
                "request_id": request_id,
                "user_text": user_text,
                "agent_text": agent_text,
                "valid_urls": valid_urls,
                "tool_calls": tool_calls,
                "trace_summary": trace_summary,
            }
        )

        history.append({"role": "user", "parts": [{"text": user_text}]})
        history.append({"role": "model", "parts": [{"text": agent_text}]})

    if error_message:
        status = "error"
    elif clarification_hit:
        status = "clarification"
    else:
        status = "success"
    return {
        "case_id": case_id,
        "category": category,
        "multi_turn": multi_turn,
        "probe": probe,
        "thread_id": thread_id,
        "status": status,
        "error_message": error_message,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "turns": turn_payloads,
    }


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G1 error analysis runner.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=here / "queries.jsonl",
        help="Path to queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "runs" / "g1_run.jsonl",
        help="Append-only output JSONL (resume supported).",
    )
    parser.add_argument(
        "--only-case-id",
        type=str,
        default=None,
        help="Run only the specified case_id (useful for retries).",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Skip the n8n pause confirmation prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_env()

    if not args.skip_confirm:
        if not _confirm_n8n_paused():
            print("Aborted: n8n not confirmed paused.")
            return 1

    queries_path = args.queries.resolve()
    if not queries_path.exists():
        print(f"Queries file not found: {queries_path}")
        return 1

    cases = _read_queries(queries_path)
    if args.only_case_id:
        cases = [c for c in cases if str(c.get("case_id")) == args.only_case_id]
        if not cases:
            print(f"No case matching case_id={args.only_case_id}")
            return 1

    output_path = args.output.resolve()
    done_ids = _read_done_case_ids(output_path)
    if done_ids:
        print(f"Resume: {len(done_ids)} case(s) already succeeded; skipping them.")

    from agent import generate_response_eval_payload  # noqa: E402
    from agent.clarification import ClarificationRequiredError  # noqa: E402

    pending = [c for c in cases if str(c.get("case_id")) not in done_ids]
    print(f"Total cases: {len(cases)}; pending: {len(pending)}")

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    success_count = 0
    clarification_count = 0
    error_count = 0
    for offset, case in enumerate(pending, start=1):
        case_id = str(case.get("case_id"))
        print(f"[{offset}/{len(pending)}] running {case_id} ...", flush=True)
        result = _run_single_case(
            case,
            generate_fn=generate_response_eval_payload,
            run_token=run_token,
            clarification_exc_cls=ClarificationRequiredError,
        )
        _append_record(output_path, result)
        status = result["status"]
        if status == "success":
            success_count += 1
            print(f"  -> success ({len(result['turns'])} turn(s))")
        elif status == "clarification":
            clarification_count += 1
            print(f"  -> clarification (agent asked for clarification on turn {len(result['turns'])})")
        else:
            error_count += 1
            print(f"  -> ERROR: {result['error_message']}")

    print("=" * 60)
    print(
        f"Done. success={success_count}, clarification={clarification_count}, "
        f"error={error_count}, output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
