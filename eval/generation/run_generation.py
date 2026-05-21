"""G3 Phase A — run the full agent and capture (question, evidence, answer).

For each question, runs the production agent (post-fix), records the final answer
and the evidence URL set the answer is grounded in (valid_urls), then fetches each
evidence URL's title + summary from the DB so the faithfulness judge has the
content the answer should be supported by.

Prerequisite: n8n ingestion paused (stable DB snapshot).
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


def _fetch_evidence(urls: list[str], db_cursor_fn: Any) -> list[dict[str, Any]]:
    if not urls:
        return []
    with db_cursor_fn() as (_, cur):
        cur.execute(
            "SELECT url, title, COALESCE(title_cn,'') AS title_cn, COALESCE(summary,'') AS summary "
            "FROM tech_news WHERE url = ANY(%s)",
            (list(urls),),
        )
        rows = {r[0]: {"url": r[0], "title": r[1], "title_cn": r[2], "summary": r[3]} for r in cur.fetchall()}
    # preserve answer's url order; include not-found urls with empty content
    out: list[dict[str, Any]] = []
    for u in urls:
        out.append(rows.get(u, {"url": u, "title": "", "title_cn": "", "summary": "(not found in DB)"}))
    return out


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="G3 generation runner.")
    parser.add_argument("--queries", type=Path, default=here / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=here / "runs" / "generation_results.jsonl")
    parser.add_argument("--only-case-id", type=str, default=None)
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument(
        "--synth-prompt-file",
        type=Path,
        default=None,
        help="Override the final_synthesizer system prompt (ablation). Reads the file "
        "and monkeypatches agent.graph.nodes._FINAL_SYSTEM_PROMPT before running.",
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
        print(f"Resume: {len(done)} case(s) already done; skipping them.")

    from agent import generate_response_eval_payload  # noqa: E402
    from agent.clarification import ClarificationRequiredError  # noqa: E402
    from services.db import db_cursor  # noqa: E402

    if args.synth_prompt_file:
        prompt_path = args.synth_prompt_file.resolve()
        if not prompt_path.exists():
            print(f"Synth prompt file not found: {prompt_path}")
            return 1
        weak_prompt = prompt_path.read_text(encoding="utf-8").strip()
        import agent.graph.nodes as _nodes  # noqa: E402
        _nodes._FINAL_SYSTEM_PROMPT = weak_prompt
        print("=" * 60)
        print(f"ABLATION: final_synthesizer prompt OVERRIDDEN from {prompt_path.name}")
        print(f"  ({len(weak_prompt)} chars) — grounding constraints removed for this run.")
        print("=" * 60)

    pending = [c for c in cases if str(c.get("case_id")) not in done]
    print(f"Total: {len(cases)}; pending: {len(pending)}")

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    success = 0
    skipped = 0
    error = 0
    for idx, case in enumerate(pending, start=1):
        case_id = str(case.get("case_id") or "")
        question = str(case.get("question") or "")
        request_id = f"eval_g3_{case_id}_{uuid.uuid4().hex[:8]}"
        thread_id = f"eval_g3_{case_id}_{run_token}"
        print(f"[{idx}/{len(pending)}] {case_id}: {question} ...", flush=True)
        try:
            payload = generate_response_eval_payload(
                [],
                question,
                request_id=request_id,
                thread_id=thread_id,
                case_id=case_id,
                experiment_group="g3_generation",
                include_trace_summary=False,
            )
            answer = str(payload.get("text") or "")
            valid_urls = [str(u).strip() for u in (payload.get("valid_urls") or []) if str(u).strip()]
            evidence = _fetch_evidence(valid_urls, db_cursor)
            record = {
                "case_id": case_id,
                "question": question,
                "answer": answer,
                "evidence_urls": valid_urls,
                "evidence": evidence,
                "request_id": request_id,
                "status": "success",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  -> answer_chars={len(answer)} evidence={len(evidence)}")
            success += 1
        except ClarificationRequiredError as exc:
            record = {
                "case_id": case_id,
                "question": question,
                "status": "clarification",
                "note": str(exc),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            print("  -> clarification (skipped for faithfulness)")
            skipped += 1
        except Exception as exc:  # noqa: BLE001
            record = {
                "case_id": case_id,
                "question": question,
                "status": "error",
                "error_message": f"{type(exc).__name__}: {exc}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  -> ERROR: {record['error_message']}")
            error += 1
        _append_jsonl(output_path, record)

    print("=" * 60)
    print(f"Done. success={success}, clarification={skipped}, error={error}, output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
