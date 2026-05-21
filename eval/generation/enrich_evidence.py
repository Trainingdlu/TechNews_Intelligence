"""G3 fix — enrich generation results with the ACTUAL evidence the synthesizer saw.

The first G3 pass gave the faithfulness judge only article summaries re-fetched
from the DB, which omitted analytical-tool outputs (compare_topics / trend /
landscape statistics) and full-text content. That produced false-positive
"hallucination" flags on real tool statistics.

This script reads the persisted final_synthesizer model input (agent_model_io,
node='final_synthesizer') for each case and extracts the exact "Evidence brief +
Tool results" block the agent generated from, writing it as `synthesizer_evidence`
so the judge can be re-run against the true context.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            out.append(json.loads(text))
    return out


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and c.get("text"):
                parts.append(str(c["text"]))
        return "\n".join(parts)
    return str(content or "")


def _extract_evidence_block(content: str) -> str:
    start = content.find("Evidence brief:")
    end = content.rfind("Write the final answer now.")
    if start >= 0:
        seg = content[start:end] if end > start else content[start:]
        return seg.strip()
    return content.strip()


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Enrich G3 results with real synthesizer evidence.")
    parser.add_argument("--results", type=Path, default=here / "runs" / "generation_results.jsonl")
    parser.add_argument("--output", type=Path, default=here / "runs" / "generation_results_enriched.jsonl")
    args = parser.parse_args()
    _load_env()

    from services.db import db_cursor  # noqa: E402

    cases = [r for r in _read_jsonl(args.results.resolve()) if r.get("status") == "success"]
    # dedupe by case_id
    by_id = {str(r.get("case_id") or ""): r for r in cases}

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = 0
    missing = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for case_id, rec in by_id.items():
            exact_request_id = str(rec.get("request_id") or "").strip()
            with db_cursor() as (_, cur):
                if exact_request_id:
                    # Precise match (avoids cross-run collisions when the same
                    # case_id is run twice, e.g. strict vs ablation).
                    cur.execute(
                        "SELECT input_messages FROM agent_model_io "
                        "WHERE request_id = %s AND node = 'final_synthesizer' "
                        "ORDER BY created_at DESC LIMIT 1",
                        (exact_request_id,),
                    )
                else:
                    cur.execute(
                        "SELECT input_messages FROM agent_model_io "
                        "WHERE request_id LIKE %s AND node = 'final_synthesizer' "
                        "ORDER BY created_at DESC LIMIT 1",
                        (f"eval_g3_{case_id}_%",),
                    )
                row = cur.fetchone()
            evidence_text = ""
            if row and row[0]:
                msgs = row[0]
                if isinstance(msgs, str):
                    try:
                        msgs = json.loads(msgs)
                    except json.JSONDecodeError:
                        msgs = []
                humans = [m for m in msgs if isinstance(m, dict) and m.get("class") == "HumanMessage"]
                if humans:
                    evidence_text = _extract_evidence_block(_coerce_content(humans[-1].get("content")))
            if evidence_text:
                enriched += 1
            else:
                missing += 1
                print(f"  [warn] no synthesizer evidence for {case_id}")
            rec = dict(rec)
            rec["synthesizer_evidence"] = evidence_text
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Enriched {enriched} case(s); missing {missing}. Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
