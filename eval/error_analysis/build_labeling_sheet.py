"""Build labeling sheet from G1 run output.

Reads runs/g1_run.jsonl and produces labeling_sheet.md grouped by case_id.
Each case shows: question, intent classification, tools called, top retrieved
URLs, and the agent's final answer — followed by a blank label slot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOP_URL_LIMIT = 8

# Status priority for deduplication when the same case_id has multiple records.
# Higher number wins. Among ties, the most recent completed_at wins.
_STATUS_PRIORITY = {
    "success": 3,
    "clarification": 2,
    "error": 1,
}


def _read_runs(path: Path) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                raw.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    # Dedupe by case_id: prefer success > clarification > error; tiebreak by completed_at.
    best: dict[str, dict[str, Any]] = {}
    for rec in raw:
        case_id = str(rec.get("case_id") or "")
        if not case_id:
            continue
        prev = best.get(case_id)
        if prev is None:
            best[case_id] = rec
            continue
        prev_score = _STATUS_PRIORITY.get(str(prev.get("status") or ""), 0)
        new_score = _STATUS_PRIORITY.get(str(rec.get("status") or ""), 0)
        if new_score > prev_score:
            best[case_id] = rec
        elif new_score == prev_score:
            prev_ts = str(prev.get("completed_at") or "")
            new_ts = str(rec.get("completed_at") or "")
            if new_ts > prev_ts:
                best[case_id] = rec
    return list(best.values())


def _extract_intent(trace_summary: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(trace_summary, dict):
        return {"route": "", "intent_type": ""}
    route = ""
    intent_type = ""
    for span in trace_summary.get("spans") or []:
        if not isinstance(span, dict):
            continue
        if str(span.get("name") or "") != "intent_router":
            continue
        output = span.get("output_summary") or {}
        if isinstance(output, dict):
            route = str(output.get("intent_route") or "") or route
    for io_row in trace_summary.get("model_io") or []:
        if not isinstance(io_row, dict):
            continue
        if str(io_row.get("node") or "") != "intent_router":
            continue
        parsed = io_row.get("parsed_output")
        if isinstance(parsed, dict):
            intent_type = str(parsed.get("intent_type") or "") or intent_type
            if not route:
                route = str(parsed.get("route") or "")
    return {"route": route, "intent_type": intent_type}


def _extract_tool_calls(trace_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(trace_summary, dict):
        return []
    out: list[dict[str, Any]] = []
    spans = [
        s for s in (trace_summary.get("spans") or [])
        if isinstance(s, dict) and s.get("span_type") == "tool_call"
    ]
    spans.sort(key=lambda s: int(s.get("started_at_ms") or 0))
    for span in spans:
        name = str(span.get("name") or "")
        if not name:
            continue
        input_summary = span.get("input_summary") or {}
        args: dict[str, Any] = {}
        if isinstance(input_summary, dict):
            nested = input_summary.get("args")
            if isinstance(nested, dict):
                args = nested
            else:
                args = {k: v for k, v in input_summary.items() if k != "tool"}
        out.append({"tool": name, "args": args})
    return out


def _extract_retrieved_urls(trace_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(trace_summary, dict):
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for span in trace_summary.get("spans") or []:
        if not isinstance(span, dict) or span.get("span_type") != "tool_call":
            continue
        out_summary = span.get("output_summary") or {}
        if not isinstance(out_summary, dict):
            continue
        for url in out_summary.get("evidence_urls") or []:
            text = str(url).strip()
            if text and text not in seen:
                seen.add(text)
                urls.append(text)
        ctx_docs = out_summary.get("context_docs")
        if isinstance(ctx_docs, list):
            for doc in ctx_docs:
                if not isinstance(doc, dict):
                    continue
                text = str(doc.get("url") or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    urls.append(text)
    return urls


def _format_args(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(args)


def _format_case(case: dict[str, Any]) -> str:
    case_id = str(case.get("case_id") or "")
    category = str(case.get("category") or "")
    probe = str(case.get("probe") or "")
    status = str(case.get("status") or "")
    multi_turn = bool(case.get("multi_turn") or False)
    turns = list(case.get("turns") or [])

    lines: list[str] = []
    lines.append(f"## {case_id}")
    lines.append("")
    header_bits = [f"Category: **{category}**", f"Status: **{status}**"]
    if multi_turn:
        header_bits.append(f"Multi-turn: **yes ({len(turns)} turns)**")
    lines.append(" | ".join(header_bits))
    lines.append("")
    lines.append(f"Probe: {probe}")
    lines.append("")

    if status != "success":
        err = str(case.get("error_message") or "")
        lines.append(f"> Error: {err}")
        lines.append("")

    for turn in turns:
        idx = int(turn.get("turn_index") or 0)
        user_text = str(turn.get("user_text") or "")
        agent_text = str(turn.get("agent_text") or "")
        trace_summary = turn.get("trace_summary")

        intent = _extract_intent(trace_summary)
        tools = _extract_tool_calls(trace_summary)
        urls = _extract_retrieved_urls(trace_summary)

        lines.append(f"### Turn {idx}")
        lines.append("")
        lines.append(f"**Question:** {user_text}")
        lines.append("")
        route = intent.get("route") or "(n/a)"
        intent_type = intent.get("intent_type") or "(n/a)"
        lines.append(f"**Intent classified:** route=`{route}`, intent_type=`{intent_type}`")
        lines.append("")
        if tools:
            lines.append("**Tools called:**")
            for call in tools:
                tool_name = call.get("tool") or ""
                args_str = _format_args(call.get("args") or {})
                lines.append(f"- `{tool_name}` args={args_str}")
            lines.append("")
        else:
            lines.append("**Tools called:** (none)")
            lines.append("")

        if urls:
            shown = urls[:TOP_URL_LIMIT]
            lines.append(f"**Retrieved URLs (top {len(shown)} of {len(urls)}):**")
            for i, url in enumerate(shown, start=1):
                lines.append(f"{i}. {url}")
            lines.append("")
        else:
            lines.append("**Retrieved URLs:** (none)")
            lines.append("")

        lines.append("**Final answer:**")
        lines.append("")
        lines.append("```")
        lines.append(agent_text)
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**Your label:** `[ ]`  *(OK / intent_wrong / tool_wrong / retrieval_miss / retrieval_noise / hallucination / format_bad / refusal_bad)*")
    lines.append("")
    lines.append("**Notes:**")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build labeling sheet from G1 run output.")
    parser.add_argument(
        "--runs",
        type=Path,
        default=here / "runs" / "g1_run.jsonl",
        help="Path to g1_run.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "labeling_sheet.md",
        help="Output markdown labeling sheet.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs_path = args.runs.resolve()
    if not runs_path.exists():
        print(f"Run file not found: {runs_path}")
        return 1
    cases = _read_runs(runs_path)
    if not cases:
        print(f"No cases parsed from {runs_path}")
        return 1

    cases.sort(key=lambda c: str(c.get("case_id") or ""))
    out_lines: list[str] = []
    out_lines.append("# G1 Error Analysis — Labeling Sheet")
    out_lines.append("")
    out_lines.append(f"Total cases: **{len(cases)}**")
    out_lines.append("")
    out_lines.append("Label legend:")
    out_lines.append("")
    out_lines.append("- `OK`: agent response is acceptable")
    out_lines.append("- `intent_wrong`: intent (route or intent_type) misclassified")
    out_lines.append("- `tool_wrong`: intent right but wrong tool called")
    out_lines.append("- `retrieval_miss`: relevant docs not in retrieved list")
    out_lines.append("- `retrieval_noise`: too many irrelevant docs in retrieved list")
    out_lines.append("- `hallucination`: answer contains claims not in evidence")
    out_lines.append("- `format_bad`: wrong format / language / length")
    out_lines.append("- `refusal_bad`: should have clarified/refused but did not, or vice versa")
    out_lines.append("")
    out_lines.append("---")
    out_lines.append("")
    for case in cases:
        out_lines.append(_format_case(case))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote labeling sheet to {output_path} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
