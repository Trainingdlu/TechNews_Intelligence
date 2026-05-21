"""Summarize G1 labeled sheet into failure taxonomy distribution.

Reads labeling_sheet.md, parses the `**Your label:**` fields you filled in,
and outputs:
  - overall tag distribution
  - per-category breakdown (A-J)
  - list of unlabeled / unknown-tag cases

Multiple tags per case can be entered comma-separated, e.g. `[retrieval_miss, hallucination]`.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

KNOWN_TAGS = {
    "OK",
    "intent_wrong",
    "tool_wrong",
    "retrieval_miss",
    "retrieval_noise",
    "hallucination",
    "format_bad",
    "refusal_bad",
}

CASE_HEADER_RE = re.compile(r"^##\s+(g1_[A-J]_\d+_\w+)\s*$", re.MULTILINE)
CATEGORY_RE = re.compile(r"^Category:\s*\*\*([A-J])\*\*", re.MULTILINE)
LABEL_RE = re.compile(r"^\*\*Your label:\*\*\s*`\[(.*?)\]`", re.MULTILINE)


def _split_cases(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    matches = list(CASE_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        case_id = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((case_id, text[start:end]))
    return parts


def _parse_block(block: str) -> tuple[str, list[str]]:
    cat_match = CATEGORY_RE.search(block)
    category = cat_match.group(1) if cat_match else "?"
    label_match = LABEL_RE.search(block)
    raw_label = (label_match.group(1) if label_match else "").strip()
    if not raw_label:
        return category, []
    tokens = [t.strip() for t in raw_label.split(",") if t.strip()]
    return category, tokens


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Summarize G1 labeled sheet.")
    parser.add_argument(
        "--sheet",
        type=Path,
        default=here / "labeling_sheet.md",
        help="Path to labeled labeling_sheet.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "failure_taxonomy.md",
        help="Output report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sheet_path = args.sheet.resolve()
    if not sheet_path.exists():
        print(f"Labeling sheet not found: {sheet_path}")
        return 1
    text = sheet_path.read_text(encoding="utf-8")

    cases = _split_cases(text)
    if not cases:
        print("No cases parsed from the sheet (header pattern not matched).")
        return 1

    overall_counter: Counter = Counter()
    per_category: dict[str, Counter] = defaultdict(Counter)
    unlabeled: list[str] = []
    unknown_tags: list[tuple[str, str]] = []

    for case_id, block in cases:
        category, tags = _parse_block(block)
        if not tags:
            unlabeled.append(case_id)
            continue
        for tag in tags:
            if tag not in KNOWN_TAGS:
                unknown_tags.append((case_id, tag))
            overall_counter[tag] += 1
            per_category[category][tag] += 1

    total_cases = len(cases)
    labeled_cases = total_cases - len(unlabeled)

    lines: list[str] = []
    lines.append("# G1 Failure Taxonomy")
    lines.append("")
    lines.append(
        f"Total cases: **{total_cases}**, "
        f"labeled: **{labeled_cases}**, "
        f"unlabeled: **{len(unlabeled)}**"
    )
    lines.append("")

    lines.append("## Overall tag distribution")
    lines.append("")
    lines.append("| Tag | Count | Share |")
    lines.append("|---|---|---|")
    for tag, count in overall_counter.most_common():
        share = (count / labeled_cases * 100.0) if labeled_cases else 0.0
        lines.append(f"| `{tag}` | {count} | {share:.1f}% |")
    lines.append("")

    lines.append("## Per-category breakdown")
    lines.append("")
    sorted_cats = sorted(per_category.keys())
    if sorted_cats:
        all_tags = sorted(overall_counter.keys())
        header = "| Category | " + " | ".join(f"`{t}`" for t in all_tags) + " |"
        sep = "|---|" + "|".join(["---"] * len(all_tags)) + "|"
        lines.append(header)
        lines.append(sep)
        for cat in sorted_cats:
            row = [f"| {cat}"]
            for tag in all_tags:
                row.append(str(per_category[cat].get(tag, 0)))
            lines.append(" | ".join(row) + " |")
        lines.append("")

    if unlabeled:
        lines.append("## Unlabeled cases")
        lines.append("")
        for cid in unlabeled:
            lines.append(f"- {cid}")
        lines.append("")

    if unknown_tags:
        lines.append("## Unknown tags (typos?)")
        lines.append("")
        for cid, tag in unknown_tags:
            lines.append(f"- {cid}: `{tag}`")
        lines.append("")

    output_path = args.output.resolve()
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {output_path}")
    print(f"  Labeled: {labeled_cases}/{total_cases}")
    print(f"  Top tags: {overall_counter.most_common(5)}")
    if unlabeled:
        print(f"  Unlabeled: {len(unlabeled)}")
    if unknown_tags:
        print(f"  Unknown tags: {len(unknown_tags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
