"""G2 Phase D (2/2) — compute LLM-vs-human agreement (Cohen's kappa).

Reads the user's filled calibration_sheet.md and the hidden answer key
(runs/calibration_key.jsonl), joins by pair_id, and reports:
  - raw agreement %
  - Cohen's kappa (nominal, 3-class)
  - quadratic-weighted kappa (ordinal 0/1/2 — the appropriate metric for graded relevance)
  - binary-relevance agreement (rel>=1), which is what Hit/MRR/Precision rely on
  - the 3x3 confusion matrix (human rows vs LLM cols)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

BLOCK_RE = re.compile(
    r"^##\s+\d+\.\s+`([^`]+)`.*?\*\*Your score:\*\*\s*`\[\s*([012]?)\s*\]`",
    re.MULTILINE | re.DOTALL,
)


def _read_key(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec.get("pair_id"))] = int(rec.get("llm_relevance") or 0)
    return out


def _parse_sheet(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for m in BLOCK_RE.finditer(text):
        pair_id = m.group(1)
        raw = m.group(2).strip()
        if raw == "":
            continue  # unlabeled
        out[pair_id] = int(raw)
    return out


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compute LLM-vs-human kappa.")
    parser.add_argument("--sheet", type=Path, default=here / "calibration_sheet.md")
    parser.add_argument("--key", type=Path, default=here / "runs" / "calibration_key.jsonl")
    parser.add_argument("--report", type=Path, default=here / "calibration_report.md")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    key = _read_key(args.key.resolve())
    human = _parse_sheet(args.sheet.resolve())

    paired = [(pid, human[pid], key[pid]) for pid in human if pid in key]
    n = len(paired)
    unlabeled = len(key) - len(human)
    if n == 0:
        print("No labeled pairs found. Fill the `[ ]` brackets in the sheet first.")
        return 1

    h = [p[1] for p in paired]
    m = [p[2] for p in paired]

    from sklearn.metrics import cohen_kappa_score  # noqa: E402

    kappa = cohen_kappa_score(h, m)
    qwk = cohen_kappa_score(h, m, weights="quadratic")
    raw_agree = sum(1 for a, b in zip(h, m) if a == b) / n

    # binary relevance agreement (rel>=1)
    hb = [1 if x >= 1 else 0 for x in h]
    mb = [1 if x >= 1 else 0 for x in m]
    bin_agree = sum(1 for a, b in zip(hb, mb) if a == b) / n
    bin_kappa = cohen_kappa_score(hb, mb)

    # 3x3 confusion (human rows, llm cols)
    conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for a, b in zip(h, m):
        conf[a][b] += 1

    lines: list[str] = []
    lines.append("# G2 相关性判分校准：LLM vs 人工")
    lines.append("")
    lines.append(f"已标注样本对：**{n}**" + (f"（剩余未标注：{unlabeled}）" if unlabeled else ""))
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 原始一致率（3 类） | {raw_agree*100:.1f}% |")
    lines.append(f"| Cohen's kappa（名义 3 类） | {kappa:.3f} |")
    lines.append(f"| 二次加权 kappa（有序 0/1/2） | {qwk:.3f} |")
    lines.append(f"| 二元相关一致率（rel>=1） | {bin_agree*100:.1f}% |")
    lines.append(f"| 二元相关 Cohen's kappa | {bin_kappa:.3f} |")
    lines.append("")
    lines.append("解释口径（Landis & Koch）：<0 差；0-0.2 轻微；0.2-0.4 一般；0.4-0.6 中等；0.6-0.8 较强；0.8-1.0 几乎完全一致。")
    lines.append("")
    lines.append("## 混淆矩阵（行=人工，列=LLM）")
    lines.append("")
    lines.append("| 人工 \\ LLM | 0 | 1 | 2 |")
    lines.append("|---|---|---|---|")
    for i in range(3):
        lines.append(f"| **{i}** | {conf[i][0]} | {conf[i][1]} | {conf[i][2]} |")
    lines.append("")

    report_path = args.report.resolve()
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    print(f"  n={n}  raw_agree={raw_agree*100:.1f}%  kappa={kappa:.3f}  qwk={qwk:.3f}")
    print(f"  binary: agree={bin_agree*100:.1f}%  kappa={bin_kappa:.3f}")
    if unlabeled:
        print(f"  WARNING: {unlabeled} pairs still unlabeled in the sheet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
