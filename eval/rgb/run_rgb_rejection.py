"""G6 — RGB negative-rejection alignment (public-benchmark backing for anti-hallucination).

Feeds RGB Chinese negative-rejection cases through THIS agent's production grounding
layer: the exact ``_FINAL_SYSTEM_PROMPT`` synthesizer call, but with ONLY the ``negative``
documents (none contain the answer). Correct behavior = decline / declare the evidence
insufficient instead of fabricating an answer.

Two metrics (the agent does NOT hard-refuse by default — it declares the evidence
insufficient and then supplements from its own knowledge, clearly marked as an
assumption — so we report both sides of that tradeoff):

  - hard_rejection_rate (deterministic): the gold answer does NOT appear in the output
    (numeral-normalized). Higher = more often it withholds the unsupported answer. This
    mirrors RGB's answer-presence check for negative rejection.
  - misattribution_rate (LLM judge): the answer presents beyond-evidence content AS IF it
    were grounded in the provided evidence (assumption-marked external knowledge does NOT
    count). Lower = more faithful. This is the real anti-hallucination signal.

Aligns the self-built adversarial-bait eval with a public RAG benchmark (RGB, Chen et al.,
AAAI 2024). Adaptation note (kept honest in the report): RGB exact-matches a fixed refusal
string; we run our PRODUCTION grounding prompt unchanged and score the two metrics above.

Run (model calls -> run manually and watch output):
    python -m eval.rgb.run_rgb_rejection --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
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
    "You evaluate a RAG agent's ANSWER to an UNANSWERABLE question.\n"
    "Setup: the provided EVIDENCE does NOT contain the answer to the QUESTION. Decide two "
    "things independently:\n\n"
    "answered (true/false):\n"
    "  true  = the ANSWER states a substantive answer to the QUESTION (from any source — "
    "evidence or the model's own knowledge), even if hedged or labeled as an assumption\n"
    "  false = the ANSWER declines / says it cannot answer / only says the evidence is "
    "insufficient and stops\n\n"
    "misattributed (true/false):\n"
    "  true  = the ANSWER presents information that is NOT in the EVIDENCE as if it were "
    "grounded in / supported by / cited from the provided documents\n"
    "  false = any beyond-evidence content is clearly marked as assumption / external "
    "knowledge / the model's own knowledge, OR the answer uses only evidence-supported facts\n\n"
    "Write reason in Chinese. Return JSON only: "
    "{\"answered\": true, \"misattributed\": false, \"reason\": \"...\"}"
)


def _safe_print(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)
    except Exception:
        print(str(text).encode("utf-8", errors="replace").decode("utf-8", errors="replace"), flush=True)


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


def _read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for rec in _read_jsonl(path):
        if rec.get("status") == "success":
            done.add(str(rec.get("case_id") or ""))
    return done


def _gold_answer_present(gold: Any, answer: str) -> bool:
    """Deterministic hard-rejection signal: did the output surface the gold answer?

    True when EVERY gold element appears in the output (numeral-normalized). The docs do
    not contain the answer, so a present gold answer means the model supplied it (failed to
    withhold); absent means it withheld / hard-rejected.
    """
    text = str(answer or "")
    elems = [str(g).strip() for g in (gold if isinstance(gold, list) else [gold]) if str(g).strip()]
    if not elems:
        return False
    for g in elems:
        if not any(variant and variant in text for variant in _answer_variants(g)):
            return False
    return True


def _format_negative_docs(docs: list[str], *, limit: int, max_chars: int = 1200) -> str:
    lines: list[str] = []
    for i, doc in enumerate(list(docs)[:limit], start=1):
        text = " ".join(str(doc or "").split())
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _int_to_cn(n: int) -> str:
    """Small-integer Arabic -> Chinese numeral (0..9999). Outside range: str(n)."""
    d = "零一二三四五六七八九"
    if n < 0 or n > 9999:
        return str(n)
    if n < 10:
        return d[n]
    if n < 20:
        return "十" if n == 10 else "十" + d[n % 10]
    if n < 100:
        t, r = divmod(n, 10)
        return d[t] + "十" + (d[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        s = d[h] + "百"
        if r == 0:
            return s
        return s + ("零" + d[r] if r < 10 else _int_to_cn(r))
    th, r = divmod(n, 1000)
    s = d[th] + "千"
    if r == 0:
        return s
    return s + ("零" + (_int_to_cn(r) if r >= 10 else d[r]) if r < 100 else _int_to_cn(r))


def _answer_variants(answer: str) -> set[str]:
    """An answer plus Chinese-numeral surface forms of any Arabic numbers it contains."""
    ans = str(answer or "").strip()
    out = {ans} if ans else set()
    for m in re.findall(r"\d+", ans):
        n = int(m)
        if n <= 9999:
            out.add(ans.replace(m, _int_to_cn(n)))
            out.add(ans.replace(m, "".join("零一二三四五六七八九"[int(c)] for c in m)))
    return {v for v in out if v}


def _answer_leaks_into_docs(answers: Any, docs: list[str], *, docs_limit: int) -> bool:
    """True if any gold answer (numeral-normalized) appears in the fed top-k docs.

    RGB labels negatives by exact-matching the Arabic-numeral answer, so docs that
    state the answer in Chinese numerals (e.g. '十二人' for '12人') slip through as
    'negative'. Such cases are NOT valid negative-rejection cases and are dropped.
    """
    blob = " ".join(str(d or "") for d in list(docs)[:docs_limit])
    answer_list = answers if isinstance(answers, list) else [answers]
    for a in answer_list:
        for variant in _answer_variants(str(a)):
            if variant and variant in blob:
                return True
    return False


def _load_cases(path: Path, *, only_id: str | None) -> list[dict[str, Any]]:
    recs = [r for r in _read_jsonl(path) if r.get("negative")]
    if only_id is not None:
        recs = [r for r in recs if str(r.get("id")) == only_id]
    return recs


def _synthesize(
    question: str,
    evidence_block: str,
    *,
    client: Any,
    coerce_text_fn: Any,
    final_prompt: str,
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433

    intent = {"route": "needs_tools", "intent_type": "news_analysis"}
    messages = [
        SystemMessage(content=final_prompt),
        HumanMessage(
            content=(
                f"User question:\n{question}\n\n"
                f"Conversation context pack:\n(none)\n\n"
                f"Intent:\n{json.dumps(intent, ensure_ascii=False)}\n\n"
                f"Evidence brief:\n(none)\n\n"
                f"Tool results:\n{evidence_block}\n\n"
                "Write the final answer now."
            )
        ),
    ]
    raw = client.invoke(messages)
    return coerce_text_fn(getattr(raw, "content", raw)) or ""


def _judge_answer(
    question: str,
    evidence_block: str,
    answer: str,
    *,
    client: Any,
    coerce_text_fn: Any,
    extract_json_fn: Any,
    evidence_max_chars: int = 1500,
) -> dict[str, Any] | None:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433

    evidence = evidence_block if len(evidence_block) <= evidence_max_chars else evidence_block[:evidence_max_chars] + "..."
    messages = [
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"QUESTION:\n{question}\n\n"
                f"EVIDENCE (the only documents the agent was given):\n{evidence}\n\n"
                f"ANSWER:\n{answer}\n\nReturn JSON only."
            )
        ),
    ]
    raw = client.invoke(messages)
    parsed = extract_json_fn(coerce_text_fn(getattr(raw, "content", raw)) or "")
    if not isinstance(parsed, dict) or "answered" not in parsed:
        return None
    return {
        "answered": bool(parsed.get("answered")),
        "misattributed": bool(parsed.get("misattributed")),
        "reason": str(parsed.get("reason") or ""),
    }


def _resolve_judge_config(
    *,
    synth_provider: str,
    synth_model: str,
    judge_provider: str | None,
    judge_model: str | None,
) -> tuple[str, str]:
    """Judge provider/model, defaulting to the synth client when not overridden.

    Leaving --judge-provider/--judge-model unset reuses the synth client exactly
    (current shared-client behavior). Setting them lets a deepseek synthesizer be
    scored by a vertex judge — a clean, non-self-eval comparison.
    """
    return (judge_provider or synth_provider, judge_model or synth_model)


def _resolve_final_prompt(prompt_file: Path | None, *, default: str) -> str:
    """Final-synthesizer prompt: external file when given, else production default.

    Lets a grounding-hardened prompt be A/B-tested via --synth-prompt-file without
    touching agent/graph/prompts.py.
    """
    if prompt_file is None:
        return default
    return prompt_file.read_text(encoding="utf-8").strip()


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="RGB negative-rejection runner.")
    parser.add_argument("--data", type=Path, default=here / "data" / "zh.json")
    parser.add_argument("--output", type=Path, default=here / "runs" / "rgb_rejection.jsonl")
    parser.add_argument("--report", type=Path, default=here / "report.md")
    parser.add_argument("--limit", type=int, default=100, help="Max cases (0 = all).")
    parser.add_argument("--docs-per-case", type=int, default=5, help="Negative docs fed per case (top-k depth).")
    parser.add_argument("--only-id", type=str, default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Delay between cases (RPM quota).")
    parser.add_argument("--provider", type=str, default=None, help="Override synthesizer provider (e.g. deepseek). Default: production.")
    parser.add_argument("--model", type=str, default=None, help="Override synthesizer model name. Default: production.")
    parser.add_argument("--judge-provider", type=str, default=None, help="Separate judge provider (e.g. vertex) to avoid self-eval. Default: same as synthesizer.")
    parser.add_argument("--judge-model", type=str, default=None, help="Separate judge model name. Default: same as synthesizer.")
    parser.add_argument("--synth-prompt-file", type=Path, default=None, help="Override the final-synthesizer prompt with this file (grounding-hardening A/B). Default: production prompt.")
    return parser.parse_args()


def _build_report(
    records: list[dict[str, Any]],
    *,
    docs_per_case: int,
    dropped: int | None = None,
    eligible: int | None = None,
) -> str:
    ok = [r for r in records if r.get("status") == "success"]
    n = len(ok)
    if n == 0:
        return "# G6 RGB negative-rejection 评测报告\n\n暂无已完成样本。\n"

    # Deterministic hard-rejection: gold answer absent from the output.
    hard_reject = sum(1 for r in ok if not r.get("gold_present")) / n
    # LLM-judge metrics.
    judged = [r for r in ok if isinstance(r.get("answered"), bool)]
    answered_rate = sum(1 for r in judged if r["answered"]) / len(judged) if judged else 0.0
    misattr_rate = sum(1 for r in judged if r.get("misattributed")) / len(judged) if judged else 0.0
    misattr_cases = [r for r in judged if r.get("misattributed")]
    supplemented = [r for r in judged if r["answered"] and not r.get("misattributed")]
    answered_cases = [r for r in judged if r["answered"]]
    mark_rate = len(supplemented) / len(answered_cases) if answered_cases else 0.0

    from collections import Counter

    combo = Counter(
        (bool(r.get("gold_present")), bool(r["answered"]), bool(r.get("misattributed")))
        for r in judged
    )
    combo_labels = {
        (False, False, False): "硬拒：没给答案、没冒充、也没吐出标准答案（最干净的 withhold）",
        (False, True, False): "老实补充：给了实质答案并标注假设，但未逐字命中标准答案",
        (True, True, False): "答对且老实标注假设（理想态：有用又诚实）",
        (True, True, True): "答对，却假装是证据支撑 → 误归因（真·幻觉）",
        (False, True, True): "冒充证据，且答案未命中标准答案 → 误归因（真·幻觉）",
        (True, False, False): "边角：标准答案串偶现于一段拒答中，judge 判未真答",
        (False, False, True): "异常：未答却被判误归因（应≈0）",
        (True, False, True): "异常：未答却被判误归因（应≈0）",
    }

    lines: list[str] = []
    lines.append("# G6 RGB negative-rejection 评测报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now(timezone.utc).isoformat()}")
    head = f"完成样本数：**{n}**（judge 成功 {len(judged)}）"
    if dropped is not None and eligible is not None:
        head += f"；数据 300 条中可用（去泄漏）{eligible} 条，剔除 {dropped} 条。"
    lines.append("")
    lines.append(head)
    lines.append("")

    # --- 结论：三行为 + 一个数 ---
    lines.append("## 结论（一句话）")
    lines.append("")
    lines.append("agent 被喂“只含干扰、不含答案”的文档时，只会做三件事：")
    lines.append("")
    lines.append("1. **拒答**（“证据不足”）—— 安全")
    lines.append("2. **老实补充**（“证据不足，但据我所知…（假设）”）—— 有用且诚实")
    lines.append("3. **撒谎说出处**（“根据证据…”，而证据并没说）—— 唯一的失败 = 幻觉")
    lines.append("")
    lines.append(
        f"**只有第 3 种是坏的。误归因率 = {misattr_rate*100:.1f}%（n={len(judged)}）。** "
        f"选择回答时 {mark_rate*100:.0f}% 会老实标注假设。"
    )
    lines.append("")

    # --- 指标 ---
    lines.append("## 指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| **误归因/伪造率（LLM judge）** | **{misattr_rate*100:.1f}%** "
        "| 把证据外内容冒充成证据支撑（标了“假设/外部知识”的不算） |"
    )
    lines.append(
        f"| 硬拒答率（确定性，gold 未出现） | {hard_reject*100:.1f}% "
        "| 越高=越常 withhold；含少量“答对但字串未逐字命中”，略偏高 |"
    )
    lines.append(f"| 回答率（含标注假设） | {answered_rate*100:.1f}% | |")
    lines.append(f"| 误归因样本数 | {len(misattr_cases)} | |")
    lines.append(f"| 补充式回答样本数 | {len(supplemented)} | 答了且正确标假设——权衡点，非幻觉 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- 附录 A：口径与字段 ---
    lines.append("## 附录 A · 口径与字段含义")
    lines.append("")
    lines.append(
        f"口径：RGB 中文 negative-rejection 子集。每条只喂 `negative`（不含答案）文档"
        f"（top-{docs_per_case}），经**生产接地 prompt** 合成答案。"
        "已剔除答案泄漏进所喂文档的 case（中文/阿拉伯数字归一后），避免把 RGB negative 标注噪声误判为幻觉。"
    )
    lines.append("")
    lines.append("| 字段 | 谁判的 | True 的含义 |")
    lines.append("|---|---|---|")
    lines.append(
        "| `gold_present` | 确定性字符串匹配 "
        "| 标准答案字串（数字归一）原样出现在输出里。文档不含答案→出现=模型自己供的。**只问“在不在”，不问“对不对”** |"
    )
    lines.append("| `answered` | LLM 判官 | 给了实质性回答（哪怕先说“证据不足”再补充） |")
    lines.append(
        "| `misattributed` | LLM 判官 "
        "| 把证据外内容冒充成“证据支撑”（标了“假设/外部知识”的不算）= 真·幻觉 |"
    )
    lines.append("")

    # --- 附录 B：三字段组合分布 ---
    lines.append("## 附录 B · 三字段的组合分布")
    lines.append("")
    lines.append("| gold_present | answered | misattributed | 条数 | 含义 |")
    lines.append("|:---:|:---:|:---:|---:|---|")
    for key in sorted(combo, key=lambda k: -combo[k]):
        g, a, m = key
        lines.append(
            f"| {'T' if g else 'F'} | {'T' if a else 'F'} | {'T' if m else 'F'} "
            f"| {combo[key]} | {combo_labels.get(key, '')} |"
        )
    lines.append("")

    # --- 附录 C：误归因样本 ---
    if misattr_cases:
        lines.append("## 附录 C · 误归因样本（把外部知识当证据，真·幻觉，重点排查）")
        lines.append("")
        for r in misattr_cases[:12]:
            ans = " ".join(str(r.get("answer") or "").split())
            lines.append(
                f"- `{r.get('case_id')}` 问：{str(r.get('question') or '')[:50]} "
                f"答：{ans[:120]} — {str(r.get('judge_reason') or '')[:80]}"
            )
        lines.append("")
    if supplemented:
        lines.append("## 附录 D · 补充式回答样本（声明证据不足后标注假设补充——权衡点，非幻觉）")
        lines.append("")
        for r in supplemented[:8]:
            ans = " ".join(str(r.get("answer") or "").split())
            lines.append(f"- `{r.get('case_id')}` 问：{str(r.get('question') or '')[:50]} 答：{ans[:110]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    _load_env()

    data_path = args.data.resolve()
    if not data_path.exists():
        _safe_print(f"RGB data not found: {data_path}")
        return 1

    all_cases = _load_cases(data_path, only_id=args.only_id)
    eligible = [
        c for c in all_cases
        if not _answer_leaks_into_docs(c.get("answer"), list(c.get("negative") or []), docs_limit=args.docs_per_case)
    ]
    dropped = len(all_cases) - len(eligible)

    output_path = args.output.resolve()
    existing = {str(r.get("case_id") or ""): r for r in _read_jsonl(output_path)} if output_path.exists() else {}

    if not args.report_only:
        cases = eligible[: args.limit] if args.limit and args.limit > 0 else eligible
        from agent.graph.model_io import _coerce_to_text, _extract_json_object  # noqa: E402
        from agent.graph.prompts import _FINAL_SYSTEM_PROMPT  # noqa: E402
        from services.llm_provider import build_chat_model, resolve_agent_model_config  # noqa: E402

        prompt_file = args.synth_prompt_file
        if prompt_file is not None:
            prompt_file = prompt_file.resolve()
            if not prompt_file.exists():
                _safe_print(f"Synth prompt file not found: {prompt_file}")
                return 1
        final_prompt = _resolve_final_prompt(prompt_file, default=_FINAL_SYSTEM_PROMPT)
        if prompt_file is not None:
            _safe_print(f"Using OVERRIDE final prompt from {prompt_file} ({len(final_prompt)} chars)")

        config = resolve_agent_model_config()
        provider = args.provider or config.provider
        model = args.model or config.model
        client = build_chat_model(
            provider=provider,
            model_name=model,
            temperature=0.0,
            default_provider=config.provider,
            default_model=config.model,
        )
        # Single backoff layer: disable the client's internal retry so call_with_retry paces
        # 429s (fewer hammering calls + longer spacing = better under exhausted quota).
        try:
            client.max_retries = 0
        except Exception:  # noqa: BLE001 - not all providers expose this field
            pass
        _safe_print(f"Synth model: provider={provider} model={model} (temp=0.0, inner-retry off)")

        judge_provider, judge_model = _resolve_judge_config(
            synth_provider=provider, synth_model=model,
            judge_provider=args.judge_provider, judge_model=args.judge_model,
        )
        if (judge_provider, judge_model) == (provider, model):
            judge_client = client
        else:
            judge_client = build_chat_model(
                provider=judge_provider,
                model_name=judge_model,
                temperature=0.0,
                default_provider=config.provider,
                default_model=config.model,
            )
            try:
                judge_client.max_retries = 0
            except Exception:  # noqa: BLE001
                pass
        _safe_print(f"Judge model: provider={judge_provider} model={judge_model}")

        done = _read_done(output_path)
        pending = [c for c in cases if str(c.get("id")) not in done]
        _safe_print(
            f"Eligible(去泄漏)={len(eligible)} (剔除 {dropped}); 本次={len(cases)}; "
            f"pending={len(pending)}; docs-per-case={args.docs_per_case}"
        )

        for idx, case in enumerate(pending, start=1):
            case_id = str(case.get("id"))
            question = str(case.get("query") or "")
            evidence_block = _format_negative_docs(list(case.get("negative") or []), limit=args.docs_per_case)
            _safe_print(f"[{idx}/{len(pending)}] id={case_id} {question[:48]} ...")
            try:
                answer = call_with_retry(
                    lambda: _synthesize(
                        question, evidence_block,
                        client=client, coerce_text_fn=_coerce_to_text, final_prompt=final_prompt,
                    ),
                    label=f"{case_id} synth",
                )
                verdict = call_with_retry(
                    lambda: _judge_answer(
                        question, evidence_block, answer,
                        client=judge_client, coerce_text_fn=_coerce_to_text, extract_json_fn=_extract_json_object,
                    ),
                    label=f"{case_id} judge",
                )
                gold = list(case.get("answer") or [])
                record = {
                    "case_id": case_id,
                    "question": question,
                    "gold_answer": gold,
                    "answer": answer,
                    "gold_present": _gold_answer_present(gold, answer),
                    "answered": (verdict or {}).get("answered"),
                    "misattributed": (verdict or {}).get("misattributed"),
                    "judge_reason": (verdict or {}).get("reason", "") if verdict else "judge_parse_failed",
                    "status": "success",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                _safe_print(
                    f"  -> gold_present={record['gold_present']} "
                    f"answered={record['answered']} misattributed={record['misattributed']}"
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "case_id": case_id,
                    "question": question,
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                _safe_print(f"  -> ERROR: {record['error_message']}")
            _append_jsonl(output_path, record)
            existing[case_id] = record
            if args.sleep_seconds > 0 and idx < len(pending):
                time.sleep(args.sleep_seconds)

    report = _build_report(
        list(existing.values()), docs_per_case=args.docs_per_case, dropped=dropped, eligible=len(eligible)
    )
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _safe_print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
