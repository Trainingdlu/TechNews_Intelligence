"""Deterministic context-pack token budget measurement (no LLM, no DB).

Measures how many tokens the assembled context pack contributes to a turn's
prompt(s), broken down by render section, with the in-turn injection multiplier,
plus the exact savings from the render de-duplication (F1).

Two counts are reported per item:
  - approx:   count_tokens_approximately (the counter ``memory_policy`` actually
              budgets with; flat chars_per_token=4.0).
  - tiktoken: cl100k_base BPE, a proxy for a real tokenizer (window/cost usage).

The gap between them exposes how the chars/4 approximation under-counts CJK
text: the production budget can believe it is spending far fewer tokens than a
real tokenizer charges. cl100k_base is a proxy (the agent uses Gemini/DeepSeek,
not GPT), but the CJK under-count holds across modern BPE tokenizers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from agent.context_manager import (
    MAX_SELECTED_TURNS,
    build_context_pack,
    build_history_manifest,
    normalize_context_curator_result,
    render_context_for_prompt,
)
from agent.core.evidence import normalize_url_for_match

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENC = None


# --- token counters ----------------------------------------------------------

_EMPTY_OVERHEAD = count_tokens_approximately([HumanMessage(content="")])


def count_approx(text: str) -> int:
    """Approximate tokens of text alone (strip the fixed per-message overhead)."""
    if not text:
        return 0
    return max(0, count_tokens_approximately([HumanMessage(content=text)]) - _EMPTY_OVERHEAD)


def count_tiktoken(text: str) -> int | None:
    if _ENC is None or not text:
        return 0 if _ENC is not None else None
    return len(_ENC.encode(text))


# --- section attribution -----------------------------------------------------

SECTION_HEADERS = (
    "Context summary:",
    "Thread memory summary:",
    "Thread evidence index:",
    "Selected prior turns:",
    "Selected memory evidence:",
    "Prior evidence URLs:",
)


def split_sections(rendered: str) -> dict[str, str]:
    """Slice the rendered pack into its top-level sections by their headers."""
    if not rendered or rendered == "(none)":
        return {}
    found = [(rendered.find(h), h) for h in SECTION_HEADERS if rendered.find(h) >= 0]
    found.sort()
    sections: dict[str, str] = {}
    for i, (start, header) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(rendered)
        sections[header.rstrip(":")] = rendered[start:end].strip()
    return sections


# --- pre-F1 renderer (no de-duplication), for an exact before/after ----------

def render_pre_f1(context_pack: dict | None) -> str:
    """Reproduce the pre-de-dup render so F1 savings can be measured exactly."""
    from agent.context_manager import MAX_SELECTED_TURNS as _MAX, _evidence_age_label

    if not isinstance(context_pack, dict):
        return "(none)"
    parts: list[str] = []
    summary = str(context_pack.get("context_summary") or "").strip()
    if summary:
        parts.append(f"Context summary:\n{summary}")
    memory = context_pack.get("thread_memory_summary")
    if isinstance(memory, dict) and memory:
        memory_text = str(memory.get("summary_text") or "").strip()
        if memory_text:
            parts.append(f"Thread memory summary:\n{memory_text}")
        memory_evidence = [
            item for item in memory.get("evidence_index", [])
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        ]
        if memory_evidence:
            lines = []
            for item in memory_evidence[:8]:
                title = str(item.get("title") or "").strip() or "previous evidence"
                url = str(item.get("url") or "").strip()
                excerpt = str(item.get("excerpt") or "").strip()
                age = _evidence_age_label(item.get("created_at"))
                header = f"- {title} | {url}" + (f" ({age})" if age else "")
                lines.append(f"{header}\n  {excerpt}" if excerpt else header)
            parts.append("Thread evidence index:\n" + "\n".join(lines))
    turns = context_pack.get("selected_turns")
    if isinstance(turns, list) and turns:
        lines = []
        for turn in turns[:_MAX]:
            if not isinstance(turn, dict):
                continue
            user = str(turn.get("user_message") or "").strip()
            assistant = str(turn.get("assistant_excerpt") or "").strip()
            evidence_urls = [
                str(item or "").strip()
                for item in turn.get("evidence_urls", [])
                if str(item or "").strip()
            ]
            lines.append(
                "\n".join(
                    part
                    for part in [
                        f"Turn {turn.get('turn_id')}:",
                        f"User: {user}" if user else "",
                        f"Assistant excerpt: {assistant}" if assistant else "",
                        f"Evidence URLs: {' '.join(evidence_urls[:6])}" if evidence_urls else "",
                    ]
                    if part
                )
            )
        if lines:
            parts.append("Selected prior turns:\n" + "\n\n".join(lines))
    memory_selected = context_pack.get("selected_memory_evidence")
    if isinstance(memory_selected, list) and memory_selected:
        lines = []
        for item in memory_selected[:8]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip() or "previous evidence"
            url = str(item.get("url") or "").strip()
            excerpt = str(item.get("excerpt") or "").strip()
            age = _evidence_age_label(item.get("created_at"))
            if url:
                header = f"- {title} | {url}" + (f" ({age})" if age else "")
                lines.append(header + (f"\n  {excerpt}" if excerpt else ""))
        if lines:
            parts.append("Selected memory evidence:\n" + "\n".join(lines))
    evidence_urls = [
        str(item or "").strip()
        for item in context_pack.get("selected_evidence_urls", [])
        if str(item or "").strip()
    ]
    if evidence_urls:
        parts.append("Prior evidence URLs:\n" + "\n".join(f"- {url}" for url in evidence_urls[:12]))
    return "\n\n".join(parts).strip() or "(none)"


# --- scenario synthesis ------------------------------------------------------

def _url(i: int) -> str:
    return f"https://news.example.com/ai/coverage-{i:04d}"


def _answer(topic: str, urls: list[str]) -> str:
    body = (
        f"关于{topic}的最新进展显示，多家厂商在该领域加速布局，竞争格局正在重塑，"
        f"产品定价与商业化路径成为焦点。Market analysis suggests continued momentum "
        f"across enterprise and consumer segments, with pricing pressure intensifying. "
    )
    cites = " ".join(f"[{i + 1}] {u}" for i, u in enumerate(urls))
    return body * 2 + "\nSources:\n" + cites


def _turn(user_text: str, model_text: str, urls: list[str]) -> list[dict]:
    return [
        {"role": "user", "parts": [{"text": user_text}]},
        {
            "role": "model",
            "parts": [{"text": model_text}],
            "citation_urls": urls,
            "url_title_map": {u: f"{['英伟达','OpenAI','谷歌','Anthropic','Meta'][i % 5]} 相关报道 {i}" for i, u in enumerate(urls)},
        },
    ]


def _history(n_turns: int, urls_per_turn: int) -> list[dict]:
    topics = ["英伟达新卡", "OpenAI 模型", "谷歌 Gemini", "AI 监管", "开源大模型", "AI 芯片"]
    history: list[dict] = []
    k = 0
    for t in range(n_turns):
        topic = topics[t % len(topics)]
        urls = [_url(k + j) for j in range(urls_per_turn)]
        k += urls_per_turn
        history.extend(_turn(f"{topic}最近怎么样？", _answer(topic, urls), urls))
    return history


def _memory(n_evidence: int, summary_turns: int) -> dict:
    evidence_index = [
        {
            "url": _url(900 + i),
            "title": f"历史证据 {i}：行业动态与商业化分析",
            "excerpt": f"该报道指出在第 {i} 阶段市场出现明显分化，enterprise adoption accelerating。",
        }
        for i in range(n_evidence)
    ]
    blocks = []
    for i in range(summary_turns):
        blocks.append(
            f"User: 第{i}轮提问关于 AI 行业格局\n"
            f"Assistant: 第{i}轮回答，综合多源证据给出趋势研判 https://news.example.com/ai/coverage-{900 + i:04d}\n"
            f"Evidence: https://news.example.com/ai/coverage-{900 + i:04d}"
        )
    return {"summary_text": "\n\n".join(blocks), "evidence_index": evidence_index}


def build_scenarios() -> list[tuple[str, dict]]:
    scenarios: list[tuple[str, dict]] = []

    # S1: first turn, no history, no memory (baseline floor)
    scenarios.append((
        "S1 首轮·无历史无记忆",
        build_context_pack(user_message="今天有什么 AI 新闻？", history=[]),
    ))

    # S2: a few turns + memory + curator hit (common "loaded" case)
    h2 = _history(n_turns=3, urls_per_turn=2)
    m2 = _memory(n_evidence=4, summary_turns=3)
    manifest2 = build_history_manifest(h2)
    raw2 = {
        "depends_on_history": True,
        "standalone_question": "英伟达新卡相对 OpenAI 模型的进展对比",
        "selected_turn_ids": [1, 2],
        "selected_evidence_urls": [_url(0), _url(2), _url(900), _url(901)],
        "context_summary": "用户在多轮中持续追踪 AI 厂商动态，本轮要求对比英伟达与 OpenAI。",
        "reason": "follow-up comparison",
        "confidence": 0.85,
    }
    cur2 = normalize_context_curator_result(raw2, manifest2, m2)
    scenarios.append((
        "S2 多轮+记忆+curator命中",
        build_context_pack(
            user_message="它们俩最近的对比？",
            history=h2,
            history_manifest=manifest2,
            memory_summary=m2,
            curator_result=cur2,
            curator_used=True,
        ),
    ))

    # S3: long history + full memory (near worst-case)
    h3 = _history(n_turns=12, urls_per_turn=3)
    m3 = _memory(n_evidence=8, summary_turns=4)
    manifest3 = build_history_manifest(h3)
    sel_urls = [_url(i) for i in range(0, 9, 3)] + [_url(900 + i) for i in range(4)]
    raw3 = {
        "depends_on_history": True,
        "standalone_question": "综合过去多轮，AI 芯片与大模型的整体格局研判",
        "selected_turn_ids": [9, 10, 11, 12],
        "selected_evidence_urls": sel_urls,
        "context_summary": "用户进行了长达十余轮的 AI 行业追踪，本轮要求综合研判整体格局与商业化前景。",
        "reason": "long-running synthesis",
        "confidence": 0.8,
    }
    cur3 = normalize_context_curator_result(raw3, manifest3, m3)
    scenarios.append((
        "S3 长历史+满记忆(近最坏)",
        build_context_pack(
            user_message="综合看整体格局怎么样？",
            history=h3,
            history_manifest=manifest3,
            memory_summary=m3,
            curator_result=cur3,
            curator_used=True,
        ),
    ))

    return scenarios


# --- reporting ---------------------------------------------------------------

INJECTIONS = (("final only", 1), ("+planner", 2), ("+1 tool round", 3), ("+2 tool rounds", 4))


def _fmt(approx: int, tk: int | None) -> str:
    tk_s = "n/a" if tk is None else str(tk)
    return f"{approx:>7}  {tk_s:>9}"


def measure_scenario(name: str, pack: dict) -> dict:
    rendered = render_context_for_prompt(pack)
    sections = split_sections(rendered)
    total_approx = count_approx(rendered) if rendered != "(none)" else 0
    total_tk = count_tiktoken(rendered) if rendered != "(none)" else 0
    pre = render_pre_f1(pack)
    pre_approx = count_approx(pre) if pre != "(none)" else 0
    pre_tk = count_tiktoken(pre) if pre != "(none)" else 0
    tool = render_context_for_prompt(pack, "tool")
    tool_approx = count_approx(tool) if tool != "(none)" else 0
    tool_tk = count_tiktoken(tool) if tool != "(none)" else 0
    return {
        "name": name,
        "rendered": rendered,
        "sections": sections,
        "total_approx": total_approx,
        "total_tk": total_tk,
        "pre_approx": pre_approx,
        "pre_tk": pre_tk,
        "tool_approx": tool_approx,
        "tool_tk": tool_tk,
    }


def print_report(results: list[dict]) -> None:
    print("=" * 72)
    print("上下文 pack token 预算测量  (approx = 代码预算计数器 / tiktoken = 真实分词器代理)")
    print("=" * 72)
    for r in results:
        print(f"\n### {r['name']}")
        print(f"{'section':<26}{'approx':>9}{'tiktoken':>11}")
        print("-" * 48)
        for header in SECTION_HEADERS:
            label = header.rstrip(":")
            text = r["sections"].get(label, "")
            if not text:
                continue
            print(f"{label:<26}{_fmt(count_approx(text), count_tiktoken(text))}")
        print("-" * 48)
        print(f"{'TOTAL pack (当前/F1后)':<26}{_fmt(r['total_approx'], r['total_tk'])}")

        # injection multiplier
        print("  每轮注入放大 (pack × 注入点):")
        for label, mult in INJECTIONS:
            a = r["total_approx"] * mult
            t = None if r["total_tk"] is None else r["total_tk"] * mult
            tk_s = "n/a" if t is None else str(t)
            print(f"    {label:<16} approx {a:>7}   tiktoken {tk_s:>9}")

        # F1 before/after
        d_approx = r["pre_approx"] - r["total_approx"]
        pct = (100.0 * d_approx / r["pre_approx"]) if r["pre_approx"] else 0.0
        d_tk = None if (r["total_tk"] is None or r["pre_tk"] is None) else r["pre_tk"] - r["total_tk"]
        tk_s = "n/a" if d_tk is None else str(d_tk)
        print(f"  F1 去重省下:        approx {d_approx:>7} ({pct:4.1f}%)   tiktoken {tk_s:>9}")

        # F2 per-node profile: the tool worker gets the lean (URL-only) profile.
        # Per-turn injections = final(full) + planner(full) + tool(profile) × rounds.
        ta = r["tool_approx"]
        tt = r["tool_tk"]
        tool_pct = (100.0 * (r["total_approx"] - ta) / r["total_approx"]) if r["total_approx"] else 0.0
        tt_s = "n/a" if tt is None else str(tt)
        print(f"  F2 tool-profile单份: approx {ta:>7} (比 full 小 {tool_pct:4.1f}%)   tiktoken {tt_s:>9}")
        print("  F2 每轮注入省下 (planner+final=full, tool节点=lean):")
        for rounds in (1, 2):
            cur = r["total_approx"] * (2 + rounds)
            f2 = r["total_approx"] * 2 + ta * rounds
            saved = cur - f2
            spct = (100.0 * saved / cur) if cur else 0.0
            print(
                f"    {rounds} 轮工具: approx {cur:>6} -> {f2:<6} 省 {saved:>5} ({spct:4.1f}%)"
            )

        # CJK under-count gap
        if r["total_tk"] and r["total_approx"]:
            ratio = r["total_tk"] / r["total_approx"]
            print(f"  approx→tiktoken 比: {ratio:4.2f}x  (>1 = 代码预算低估真实 token，CJK 越多越严重)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-rendered", action="store_true", help="print the rendered pack text per scenario")
    args = parser.parse_args()

    if _ENC is None:
        print("[warn] tiktoken 不可用，仅报告 approx 计数。", file=sys.stderr)

    results = [measure_scenario(name, pack) for name, pack in build_scenarios()]
    print_report(results)

    if args.show_rendered:
        for r in results:
            print("=" * 72)
            print(f"# {r['name']} — rendered pack")
            print("=" * 72)
            print(r["rendered"])
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
