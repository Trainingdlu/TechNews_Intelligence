"""Output URL guard and final fallback helpers for graph nodes."""

from __future__ import annotations

import re
from typing import Any

from agent.core.evidence import extract_urls, normalize_url_for_match


def _guard_output_urls(text: str, valid_urls: list[str]) -> tuple[str, dict[str, Any]]:
    raw = str(text or "").strip()
    allowed: dict[str, str] = {}
    for item in valid_urls or []:
        url = str(item or "").strip()
        normalized = normalize_url_for_match(url)
        if normalized and normalized not in allowed:
            allowed[normalized] = url

    output_urls = extract_urls(raw)
    removed_count = 0
    has_valid_body_url = False
    guarded = raw
    for url in output_urls:
        normalized = normalize_url_for_match(url)
        if normalized and normalized in allowed:
            has_valid_body_url = True
            continue
        guarded = guarded.replace(url, "").strip()
        removed_count += 1

    guarded = re.sub(r"[ \t]+([，,。.;；:：!?！？])", r"\1", guarded)
    guarded = re.sub(r"\n{3,}", "\n\n", guarded).strip()

    appended_evidence_url = False
    if allowed and not has_valid_body_url:
        first_url = next(iter(allowed.values()))
        guarded = f"{guarded.rstrip()}\n\n来源：{first_url}".strip()
        appended_evidence_url = True

    return guarded, {
        "removed_unknown_url_count": removed_count,
        "appended_evidence_url": appended_evidence_url,
    }


def _fallback_final_text(user_message: str, state: dict[str, Any]) -> str:
    if str((state.get("intent") or {}).get("route") or "") == "direct_answer":
        if re.search(r"你好|您好|hello|hi", user_message, re.IGNORECASE):
            return "你好，我可以帮助你检索和分析科技新闻，包括趋势、对比、时间线、来源差异和行业格局。"
        return "我可以帮助你基于新闻证据做科技情报分析。你可以直接提出公司、产品、主题、时间范围或要比较的对象。"
    urls = list(state.get("evidence_urls") or [])
    if urls:
        return f"已找到相关证据，但当前模型暂时无法完成完整综合。建议先查看核心来源：{urls[0]}"
    return "目前没有足够可靠的证据生成结论。请补充更具体的时间范围、实体或来源。"
