"""Final fallback text helper for graph nodes."""

from __future__ import annotations

import re
from typing import Any


def _fallback_final_text(user_message: str, state: dict[str, Any]) -> str:
    if str((state.get("intent") or {}).get("route") or "") == "direct_answer":
        if re.search(r"你好|您好|hello|hi", user_message, re.IGNORECASE):
            return "你好，我可以帮助你检索和分析科技新闻，包括趋势、对比、时间线、来源差异和行业格局。"
        return "我可以帮助你基于新闻证据做科技情报分析。你可以直接提出公司、产品、主题、时间范围或要比较的对象。"
    urls = list(state.get("evidence_urls") or [])
    if urls:
        return (
            "抱歉，我暂时没能就这个问题生成可靠的结论，"
            "数据库中可能没有与之直接相关的信息。"
            f"可参考检索到的来源：{urls[0]}"
        )
    return "目前没有足够可靠的证据生成结论。请补充更具体的时间范围、实体或来源。"
