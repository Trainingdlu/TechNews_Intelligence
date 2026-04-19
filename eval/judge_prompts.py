"""Prompt templates for Judge-layer LLM evaluation."""

from __future__ import annotations

import json
from typing import Any

SCORE_DIMENSIONS: tuple[str, ...] = (
    "accuracy",
    "groundedness",
    "coherence",
    "completeness",
    "helpfulness",
)

SCORE_WEIGHTS: dict[str, float] = {
    "accuracy": 0.30,
    "groundedness": 0.25,
    "completeness": 0.20,
    "coherence": 0.15,
    "helpfulness": 0.10,
}

JUDGE_SYSTEM_PROMPT = (
    "你是一个用来评估问答(QA)输出的裁判模型。\n"
    "请使用1到5的整数分数在5个维度上对答案进行评分。\n"
    "你必须只输出有效的JSON数据，不要包含markdown，也不要包含额外键值。\n"
    "Scoring dimensions:\n"
    "- accuracy: factual correctness, consistency with provided references.\n"
    "- groundedness: claims supported by provided contexts/evidence.\n"
    "- coherence: logical flow and no contradictions.\n"
    "- completeness: coverage of user question and constraints.\n"
    "- helpfulness: actionable, clear, and decision-useful response.\n"
    "If answer is empty, error-like, or non-responsive, score low and explain why.\n"
    "Return JSON schema:\n"
    "{\n"
    '  "scores": {\n'
    '    "accuracy": 1,\n'
    '    "groundedness": 1,\n'
    '    "coherence": 1,\n'
    '    "completeness": 1,\n'
    '    "helpfulness": 1\n'
    "  },\n"
    '  "evidence": {\n'
    '    "accuracy": "string",\n'
    '    "groundedness": "string",\n'
    '    "coherence": "string",\n'
    '    "completeness": "string",\n'
    '    "helpfulness": "string"\n'
    "  }\n"
    "}"
)


def _truncate_text(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}...[truncated]"


def build_judge_user_payload(
    case: dict[str, Any],
    *,
    max_answer_chars: int = 5000,
    max_context_chars: int = 4000,
    max_constraints_chars: int = 3000,
) -> str:
    """Build compact JSON payload for judge prompt input."""
    contexts = case.get("contexts", [])
    if not isinstance(contexts, list):
        contexts = []
    normalized_contexts = [_truncate_text(str(item), 700) for item in contexts]
    normalized_contexts = [item for item in normalized_contexts if item]

    constraints = case.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    payload = {
        "case_id": str(case.get("case_id", "")).strip(),
        "question": _truncate_text(str(case.get("question", "")), 2500),
        "answer": _truncate_text(str(case.get("answer", "")), max_answer_chars),
        "contexts": normalized_contexts,
        "constraints": _truncate_text(
            json.dumps(constraints, ensure_ascii=False),
            max_constraints_chars,
        ),
    }

    # Keep the final payload bounded for better latency and deterministic behavior.
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if max_context_chars > 0 and len(serialized) > max_context_chars:
        payload["contexts"] = []
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return serialized


def build_judge_messages(
    case: dict[str, Any],
    *,
    max_answer_chars: int = 5000,
    max_context_chars: int = 4000,
    max_constraints_chars: int = 3000,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the judge call."""
    user_payload = build_judge_user_payload(
        case,
        max_answer_chars=max_answer_chars,
        max_context_chars=max_context_chars,
        max_constraints_chars=max_constraints_chars,
    )
    user_prompt = (
        "评估这个问答(QA)用例。请给每个维度打分(1到5)。\n"
        "只返回JSON格式。\n\n"
        f"{user_payload}"
    )
    return JUDGE_SYSTEM_PROMPT, user_prompt
