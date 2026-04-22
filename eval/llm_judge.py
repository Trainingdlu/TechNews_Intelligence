"""LLM-as-a-Judge evaluator for generation quality metrics.

Provides Faithfulness and Answer Relevancy scoring using the same
LLM infrastructure as the agent (Vertex/Gemini), avoiding external
dependencies like RAGAS.

Implements CoT prompting for transparent, auditable reasoning.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# LLM builder (reused from build_task_dataset_v1.py pattern)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_PROVIDER = "vertex"


def _build_judge_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> Any:
    """Build a chat model for judge evaluation."""
    provider = str(provider or os.getenv("TASK_EVAL_PROVIDER", DEFAULT_PROVIDER)).strip().lower()
    model = str(model or os.getenv("TASK_EVAL_MODEL", DEFAULT_MODEL)).strip()

    if provider in {"gemini_api", "gemini", "google_ai_studio", "developer_api"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = str(os.getenv("GEMINI_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for provider=gemini_api.")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

    if provider in {"vertex", "vertex_ai", "gcp"}:
        from langchain_google_vertexai import ChatVertexAI

        project = str(os.getenv("VERTEX_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", ""))).strip()
        if not project:
            raise ValueError("VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT is required for provider=vertex.")
        location = str(
            os.getenv(
                "VERTEX_GENERATION_LOCATION",
                os.getenv("VERTEX_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global")),
            )
        ).strip()
        return ChatVertexAI(
            model=model,
            project=project,
            location=location,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

FAITHFULNESS_SYSTEM = (
    "You are a strict evaluation judge assessing the faithfulness of an AI-generated answer.\n\n"
    "Faithfulness measures whether the answer's claims are supported by the provided context "
    "(tool outputs / retrieved evidence). Claims not present in the context are unfaithful.\n\n"
    "Think step by step:\n"
    "1) Identify all factual claims in the answer.\n"
    "2) For each claim, check if it is directly supported by the context.\n"
    "3) Count supported vs unsupported claims.\n"
    "4) Assign a score from 1 to 5.\n\n"
    "Scoring rubric:\n"
    "- 5: All claims are fully supported by context.\n"
    "- 4: Most claims supported, 1 minor unsupported detail.\n"
    "- 3: Several claims supported, but some lack evidence.\n"
    "- 2: Many claims unsupported or extrapolated.\n"
    "- 1: Answer contradicts context or is entirely fabricated.\n\n"
    "Return a JSON object with keys: score (int 1-5), reasoning (string).\n"
    "Do not output markdown fences."
)

RELEVANCY_SYSTEM = (
    "You are a strict evaluation judge assessing answer relevancy.\n\n"
    "Answer Relevancy measures whether the answer directly addresses the user's question "
    "and provides useful information for the specific intent.\n\n"
    "Think step by step:\n"
    "1) Identify the user's core question and intent.\n"
    "2) Check if the answer addresses that specific question.\n"
    "3) Evaluate completeness — does it cover the key aspects?\n"
    "4) Assign a score from 1 to 5.\n\n"
    "Scoring rubric:\n"
    "- 5: Directly and completely answers the question with relevant details.\n"
    "- 4: Mostly answers the question, minor gaps.\n"
    "- 3: Partially relevant, misses key aspects.\n"
    "- 2: Tangentially related, mostly off-topic.\n"
    "- 1: Completely irrelevant or does not address the question.\n\n"
    "Return a JSON object with keys: score (int 1-5), reasoning (string).\n"
    "Do not output markdown fences."
)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """Extract first JSON object from LLM response."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON found in judge response.")
    depth = 0
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("Incomplete JSON in judge response.")


def _parse_judge_response(raw: Any) -> dict[str, Any]:
    """Parse and validate judge model response."""
    content = raw
    if hasattr(raw, "content"):
        content = raw.content
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", item) if isinstance(item, dict) else item)
            for item in content
        )
    text = str(content or "").strip()
    if not text:
        return {"score": 0, "reasoning": "Empty judge response."}

    try:
        result = _extract_json(text)
        score = int(result.get("score", 0))
        score = max(1, min(5, score))
        reasoning = str(result.get("reasoning", "")).strip()
        return {"score": score, "reasoning": reasoning}
    except Exception as exc:
        return {"score": 0, "reasoning": f"Failed to parse judge response: {exc}. Raw: {text[:300]}"}


# ---------------------------------------------------------------------------
# LLMJudge class
# ---------------------------------------------------------------------------

class LLMJudge:
    """Faithfulness and Answer Relevancy evaluator using LLM-as-a-Judge.

    Uses CoT prompting with 1-5 scoring scale, aligned with the
    rag-evaluation skill's rubric.

    Parameters
    ----------
    provider:
        LLM provider (vertex, gemini_api, etc.).
    model:
        Model name.
    temperature:
        Judge temperature (0.0 recommended for consistency).
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ):
        self._llm = _build_judge_model(
            provider=provider, model=model, temperature=temperature,
        )

    def judge_faithfulness(
        self, context: str, answer: str,
    ) -> dict[str, Any]:
        """Evaluate faithfulness of answer against context.

        Parameters
        ----------
        context:
            Tool outputs / retrieved evidence (concatenated).
        answer:
            The AI-generated answer to evaluate.

        Returns
        -------
        dict with ``score`` (1-5) and ``reasoning``.
        """
        user_msg = (
            f"## Context (Tool Outputs)\n{context}\n\n"
            f"## Answer to Evaluate\n{answer}"
        )
        try:
            result = self._llm.invoke([
                SystemMessage(content=FAITHFULNESS_SYSTEM),
                HumanMessage(content=user_msg),
            ])
            return _parse_judge_response(result)
        except Exception as exc:
            return {"score": 0, "reasoning": f"Judge invocation failed: {exc}"}

    def judge_relevancy(
        self, question: str, answer: str,
    ) -> dict[str, Any]:
        """Evaluate relevancy of answer to the question.

        Parameters
        ----------
        question:
            The user's original question.
        answer:
            The AI-generated answer to evaluate.

        Returns
        -------
        dict with ``score`` (1-5) and ``reasoning``.
        """
        user_msg = (
            f"## User Question\n{question}\n\n"
            f"## Answer to Evaluate\n{answer}"
        )
        try:
            result = self._llm.invoke([
                SystemMessage(content=RELEVANCY_SYSTEM),
                HumanMessage(content=user_msg),
            ])
            return _parse_judge_response(result)
        except Exception as exc:
            return {"score": 0, "reasoning": f"Judge invocation failed: {exc}"}

    def judge_both(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> dict[str, Any]:
        """Run both faithfulness and relevancy evaluations.

        Returns dict with ``faithfulness`` and ``relevancy`` sub-dicts.
        """
        faith = self.judge_faithfulness(context=context, answer=answer)
        relev = self.judge_relevancy(question=question, answer=answer)
        return {
            "faithfulness": faith,
            "relevancy": relev,
            "composite_score": round(
                (faith.get("score", 0) + relev.get("score", 0)) / 2.0, 2
            ),
        }
