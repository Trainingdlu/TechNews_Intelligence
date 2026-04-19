"""LLM-as-a-Judge entrypoint for evaluation report scoring."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

try:
    from judge_prompts import SCORE_DIMENSIONS, SCORE_WEIGHTS, build_judge_messages
except ImportError:  # package-style import fallback
    from .judge_prompts import SCORE_DIMENSIONS, SCORE_WEIGHTS, build_judge_messages


DEFAULT_THRESHOLDS = {
    "excellent": 4.5,
    "good": 3.5,
    "adequate": 2.5,
    "poor": 1.5,
}


@dataclass
class JudgeRuntimeConfig:
    model_name: str = "gemini-2.5-pro"
    provider: str = "vertex"
    backend: str = "llm"
    temperature: float = 0.0
    batch_size: int = 4
    max_retries: int = 2
    retry_backoff_sec: float = 1.5
    request_timeout_sec: float = 120.0
    skip_failed_cases: bool = True
    max_answer_chars: int = 5000
    max_context_chars: int = 4000
    max_constraints_chars: int = 3000
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from: {path}")
    return payload


def _resolve_env_value(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _load_eval_env(env_file: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env = project_root / "agent" / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)

    if env_file:
        candidate = env_file.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Env file not found: {candidate}")
        load_dotenv(dotenv_path=candidate, override=True)

    # Prefer Vertex for judge LLM unless caller explicitly sets a provider.
    if not _resolve_env_value("JUDGE_PROVIDER", "EVAL_JUDGE_PROVIDER"):
        os.environ["JUDGE_PROVIDER"] = "vertex"


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Failed to parse judge config. Install PyYAML or use JSON syntax in judge.yaml."
            ) from exc
    else:
        parsed = yaml.safe_load(raw)

    if not parsed:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return parsed


def _extract_contexts_from_trace_summary(summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    contexts: list[str] = []
    seen: set[str] = set()
    for event in summary.get("tool_events", []) or []:
        if not isinstance(event, dict):
            continue
        output_summary = event.get("output_summary", {})
        if not isinstance(output_summary, dict):
            continue
        docs = output_summary.get("context_docs", [])
        if not isinstance(docs, list):
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            snippet = str(doc.get("summary") or doc.get("title") or doc.get("url") or "").strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            contexts.append(snippet)
    return contexts


def _coerce_reference_from_constraints(constraints: dict[str, Any]) -> str:
    ground_truth = str(constraints.get("ground_truth", "")).strip()
    if ground_truth:
        return ground_truth
    expected_facts = constraints.get("expected_facts", [])
    if isinstance(expected_facts, list):
        tokens = [str(item).strip() for item in expected_facts if str(item).strip()]
        if tokens:
            return "; ".join(tokens)
    return ""


def _first_nonempty_text(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    for item in values:
        text = str(item or "").strip()
        if text:
            return text
    return ""


def _dedupe_texts(items: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _build_cases_from_run_eval(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    cases = source_report.get("cases", [])
    if not isinstance(cases, list):
        return []

    normalized_cases: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id", "")).strip() or f"case_{idx}"
        question = str(case.get("question", "")).strip()
        answer = _first_nonempty_text(case.get("outputs", []))

        constraints = case.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        contexts: list[str] = []
        runs = case.get("runs", [])
        if isinstance(runs, list) and runs:
            first_run = runs[0]
            if isinstance(first_run, dict):
                summary = first_run.get("trace_summary")
                if isinstance(summary, dict):
                    contexts.extend(_extract_contexts_from_trace_summary(summary))

        reference = _coerce_reference_from_constraints(constraints)
        if reference and "ground_truth" not in constraints:
            constraints = dict(constraints)
            constraints["ground_truth"] = reference

        normalized_cases.append(
            {
                "case_id": case_id,
                "question": question,
                "answer": answer,
                "contexts": _dedupe_texts(contexts),
                "constraints": constraints,
            }
        )
    return normalized_cases


def _build_cases_from_rows(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = source_report.get("rows", [])
    if not isinstance(rows, list):
        return []

    normalized_cases: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", "")).strip() or f"row_{idx}"
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        contexts = row.get("contexts", [])
        if not isinstance(contexts, list):
            contexts = []

        constraints: dict[str, Any] = {}
        reference = str(row.get("reference", "")).strip()
        if reference:
            constraints["ground_truth"] = reference

        normalized_cases.append(
            {
                "case_id": case_id,
                "question": question,
                "answer": answer,
                "contexts": _dedupe_texts(str(item).strip() for item in contexts),
                "constraints": constraints,
            }
        )
    return normalized_cases


def _build_cases(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(source_report.get("cases"), list):
        return _build_cases_from_run_eval(source_report)
    if isinstance(source_report.get("rows"), list):
        return _build_cases_from_rows(source_report)
    raise ValueError("Unsupported source report format: expected `cases` or `rows`.")


def _parse_score(raw: Any) -> int:
    if isinstance(raw, bool):
        return 1
    try:
        value = round(float(raw))
    except Exception:
        value = 1
    return max(1, min(5, int(value)))


def _compute_composite(scores: dict[str, int]) -> float:
    return sum(float(scores.get(dim, 1)) * weight for dim, weight in SCORE_WEIGHTS.items())


def _resolve_verdict(composite: float, thresholds: dict[str, float]) -> str:
    excellent = _safe_float(thresholds.get("excellent"), DEFAULT_THRESHOLDS["excellent"])
    good = _safe_float(thresholds.get("good"), DEFAULT_THRESHOLDS["good"])
    adequate = _safe_float(thresholds.get("adequate"), DEFAULT_THRESHOLDS["adequate"])
    poor = _safe_float(thresholds.get("poor"), DEFAULT_THRESHOLDS["poor"])
    if composite >= excellent:
        return "excellent"
    if composite >= good:
        return "good"
    if composite >= adequate:
        return "adequate"
    if composite >= poor:
        return "poor"
    return "failing"


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content)


def _extract_first_json_object(raw: str) -> str:
    text = str(raw or "")
    if not text.strip():
        raise ValueError("Judge response is empty.")

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1)

    start = text.find("{")
    if start < 0:
        raise ValueError("Judge response does not contain a JSON object.")
    depth = 0
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Judge response JSON object is incomplete.")


def _normalize_judge_payload(payload: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    raw_scores = payload.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    raw_evidence = payload.get("evidence", {})
    if not isinstance(raw_evidence, dict):
        raw_evidence = {}

    scores: dict[str, int] = {}
    evidence: dict[str, str] = {}
    for dim in SCORE_DIMENSIONS:
        scores[dim] = _parse_score(raw_scores.get(dim, 1))
        ev = str(raw_evidence.get(dim, "")).strip()
        evidence[dim] = ev or "Judge model did not provide evidence."
    return scores, evidence


def _parse_judge_text(raw_text: str) -> tuple[dict[str, int], dict[str, str]]:
    payload_json = _extract_first_json_object(raw_text)
    parsed = json.loads(payload_json)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON root must be an object.")
    return _normalize_judge_payload(parsed)


def _build_empty_row(
    case: dict[str, Any],
    reason: str,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    evidence = {dim: reason for dim in SCORE_DIMENSIONS}
    scores = {dim: 1 for dim in SCORE_DIMENSIONS}
    composite = round(_compute_composite(scores), 4)
    scores_out: dict[str, float | int] = dict(scores)
    scores_out["composite"] = composite
    return {
        "case_id": str(case.get("case_id", "")).strip(),
        "scores": scores_out,
        "evidence": evidence,
        "verdict": _resolve_verdict(composite, thresholds or DEFAULT_THRESHOLDS),
    }


def _evaluate_case_with_retry(
    case: dict[str, Any],
    config: JudgeRuntimeConfig,
    invoker: Callable[[dict[str, Any]], tuple[dict[str, int], dict[str, str]]],
) -> dict[str, Any]:
    answer = str(case.get("answer", "")).strip()
    if not answer:
        return _build_empty_row(
            case,
            "Empty answer; assigned failing scores.",
            thresholds=config.thresholds,
        )
    if answer.startswith("[EVAL_ERROR]"):
        return _build_empty_row(
            case,
            "Upstream eval error answer; assigned failing scores.",
            thresholds=config.thresholds,
        )

    attempts = max(1, int(config.max_retries) + 1)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            scores, evidence = invoker(case)
            composite = round(_compute_composite(scores), 4)
            verdict = _resolve_verdict(composite, config.thresholds)
            scores_out: dict[str, float | int] = dict(scores)
            scores_out["composite"] = composite
            return {
                "case_id": str(case.get("case_id", "")).strip(),
                "scores": scores_out,
                "evidence": evidence,
                "verdict": verdict,
            }
        except Exception as exc:  # noqa: BLE001 - each case is isolated
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, config.retry_backoff_sec) * attempt)

    message = f"Judge invocation failed after retries: {type(last_error).__name__}: {last_error}"
    if not config.skip_failed_cases:
        raise RuntimeError(message) from last_error
    return _build_empty_row(case, message, thresholds=config.thresholds)


def _chunked(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    chunk_size = max(1, int(size))
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def _build_heuristic_invoker(
    config: JudgeRuntimeConfig,
) -> Callable[[dict[str, Any]], tuple[dict[str, int], dict[str, str]]]:
    _ = config
    url_pattern = re.compile(r"https?://", flags=re.IGNORECASE)

    def _score(case: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
        answer = str(case.get("answer", "")).strip()
        question = str(case.get("question", "")).strip()
        constraints = case.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        reference = _coerce_reference_from_constraints(constraints)
        expected_facts = constraints.get("expected_facts", [])
        if not isinstance(expected_facts, list):
            expected_facts = []
        expected_facts = [str(item).strip().lower() for item in expected_facts if str(item).strip()]
        answer_lower = answer.lower()

        fact_hits = 0
        if expected_facts:
            fact_hits = sum(1 for token in expected_facts if token and token in answer_lower)
        fact_ratio = (fact_hits / len(expected_facts)) if expected_facts else 0.0

        has_url = bool(url_pattern.search(answer))
        long_enough = len(answer) >= 120
        sentence_like = answer.count("。") + answer.count(".") + answer.count("!") + answer.count("?")

        accuracy = 3
        if expected_facts:
            if fact_ratio >= 0.9:
                accuracy = 5
            elif fact_ratio >= 0.6:
                accuracy = 4
            elif fact_ratio >= 0.3:
                accuracy = 3
            elif fact_ratio > 0.0:
                accuracy = 2
            else:
                accuracy = 1
        elif reference:
            accuracy = 4 if reference.lower()[:30] in answer_lower else 3

        groundedness = 4 if has_url else 2
        coherence = 4 if sentence_like >= 2 else 2
        completeness = 4 if long_enough and (question.lower()[:15] not in answer_lower or len(answer) >= 200) else 3
        helpfulness = 4 if long_enough else 2

        scores = {
            "accuracy": max(1, min(5, accuracy)),
            "groundedness": max(1, min(5, groundedness)),
            "coherence": max(1, min(5, coherence)),
            "completeness": max(1, min(5, completeness)),
            "helpfulness": max(1, min(5, helpfulness)),
        }
        evidence = {
            "accuracy": (
                f"Heuristic fact coverage ratio={fact_ratio:.2f}, "
                f"matched={fact_hits}/{len(expected_facts)}."
                if expected_facts
                else "No expected_facts provided; using heuristic baseline."
            ),
            "groundedness": (
                "Detected URL/citation pattern in answer."
                if has_url
                else "No explicit URL/citation pattern detected in answer."
            ),
            "coherence": f"Heuristic sentence_like_segments={sentence_like}.",
            "completeness": f"Heuristic answer_length={len(answer)} chars.",
            "helpfulness": "Heuristic score based on response length and structure.",
        }
        return scores, evidence

    return _score


def _build_llm_invoker(
    config: JudgeRuntimeConfig,
) -> Callable[[dict[str, Any]], tuple[dict[str, int], dict[str, str]]]:
    from langchain_core.messages import HumanMessage, SystemMessage

    provider = str(config.provider or "gemini_api").strip().lower()
    if provider in {"gemini_api", "gemini", "google_ai_studio", "developer_api"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = str(os.getenv("GEMINI_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for provider=gemini_api.")
        llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=api_key,
            temperature=float(config.temperature),
        )
    elif provider in {"vertex", "vertex_ai", "gcp"}:
        from langchain_google_vertexai import ChatVertexAI

        project = str(
            os.getenv("VERTEX_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
        ).strip()
        if not project:
            raise ValueError("VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT is required for provider=vertex.")
        location = str(
            os.getenv(
                "VERTEX_GENERATION_LOCATION",
                os.getenv("VERTEX_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global")),
            )
        ).strip()
        llm = ChatVertexAI(
            model=config.model_name,
            project=project,
            location=location,
            temperature=float(config.temperature),
        )
    else:
        raise ValueError(f"Unsupported judge provider: {config.provider}")

    def _invoke(case: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
        system_prompt, user_prompt = build_judge_messages(
            case,
            max_answer_chars=config.max_answer_chars,
            max_context_chars=config.max_context_chars,
            max_constraints_chars=config.max_constraints_chars,
        )
        result = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        text = _coerce_text_content(getattr(result, "content", result))
        return _parse_judge_text(text)

    return _invoke


def _build_runtime_config(args: argparse.Namespace, eval_dir: Path) -> JudgeRuntimeConfig:
    config_path = args.config.resolve() if args.config else (eval_dir / "config" / "judge.yaml").resolve()
    file_config = _load_yaml_config(config_path)

    thresholds = dict(DEFAULT_THRESHOLDS)
    raw_thresholds = file_config.get("thresholds", {})
    if isinstance(raw_thresholds, dict):
        for key in DEFAULT_THRESHOLDS:
            if key in raw_thresholds:
                thresholds[key] = _safe_float(raw_thresholds.get(key), thresholds[key])

    model_name = str(file_config.get("judge_model", "gemini-2.5-pro")).strip() or "gemini-2.5-pro"
    provider = str(file_config.get("judge_provider", "vertex")).strip() or "vertex"
    backend = str(file_config.get("judge_backend", "llm")).strip() or "llm"
    temperature = _safe_float(file_config.get("temperature"), 0.0)
    batch_size = max(1, _safe_int(file_config.get("batch_size"), 4))
    max_retries = max(0, _safe_int(file_config.get("max_retries"), 2))
    retry_backoff_sec = max(0.0, _safe_float(file_config.get("retry_backoff_sec"), 1.5))
    request_timeout_sec = max(10.0, _safe_float(file_config.get("request_timeout_sec"), 120.0))
    skip_failed_cases = _safe_bool(file_config.get("skip_failed_cases"), True)
    max_answer_chars = max(500, _safe_int(file_config.get("max_answer_chars"), 5000))
    max_context_chars = max(1000, _safe_int(file_config.get("max_context_chars"), 4000))
    max_constraints_chars = max(800, _safe_int(file_config.get("max_constraints_chars"), 3000))

    env_model = _resolve_env_value("JUDGE_MODEL", "EVAL_JUDGE_MODEL")
    env_provider = _resolve_env_value("JUDGE_PROVIDER", "EVAL_JUDGE_PROVIDER")
    env_backend = _resolve_env_value("JUDGE_BACKEND", "EVAL_JUDGE_BACKEND")
    env_temperature = _resolve_env_value("JUDGE_TEMPERATURE", "EVAL_JUDGE_TEMPERATURE")
    env_batch = _resolve_env_value("JUDGE_BATCH_SIZE", "EVAL_JUDGE_BATCH_SIZE")
    env_retries = _resolve_env_value("JUDGE_MAX_RETRIES", "EVAL_JUDGE_MAX_RETRIES")
    env_backoff = _resolve_env_value("JUDGE_RETRY_BACKOFF_SEC", "EVAL_JUDGE_RETRY_BACKOFF_SEC")
    env_timeout = _resolve_env_value("JUDGE_TIMEOUT_SEC", "EVAL_JUDGE_TIMEOUT_SEC")
    env_skip_failed = _resolve_env_value("JUDGE_SKIP_FAILED_CASES", "EVAL_JUDGE_SKIP_FAILED_CASES")

    if env_model:
        model_name = env_model
    if env_provider:
        provider = env_provider
    if env_backend:
        backend = env_backend
    if env_temperature is not None:
        temperature = _safe_float(env_temperature, temperature)
    if env_batch is not None:
        batch_size = max(1, _safe_int(env_batch, batch_size))
    if env_retries is not None:
        max_retries = max(0, _safe_int(env_retries, max_retries))
    if env_backoff is not None:
        retry_backoff_sec = max(0.0, _safe_float(env_backoff, retry_backoff_sec))
    if env_timeout is not None:
        request_timeout_sec = max(10.0, _safe_float(env_timeout, request_timeout_sec))
    if env_skip_failed is not None:
        skip_failed_cases = _safe_bool(env_skip_failed, skip_failed_cases)

    env_threshold_keys = {
        "excellent": _resolve_env_value("JUDGE_THRESHOLD_EXCELLENT", "EVAL_JUDGE_THRESHOLD_EXCELLENT"),
        "good": _resolve_env_value("JUDGE_THRESHOLD_GOOD", "EVAL_JUDGE_THRESHOLD_GOOD"),
        "adequate": _resolve_env_value("JUDGE_THRESHOLD_ADEQUATE", "EVAL_JUDGE_THRESHOLD_ADEQUATE"),
        "poor": _resolve_env_value("JUDGE_THRESHOLD_POOR", "EVAL_JUDGE_THRESHOLD_POOR"),
    }
    for key, value in env_threshold_keys.items():
        if value is not None:
            thresholds[key] = _safe_float(value, thresholds[key])

    if args.model:
        model_name = str(args.model).strip()
    if args.provider:
        provider = str(args.provider).strip()
    if args.backend:
        backend = str(args.backend).strip()
    if args.temperature is not None:
        temperature = float(args.temperature)
    if args.batch_size is not None:
        batch_size = max(1, int(args.batch_size))
    if args.max_retries is not None:
        max_retries = max(0, int(args.max_retries))
    if args.retry_backoff_sec is not None:
        retry_backoff_sec = max(0.0, float(args.retry_backoff_sec))
    if args.no_skip_failed_cases:
        skip_failed_cases = False

    return JudgeRuntimeConfig(
        model_name=model_name,
        provider=provider,
        backend=backend,
        temperature=temperature,
        batch_size=batch_size,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
        request_timeout_sec=request_timeout_sec,
        skip_failed_cases=skip_failed_cases,
        max_answer_chars=max_answer_chars,
        max_context_chars=max_context_chars,
        max_constraints_chars=max_constraints_chars,
        thresholds=thresholds,
    )


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "avg_accuracy": 0.0,
            "avg_groundedness": 0.0,
            "avg_coherence": 0.0,
            "avg_completeness": 0.0,
            "avg_helpfulness": 0.0,
            "avg_composite": 0.0,
        }

    def _avg(metric: str) -> float:
        values: list[float] = []
        for row in rows:
            scores = row.get("scores", {})
            if not isinstance(scores, dict):
                continue
            value = scores.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)

    return {
        "avg_accuracy": _avg("accuracy"),
        "avg_groundedness": _avg("groundedness"),
        "avg_coherence": _avg("coherence"),
        "avg_completeness": _avg("completeness"),
        "avg_helpfulness": _avg("helpfulness"),
        "avg_composite": _avg("composite"),
    }


def _build_arg_parser(eval_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge scoring on eval reports.")
    parser.add_argument(
        "--report",
        type=Path,
        default=eval_dir / "reports" / "latest.json",
        help="Source report JSON from eval/run_eval.py (or rows-style JSON).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=eval_dir / "reports" / "judge" / "latest.json",
        help="Output path for judge results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=eval_dir / "config" / "judge.yaml",
        help="Optional judge config file (YAML).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file loaded after agent/.env.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override judge model name.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override judge provider (gemini_api|vertex).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Judge backend (llm|heuristic).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Judge temperature (recommend 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Case batch size.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Retries per case when judge call fails.",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=None,
        help="Linear retry backoff seconds.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, only process first N cases.",
    )
    parser.add_argument(
        "--no-skip-failed-cases",
        action="store_true",
        help="Fail entire run when a case exhausts retries.",
    )
    return parser


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    parser = _build_arg_parser(eval_dir)
    args = parser.parse_args()
    _load_eval_env(args.env_file)

    source_report_path = args.report.resolve()
    if not source_report_path.exists():
        raise FileNotFoundError(f"Source report not found: {source_report_path}")

    config = _build_runtime_config(args, eval_dir)
    source_report = _read_json(source_report_path)

    cases = _build_cases(source_report)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: int(args.max_cases)]
    if not cases:
        raise ValueError("No cases found in source report.")

    backend = str(config.backend or "llm").strip().lower()
    if backend == "heuristic":
        invoker = _build_heuristic_invoker(config)
    elif backend == "llm":
        invoker = _build_llm_invoker(config)
    else:
        raise ValueError(f"涓嶆敮鎸佺殑judge鍚庣: {config.backend}")

    rows: list[dict[str, Any]] = []
    for batch in _chunked(cases, config.batch_size):
        for case in batch:
            rows.append(_evaluate_case_with_retry(case, config, invoker))

    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(source_report_path),
        "judge_model": config.model_name,
        "case_count": len(rows),
        "rows": rows,
        "summary": _build_summary(rows),
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[JudgeEval] cases=%s avg_composite=%.4f output=%s"
        % (
            result["case_count"],
            result["summary"]["avg_composite"],
            output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
