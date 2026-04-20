"""Build task-driven eval dataset (v1).

Flow:
1) Load task types (task-driven).
2) Sample news pools per task type.
3) One LLM call per task type to generate expected question/answer/path.
4) Contract validation + optional audit model.
5) Export JSONL cases + manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from services.db import get_conn, put_conn

try:
    from task_eval_v1_schema import (
        build_news_pool_hash,
        load_task_types,
        normalize_case,
        validate_case,
    )
except ImportError:  # package-style import fallback
    from .task_eval_v1_schema import (
        build_news_pool_hash,
        load_task_types,
        normalize_case,
        validate_case,
    )


DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_PROVIDER = "vertex"
T = TypeVar("T")


@dataclass
class Pool:
    pool_id: str
    docs: list[dict[str, Any]]


def _resolve_preferred_provider() -> str:
    explicit = str(os.getenv("TASK_EVAL_PROVIDER", "")).strip()
    if explicit:
        return explicit
    agent_provider = str(os.getenv("AGENT_MODEL_PROVIDER", "")).strip()
    if agent_provider:
        return agent_provider
    return DEFAULT_PROVIDER


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

    # Prefer Vertex for all LLM invocations in this chain.
    if not str(os.getenv("TASK_EVAL_PROVIDER", "")).strip():
        os.environ["TASK_EVAL_PROVIDER"] = _resolve_preferred_provider()


def _extract_first_json_object(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("LLM output is empty.")
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM output.")
    depth = 0
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Incomplete JSON object in LLM output.")


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    chunks.append(str(txt))
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content)


def _build_chat_model(provider: str, model_name: str, temperature: float) -> Any:
    normalized = str(provider or DEFAULT_PROVIDER).strip().lower()
    if normalized in {"gemini_api", "gemini", "google_ai_studio", "developer_api"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = str(os.getenv("GEMINI_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for provider=gemini_api.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=float(temperature),
        )

    if normalized in {"vertex", "vertex_ai", "gcp"}:
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
            model=model_name,
            project=project,
            location=location,
            temperature=float(temperature),
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _is_retryable_llm_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    retry_markers = (
        "429",
        "rate limit",
        "resource exhausted",
        "quota",
        "too many requests",
        "deadline exceeded",
        "timeout",
        "timed out",
        "service unavailable",
        "503",
    )
    return any(marker in text for marker in retry_markers)


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}")
QUESTION_STOPWORDS = {
    "请", "帮我", "过去", "最近", "新闻", "分析", "比较", "构建", "时间线", "趋势", "关于",
    "the", "and", "for", "with", "over", "last", "days", "news", "compare", "analyze", "build",
}


def _contains_zh(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _canonical_path(path: list[dict[str, Any]]) -> str:
    return json.dumps(path, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_path_list(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list):
        return []
    out: list[list[dict[str, Any]]] = []
    for item in value:
        if not isinstance(item, list) or not item:
            continue
        path: list[dict[str, Any]] = []
        for step in item:
            if isinstance(step, str):
                tool = step.strip()
                if not tool:
                    continue
                path.append({"tool": tool, "args": {}})
                continue
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool", "")).strip()
            if not tool:
                continue
            args = step.get("args", {})
            if not isinstance(args, dict):
                args = {}
            path.append({"tool": tool, "args": args})
        if path:
            out.append(path)
    return out


def _ordered_tool_matches(actual_tools: list[str], expected_tools: list[str]) -> int:
    if not actual_tools or not expected_tools:
        return 0
    cursor = 0
    for tool in actual_tools:
        if cursor < len(expected_tools) and tool == expected_tools[cursor]:
            cursor += 1
            if cursor == len(expected_tools):
                return cursor
    return cursor


def _select_best_acceptable_path(
    raw_paths: list[list[dict[str, Any]]],
    acceptable_paths: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not acceptable_paths:
        return []
    if not raw_paths:
        return acceptable_paths[0]

    raw_tools = [str(step.get("tool", "")).strip() for step in raw_paths[0] if isinstance(step, dict)]
    best_idx = 0
    best_score: tuple[int, int] | None = None
    for idx, path in enumerate(acceptable_paths):
        exp_tools = [str(step.get("tool", "")).strip() for step in path if isinstance(step, dict)]
        score = (_ordered_tool_matches(raw_tools, exp_tools), -len(exp_tools))
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
    return acceptable_paths[best_idx]


def _coerce_expected_paths_to_acceptable(
    raw_paths: Any,
    acceptable_paths: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    acceptable = _coerce_path_list(acceptable_paths)
    if not acceptable:
        return _coerce_path_list(raw_paths)

    raw = _coerce_path_list(raw_paths)
    acceptable_set = {_canonical_path(path) for path in acceptable}
    exact = [path for path in raw if _canonical_path(path) in acceptable_set]
    if exact:
        return exact
    return [_select_best_acceptable_path(raw, acceptable)]


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def _question_grounded(question: str, pool_docs: list[dict[str, Any]]) -> bool:
    q_tokens = [token for token in _tokenize(question) if token not in QUESTION_STOPWORDS]
    if not q_tokens:
        return False

    pool_text = " ".join(
        f"{doc.get('title', '')} {doc.get('summary', '')}"
        for doc in pool_docs
        if isinstance(doc, dict)
    )
    pool_tokens = set(_tokenize(pool_text))
    if not pool_tokens:
        return False

    overlap = [token for token in q_tokens if token in pool_tokens]
    if len(overlap) >= 2:
        return True
    return any(len(token) >= 4 for token in overlap)


def _title_topic(pool_docs: list[dict[str, Any]]) -> str:
    for doc in pool_docs:
        if not isinstance(doc, dict):
            continue
        title = str(doc.get("title", "")).strip()
        if not title:
            continue
        title = re.sub(r"^\[[^\]]+\]\s*", "", title).strip()
        if title:
            return title[:28]
    return "相关新闻"


def _first_arg(expected_paths: list[list[dict[str, Any]]], key: str) -> str:
    for path in expected_paths:
        if not isinstance(path, list):
            continue
        for step in path:
            if not isinstance(step, dict):
                continue
            args = step.get("args", {})
            if not isinstance(args, dict):
                continue
            value = args.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return ""


def _build_grounded_question(
    task: dict[str, Any],
    pool_docs: list[dict[str, Any]],
    expected_paths: list[list[dict[str, Any]]],
) -> str:
    skill = str(task.get("skill", "")).strip()
    days = _safe_int(
        _first_arg(expected_paths, "days") or task.get("sampling", {}).get("days", 30),
        30,
    )
    limit = _safe_int(_first_arg(expected_paths, "limit") or 10, 10)

    topic = (
        _first_arg(expected_paths, "query")
        or _first_arg(expected_paths, "topic")
        or _title_topic(pool_docs)
    )
    topic_a = _first_arg(expected_paths, "topic_a")
    topic_b = _first_arg(expected_paths, "topic_b")

    if skill == "compare_sources":
        return f"请比较过去{days}天「{topic}」在 HackerNews 与 TechCrunch 的覆盖和情绪差异。"
    if skill == "compare_topics":
        a = topic_a or topic or "主题A"
        b = topic_b or "主题B"
        return f"请比较过去{days}天「{a}」与「{b}」的热度、情绪与来源结构。"
    if skill == "build_timeline":
        return f"请构建过去{days}天「{topic}」的关键事件时间线，最多{limit}条。"
    if skill == "trend_analysis":
        return f"请分析「{topic}」最近{days}天相对前{days}天的趋势变化。"
    if skill == "analyze_landscape":
        return f"请分析过去{days}天「{topic}」相关赛道的竞争格局。"
    if skill == "fulltext_batch":
        return f"请围绕「{topic}」筛选最近{days}天相关新闻并批量读取全文，提炼关键结论。"
    if skill == "search_news":
        return f"请检索最近{days}天与「{topic}」相关的新闻，并返回最相关结果。"
    if skill == "query_news":
        return f"请查询最近{days}天与「{topic}」相关的新闻，并按相关性返回结果。"
    if skill == "read_news_content":
        url = _first_arg(expected_paths, "url")
        if url:
            return f"请读取该新闻链接全文并提炼要点：{url}"
        return "请读取指定新闻全文并提炼关键要点。"
    if skill == "list_topics":
        return "请给出最近一段时间的主题分布与数量统计。"
    if skill == "get_db_stats":
        return "请返回当前新闻库总量与最新数据时间。"
    return f"请基于最近{days}天新闻，围绕「{topic}」完成分析。"


def _repair_generated_case(
    raw_case: dict[str, Any],
    task: dict[str, Any],
    pool_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(raw_case)
    acceptable_paths = _coerce_path_list(task.get("acceptable_tool_paths", []))
    out["expected_tool_paths"] = _coerce_expected_paths_to_acceptable(
        out.get("expected_tool_paths"),
        acceptable_paths,
    )

    scenario = str(task.get("scenario", "")).strip().lower()
    question = str(out.get("expected_question", "")).strip()
    requires_grounding = scenario != "empty"
    if (not _contains_zh(question)) or (requires_grounding and not _question_grounded(question, pool_docs)):
        out["expected_question"] = _build_grounded_question(task, pool_docs, out["expected_tool_paths"])

    answer = str(out.get("expected_answer", "")).strip()
    if not _contains_zh(answer):
        out["expected_answer"] = f"参考答案（中文）：{answer}"
    return out


def _doc_language(title: str, title_cn: str) -> str:
    if title_cn.strip():
        return "zh"
    if any("\u4e00" <= ch <= "\u9fff" for ch in title):
        return "zh"
    return "en"


def _doc_id_from_url(url: str) -> str:
    digest = hashlib.sha1(str(url).strip().encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _sample_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = task.get("sampling", {})
    days = int(sampling.get("days", 30))
    candidate_limit = int(sampling.get("candidate_limit", 300))
    keywords = [str(item).strip() for item in sampling.get("keywords", []) if str(item).strip()]
    sources = [str(item).strip() for item in sampling.get("sources", []) if str(item).strip()]
    languages = {str(item).strip().lower() for item in sampling.get("languages", []) if str(item).strip()}

    where_parts = ["created_at >= NOW() - %s::interval"]
    params: list[Any] = [f"{days} days"]

    if keywords:
        keyword_clauses: list[str] = []
        for token in keywords:
            keyword_clauses.append(
                "(title ILIKE %s OR COALESCE(title_cn,'') ILIKE %s OR COALESCE(summary,'') ILIKE %s)"
            )
            token_like = f"%{token}%"
            params.extend([token_like, token_like, token_like])
        where_parts.append("(" + " OR ".join(keyword_clauses) + ")")

    if sources:
        where_parts.append("COALESCE(source_type,'') = ANY(%s)")
        params.append(sources)

    sql = f"""
        SELECT
            COALESCE(title_cn, title) AS title_norm,
            title,
            COALESCE(summary, '') AS summary,
            url,
            source_type,
            created_at,
            COALESCE(sentiment, '') AS sentiment,
            COALESCE(points, 0) AS points,
            COALESCE(title_cn, '') AS title_cn
        FROM view_dashboard_news
        WHERE {' AND '.join(where_parts)}
        ORDER BY created_at DESC, points DESC NULLS LAST
        LIMIT %s
    """
    params.append(candidate_limit)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()
    finally:
        put_conn(conn)

    docs: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for row in rows:
        title_norm, title, summary, url, source_type, created_at, sentiment, points, title_cn = row
        url_text = str(url or "").strip()
        if not url_text or url_text in seen_urls:
            continue
        seen_urls.add(url_text)
        language = _doc_language(str(title or ""), str(title_cn or ""))
        if languages and language not in languages:
            continue
        docs.append(
            {
                "doc_id": _doc_id_from_url(url_text),
                "url": url_text,
                "title": str(title_norm or "").strip() or str(title or "").strip() or "(untitled)",
                "summary": str(summary or "").strip(),
                "published_at": created_at.isoformat() if created_at else "",
                "source": str(source_type or "").strip() or "unknown",
                "sentiment": str(sentiment or "").strip(),
                "points": int(points or 0),
                "language": language,
            }
        )
    return docs


def _round_robin_by_source(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        source = str(item.get("source", "unknown")).strip() or "unknown"
        groups.setdefault(source, []).append(item)
    ordered_sources = sorted(groups.keys())
    cursors = {source: 0 for source in ordered_sources}
    output: list[dict[str, Any]] = []
    while True:
        progressed = False
        for source in ordered_sources:
            idx = cursors[source]
            bucket = groups[source]
            if idx >= len(bucket):
                continue
            output.append(bucket[idx])
            cursors[source] = idx + 1
            progressed = True
        if not progressed:
            break
    return output


def _build_pools(task: dict[str, Any], candidates: list[dict[str, Any]], pools_per_task: int, rng: random.Random) -> list[Pool]:
    pool_size = int(task.get("sampling", {}).get("pool_size", 12))
    ordered = _round_robin_by_source(candidates)
    if ordered:
        # Keep deterministic but avoid repeated same-front slices.
        shift = rng.randint(0, max(0, len(ordered) - 1))
        ordered = ordered[shift:] + ordered[:shift]

    pools: list[Pool] = []
    for idx in range(1, pools_per_task + 1):
        pool_id = f"{task['task_id']}.pool.{idx:03d}"
        if not ordered:
            pools.append(Pool(pool_id=pool_id, docs=[]))
            continue
        start = ((idx - 1) * pool_size) % len(ordered)
        docs: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()
        for offset in range(len(ordered)):
            item = ordered[(start + offset) % len(ordered)]
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            docs.append(item)
            if len(docs) >= pool_size:
                break
        pools.append(Pool(pool_id=pool_id, docs=docs))
    return pools


def _chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    size = int(chunk_size)
    if size <= 0 or len(items) <= size:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _pools_for_prompt(pools: list[Pool], *, summary_chars: int = 220) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pool in pools:
        docs = []
        for item in pool.docs:
            docs.append(
                {
                    "doc_id": item["doc_id"],
                    "url": item["url"],
                    "title": item["title"],
                    "summary": str(item.get("summary", ""))[:summary_chars],
                    "published_at": item["published_at"],
                    "source": item["source"],
                    "sentiment": item.get("sentiment", ""),
                    "points": item.get("points", 0),
                    "language": item.get("language", ""),
                }
            )
        out.append(
            {
                "pool_id": pool.pool_id,
                "input_news_pool_hash": build_news_pool_hash(pool.docs),
                "docs": docs,
            }
        )
    return out


def _generator_prompts(task: dict[str, Any], pools: list[Pool]) -> tuple[str, str]:
    schema_hint = {
        "task_id": task["task_id"],
        "cases": [
            {
                "pool_id": "string",
                "expected_question": "string",
                "expected_answer": "string",
                "expected_tool_paths": [
                    [{"tool": task["skill"], "args": task["parameter_template"]}]
                ],
                "required_tools": task["required_tools"],
                "forbidden_tools": task["forbidden_tools"],
                "retrieval_gold_doc_ids": ["doc_id from same pool"],
                "retrieval_gold_urls": ["url from same pool"],
                "verifiable_claims": [
                    {
                        "claim": "verifiable claim from expected_answer",
                        "evidence_doc_ids": ["doc_id from same pool"],
                        "claim_type": "fact|number|comparison",
                    }
                ],
                "should_clarify": task["should_clarify"],
                "retrieval_evaluable": task["retrieval_mode"] == "evaluable",
                "difficulty": task["difficulty"],
                "tags": task["tags"],
            }
        ],
    }

    system_prompt = (
        "You generate task-driven evaluation cases.\n"
        "Rules:\n"
        "1) Return strict JSON object only.\n"
        "2) One case per pool_id, no missing and no extra pools.\n"
        "3) Use only tools from acceptable_tool_paths.\n"
        "4) retrieval_gold_doc_ids/retrieval_gold_urls must come from the SAME input_news_pool.\n"
        "5) If retrieval_evaluable=true, retrieval_gold_doc_ids must be non-empty.\n"
        "6) expected_answer must be grounded in pool docs.\n"
        "7) verifiable_claims must be checkable and linked to evidence_doc_ids in pool.\n"
        "8) expected_question and expected_answer must be written in Chinese (entity/product names may stay in English).\n"
        "9) expected_tool_paths must be an exact subset of acceptable_tool_paths; do not alter tool args.\n"
        "10) For non-empty scenarios, expected_question must be answerable by the pool and mention pool entities/topics.\n"
        "11) Do not output markdown fences."
    )

    payload = {
        "task_definition": {
            "task_id": task["task_id"],
            "skill": task["skill"],
            "intent_label": task["intent_label"],
            "retrieval_mode": task["retrieval_mode"],
            "scenario": task["scenario"],
            "example_question": task["example_question"],
            "parameter_template": task["parameter_template"],
            "acceptable_tool_paths": task["acceptable_tool_paths"],
            "required_tools": task["required_tools"],
            "forbidden_tools": task["forbidden_tools"],
            "should_clarify": task["should_clarify"],
            "difficulty": task["difficulty"],
            "tags": task["tags"],
        },
        "news_pools": _pools_for_prompt(pools),
        "output_schema_example": schema_hint,
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def _audit_prompts(task: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[str, str]:
    system_prompt = (
        "You are an evaluation-case auditor.\n"
        "Check each case for contract consistency and evidence grounding.\n"
        "Reject any case whose expected_question or expected_answer is not Chinese.\n"
        "Reject any case whose expected_tool_paths is not an exact subset of task.acceptable_tool_paths.\n"
        "For non-empty scenarios, reject if expected_question is not grounded in input_news_pool topics/entities.\n"
        "If rejected for language, set reason to non_chinese_expected_text.\n"
        "If rejected for path drift, set reason to path_not_in_acceptable.\n"
        "If rejected for question grounding, set reason to question_not_grounded.\n"
        "Return strict JSON object only."
    )
    payload = {
        "task": {
            "task_id": task["task_id"],
            "skill": task["skill"],
            "retrieval_mode": task["retrieval_mode"],
            "scenario": task["scenario"],
            "acceptable_tool_paths": task["acceptable_tool_paths"],
            "required_tools": task["required_tools"],
            "forbidden_tools": task["forbidden_tools"],
        },
        "cases": cases,
        "output_schema": {
            "task_id": task["task_id"],
            "verdicts": [
                {
                    "case_id": "string",
                    "accepted": True,
                    "reason": "string",
                }
            ],
        },
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def _invoke_json(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    max_retries: int,
    backoff_sec: float,
) -> dict[str, Any]:
    attempts = max(1, int(max_retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            text = _coerce_text_content(getattr(result, "content", result))
            payload = json.loads(_extract_first_json_object(text))
            if not isinstance(payload, dict):
                raise ValueError("LLM output JSON root must be object.")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts or not _is_retryable_llm_error(exc):
                raise
            sleep_sec = max(0.0, float(backoff_sec)) * (2 ** (attempt - 1))
            print(
                "[TaskDatasetV1][Retry] attempt=%s/%s backoff=%.2fs error=%s"
                % (attempt, attempts, sleep_sec, exc)
            )
            time.sleep(sleep_sec)

    raise RuntimeError(f"LLM invocation failed after retries: {last_exc}")


def _generate_for_task(
    llm: Any,
    task: dict[str, Any],
    pools: list[Pool],
    *,
    llm_max_retries: int,
    llm_backoff_sec: float,
    pools_per_generation_call: int,
    inter_llm_call_sleep_sec: float,
) -> list[dict[str, Any]]:
    pool_map = {pool.pool_id: pool.docs for pool in pools}
    out_cases: list[dict[str, Any]] = []
    pool_chunks = _chunk_list(pools, int(pools_per_generation_call))

    for chunk_idx, pools_chunk in enumerate(pool_chunks, 1):
        system_prompt, user_prompt = _generator_prompts(task, pools_chunk)
        payload = _invoke_json(
            llm,
            system_prompt,
            user_prompt,
            max_retries=llm_max_retries,
            backoff_sec=llm_backoff_sec,
        )

        rows = payload.get("cases", [])
        if not isinstance(rows, list):
            raise ValueError(f"{task['task_id']}: generator output missing cases list.")

        by_pool: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            pool_id = str(row.get("pool_id", "")).strip()
            if not pool_id:
                continue
            by_pool[pool_id] = row

        missing = [pool.pool_id for pool in pools_chunk if pool.pool_id not in by_pool]
        if missing:
            raise ValueError(f"{task['task_id']}: missing generated cases for pools={missing}")

        for pool in pools_chunk:
            suffix_match = re.search(r"\.pool\.(\d+)$", pool.pool_id)
            if suffix_match:
                suffix = f"{int(suffix_match.group(1)):03d}"
            else:
                suffix = hashlib.sha1(pool.pool_id.encode("utf-8")).hexdigest()[:8]
            case_id = f"{task['task_id']}.{suffix}"
            raw_case = _repair_generated_case(
                by_pool[pool.pool_id],
                task,
                pool_map[pool.pool_id],
            )
            normalized = normalize_case(
                raw_case,
                task_type=task,
                case_id=case_id,
                pool_id=pool.pool_id,
                input_news_pool=pool_map[pool.pool_id],
            )
            out_cases.append(normalized)

        if chunk_idx < len(pool_chunks) and float(inter_llm_call_sleep_sec) > 0:
            sleep_sec = float(inter_llm_call_sleep_sec)
            print(
                "[TaskDatasetV1][Throttle] task=%s phase=generation chunk=%s/%s sleep=%.2fs"
                % (task["task_id"], chunk_idx, len(pool_chunks), sleep_sec)
            )
            time.sleep(sleep_sec)
    return out_cases


def _audit_cases(
    llm: Any,
    task: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    llm_max_retries: int,
    llm_backoff_sec: float,
    cases_per_audit_call: int,
    inter_llm_call_sleep_sec: float,
) -> dict[str, str]:
    if not cases:
        return {}

    rejected: dict[str, str] = {}
    case_chunks = _chunk_list(cases, int(cases_per_audit_call))
    for chunk_idx, cases_chunk in enumerate(case_chunks, 1):
        system_prompt, user_prompt = _audit_prompts(task, cases_chunk)
        payload = _invoke_json(
            llm,
            system_prompt,
            user_prompt,
            max_retries=llm_max_retries,
            backoff_sec=llm_backoff_sec,
        )
        verdicts = payload.get("verdicts", [])
        if not isinstance(verdicts, list):
            continue
        for row in verdicts:
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case_id", "")).strip()
            accepted = bool(row.get("accepted", False))
            reason = str(row.get("reason", "")).strip() or "audit_rejected"
            if case_id and not accepted:
                rejected[case_id] = reason

        if chunk_idx < len(case_chunks) and float(inter_llm_call_sleep_sec) > 0:
            sleep_sec = float(inter_llm_call_sleep_sec)
            print(
                "[TaskDatasetV1][Throttle] task=%s phase=audit chunk=%s/%s sleep=%.2fs"
                % (task["task_id"], chunk_idx, len(case_chunks), sleep_sec)
            )
            time.sleep(sleep_sec)
    return rejected


def _parse_args() -> argparse.Namespace:
    eval_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build task-driven eval dataset v1.")
    parser.add_argument(
        "--task-types",
        type=Path,
        default=eval_dir / "config" / "task_types_v1.json",
        help="Task type config JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=eval_dir / "datasets" / "task_eval_v1_cases.jsonl",
        help="Output dataset JSONL path.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=eval_dir / "datasets" / "task_eval_v1_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--strict-skill-check",
        action="store_true",
        default=True,
        help="Validate task/case tools against live skill catalog.",
    )
    parser.add_argument(
        "--no-strict-skill-check",
        dest="strict_skill_check",
        action="store_false",
        help="Skip strict skill catalog validation.",
    )
    parser.add_argument(
        "--enforce-coverage-policy",
        action="store_true",
        default=True,
        help="Enforce scenario coverage policy for task type files.",
    )
    parser.add_argument(
        "--no-enforce-coverage-policy",
        dest="enforce_coverage_policy",
        action="store_false",
        help="Allow partial task subsets that do not satisfy full scenario coverage policy.",
    )
    parser.add_argument(
        "--pools-per-task",
        type=int,
        default=0,
        help="Override n_min per task. If 0, use task sampling.n_min.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic pool construction.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=_resolve_preferred_provider(),
        help="LLM provider: gemini_api|vertex.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("TASK_EVAL_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_MODEL)),
        help="LLM model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TASK_EVAL_TEMPERATURE", "0")),
        help="LLM temperature for generation.",
    )
    parser.add_argument(
        "--disable-audit",
        action="store_true",
        help="Disable the second-pass audit model validation.",
    )
    parser.add_argument(
        "--audit-max-regen-rounds",
        type=int,
        default=int(os.getenv("TASK_EVAL_AUDIT_MAX_REGEN_ROUNDS", "3")),
        help=(
            "Max regeneration rounds after audit rejection (0 means no regen, only one audit pass)."
        ),
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=int(os.getenv("TASK_EVAL_LLM_MAX_RETRIES", "2")),
        help="Retry times for retryable LLM failures (429/quota/timeout).",
    )
    parser.add_argument(
        "--llm-backoff-sec",
        type=float,
        default=float(os.getenv("TASK_EVAL_LLM_BACKOFF_SEC", "2.0")),
        help="Base backoff seconds for LLM retries (exponential).",
    )
    parser.add_argument(
        "--pools-per-generation-call",
        type=int,
        default=int(os.getenv("TASK_EVAL_POOLS_PER_GENERATION_CALL", "0")),
        help="If > 0, split one task's pools into multiple generation calls (0 means all pools in one call).",
    )
    parser.add_argument(
        "--cases-per-audit-call",
        type=int,
        default=int(os.getenv("TASK_EVAL_CASES_PER_AUDIT_CALL", "0")),
        help="If > 0, split one task's cases into multiple audit calls (0 means all cases in one call).",
    )
    parser.add_argument(
        "--inter-llm-call-sleep-sec",
        type=float,
        default=float(os.getenv("TASK_EVAL_INTER_LLM_CALL_SLEEP_SEC", "0.0")),
        help="Sleep seconds between chunked generation/audit calls for throttling.",
    )
    parser.add_argument(
        "--inter-task-sleep-sec",
        type=float,
        default=float(os.getenv("TASK_EVAL_INTER_TASK_SLEEP_SEC", "0.0")),
        help="Sleep seconds between task types to reduce sustained quota pressure.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint JSON path for restore-point resume.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        default=True,
        help="Resume from checkpoint if exists.",
    )
    parser.add_argument(
        "--no-resume-from-checkpoint",
        dest="resume_from_checkpoint",
        action="store_false",
        help="Ignore existing checkpoint and rebuild from scratch.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file loaded after agent/.env.",
    )
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        return None
    return payload


def main() -> int:
    args = _parse_args()
    _load_eval_env(args.env_file)

    task_types = load_task_types(
        args.task_types.resolve(),
        strict_skill=bool(args.strict_skill_check),
        enforce_coverage_policy=bool(args.enforce_coverage_policy),
    )
    llm = _build_chat_model(
        provider=str(args.provider),
        model_name=str(args.model),
        temperature=float(args.temperature),
    )

    rng = random.Random(int(args.seed))
    output_path = args.output.resolve()
    checkpoint_path = (
        args.checkpoint_path.resolve()
        if args.checkpoint_path
        else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    )

    all_cases: list[dict[str, Any]] = []
    manifest_tasks: list[dict[str, Any]] = []
    completed_task_ids: set[str] = set()

    if bool(args.resume_from_checkpoint):
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint:
            cp_task_file = str(checkpoint.get("task_type_file", "")).strip()
            expected_task_file = str(args.task_types.resolve())
            if cp_task_file == expected_task_file:
                cp_cases = checkpoint.get("cases", [])
                cp_tasks = checkpoint.get("tasks", [])
                cp_completed = checkpoint.get("completed_task_ids", [])
                if isinstance(cp_cases, list):
                    all_cases = [row for row in cp_cases if isinstance(row, dict)]
                if isinstance(cp_tasks, list):
                    manifest_tasks = [row for row in cp_tasks if isinstance(row, dict)]
                if isinstance(cp_completed, list):
                    completed_task_ids = {
                        str(item).strip() for item in cp_completed if str(item).strip()
                    }
                print(
                    "[TaskDatasetV1][Resume] checkpoint=%s completed_tasks=%s cases=%s"
                    % (checkpoint_path, len(completed_task_ids), len(all_cases))
                )
            else:
                print(
                    "[TaskDatasetV1][Resume] skip checkpoint due to task file mismatch: cp=%s current=%s"
                    % (cp_task_file, expected_task_file)
                )

    total_tasks = len(task_types)
    for task_idx, task in enumerate(task_types, 1):
        task_id = str(task.get("task_id", "")).strip()
        if task_id in completed_task_ids:
            print("[TaskDatasetV1][Skip] task already completed in checkpoint: %s" % task_id)
            continue

        n_min = int(task.get("sampling", {}).get("n_min", 30))
        pools_per_task = int(args.pools_per_task) if int(args.pools_per_task) > 0 else n_min

        candidates = _sample_candidates(task)
        pools = _build_pools(task, candidates, pools_per_task=pools_per_task, rng=rng)
        generated_cases = _generate_for_task(
            llm,
            task,
            pools,
            llm_max_retries=int(args.llm_max_retries),
            llm_backoff_sec=float(args.llm_backoff_sec),
            pools_per_generation_call=int(args.pools_per_generation_call),
            inter_llm_call_sleep_sec=float(args.inter_llm_call_sleep_sec),
        )

        rejected: dict[str, str] = {}
        if not args.disable_audit:
            max_regen_rounds = max(0, int(args.audit_max_regen_rounds))
            for regen_round in range(0, max_regen_rounds + 1):
                rejected = _audit_cases(
                    llm,
                    task,
                    generated_cases,
                    llm_max_retries=int(args.llm_max_retries),
                    llm_backoff_sec=float(args.llm_backoff_sec),
                    cases_per_audit_call=int(args.cases_per_audit_call),
                    inter_llm_call_sleep_sec=float(args.inter_llm_call_sleep_sec),
                )
                if not rejected:
                    break
                if regen_round >= max_regen_rounds:
                    break

                rejected_case_ids = set(rejected.keys())
                regen_pools = [
                    pool
                    for pool in pools
                    if any(
                        case["pool_id"] == pool.pool_id and case["case_id"] in rejected_case_ids
                        for case in generated_cases
                    )
                ]
                if not regen_pools:
                    break

                print(
                    "[TaskDatasetV1][AuditRegen] task=%s round=%s/%s rejected=%s regen_pools=%s"
                    % (
                        task["task_id"],
                        regen_round + 1,
                        max_regen_rounds,
                        len(rejected),
                        len(regen_pools),
                    )
                )
                regenerated = _generate_for_task(
                    llm,
                    task,
                    regen_pools,
                    llm_max_retries=int(args.llm_max_retries),
                    llm_backoff_sec=float(args.llm_backoff_sec),
                    pools_per_generation_call=int(args.pools_per_generation_call),
                    inter_llm_call_sleep_sec=float(args.inter_llm_call_sleep_sec),
                )
                regen_by_pool = {row["pool_id"]: row for row in regenerated}
                for idx, row in enumerate(generated_cases):
                    pool_id = row["pool_id"]
                    if pool_id in regen_by_pool:
                        generated_cases[idx] = regen_by_pool[pool_id]

            if rejected:
                print(f"[TaskDatasetV1][Warning] {task['task_id']}: reached max retries. Dropping rejected cases={rejected}")
                generated_cases =[case for case in generated_cases if case.get("case_id") not in rejected]

        for case in generated_cases:
            validate_case(case, strict_skill=bool(args.strict_skill_check))
        all_cases.extend(generated_cases)

        manifest_tasks.append(
            {
                "task_id": task["task_id"],
                "skill": task["skill"],
                "retrieval_mode": task["retrieval_mode"],
                "scenario": task["scenario"],
                "candidate_docs": len(candidates),
                "pool_count": len(pools),
                "generated_cases": len(generated_cases),
            }
        )
        completed_task_ids.add(task["task_id"])
        checkpoint_payload = {
            "status": "in_progress",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_type_file": str(args.task_types.resolve()),
            "dataset_path": str(output_path),
            "coverage_policy_enforced": bool(args.enforce_coverage_policy),
            "provider": str(args.provider),
            "model": str(args.model),
            "temperature": float(args.temperature),
            "llm_max_retries": int(args.llm_max_retries),
            "llm_backoff_sec": float(args.llm_backoff_sec),
            "audit_max_regen_rounds": int(args.audit_max_regen_rounds),
            "pools_per_generation_call": int(args.pools_per_generation_call),
            "cases_per_audit_call": int(args.cases_per_audit_call),
            "inter_llm_call_sleep_sec": float(args.inter_llm_call_sleep_sec),
            "inter_task_sleep_sec": float(args.inter_task_sleep_sec),
            "completed_task_ids": sorted(completed_task_ids),
            "tasks": manifest_tasks,
            "cases": all_cases,
        }
        _write_json_atomic(checkpoint_path, checkpoint_payload)
        print(
            "[TaskDatasetV1] task=%s pools=%s candidates=%s generated=%s"
            % (task["task_id"], len(pools), len(candidates), len(generated_cases))
        )
        if float(args.inter_task_sleep_sec) > 0 and task_idx < total_tasks:
            sleep_sec = float(args.inter_task_sleep_sec)
            print(
                "[TaskDatasetV1][Throttle] after task=%s sleep=%.2fs"
                % (task["task_id"], sleep_sec)
            )
            time.sleep(sleep_sec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_type_file": str(args.task_types.resolve()),
        "dataset_path": str(output_path),
        "coverage_policy_enforced": bool(args.enforce_coverage_policy),
        "case_count": len(all_cases),
        "provider": str(args.provider),
        "model": str(args.model),
        "temperature": float(args.temperature),
        "audit_enabled": not bool(args.disable_audit),
        "llm_max_retries": int(args.llm_max_retries),
        "llm_backoff_sec": float(args.llm_backoff_sec),
        "audit_max_regen_rounds": int(args.audit_max_regen_rounds),
        "pools_per_generation_call": int(args.pools_per_generation_call),
        "cases_per_audit_call": int(args.cases_per_audit_call),
        "inter_llm_call_sleep_sec": float(args.inter_llm_call_sleep_sec),
        "inter_task_sleep_sec": float(args.inter_task_sleep_sec),
        "tasks": manifest_tasks,
    }
    manifest_path = args.manifest_output.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_json_atomic(
        checkpoint_path,
        {
            "status": "completed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_type_file": str(args.task_types.resolve()),
            "dataset_path": str(output_path),
            "coverage_policy_enforced": bool(args.enforce_coverage_policy),
            "provider": str(args.provider),
            "model": str(args.model),
            "temperature": float(args.temperature),
            "llm_max_retries": int(args.llm_max_retries),
            "llm_backoff_sec": float(args.llm_backoff_sec),
            "audit_max_regen_rounds": int(args.audit_max_regen_rounds),
            "pools_per_generation_call": int(args.pools_per_generation_call),
            "cases_per_audit_call": int(args.cases_per_audit_call),
            "inter_llm_call_sleep_sec": float(args.inter_llm_call_sleep_sec),
            "inter_task_sleep_sec": float(args.inter_task_sleep_sec),
            "completed_task_ids": sorted(completed_task_ids),
            "tasks": manifest_tasks,
            "cases": all_cases,
            "manifest_path": str(manifest_path),
        },
    )

    print("[TaskDatasetV1] output=%s cases=%s" % (output_path, len(all_cases)))
    print("[TaskDatasetV1] manifest=%s" % manifest_path)
    print("[TaskDatasetV1] checkpoint=%s" % checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
