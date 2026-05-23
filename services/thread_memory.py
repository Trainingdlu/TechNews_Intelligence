"""Thread-level memory storage for conversation context assembly."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from psycopg2.extras import Json

from agent.context_manager import extract_history_evidence, message_text
from services.db import db_cursor, db_transaction

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="thread-memory")

MAX_ROLLING_SUMMARY_TURNS = 4


def load_thread_memory_summary(thread_id: str | None) -> dict[str, Any]:
    key = str(thread_id or "").strip()
    if not key:
        return {}
    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT summary_text, summary_payload, last_summarized_message_id,
                   updated_at
            FROM public.thread_memory_summaries
            WHERE thread_id = %s
            LIMIT 1
            """,
            (key,),
        )
        row = cur.fetchone()
    if row is None:
        return {}
    return {
        "summary_text": str(row[0] or ""),
        "summary_payload": row[1] if isinstance(row[1], dict) else {},
        "last_summarized_message_id": row[2],
        "updated_at": row[3].isoformat() if hasattr(row[3], "isoformat") else row[3],
        "evidence_index": load_thread_evidence_index(key, limit=40),
    }


def load_thread_evidence_index(thread_id: str | None, *, limit: int = 80) -> list[dict[str, Any]]:
    key = str(thread_id or "").strip()
    if not key:
        return []
    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT message_id, turn_request_id, evidence_url, title,
                   source_index, excerpt, created_at
            FROM public.thread_evidence_index
            WHERE thread_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (key, max(1, min(int(limit), 200))),
        )
        rows = cur.fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "message_id": row[0],
                "request_id": row[1],
                "url": str(row[2] or ""),
                "title": str(row[3] or ""),
                "index": row[4],
                "excerpt": str(row[5] or ""),
                "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else row[6],
            }
        )
    return items


def schedule_thread_memory_update(
    *,
    thread_id: str,
    user_message: str,
    model_message: dict[str, Any],
    model_message_id: int | None,
    request_id: str,
) -> None:
    """Update thread memory outside the request critical path."""
    if not str(thread_id or "").strip():
        return
    _EXECUTOR.submit(
        _safe_update,
        thread_id=str(thread_id),
        user_message=str(user_message or ""),
        model_message=dict(model_message or {}),
        model_message_id=model_message_id,
        request_id=str(request_id or ""),
    )


def _safe_update(
    *,
    thread_id: str,
    user_message: str,
    model_message: dict[str, Any],
    model_message_id: int | None,
    request_id: str,
) -> None:
    try:
        update_thread_memory_from_turn(
            thread_id=thread_id,
            user_message=user_message,
            model_message=model_message,
            model_message_id=model_message_id,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("thread memory update failed thread_id=%s error=%s", thread_id, exc)


def update_thread_memory_from_turn(
    *,
    thread_id: str,
    user_message: str,
    model_message: dict[str, Any],
    model_message_id: int | None,
    request_id: str,
) -> None:
    evidence = extract_history_evidence(model_message, message_text(model_message))
    answer_text = message_text(model_message)
    previous_summary = load_thread_memory_summary(thread_id)
    summary_payload = _build_summary_payload(
        previous_summary=previous_summary,
        user_message=user_message,
        answer_text=answer_text,
        evidence=evidence,
    )
    summary_text = str(summary_payload.get("summary_text") or "").strip() or _build_summary_text(summary_payload)
    summary_payload = {
        **summary_payload,
        "recent_user_message": user_message[:1000],
        "recent_answer_excerpt": answer_text[:1400],
        "recent_evidence_urls": [item.url for item in evidence[:12]],
        "recent_evidence_titles": [item.title for item in evidence[:12] if item.title],
    }

    with db_transaction() as (_, cur):
        cur.execute(
            """
            INSERT INTO public.thread_memory_summaries (
                thread_id, summary_text, summary_payload,
                last_summarized_message_id, updated_at
            )
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (thread_id) DO UPDATE
            SET summary_text = EXCLUDED.summary_text,
                summary_payload = EXCLUDED.summary_payload,
                last_summarized_message_id = COALESCE(
                    EXCLUDED.last_summarized_message_id,
                    public.thread_memory_summaries.last_summarized_message_id
                ),
                updated_at = NOW()
            """,
            (
                thread_id,
                summary_text,
                Json(summary_payload),
                model_message_id,
            ),
        )
        for item in evidence:
            cur.execute(
                """
                INSERT INTO public.thread_evidence_index (
                    thread_id, message_id, turn_request_id, evidence_url,
                    title, source_index, excerpt
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, evidence_url) DO UPDATE
                SET message_id = COALESCE(EXCLUDED.message_id, public.thread_evidence_index.message_id),
                    turn_request_id = EXCLUDED.turn_request_id,
                    title = COALESCE(NULLIF(EXCLUDED.title, ''), public.thread_evidence_index.title),
                    source_index = COALESCE(EXCLUDED.source_index, public.thread_evidence_index.source_index),
                    excerpt = COALESCE(NULLIF(EXCLUDED.excerpt, ''), public.thread_evidence_index.excerpt),
                    created_at = NOW()
                """,
                (
                    thread_id,
                    model_message_id,
                    request_id,
                    item.url,
                    item.title,
                    item.index,
                    _evidence_excerpt(answer_text, item.url),
                ),
            )


def _build_summary_text(payload: dict[str, Any]) -> str:
    existing = str(payload.get("summary_text") or "").strip()
    if existing:
        return existing[:3600]
    question = str(payload.get("recent_user_message") or "").strip()
    answer = str(payload.get("recent_answer_excerpt") or "").strip()
    urls = [str(item).strip() for item in payload.get("recent_evidence_urls", []) if str(item).strip()]
    parts = []
    if question:
        parts.append(f"Recent user question: {question}")
    if answer:
        parts.append(f"Recent assistant answer excerpt: {answer}")
    if urls:
        parts.append("Recent evidence URLs:\n" + "\n".join(f"- {url}" for url in urls[:8]))
    return "\n\n".join(parts)[:3600]


def _build_summary_payload(
    *,
    previous_summary: dict[str, Any],
    user_message: str,
    answer_text: str,
    evidence: list[Any],
) -> dict[str, Any]:
    fallback = _deterministic_summary_payload(
        previous_summary=previous_summary,
        user_message=user_message,
        answer_text=answer_text,
        evidence=evidence,
    )
    if not _memory_llm_enabled():
        return fallback
    model_payload = _generate_llm_summary_payload(
        previous_summary=previous_summary,
        user_message=user_message,
        answer_text=answer_text,
        evidence=evidence,
    )
    if not model_payload:
        return fallback
    allowed_urls = {item.url for item in evidence}
    raw_urls = model_payload.get("evidence_urls")
    if isinstance(raw_urls, list):
        model_payload["evidence_urls"] = [
            str(url).strip()
            for url in raw_urls
            if str(url).strip() in allowed_urls
        ]
    model_payload["summary_source"] = "llm"
    return {**fallback, **model_payload}


def _deterministic_summary_payload(
    *,
    previous_summary: dict[str, Any],
    user_message: str,
    answer_text: str,
    evidence: list[Any],
) -> dict[str, Any]:
    recent_turns = _prior_recent_turns(previous_summary)
    recent_turns.append(
        {
            "question": user_message[:500],
            "answer_excerpt": answer_text[:700],
            "evidence_urls": [item.url for item in evidence[:6]],
        }
    )
    recent_turns = recent_turns[-MAX_ROLLING_SUMMARY_TURNS:]
    return {
        "summary_source": "deterministic",
        "summary_text": _render_rolling_summary(recent_turns),
        "recent_turns": recent_turns,
        "evidence_urls": [item.url for item in evidence[:12]],
    }


def _prior_recent_turns(previous_summary: dict[str, Any]) -> list[dict[str, Any]]:
    payload = previous_summary.get("summary_payload") if isinstance(previous_summary, dict) else None
    if not isinstance(payload, dict):
        return []
    raw = payload.get("recent_turns")
    if not isinstance(raw, list):
        return []
    turns: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        turns.append(
            {
                "question": str(item.get("question") or "")[:500],
                "answer_excerpt": str(item.get("answer_excerpt") or "")[:700],
                "evidence_urls": [
                    str(url).strip()
                    for url in (item.get("evidence_urls") or [])
                    if str(url).strip()
                ][:6],
            }
        )
    return turns


def _render_rolling_summary(recent_turns: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for turn in recent_turns:
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer_excerpt") or "").strip()
        urls = [str(url).strip() for url in turn.get("evidence_urls", []) if str(url).strip()]
        lines: list[str] = []
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")
        if urls:
            lines.append("Evidence: " + " ".join(urls[:6]))
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)[:3600]


def _generate_llm_summary_payload(
    *,
    previous_summary: dict[str, Any],
    user_message: str,
    answer_text: str,
    evidence: list[Any],
) -> dict[str, Any] | None:
    try:
        client = _build_memory_model()
        evidence_payload = [
            {"url": item.url, "title": item.title, "index": item.index}
            for item in evidence[:12]
        ]
        messages = [
            SystemMessage(
                content=(
                    "You update thread memory for a tech-news intelligence assistant. "
                    "Return JSON only. Do not invent evidence URLs. "
                    "Keep the summary concise but specific."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "previous_summary": previous_summary,
                        "new_user_message": user_message,
                        "new_assistant_answer": answer_text[:5000],
                        "new_evidence": evidence_payload,
                        "required_json": {
                            "summary_text": "string",
                            "current_topic": "string",
                            "key_entities": ["string"],
                            "confirmed_facts": ["string"],
                            "open_questions": ["string"],
                            "evidence_urls": ["string"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
        result = client.invoke(messages)
        text = _coerce_model_text(getattr(result, "content", result))
        parsed = _extract_json_object(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.info("thread memory llm summary skipped: %s", exc)
        return None


def _build_memory_model() -> Any:
    from services.llm_provider import DEFAULT_DEEPSEEK_MODEL, build_chat_model

    provider = _first_env("AGENT_MEMORY_PROVIDER", "AGENT_GRAPH_CONTEXT_PROVIDER", default="deepseek")
    model = _first_env("AGENT_MEMORY_MODEL", "AGENT_GRAPH_CONTEXT_MODEL", default=DEFAULT_DEEPSEEK_MODEL)
    return build_chat_model(
        provider=provider,
        model_name=model,
        temperature=_env_float("AGENT_MEMORY_TEMPERATURE", 0.1),
        default_provider="deepseek",
        default_model=DEFAULT_DEEPSEEK_MODEL,
    )


def _memory_llm_enabled() -> bool:
    return _env_flag("AGENT_THREAD_MEMORY_LLM_ENABLED", default=False)


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return default


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except Exception:
        return default


def _coerce_model_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("text"):
                chunks.append(str(item.get("text")))
        return "\n".join(chunks)
    return "" if content is None else str(content)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _evidence_excerpt(text: str, url: str) -> str:
    clean_url = str(url or "").strip()
    for line in str(text or "").splitlines():
        if clean_url and clean_url in line:
            return line.strip()[:1000]
    return str(text or "").strip()[:700]
