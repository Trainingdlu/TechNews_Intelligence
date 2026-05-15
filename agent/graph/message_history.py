"""Conversation history helpers for graph nodes."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.core.evidence import extract_urls, normalize_url_for_match


def _history_to_messages(history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        text = _message_text(item)
        if not text:
            continue
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def _message_text(item: dict[str, Any]) -> str:
    parts = item.get("parts")
    if isinstance(parts, list):
        return "\n".join(
            str(part.get("text", "")).strip()
            for part in parts
            if isinstance(part, dict) and str(part.get("text", "")).strip()
        ).strip()
    return str(item.get("text", "") or "").strip()


def _load_thread_memory_summary_safe(thread_id: str) -> dict[str, Any]:
    if not thread_id:
        return {}
    try:
        from services.thread_memory import load_thread_memory_summary

        return load_thread_memory_summary(thread_id)
    except Exception:
        return {}


def _human_texts(messages: list[BaseMessage]) -> list[str]:
    return [
        str(getattr(message, "content", "") or "")
        for message in messages
        if isinstance(message, HumanMessage)
    ]


def _recent_context_snippet(history: list[dict], max_messages: int = 4, max_chars: int = 2400) -> str:
    if not history:
        return "(none)"
    chunks: list[str] = []
    for item in history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        text = _message_text(item)
        if not text:
            continue
        label = "User" if role == "user" else "Assistant"
        urls = _history_item_context_urls(item)
        text_limit = 360 if role == "user" else 900
        clipped_text = _truncate_context_text(text, max_chars=text_limit)
        if urls:
            url_context = " ".join(urls[:3])
            chunks.append(f"[{label}] {clipped_text}\nEvidence URLs: {url_context}")
        else:
            chunks.append(f"[{label}] {clipped_text}")
    merged = "\n".join(chunks).strip()
    if not merged:
        return "(none)"
    if len(merged) > max_chars:
        prefix = "[...truncated...]\n"
        tail = merged[-(max_chars - len(prefix)) :]
        first_break = tail.find("\n")
        if 0 < first_break < 180:
            tail = tail[first_break + 1 :]
        return prefix + tail
    return merged


def _truncate_context_text(text: str, *, max_chars: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rstrip()
    boundaries = [
        clipped.rfind("\n"),
        clipped.rfind("。"),
        clipped.rfind("；"),
        clipped.rfind(";"),
        clipped.rfind("."),
        clipped.rfind("，"),
        clipped.rfind(","),
    ]
    boundary = max(boundaries)
    if boundary >= int(max_chars * 0.55):
        clipped = clipped[: boundary + 1].rstrip()
    return f"{clipped}…"


def _history_item_context_urls(item: dict[str, Any], max_urls: int = 3) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    raw_urls = item.get("citation_urls")
    if isinstance(raw_urls, list):
        for raw in raw_urls:
            url = str(raw or "").strip()
            normalized = normalize_url_for_match(url)
            if normalized and normalized not in seen:
                seen.add(normalized)
                urls.append(url)
                if len(urls) >= max_urls:
                    return urls
    for raw in extract_urls(_message_text(item)):
        normalized = normalize_url_for_match(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(raw)
            if len(urls) >= max_urls:
                return urls
    return urls
