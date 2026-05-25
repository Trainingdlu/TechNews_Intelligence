"""Conversation context assembly for the LangGraph agent.

This module builds a structured context pack from persisted conversation
history. The context pack is the single model-facing view of prior turns; raw
conversation history remains persisted separately.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.core.evidence import extract_urls, normalize_url_for_match


MAX_MANIFEST_TURNS = 20
MAX_SELECTED_TURNS = 4
MAX_RECENT_TURNS_WITHOUT_CURATOR = 2
MAX_ASSISTANT_PREVIEW_CHARS = 700
MAX_CONTEXT_EXCERPT_CHARS = 1200
MAX_MEMORY_SUMMARY_CHARS = 2400

_CURATOR_SYSTEM_PROMPT = """You select prior conversation context for a tech-news intelligence agent.
Return JSON only. Do not answer the user.

You may only select turn_id values and evidence URLs that appear in the provided history_manifest
or thread_memory_summary.evidence_index.
If the current question is independent, return depends_on_history=false and an empty selected_turn_ids list.
If it depends on prior context, rewrite it as a standalone question and select only clearly useful turns.

Required JSON shape:
{
  "depends_on_history": boolean,
  "standalone_question": string,
  "selected_turn_ids": number[],
  "selected_evidence_urls": string[],
  "context_summary": string,
  "reason": string,
  "confidence": number
}
"""


@dataclass(frozen=True)
class HistoryEvidence:
    index: int | None
    title: str
    url: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
        }


@dataclass(frozen=True)
class HistoryTurn:
    turn_id: int
    user_message: str
    assistant_message: str
    user_index: int
    assistant_index: int | None
    evidence: list[HistoryEvidence] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_message": _clip(self.user_message, 420),
            "assistant_preview": _clip(self.assistant_message, MAX_ASSISTANT_PREVIEW_CHARS),
            "assistant_chars": len(self.assistant_message),
            "evidence_count": len(self.evidence),
            "evidence": [item.to_manifest() for item in self.evidence[:8]],
        }


def context_curator_enabled() -> bool:
    return _env_flag("AGENT_CONTEXT_CURATOR_ENABLED", default=True)


def build_history_manifest(history: list[dict] | None) -> list[dict[str, Any]]:
    turns = build_history_turns(history)
    return [turn.to_manifest() for turn in turns[-MAX_MANIFEST_TURNS:]]


def build_history_turns(history: list[dict] | None) -> list[HistoryTurn]:
    turns: list[HistoryTurn] = []
    current_user: tuple[int, str] | None = None
    for idx, item in enumerate(history or []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        text = message_text(item)
        if not text:
            continue
        if role == "user":
            if current_user is not None:
                user_index, user_text = current_user
                turns.append(
                    HistoryTurn(
                        turn_id=len(turns) + 1,
                        user_message=user_text,
                        assistant_message="",
                        user_index=user_index,
                        assistant_index=None,
                        evidence=[],
                    )
                )
            current_user = (idx, text)
            continue
        if current_user is None:
            continue
        user_index, user_text = current_user
        turns.append(
            HistoryTurn(
                turn_id=len(turns) + 1,
                user_message=user_text,
                assistant_message=text,
                user_index=user_index,
                assistant_index=idx,
                evidence=extract_history_evidence(item, text),
            )
        )
        current_user = None

    if current_user is not None:
        user_index, user_text = current_user
        turns.append(
            HistoryTurn(
                turn_id=len(turns) + 1,
                user_message=user_text,
                assistant_message="",
                user_index=user_index,
                assistant_index=None,
                evidence=[],
            )
        )
    return turns


def build_context_curator_messages(
    *,
    user_message: str,
    history_manifest: list[dict[str, Any]],
    memory_summary: dict[str, Any] | None = None,
) -> list[BaseMessage]:
    payload = {
        "current_user_message": str(user_message or "").strip(),
        "thread_memory_summary": _compact_memory_summary(memory_summary),
        "history_manifest": history_manifest,
    }
    return [
        SystemMessage(content=_CURATOR_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]


def normalize_context_curator_result(
    value: dict[str, Any] | None,
    history_manifest: list[dict[str, Any]],
    memory_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    valid_turn_ids = {
        int(item["turn_id"])
        for item in history_manifest
        if isinstance(item, dict) and _is_int_like(item.get("turn_id"))
    }
    valid_urls: set[str] = set()
    canonical_url: dict[str, str] = {}
    for turn in history_manifest:
        if not isinstance(turn, dict):
            continue
        evidence = turn.get("evidence")
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            normalized = normalize_url_for_match(url)
            if normalized:
                valid_urls.add(normalized)
                canonical_url[normalized] = url
    for item in _memory_evidence_items(memory_summary):
        url = str(item.get("url") or "").strip()
        normalized = normalize_url_for_match(url)
        if normalized:
            valid_urls.add(normalized)
            canonical_url[normalized] = url

    selected_turn_ids: list[int] = []
    raw_turns = value.get("selected_turn_ids")
    if isinstance(raw_turns, list):
        for raw in raw_turns:
            if not _is_int_like(raw):
                continue
            turn_id = int(raw)
            if turn_id in valid_turn_ids and turn_id not in selected_turn_ids:
                selected_turn_ids.append(turn_id)
            if len(selected_turn_ids) >= MAX_SELECTED_TURNS:
                break

    selected_evidence_urls: list[str] = []
    raw_urls = value.get("selected_evidence_urls")
    if isinstance(raw_urls, list):
        for raw in raw_urls:
            normalized = normalize_url_for_match(str(raw or "").strip())
            if normalized and normalized in valid_urls:
                url = canonical_url[normalized]
                if url not in selected_evidence_urls:
                    selected_evidence_urls.append(url)

    return {
        "depends_on_history": bool(value.get("depends_on_history")),
        "standalone_question": _clip(str(value.get("standalone_question") or ""), 600),
        "selected_turn_ids": selected_turn_ids,
        "selected_evidence_urls": selected_evidence_urls[:12],
        "context_summary": _clip(str(value.get("context_summary") or ""), 1200),
        "reason": _clip(str(value.get("reason") or ""), 600),
        "confidence": _curator_confidence(value),
    }


def should_use_context_curator(
    *,
    user_message: str,
    history_manifest: list[dict[str, Any]],
    memory_summary: dict[str, Any] | None = None,
) -> bool:
    if not context_curator_enabled() or not history_manifest:
        return False
    if memory_summary:
        return True
    evidence_count = sum(int(item.get("evidence_count") or 0) for item in history_manifest)
    if evidence_count > 0:
        return True
    if len(history_manifest) >= 3:
        return True
    return len(str(user_message or "").strip()) <= 80 and len(history_manifest) >= 2


def build_context_pack(
    *,
    user_message: str,
    history: list[dict] | None,
    history_manifest: list[dict[str, Any]] | None = None,
    memory_summary: dict[str, Any] | None = None,
    curator_result: dict[str, Any] | None = None,
    curator_used: bool = False,
    curator_error: str | None = None,
) -> dict[str, Any]:
    turns = build_history_turns(history)
    manifest = history_manifest if history_manifest is not None else [turn.to_manifest() for turn in turns]
    selected_turn_ids, selected_evidence_urls = _selected_ids_from_curator(curator_result, turns, memory_summary)
    strategy = "context_curator" if curator_used and curator_result else "recent_context"
    depends_on_history = bool(curator_result.get("depends_on_history")) if isinstance(curator_result, dict) else False
    standalone_question = _curator_text(curator_result, "standalone_question") or str(user_message or "").strip()
    context_summary = _curator_text(curator_result, "context_summary")
    reason = _curator_text(curator_result, "reason")
    confidence = _curator_confidence(curator_result)

    low_confidence = (
        curator_used
        and confidence is not None
        and confidence < context_curator_min_confidence()
    )
    if low_confidence:
        selected_turn_ids = []
        selected_evidence_urls = []
        depends_on_history = False
        standalone_question = str(user_message or "").strip()
        context_summary = ""
        strategy = "recent_context_low_confidence"

    if not selected_turn_ids and not selected_evidence_urls:
        selected_turn_ids = [turn.turn_id for turn in turns[-MAX_RECENT_TURNS_WITHOUT_CURATOR:]]
        for turn in turns[-MAX_RECENT_TURNS_WITHOUT_CURATOR:]:
            for evidence in turn.evidence:
                if evidence.url not in selected_evidence_urls:
                    selected_evidence_urls.append(evidence.url)
        if not low_confidence:
            strategy = "recent_context_fallback" if curator_used else strategy

    selected_turns = [
        _selected_turn_payload(turn, evidence_urls=selected_evidence_urls)
        for turn in turns
        if turn.turn_id in set(selected_turn_ids)
    ][:MAX_SELECTED_TURNS]
    selected_memory_evidence = _selected_memory_evidence(memory_summary, selected_evidence_urls)

    memory = _compact_memory_summary(memory_summary)
    if not context_summary:
        context_summary = _fallback_context_summary(selected_turns, memory)

    return {
        "current_question": str(user_message or "").strip(),
        "standalone_question": standalone_question,
        "depends_on_history": depends_on_history or bool(selected_turns) or bool(selected_evidence_urls),
        "context_summary": context_summary,
        "selected_turns": selected_turns,
        "selected_memory_evidence": selected_memory_evidence,
        "selected_evidence_urls": selected_evidence_urls[:12],
        "thread_memory_summary": memory,
        "history_manifest_count": len(manifest),
        "trim_report": {
            "strategy": strategy,
            "history_turns_total": len(turns),
            "history_turns_in_manifest": len(manifest),
            "history_turns_selected": len(selected_turns),
            "memory_evidence_selected": len(selected_memory_evidence),
            "history_compacted": len(turns) > len(manifest),
            "curator_used": curator_used,
            "curator_error": curator_error or "",
            "curator_reason": reason,
            "curator_confidence": confidence,
            "curator_low_confidence": low_confidence,
        },
    }


def render_context_for_prompt(
    context_pack: dict[str, Any] | None, profile: str = "full"
) -> str:
    """Render the context pack for a model prompt.

    profile="full" renders every section (default, unchanged). profile="tool"
    renders only the candidate-URL sections the tool worker needs (the question
    is injected separately as the curator-rewritten standalone question), dropping
    the prose sections that dominate the pack's size.
    """
    if not isinstance(context_pack, dict):
        return "(none)"
    tool_only = str(profile or "full").strip().lower() == "tool"
    parts: list[str] = []
    rendered_url_norms: set[str] = set()
    selected_norms = {
        norm
        for norm in (
            normalize_url_for_match(str(item or "").strip())
            for item in context_pack.get("selected_evidence_urls", [])
        )
        if norm
    }
    memory = context_pack.get("thread_memory_summary")
    memory_text = (
        str(memory.get("summary_text") or "").strip() if isinstance(memory, dict) else ""
    )
    summary = str(context_pack.get("context_summary") or "").strip()
    if not tool_only and summary and not _summary_covered_by(summary, memory_text):
        parts.append(f"Context summary:\n{summary}")
    if not tool_only and isinstance(memory, dict) and memory:
        if memory_text:
            parts.append(f"Thread memory summary:\n{memory_text}")
        memory_evidence = [
            item for item in memory.get("evidence_index", [])
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        ]
        lines: list[str] = []
        for item in memory_evidence[:8]:
            url = str(item.get("url") or "").strip()
            norm = normalize_url_for_match(url)
            if norm and norm in selected_norms:
                continue
            title = str(item.get("title") or "").strip() or "previous evidence"
            excerpt = str(item.get("excerpt") or "").strip()
            age = _evidence_age_label(item.get("created_at"))
            header = f"- {title} | {url}" + (f" ({age})" if age else "")
            lines.append(f"{header}\n  {excerpt}" if excerpt else header)
            if norm:
                rendered_url_norms.add(norm)
        if lines:
            parts.append("Thread evidence index:\n" + "\n".join(lines))
    turns = context_pack.get("selected_turns")
    if not tool_only and isinstance(turns, list) and turns:
        lines = []
        for turn in turns[:MAX_SELECTED_TURNS]:
            if not isinstance(turn, dict):
                continue
            user = str(turn.get("user_message") or "").strip()
            assistant = str(turn.get("assistant_excerpt") or "").strip()
            evidence_urls = [
                str(item or "").strip()
                for item in turn.get("evidence_urls", [])
                if str(item or "").strip()
            ][:6]
            for url in evidence_urls:
                norm = normalize_url_for_match(url)
                if norm:
                    rendered_url_norms.add(norm)
            lines.append(
                "\n".join(
                    part
                    for part in [
                        f"Turn {turn.get('turn_id')}:",
                        f"User: {user}" if user else "",
                        f"Assistant excerpt: {assistant}" if assistant else "",
                        f"Evidence URLs: {' '.join(evidence_urls)}" if evidence_urls else "",
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
                if tool_only:
                    lines.append(header)
                else:
                    lines.append(header + (f"\n  {excerpt}" if excerpt else ""))
                norm = normalize_url_for_match(url)
                if norm:
                    rendered_url_norms.add(norm)
        if lines:
            parts.append("Selected memory evidence:\n" + "\n".join(lines))
    remaining_urls: list[str] = []
    for item in context_pack.get("selected_evidence_urls", []):
        url = str(item or "").strip()
        if not url:
            continue
        norm = normalize_url_for_match(url)
        if norm and norm in rendered_url_norms:
            continue
        remaining_urls.append(url)
    if remaining_urls:
        parts.append("Prior evidence URLs:\n" + "\n".join(f"- {url}" for url in remaining_urls[:12]))
    return "\n\n".join(parts).strip() or "(none)"


def active_question(context_pack: dict[str, Any] | None, fallback: str) -> str:
    if isinstance(context_pack, dict):
        question = str(context_pack.get("standalone_question") or "").strip()
        if question:
            return question
    return str(fallback or "").strip()


def message_text(item: dict[str, Any]) -> str:
    parts = item.get("parts")
    if isinstance(parts, list):
        return "\n".join(
            str(part.get("text", "")).strip()
            for part in parts
            if isinstance(part, dict) and str(part.get("text", "")).strip()
        ).strip()
    return str(item.get("text", "") or "").strip()


def extract_history_evidence(item: dict[str, Any], text: str | None = None) -> list[HistoryEvidence]:
    title_map = _history_title_map(item)
    citation_urls = _history_citation_urls(item)
    evidence: list[HistoryEvidence] = []
    seen: set[str] = set()
    raw_text = text if text is not None else message_text(item)
    for idx, url in enumerate(citation_urls, 1):
        _append_evidence(evidence, seen, index=idx, title=title_map.get(url, ""), url=url)
    for url in extract_urls(raw_text):
        _append_evidence(evidence, seen, index=None, title=title_map.get(url, ""), url=url)
    return evidence


def _history_citation_urls(item: dict[str, Any]) -> list[str]:
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
    return urls


def _history_title_map(item: dict[str, Any]) -> dict[str, str]:
    raw = item.get("url_title_map")
    if not isinstance(raw, dict):
        return {}
    return {
        str(url).strip(): str(title).strip()
        for url, title in raw.items()
        if str(url).strip() and str(title).strip()
    }


def _append_evidence(
    evidence: list[HistoryEvidence],
    seen: set[str],
    *,
    index: int | None,
    title: str,
    url: str,
) -> None:
    clean_url = str(url or "").strip()
    normalized = normalize_url_for_match(clean_url)
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    evidence.append(HistoryEvidence(index=index, title=str(title or "").strip(), url=clean_url))


def _selected_ids_from_curator(
    curator_result: dict[str, Any] | None,
    turns: list[HistoryTurn],
    memory_summary: dict[str, Any] | None = None,
) -> tuple[list[int], list[str]]:
    if not isinstance(curator_result, dict):
        return [], []
    valid_turns = {turn.turn_id: turn for turn in turns}
    raw_ids = curator_result.get("selected_turn_ids")
    selected_ids: list[int] = []
    if isinstance(raw_ids, list):
        for raw in raw_ids:
            try:
                turn_id = int(raw)
            except Exception:
                continue
            if turn_id in valid_turns and turn_id not in selected_ids:
                selected_ids.append(turn_id)
            if len(selected_ids) >= MAX_SELECTED_TURNS:
                break

    selected_urls: list[str] = []
    valid_urls: set[str] = set()
    url_by_norm: dict[str, str] = {}
    for turn in turns:
        for evidence in turn.evidence:
            normalized = normalize_url_for_match(evidence.url)
            if normalized:
                valid_urls.add(normalized)
                url_by_norm[normalized] = evidence.url
    for item in _memory_evidence_items(memory_summary):
        url = str(item.get("url") or "").strip()
        normalized = normalize_url_for_match(url)
        if normalized:
            valid_urls.add(normalized)
            url_by_norm[normalized] = url
    raw_urls = curator_result.get("selected_evidence_urls")
    if isinstance(raw_urls, list):
        for raw in raw_urls:
            normalized = normalize_url_for_match(str(raw or "").strip())
            if normalized and normalized in valid_urls:
                url = url_by_norm[normalized]
                if url not in selected_urls:
                    selected_urls.append(url)
    for turn_id in selected_ids:
        for evidence in valid_turns[turn_id].evidence:
            if evidence.url not in selected_urls:
                selected_urls.append(evidence.url)
    return selected_ids, selected_urls


def _selected_memory_evidence(
    memory_summary: dict[str, Any] | None,
    selected_evidence_urls: list[str],
) -> list[dict[str, str]]:
    selected_norms = {normalize_url_for_match(url) for url in selected_evidence_urls}
    items: list[dict[str, str]] = []
    for item in _memory_evidence_items(memory_summary):
        url = str(item.get("url") or "").strip()
        if normalize_url_for_match(url) not in selected_norms:
            continue
        items.append(
            {
                "url": url,
                "title": _clip(str(item.get("title") or ""), 220),
                "excerpt": _clip(str(item.get("excerpt") or ""), 700),
                "created_at": str(item.get("created_at") or ""),
            }
        )
    return items[:8]


def _selected_turn_payload(turn: HistoryTurn, *, evidence_urls: list[str]) -> dict[str, Any]:
    selected_norms = {normalize_url_for_match(url) for url in evidence_urls}
    turn_urls = [
        evidence.url
        for evidence in turn.evidence
        if normalize_url_for_match(evidence.url) in selected_norms
    ]
    if not turn_urls:
        turn_urls = [evidence.url for evidence in turn.evidence[:6]]
    return {
        "turn_id": turn.turn_id,
        "user_message": _clip(turn.user_message, 500),
        "assistant_excerpt": _best_assistant_excerpt(turn.assistant_message, turn_urls),
        "evidence_urls": turn_urls[:8],
    }


def _best_assistant_excerpt(text: str, urls: list[str]) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    if urls:
        lines = []
        for line in clean.splitlines():
            stripped = line.strip()
            if stripped and any(url in stripped for url in urls):
                lines.append(stripped)
        if lines:
            return _clip("\n".join(lines), MAX_CONTEXT_EXCERPT_CHARS)
    return _clip(clean, MAX_CONTEXT_EXCERPT_CHARS)


def _fallback_context_summary(selected_turns: list[dict[str, Any]], memory: dict[str, Any]) -> str:
    memory_text = str(memory.get("summary_text") or "").strip() if isinstance(memory, dict) else ""
    if memory_text:
        return _clip(memory_text, 800)
    if not selected_turns:
        return ""
    users = [str(turn.get("user_message") or "").strip() for turn in selected_turns if isinstance(turn, dict)]
    return _clip("Recent relevant user questions: " + " | ".join([item for item in users if item]), 800)


def _summary_covered_by(summary: str, memory_text: str) -> bool:
    """True when the context summary merely repeats the thread memory summary."""
    candidate = str(summary or "").strip()
    full = str(memory_text or "").strip()
    if not candidate or not full:
        return False
    if candidate == full:
        return True
    core = candidate[:-3].strip() if candidate.endswith("...") else candidate
    return bool(core) and full.startswith(core)


def _curator_text(curator_result: dict[str, Any] | None, key: str) -> str:
    if not isinstance(curator_result, dict):
        return ""
    return str(curator_result.get(key) or "").strip()


def _curator_confidence(curator_result: dict[str, Any] | None) -> float | None:
    if not isinstance(curator_result, dict):
        return None
    try:
        return max(0.0, min(float(curator_result.get("confidence")), 1.0))
    except Exception:
        return None


def _compact_memory_summary(memory_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory_summary, dict):
        return {}
    summary_text = str(memory_summary.get("summary_text") or "").strip()
    payload = memory_summary.get("summary_payload")
    result = {
        "summary_text": _clip(summary_text, MAX_MEMORY_SUMMARY_CHARS),
    }
    if isinstance(payload, dict):
        result["summary_payload"] = copy.deepcopy(payload)
    evidence_index = _memory_evidence_items(memory_summary)
    if evidence_index:
        result["evidence_index"] = [
            {
                "url": str(item.get("url") or "").strip(),
                "title": _clip(str(item.get("title") or ""), 220),
                "excerpt": _clip(str(item.get("excerpt") or ""), 500),
                "created_at": str(item.get("created_at") or ""),
            }
            for item in evidence_index[:12]
        ]
    return {key: value for key, value in result.items() if value}


def _memory_evidence_items(memory_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(memory_summary, dict):
        return []
    raw = memory_summary.get("evidence_index")
    if not isinstance(raw, list):
        return []
    max_age = _memory_evidence_max_age_days()
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        normalized = normalize_url_for_match(url)
        if not normalized or normalized in seen:
            continue
        if max_age > 0:
            age = _evidence_age_days(item.get("created_at"))
            if age is not None and age > max_age:
                continue
        seen.add(normalized)
        items.append(item)
    return items


def _clip(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    clipped = clean[: max_chars - 1].rstrip()
    for token in ("\n", ".", ";", ","):
        idx = clipped.rfind(token)
        if idx >= int(max_chars * 0.55):
            clipped = clipped[: idx + 1].rstrip()
            break
    return f"{clipped}..."


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def context_tool_profile_enabled() -> bool:
    """Whether the tool worker receives the lean (URL-only) context profile (F2)."""
    return _env_flag("AGENT_CONTEXT_TOOL_PROFILE", default=True)


def context_curator_min_confidence() -> float:
    raw = os.getenv("AGENT_CONTEXT_CURATOR_MIN_CONFIDENCE")
    if raw is None:
        return 0.4
    try:
        return max(0.0, min(float(str(raw).strip()), 1.0))
    except Exception:
        return 0.4


def _memory_evidence_max_age_days() -> float:
    raw = os.getenv("AGENT_MEMORY_EVIDENCE_MAX_AGE_DAYS")
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(str(raw).strip()))
    except Exception:
        return 0.0


def _evidence_age_days(created_at: Any) -> float | None:
    raw = str(created_at or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - parsed
    return max(0.0, delta.total_seconds() / 86400.0)


def _evidence_age_label(created_at: Any) -> str:
    days = _evidence_age_days(created_at)
    if days is None:
        return ""
    whole = int(days)
    if whole <= 0:
        return "今天检索"
    return f"{whole}天前检索"


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False
