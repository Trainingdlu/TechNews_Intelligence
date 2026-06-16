"""Chat orchestration shared by regular and SSE HTTP endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from fastapi import HTTPException

from agent import AgentGenerationError, generate_response_payload
from agent.clarification import (
    build_clarification_history_item,
    resolve_user_message_with_history_clarification,
)
from app import streaming
from app.schemas import ChatRequest, ChatResponse
from app.services import quota_service
from services.conversations import (
    ConversationThreadNotFoundError,
    append_message_for_token,
    create_thread,
    delete_thread_for_token,
    load_history_for_token,
)
from services.thread_memory import schedule_thread_memory_update

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatTurn:
    body: ChatRequest
    token_info: dict
    request_id: str
    thread_id: str
    history: list[dict]
    effective_message: str
    reservation: dict


def agent_memory_load_limit() -> int:
    try:
        return max(1, int(os.getenv("AGENT_MEMORY_LOAD_LIMIT", "80")))
    except Exception:
        return 80


def normalize_request_thread_id(thread_id: str | None) -> str | None:
    value = str(thread_id or "").strip()
    return value or None


WEB_HISTORY_DISPLAY_LIMIT = 200


def load_thread_history_or_404(
    thread_id: str, *, token_id: int, limit: int | None = None
) -> list[dict]:
    try:
        return load_history_for_token(
            thread_id,
            token_id,
            limit=limit if limit is not None else agent_memory_load_limit(),
        )
    except ConversationThreadNotFoundError:
        raise HTTPException(status_code=404, detail="conversation_thread_not_found") from None


def delete_thread_or_404(thread_id: str, *, token_id: int) -> None:
    try:
        delete_thread_for_token(thread_id, token_id)
    except ConversationThreadNotFoundError:
        raise HTTPException(status_code=404, detail="conversation_thread_not_found") from None


def create_thread_for_request(body: ChatRequest, token_info: dict) -> str:
    subject = str(body.message or "").strip()[:80] or None
    metadata = {
        "channel": "web",
        "email": str(token_info.get("email", "") or ""),
        "token_id": token_info.get("id"),
    }
    row = create_thread(channel="web", subject=subject, metadata=metadata)
    thread_id = str(row.get("thread_id", "")).strip()
    if not thread_id:
        raise RuntimeError("conversation thread creation returned empty thread_id")
    return thread_id


def user_history_item(message: str) -> dict:
    return {"role": "user", "parts": [{"text": str(message or "")}]}


def model_history_item_from_payload(payload: dict) -> dict:
    kind = str(payload.get("kind", "answer")).strip().lower()
    if kind == "clarification_required":
        clarification_payload = payload.get("clarification", {})
        if isinstance(clarification_payload, dict):
            return build_clarification_history_item(clarification_payload)
    title_map = payload.get("url_title_map", {})
    if not isinstance(title_map, dict):
        title_map = {}
    return {
        "role": "model",
        "kind": "answer",
        "parts": [{"text": str(payload.get("text", "") or "")}],
        "citation_urls": [
            str(url).strip()
            for url in payload.get("citation_urls", [])
            if str(url).strip()
        ] if isinstance(payload.get("citation_urls", []), list) else [],
        "url_title_map": {
            str(url).strip(): str(title).strip()
            for url, title in title_map.items()
            if str(url).strip() and str(title).strip()
        },
    }


def persist_conversation_turn(
    *,
    thread_id: str,
    token_id: int,
    user_message: str,
    effective_message: str,
    payload: dict,
    request_id: str,
) -> None:
    metadata = {
        "request_id": request_id,
        "effective_message": effective_message,
    }
    append_message_for_token(
        thread_id,
        token_id,
        user_history_item(user_message),
        metadata=metadata,
    )
    model_message = model_history_item_from_payload(payload)
    model_row = append_message_for_token(
        thread_id,
        token_id,
        model_message,
        metadata=metadata,
    )
    schedule_thread_memory_update(
        thread_id=thread_id,
        user_message=user_message,
        model_message=model_message,
        model_message_id=int(model_row.get("id")) if isinstance(model_row, dict) and model_row.get("id") else None,
        request_id=request_id,
    )


def prepare_chat_turn(body: ChatRequest, token_info: dict, request_id: str, *, stream: bool = False) -> ChatTurn:
    thread_id = normalize_request_thread_id(body.thread_id)
    history = load_thread_history_or_404(thread_id, token_id=int(token_info["id"])) if thread_id else []
    reservation = quota_service.reserve_quota_or_notify_403(token_info)

    if stream:
        logger.info(
            "[%s] 对话(流式): request_id=%s message=%s...",
            token_info["email"],
            request_id,
            body.message[:50],
        )
    else:
        logger.info(
            "[%s] 对话: request_id=%s message=%s...",
            token_info["email"],
            request_id,
            body.message[:50],
        )
    try:
        if thread_id is None:
            thread_id = create_thread_for_request(body, token_info)
        effective_message, pending = resolve_user_message_with_history_clarification(history, body.message)
        if pending is not None:
            if stream:
                logger.info(
                    "[%s] clarification follow-up detected(stream): request_id=%s reason=%s",
                    token_info["email"],
                    request_id,
                    pending.reason,
                )
            else:
                logger.info(
                    "[%s] clarification follow-up detected: request_id=%s reason=%s",
                    token_info["email"],
                    request_id,
                    pending.reason,
                )
    except Exception:
        quota_service.refund_reserved_quota(token_info)
        raise

    return ChatTurn(
        body=body,
        token_info=token_info,
        request_id=request_id,
        thread_id=thread_id,
        history=history,
        effective_message=effective_message,
        reservation=reservation,
    )


def run_agent_turn(turn: ChatTurn, *, progress_callback: Callable[[dict], None] | None = None) -> dict:
    payload = generate_response_payload(
        turn.history,
        turn.effective_message,
        progress_callback=progress_callback,
        request_id=turn.request_id,
        thread_id=turn.thread_id,
    )
    persist_conversation_turn(
        thread_id=turn.thread_id,
        token_id=int(turn.token_info["id"]),
        user_message=turn.body.message,
        effective_message=turn.effective_message,
        payload=payload,
        request_id=turn.request_id,
    )
    return payload


def chat_response_from_payload(turn: ChatTurn, payload: dict) -> ChatResponse:
    kind = str(payload.get("kind", "answer")).strip().lower()
    if kind == "clarification_required":
        clarification_payload = payload.get("clarification", {})
        question = str(payload.get("text", "")).strip()
        return ChatResponse(
            reply=question,
            thread_id=turn.thread_id,
            kind="clarification_required",
            clarification=clarification_payload if isinstance(clarification_payload, dict) else None,
            citation_urls=[],
            remaining=quota_service.remaining_after_refund(turn.reservation),
            quota=int(turn.reservation["quota"]),
            unlimited=bool(turn.reservation.get("unlimited", False)),
            status="active",
        )

    reply = str(payload.get("text", "")).strip()
    citation_urls = payload.get("citation_urls", [])
    if not isinstance(citation_urls, list):
        citation_urls = []
    return ChatResponse(
        reply=reply,
        thread_id=turn.thread_id,
        kind="answer",
        citation_urls=[str(url).strip() for url in citation_urls if str(url).strip()],
        remaining=max(int(turn.reservation["remaining"]), 0),
        quota=int(turn.reservation["quota"]),
        unlimited=bool(turn.reservation.get("unlimited", False)),
        status=quota_service.status_from_reservation(turn.reservation),
    )


async def handle_chat_request(body: ChatRequest, token_info: dict, request_id: str) -> ChatResponse:
    turn = prepare_chat_turn(body, token_info, request_id)
    try:
        payload = await asyncio.to_thread(
            generate_response_payload,
            turn.history,
            turn.effective_message,
            request_id=turn.request_id,
            thread_id=turn.thread_id,
        )
        persist_conversation_turn(
            thread_id=turn.thread_id,
            token_id=int(turn.token_info["id"]),
            user_message=turn.body.message,
            effective_message=turn.effective_message,
            payload=payload,
            request_id=turn.request_id,
        )
    except HTTPException:
        quota_service.refund_reserved_quota(token_info)
        raise
    except asyncio.CancelledError:
        quota_service.refund_reserved_quota(token_info)
        raise
    except AgentGenerationError as e:
        quota_service.refund_reserved_quota(token_info)
        logger.warning(
            "[%s] generation blocked: request_id=%s error=%s",
            token_info["email"],
            request_id,
            e,
        )
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        quota_service.refund_reserved_quota(token_info)
        logger.error(
            "[%s] 处理失败: request_id=%s error=%s",
            token_info["email"],
            request_id,
            e,
        )
        raise HTTPException(status_code=500, detail="服务暂时不可用，请稍后重试")

    kind = str(payload.get("kind", "answer")).strip().lower()
    if kind == "clarification_required":
        quota_service.refund_reserved_quota(token_info)
    else:
        quota_service.maybe_send_quota_exhausted_notifications(token_info, turn.reservation)
    return chat_response_from_payload(turn, payload)


async def stream_chat_events(turn: ChatTurn) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue[dict] = asyncio.Queue()
    done_event = asyncio.Event()
    result_holder: dict[str, dict] = {}
    error_holder: dict[str, Exception] = {}
    refunded = False
    completed = False
    worker_task: asyncio.Task | None = None

    def refund_once() -> None:
        nonlocal refunded
        if refunded:
            return
        refunded = True
        quota_service.refund_reserved_quota(turn.token_info)

    def progress_callback(payload: dict[str, str]) -> None:
        stage = str(payload.get("stage", "")).strip()
        if not stage:
            return
        loop.call_soon_threadsafe(progress_queue.put_nowait, payload)

    def worker() -> None:
        try:
            result_holder["payload"] = run_agent_turn(turn, progress_callback=progress_callback)
        except Exception as exc:
            error_holder["exc"] = exc
        finally:
            loop.call_soon_threadsafe(done_event.set)

    try:
        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        last_status = ""

        while True:
            if done_event.is_set() and progress_queue.empty():
                break
            try:
                payload = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            event_type = str(payload.get("event") or "").strip().lower()
            if event_type == "progress" or payload.get("tool") or payload.get("tool_name"):
                progress_payload = streaming.progress_event_payload(payload)
                yield streaming.sse_event("progress", progress_payload)
                continue
            if event_type == "evidence":
                evidence_payload = streaming.evidence_event_payload(payload)
                if evidence_payload.get("url") or evidence_payload.get("title"):
                    yield streaming.sse_event("evidence", evidence_payload)
                continue

            status_text = streaming.status_text_from_progress(payload)
            if status_text and status_text != last_status:
                last_status = status_text
                yield streaming.sse_event("status", {"text": status_text})

        await worker_task

        if "exc" in error_holder:
            exc = error_holder["exc"]
            refund_once()
            if isinstance(exc, AgentGenerationError):
                logger.warning(
                    "[%s] generation blocked: request_id=%s error=%s",
                    turn.token_info["email"],
                    turn.request_id,
                    exc,
                )
                yield streaming.sse_event("error", {"detail": str(exc), "status_code": 503})
                return
            logger.error(
                "[%s] 处理失败: request_id=%s error=%s",
                turn.token_info["email"],
                turn.request_id,
                exc,
            )
            yield streaming.sse_event("error", {"detail": "服务暂时不可用，请稍后重试", "status_code": 500})
            return

        payload = result_holder.get("payload", {})
        kind = str(payload.get("kind", "answer")).strip().lower()
        if kind == "clarification_required":
            clarification_payload = payload.get("clarification", {})
            question = str(payload.get("text", "")).strip()
            refund_once()
            completed = True
            yield streaming.sse_event("final", {
                "kind": "clarification_required",
                "reply": question,
                "thread_id": turn.thread_id,
                "clarification": clarification_payload if isinstance(clarification_payload, dict) else None,
                "citation_urls": [],
                "remaining": quota_service.remaining_after_refund(turn.reservation),
                "quota": int(turn.reservation["quota"]),
                "unlimited": bool(turn.reservation.get("unlimited", False)),
                "status": "active",
            })
            return

        reply = str(payload.get("text", "")).strip()
        citation_urls = payload.get("citation_urls", [])
        if not isinstance(citation_urls, list):
            citation_urls = []
        quota_service.maybe_send_quota_exhausted_notifications(turn.token_info, turn.reservation)
        completed = True
        yield streaming.sse_event("final", {
            "kind": "answer",
            "reply": reply,
            "thread_id": turn.thread_id,
            "citation_urls": [str(url).strip() for url in citation_urls if str(url).strip()],
            "remaining": max(int(turn.reservation["remaining"]), 0),
            "quota": int(turn.reservation["quota"]),
            "unlimited": bool(turn.reservation.get("unlimited", False)),
            "status": quota_service.status_from_reservation(turn.reservation),
        })
    finally:
        if worker_task is not None and not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except Exception:
                pass
        if not completed:
            refund_once()
