"""Model invocation and JSON parsing helpers for graph nodes."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage

from agent.core.trace import extract_token_usage, set_request_token_usage, trace_span

from .state import GraphModelHandle


def _invoke_json_model(handle: GraphModelHandle, *, node: str, messages: list[BaseMessage]) -> dict[str, Any] | None:
    text = _invoke_text_model(handle, node=node, messages=messages)
    if not text:
        return None
    return _extract_json_object(text)


def _invoke_text_model(handle: GraphModelHandle, *, node: str, messages: list[BaseMessage]) -> str:
    with trace_span(
        "model_call",
        node,
        input_summary={"message_count": len(messages)},
        metadata={
            "node": node,
            "provider": handle.provider,
            "model": handle.model,
            "fallback": handle.fallback,
            "handle_error": handle.error,
        },
    ) as span:
        if handle.client is None:
            span.set_output({"status": "skipped", "reason": "missing_client"})
            span.set_model_io(
                node=node,
                provider=handle.provider,
                model=handle.model,
                input_messages=messages,
                raw_output={"status": "skipped", "reason": "missing_client"},
            )
            return ""
        try:
            result = handle.client.invoke(messages)
            usage = extract_token_usage([result])
            if usage:
                set_request_token_usage(usage)
            text = _coerce_to_text(getattr(result, "content", result)).strip()
            span.set_output({"status": "success", "output_chars": len(text), "token_usage": usage})
            span.set_model_io(
                node=node,
                provider=handle.provider,
                model=handle.model,
                input_messages=messages,
                raw_output=result,
                token_usage=usage,
            )
            return text
        except Exception as exc:  # noqa: BLE001
            span.set_error(
                error_code=f"model_{type(exc).__name__.lower()}",
                error_message=str(exc),
                error=exc,
            )
            span.set_output({"status": "error", "fallback": True, "error": str(exc)})
            span.set_model_io(
                node=node,
                provider=handle.provider,
                model=handle.model,
                input_messages=messages,
                raw_output={"error": str(exc), "exception_type": type(exc).__name__},
            )
            return ""


def _coerce_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and part.get("text"):
                chunks.append(str(part.get("text")))
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
