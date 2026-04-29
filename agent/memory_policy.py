"""Runtime memory policy for LangGraph model inputs.

The conversation store keeps the complete persisted history. This module only
decides how much of the in-memory graph state is sent to the LLM on a given
model call.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately


DEFAULT_CONTEXT_MAX_TOKENS = 12000
DEFAULT_CONTEXT_KEEP_LAST_MESSAGES = 24


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def context_trim_enabled() -> bool:
    return _env_bool("AGENT_CONTEXT_TRIM_ENABLED", True)


def context_keep_last_messages() -> int:
    return _env_int(
        "AGENT_CONTEXT_KEEP_LAST_MESSAGES",
        DEFAULT_CONTEXT_KEEP_LAST_MESSAGES,
        minimum=1,
    )


def context_max_tokens() -> int:
    return _env_int(
        "AGENT_CONTEXT_MAX_TOKENS",
        DEFAULT_CONTEXT_MAX_TOKENS,
        minimum=100,
    )


def build_llm_input_messages(messages: Sequence[BaseMessage] | None) -> list[BaseMessage]:
    """Build the trimmed model-input view without mutating graph state."""
    message_list = list(messages or [])
    if not message_list:
        return []
    if not context_trim_enabled():
        return message_list

    trimmed = _trim_by_message_count(message_list, context_keep_last_messages())
    trimmed = _trim_by_token_budget(trimmed, context_max_tokens())
    return trimmed or [message_list[-1]]


def _trim_by_message_count(
    messages: Sequence[BaseMessage],
    max_messages: int,
) -> list[BaseMessage]:
    if len(messages) <= max_messages:
        return list(messages)
    try:
        return list(
            trim_messages(
                list(messages),
                max_tokens=max_messages,
                token_counter=len,
                strategy="last",
                start_on="human",
                allow_partial=False,
            )
        )
    except Exception:
        return list(messages)[-max_messages:]


def _trim_by_token_budget(
    messages: Sequence[BaseMessage],
    max_tokens: int,
) -> list[BaseMessage]:
    try:
        return list(
            trim_messages(
                list(messages),
                max_tokens=max_tokens,
                token_counter=count_tokens_approximately,
                strategy="last",
                start_on="human",
                allow_partial=False,
            )
        )
    except Exception:
        return list(messages)

