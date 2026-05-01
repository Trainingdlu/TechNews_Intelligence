"""Pre-execution policy checks for model-proposed tool calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .core.evidence import extract_urls, normalize_url_for_match

ToolPolicyAction = Literal["allow", "clarify"]


@dataclass(frozen=True)
class ToolPolicyDecision:
    action: ToolPolicyAction
    reason: str = ""
    details: dict[str, Any] | None = None

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


def allow_tool_use() -> ToolPolicyDecision:
    return ToolPolicyDecision(action="allow")


def evaluate_tool_calls(
    messages: list[BaseMessage] | tuple[BaseMessage, ...],
    *,
    allowed_tool_names: set[str],
    input_schemas: dict[str, dict[str, Any]] | None = None,
    evidence_urls: list[str] | set[str] | None = None,
) -> ToolPolicyDecision:
    """Validate pending tool calls before the LangGraph tools node runs."""
    pending_calls = _pending_tool_calls(messages)
    user_messages = [
        str(getattr(message, "content", "") or "")
        for message in messages
        if str(getattr(message, "type", "")).strip().lower() == "human"
    ]
    previous_tool_calls = _tool_call_names_from_messages(messages)
    return evaluate_pending_tool_calls(
        pending_calls,
        allowed_tool_names=allowed_tool_names,
        input_schemas=input_schemas,
        evidence_urls=evidence_urls,
        user_messages=user_messages,
        previous_tool_calls=previous_tool_calls,
    )


def evaluate_pending_tool_calls(
    pending_calls: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    allowed_tool_names: set[str],
    input_schemas: dict[str, dict[str, Any]] | None = None,
    evidence_urls: list[str] | set[str] | None = None,
    user_messages: list[str] | tuple[str, ...] | None = None,
    previous_tool_calls: list[str] | tuple[str, ...] | None = None,
) -> ToolPolicyDecision:
    """Validate graph-native pending tool calls before ToolRuntime execution."""
    if not pending_calls:
        return allow_tool_use()

    if len(pending_calls) > _env_int("AGENT_TOOL_MAX_PENDING_CALLS", 6, minimum=1):
        return ToolPolicyDecision(
            action="clarify",
            reason="too_many_parallel_tool_calls",
            details={"pending_count": len(pending_calls)},
        )

    names_in_latest: dict[str, int] = {}
    for call in pending_calls:
        name = str(call.get("name", "")).strip()
        if not name:
            return ToolPolicyDecision(action="clarify", reason="missing_tool_name")
        if name not in allowed_tool_names:
            return ToolPolicyDecision(
                action="clarify",
                reason="unknown_tool",
                details={"tool": name},
            )

        names_in_latest[name] = names_in_latest.get(name, 0) + 1
        if names_in_latest[name] > _env_int("AGENT_TOOL_MAX_SAME_TOOL_PENDING", 3, minimum=1):
            return ToolPolicyDecision(
                action="clarify",
                reason="repeated_parallel_tool_call",
                details={"tool": name, "count": names_in_latest[name]},
            )

        args = call.get("args")
        if not isinstance(args, dict):
            return ToolPolicyDecision(
                action="clarify",
                reason="invalid_tool_args",
                details={"tool": name, "args_type": type(args).__name__},
            )

        schema_decision = _validate_schema_args(name, args, input_schemas or {})
        if not schema_decision.allowed:
            return schema_decision

        numeric_decision = _validate_numeric_args(name, args, input_schemas or {})
        if not numeric_decision.allowed:
            return numeric_decision

        if name == "fulltext_batch":
            decision = _validate_fulltext_batch(args)
            if not decision.allowed:
                return decision

        if name == "read_news_content":
            decision = _validate_read_news_content_context(args, evidence_urls, user_messages)
            if not decision.allowed:
                return decision

    loop_decision = _validate_pending_tool_loop(
        pending_calls,
        previous_tool_calls=previous_tool_calls,
    )
    if not loop_decision.allowed:
        return loop_decision

    return allow_tool_use()


def _validate_schema_args(
    tool_name: str,
    args: dict[str, Any],
    input_schemas: dict[str, dict[str, Any]],
) -> ToolPolicyDecision:
    schema = input_schemas.get(tool_name)
    if not isinstance(schema, dict):
        return allow_tool_use()

    required = schema.get("required")
    if isinstance(required, list):
        for field_name in required:
            key = str(field_name)
            if key not in args or args.get(key) is None:
                return ToolPolicyDecision(
                    action="clarify",
                    reason="missing_required_tool_arg",
                    details={
                        "tool": tool_name,
                        "arg": key,
                        "source": "pydantic_schema",
                    },
                )

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return allow_tool_use()

    for field_name, field_schema in properties.items():
        key = str(field_name)
        if key not in args or args.get(key) is None:
            continue
        if not isinstance(field_schema, dict):
            continue

        min_length = _schema_int(field_schema.get("minLength"))
        if min_length is None:
            continue
        value = args.get(key)
        if not isinstance(value, str):
            return ToolPolicyDecision(
                action="clarify",
                reason="invalid_string_tool_arg",
                details={
                    "tool": tool_name,
                    "arg": key,
                    "value_type": type(value).__name__,
                    "source": "pydantic_schema",
                },
            )
        if len(value) < min_length:
            return ToolPolicyDecision(
                action="clarify",
                reason="tool_arg_too_short",
                details={
                    "tool": tool_name,
                    "arg": key,
                    "minimum_length": min_length,
                    "source": "pydantic_schema",
                },
            )

    return allow_tool_use()


def _pending_tool_calls(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> list[dict[str, Any]]:
    if not messages:
        return []
    answered_tool_ids = {
        str(getattr(message, "tool_call_id", "")).strip()
        for message in messages
        if isinstance(message, ToolMessage)
    }
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        calls = list(getattr(message, "tool_calls", None) or [])
        return [
            call
            for call in calls
            if isinstance(call, dict)
            and str(call.get("id", "")).strip() not in answered_tool_ids
        ]
    return []


def _tool_call_names_from_messages(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name", "")).strip()
                if name:
                    names.append(name)
        elif isinstance(message, ToolMessage):
            name = str(getattr(message, "name", "")).strip()
            if name:
                names.append(name)
    return names


def _validate_numeric_args(
    tool_name: str,
    args: dict[str, Any],
    input_schemas: dict[str, dict[str, Any]],
) -> ToolPolicyDecision:
    for arg_name, bounds in _numeric_bounds_for_tool(tool_name, input_schemas).items():
        if arg_name not in args or args.get(arg_name) is None:
            continue
        try:
            value = int(args[arg_name])
        except (TypeError, ValueError):
            return ToolPolicyDecision(
                action="clarify",
                reason="invalid_numeric_tool_arg",
                details={"tool": tool_name, "arg": arg_name, "value": args.get(arg_name)},
            )
        minimum = bounds.get("minimum")
        maximum = bounds.get("maximum")
        if minimum is not None and value < minimum:
            return ToolPolicyDecision(
                action="clarify",
                reason="tool_arg_out_of_range",
                details={
                    "tool": tool_name,
                    "arg": arg_name,
                    "value": value,
                    "minimum": minimum,
                    "maximum": maximum,
                    "source": "pydantic_schema",
                },
            )
        if maximum is not None and value > maximum:
            return ToolPolicyDecision(
                action="clarify",
                reason="tool_arg_out_of_range",
                details={
                    "tool": tool_name,
                    "arg": arg_name,
                    "value": value,
                    "minimum": minimum,
                    "maximum": maximum,
                    "source": "pydantic_schema",
                },
            )
    return allow_tool_use()


def _numeric_bounds_for_tool(
    tool_name: str,
    input_schemas: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int | None]]:
    schema = input_schemas.get(tool_name)
    if not isinstance(schema, dict):
        return {}

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}

    bounds: dict[str, dict[str, int | None]] = {}
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        field_type = field_schema.get("type")
        any_of = field_schema.get("anyOf")
        numeric = field_type in {"integer", "number"}
        if not numeric and isinstance(any_of, list):
            numeric = any(
                isinstance(option, dict) and option.get("type") in {"integer", "number"}
                for option in any_of
            )
        if not numeric:
            continue

        minimum = _schema_int(field_schema.get("minimum"))
        maximum = _schema_int(field_schema.get("maximum"))
        if minimum is None and maximum is None:
            continue
        bounds[str(field_name)] = {"minimum": minimum, "maximum": maximum}
    return bounds


def _schema_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_fulltext_batch(args: dict[str, Any]) -> ToolPolicyDecision:
    raw_urls = str(args.get("urls", "") or "")
    urls = extract_urls(raw_urls)
    max_urls = _env_int("AGENT_TOOL_FULLTEXT_MAX_URLS", 10, minimum=1)
    if len(urls) > max_urls:
        return ToolPolicyDecision(
            action="clarify",
            reason="fulltext_batch_too_many_urls",
            details={"url_count": len(urls), "maximum": max_urls},
        )
    return allow_tool_use()


def _validate_read_news_content(
    messages: list[BaseMessage] | tuple[BaseMessage, ...],
    args: dict[str, Any],
    evidence_urls: list[str] | set[str] | None,
) -> ToolPolicyDecision:
    user_messages = [
        str(getattr(message, "content", "") or "")
        for message in messages
        if str(getattr(message, "type", "")).strip().lower() == "human"
    ]
    return _validate_read_news_content_context(args, evidence_urls, user_messages)


def _validate_read_news_content_context(
    args: dict[str, Any],
    evidence_urls: list[str] | set[str] | None,
    user_messages: list[str] | tuple[str, ...] | None,
) -> ToolPolicyDecision:
    url = str(args.get("url", "") or "").strip()
    if not url:
        return ToolPolicyDecision(action="clarify", reason="read_news_content_missing_url")

    allowed_urls = set()
    for item in evidence_urls or []:
        normalized = normalize_url_for_match(str(item))
        if normalized:
            allowed_urls.add(normalized)
    for message in user_messages or []:
        for item in extract_urls(str(message or "")):
            normalized = normalize_url_for_match(item)
            if normalized:
                allowed_urls.add(normalized)

    normalized_url = normalize_url_for_match(url)
    if not allowed_urls:
        return ToolPolicyDecision(
            action="clarify",
            reason="read_news_content_no_context_urls",
            details={"url": url},
        )
    if allowed_urls and normalized_url not in allowed_urls:
        return ToolPolicyDecision(
            action="clarify",
            reason="read_news_content_url_not_in_context",
            details={"url": url},
        )
    return allow_tool_use()


def _validate_pending_tool_loop(
    pending_calls: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    previous_tool_calls: list[str] | tuple[str, ...] | None,
) -> ToolPolicyDecision:
    max_repeats = _env_int("AGENT_TOOL_MAX_REPEATS_PER_RUN", 6, minimum=1)
    counts: dict[str, int] = {}
    for raw_name in previous_tool_calls or []:
        name = str(raw_name or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    for call in pending_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name", "")).strip()
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > max_repeats:
            return ToolPolicyDecision(
                action="clarify",
                reason="tool_call_loop_detected",
                details={"tool": name, "count": counts[name], "maximum": max_repeats},
            )
    return allow_tool_use()


def _validate_tool_loop(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> ToolPolicyDecision:
    max_repeats = _env_int("AGENT_TOOL_MAX_REPEATS_PER_RUN", 6, minimum=1)
    counts: dict[str, int] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "")).strip()
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
            if counts[name] > max_repeats:
                return ToolPolicyDecision(
                    action="clarify",
                    reason="tool_call_loop_detected",
                    details={"tool": name, "count": counts[name], "maximum": max_repeats},
                )
    return allow_tool_use()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, value)
