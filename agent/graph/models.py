"""Role-based model factory for the custom agent graph."""

from __future__ import annotations

import os
from typing import Any

from services.llm_provider import (
    DEFAULT_DEEPSEEK_FLASH_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_VERTEX_MODEL,
    build_chat_model,
    resolve_agent_model_config,
    resolve_model_config,
)

from .state import GraphModelHandle, GraphModels


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _temperature(role: str) -> float:
    specific = _clean(os.getenv(f"AGENT_GRAPH_{role.upper()}_TEMPERATURE", ""))
    raw = specific or _clean(os.getenv("AGENT_TEMPERATURE", "0.1"))
    try:
        return float(raw)
    except Exception:
        return 0.1


def build_graph_models() -> GraphModels:
    """Build model clients for graph roles.

    Missing credentials do not fail graph construction. Nodes can still use
    deterministic fallbacks, and production traces will record unavailable
    role clients.
    """

    return GraphModels(
        context_curator=_build_role_model(
            role="context",
            default_provider="deepseek",
            default_model=DEFAULT_DEEPSEEK_MODEL,
        ),
        intent_router=_build_role_model(
            role="intent",
            default_provider="deepseek",
            default_model=DEFAULT_DEEPSEEK_FLASH_MODEL,
        ),
        tool_worker=_build_role_model(
            role="tool",
            default_provider="deepseek",
            default_model=DEFAULT_DEEPSEEK_FLASH_MODEL,
        ),
        final_synthesizer=_build_role_model(
            role="final",
            default_provider="vertex",
            default_model=DEFAULT_VERTEX_MODEL,
        ),
    )


def _build_role_model(
    *,
    role: str,
    default_provider: str,
    default_model: str,
) -> GraphModelHandle:
    provider_env = _clean(os.getenv(f"AGENT_GRAPH_{role.upper()}_PROVIDER", ""))
    model_env = _clean(os.getenv(f"AGENT_GRAPH_{role.upper()}_MODEL", ""))
    config = resolve_model_config(
        provider=provider_env or default_provider,
        model_name=model_env or None,
        default_provider=default_provider,
        default_model=default_model,
    )
    try:
        client = build_chat_model(
            provider=config.provider,
            model_name=config.model,
            temperature=_temperature(role),
            default_provider=default_provider,
            default_model=default_model,
        )
        return GraphModelHandle(
            role=role,
            provider=config.provider,
            model=config.model,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001
        fallback = _build_default_agent_fallback(role=role, error=exc)
        if fallback is not None:
            return fallback
        return GraphModelHandle(
            role=role,
            provider=config.provider,
            model=config.model,
            client=None,
            fallback=True,
            error=str(exc),
        )


def _build_default_agent_fallback(*, role: str, error: Exception) -> GraphModelHandle | None:
    try:
        config = resolve_agent_model_config()
        client = build_chat_model(
            provider=config.provider,
            model_name=config.model,
            temperature=_temperature(role),
            default_provider=config.provider,
            default_model=config.model,
        )
        return GraphModelHandle(
            role=role,
            provider=config.provider,
            model=config.model,
            client=client,
            fallback=True,
            error=str(error),
        )
    except Exception:
        return None
