"""Request-scoped runtime context backed by ContextVar."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class AgentRunState:
    tool_calls: list[str] = field(default_factory=list)
    tool_call_chain: list[str] = field(default_factory=list)
    progress_callback: ProgressCallback | None = None


_RUN_STATE: ContextVar[AgentRunState | None] = ContextVar("agent_run_state", default=None)


def _get_state(create_if_missing: bool = True) -> AgentRunState | None:
    state = _RUN_STATE.get()
    if state is None and create_if_missing:
        state = AgentRunState()
        _RUN_STATE.set(state)
    return state


@contextmanager
def agent_run_context(progress_callback: ProgressCallback | None = None) -> Iterator[None]:
    """Bind a fresh request context for tool calls/evidence/progress events."""
    token: Token[AgentRunState | None] = _RUN_STATE.set(
        AgentRunState(progress_callback=progress_callback)
    )
    try:
        yield
    finally:
        _RUN_STATE.reset(token)


def emit_progress(
    stage: str,
    tool_name: str = "",
    *,
    title: str | None = None,
    detail: str | None = None,
    items: list[str] | None = None,
    status: str | None = None,
    phase: str | None = None,
    event: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    state = _get_state(create_if_missing=False)
    if state is None or not callable(state.progress_callback):
        return

    payload: dict[str, Any] = {"stage": str(stage or "").strip()}
    if tool_name:
        payload["tool"] = str(tool_name).strip()
    if phase:
        payload["phase"] = str(phase).strip()
    if title:
        payload["title"] = str(title).strip()
    if detail:
        payload["detail"] = str(detail).strip()
    if items:
        payload["items"] = [str(item).strip() for item in items if str(item).strip()]
    if status:
        payload["status"] = str(status).strip()
    if event:
        payload["event"] = str(event).strip()
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key and key not in payload:
                payload[str(key)] = value
    try:
        state.progress_callback(payload)
    except Exception:
        pass


def add_tool_call(tool_name: str) -> None:
    state = _get_state(create_if_missing=True)
    if state is None:
        return
    name = str(tool_name or "").strip()
    if not name:
        return
    state.tool_call_chain.append(name)
    if name not in state.tool_calls:
        state.tool_calls.append(name)


def get_tool_calls() -> set[str]:
    state = _get_state(create_if_missing=False)
    if state is None:
        return set()
    return set(state.tool_calls)


def get_tool_call_chain() -> list[str]:
    state = _get_state(create_if_missing=False)
    if state is None:
        return []
    return list(state.tool_call_chain)

