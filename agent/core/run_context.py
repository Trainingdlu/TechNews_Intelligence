"""Request-scoped runtime context backed by ContextVar."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Callable, Iterator


ProgressCallback = Callable[[dict[str, str]], None]


@dataclass
class AgentRunState:
    evidence_urls: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
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


def set_progress_callback(callback: ProgressCallback | None) -> None:
    state = _get_state(create_if_missing=True)
    if state is not None:
        state.progress_callback = callback


def emit_progress(stage: str, tool_name: str = "") -> None:
    state = _get_state(create_if_missing=False)
    if state is None or not callable(state.progress_callback):
        return

    payload: dict[str, str] = {"stage": str(stage or "").strip()}
    if tool_name:
        payload["tool"] = str(tool_name).strip()
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
    if name not in state.tool_calls:
        state.tool_calls.append(name)


def get_tool_calls() -> set[str]:
    state = _get_state(create_if_missing=False)
    if state is None:
        return set()
    return set(state.tool_calls)


def add_evidence_urls(urls: list[str]) -> None:
    state = _get_state(create_if_missing=True)
    if state is None:
        return
    for url in urls:
        clean = str(url or "").strip()
        if clean and clean not in state.evidence_urls:
            state.evidence_urls.append(clean)


def get_evidence_urls() -> set[str]:
    state = _get_state(create_if_missing=False)
    if state is None:
        return set()
    return set(state.evidence_urls)

