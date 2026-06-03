"""State and runtime contracts for the custom LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage

from agent.core.tool_contracts import ToolEnvelope

IntentRoute = Literal["direct_answer", "needs_clarification", "needs_tools"]


class AgentGraphState(TypedDict, total=False):
    request_id: str
    thread_id: str | None
    user_message: str
    history: list[dict]
    llm_input_messages: list[BaseMessage]
    context_pack: dict[str, Any]

    intent: dict[str, Any]
    selected_tools: list[str]
    pending_tool_calls: list[dict[str, Any]]
    tool_results: list[ToolEnvelope]
    tool_round: int
    max_tool_rounds: int

    evidence_urls: list[str]
    evidence_brief: str
    final_text: str
    valid_urls: list[str]

    clarification: dict[str, Any] | None

    next_step: str

    # Human-in-the-loop clarification (interrupt/resume) bookkeeping.
    clarified: bool
    clarify_count: int
    clar_route: str


@dataclass
class AgentRunResult:
    text: str
    urls: list[str]
    clarification: dict[str, Any] | None = None
    trace_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class GraphModelHandle:
    role: str
    provider: str
    model: str
    client: Any | None
    fallback: bool = False
    error: str | None = None


@dataclass(frozen=True)
class GraphModels:
    context_curator: GraphModelHandle
    intent_router: GraphModelHandle
    tool_worker: GraphModelHandle
    final_synthesizer: GraphModelHandle


@dataclass(frozen=True)
class GraphRuntimeConfig:
    max_tool_rounds: int = 2
    max_evidence_events: int = 3
    max_status_items: int = 3
