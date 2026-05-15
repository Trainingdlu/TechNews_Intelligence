"""System prompts used by graph nodes."""

from __future__ import annotations

from agent.prompts import SYSTEM_INSTRUCTION

_INTENT_ROUTER_SYSTEM_PROMPT = (
    "You classify user requests for a tech-news intelligence agent. "
    "Return JSON only with route, intent_type, reason, confidence, requires_tools, "
    "analysis_depth, entities, time_window, risk_flags. "
    "route must be one of direct_answer, needs_clarification, needs_tools."
)

_TOOL_WORKER_SYSTEM_PROMPT = (
    "You are a tool-planning worker. Return JSON only: "
    '{"tool_calls":[{"name":"tool_name","args":{}}]}. '
    "Use only selected tools. Do not answer the user."
)

_FINAL_SYSTEM_PROMPT = (
    SYSTEM_INSTRUCTION
    + "\n\nYou are now the final synthesis node. Do not call tools. "
    "Use only the provided ToolEnvelope summaries and evidence brief. "
    "If evidence is insufficient, say so clearly. "
    "When evidence URLs are provided, include at least one exact raw URL in the answer body."
)
