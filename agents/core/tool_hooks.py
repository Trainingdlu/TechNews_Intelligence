"""Pre/Post hooks for structured skill execution auditing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .skill_contracts import SkillEnvelope

HookAction = Literal["allow", "warn", "deny"]
PreHook = Callable[[str, dict[str, Any]], "HookDecision"]
PostHook = Callable[[str, dict[str, Any], SkillEnvelope], "HookDecision"]


@dataclass
class HookDecision:
    """Hook decision payload."""

    action: HookAction = "allow"
    reason: str | None = None
    updated_payload: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ToolHookRunner:
    """Runs pre/post hooks around skill dispatch."""

    def __init__(
        self,
        pre_hooks: list[PreHook] | None = None,
        post_hooks: list[PostHook] | None = None,
    ) -> None:
        self.pre_hooks = pre_hooks or [self._pre_time_window_guard]
        self.post_hooks = post_hooks or [self._post_evidence_guard]

    def pre_tool_use(self, tool_name: str, payload: dict[str, Any]) -> HookDecision:
        current_payload = dict(payload)
        for hook in self.pre_hooks:
            decision = hook(tool_name, current_payload)
            if decision.updated_payload is not None:
                current_payload = dict(decision.updated_payload)
            if decision.action == "deny":
                if decision.updated_payload is None:
                    decision.updated_payload = current_payload
                return decision
        return HookDecision(action="allow", updated_payload=current_payload)

    def post_tool_use(
        self,
        tool_name: str,
        payload: dict[str, Any],
        output: SkillEnvelope,
    ) -> HookDecision:
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {}

        for hook in self.post_hooks:
            decision = hook(tool_name, payload, output)
            if decision.action == "deny":
                return decision
            if decision.action == "warn" and decision.reason:
                warnings.append(decision.reason)
            if decision.diagnostics:
                diagnostics.update(decision.diagnostics)

        if warnings:
            return HookDecision(
                action="warn",
                reason="; ".join(warnings),
                diagnostics=diagnostics,
            )
        return HookDecision(action="allow", diagnostics=diagnostics)

    @staticmethod
    def _pre_time_window_guard(tool_name: str, payload: dict[str, Any]) -> HookDecision:
        if tool_name == "query_news":
            raw_days = payload.get("days", 21)
            try:
                days = int(raw_days)
            except (TypeError, ValueError):
                return HookDecision(
                    action="deny",
                    reason="query_news.days must be an integer",
                    diagnostics={"days": raw_days, "error": "invalid_integer"},
                )
            if days < 1 or days > 365:
                return HookDecision(
                    action="deny",
                    reason="query_news.days must be between 1 and 365",
                    diagnostics={"days": days},
                )

        if tool_name == "trend_analysis":
            raw_window = payload.get("window", 7)
            try:
                window = int(raw_window)
            except (TypeError, ValueError):
                return HookDecision(
                    action="deny",
                    reason="trend_analysis.window must be an integer",
                    diagnostics={"window": raw_window, "error": "invalid_integer"},
                )
            if window < 3 or window > 60:
                return HookDecision(
                    action="deny",
                    reason="trend_analysis.window must be between 3 and 60",
                    diagnostics={"window": window},
                )

        if tool_name == "compare_topics":
            topic_a = str(payload.get("topic_a", "")).strip().lower()
            topic_b = str(payload.get("topic_b", "")).strip().lower()
            if topic_a and topic_b and topic_a == topic_b:
                return HookDecision(
                    action="deny",
                    reason="compare_topics requires two distinct topics",
                    diagnostics={"topic_a": topic_a, "topic_b": topic_b},
                )

        if tool_name == "build_timeline":
            topic = str(payload.get("topic", "")).strip()
            if not topic:
                return HookDecision(
                    action="deny",
                    reason="build_timeline requires a non-empty topic",
                    diagnostics={"topic": topic},
                )

        # Generic days range guard for time-windowed skills
        if tool_name in {"compare_sources", "compare_topics", "build_timeline", "analyze_landscape"}:
            raw_days = payload.get("days")
            if raw_days is not None:
                try:
                    days = int(raw_days)
                except (TypeError, ValueError):
                    return HookDecision(
                        action="deny",
                        reason=f"{tool_name}.days must be an integer",
                        diagnostics={"days": raw_days},
                    )
                if days < 1 or days > 365:
                    return HookDecision(
                        action="deny",
                        reason=f"{tool_name}.days must be between 1 and 365",
                        diagnostics={"days": days},
                    )

        return HookDecision(action="allow")

    @staticmethod
    def _post_evidence_guard(
        tool_name: str,
        payload: dict[str, Any],
        output: SkillEnvelope,
    ) -> HookDecision:
        del payload

        if output.status == "ok":
            evidence_count = len(output.evidence)
            evidence_tools = {
                "query_news", "trend_analysis", "search_news",
                "compare_sources", "compare_topics", "build_timeline",
                "analyze_landscape", "fulltext_batch",
            }
            if evidence_count == 0 and tool_name in evidence_tools:
                return HookDecision(
                    action="warn",
                    reason="no_evidence_urls_in_skill_output",
                    diagnostics={"tool": tool_name, "status": output.status},
                )

        return HookDecision(action="allow")
