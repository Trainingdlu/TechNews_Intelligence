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
            days = int(payload.get("days", 21))
            if days < 1 or days > 365:
                return HookDecision(
                    action="deny",
                    reason="query_news.days must be between 1 and 365",
                    diagnostics={"days": days},
                )

        if tool_name == "trend_analysis":
            window = int(payload.get("window", 7))
            if window < 3 or window > 60:
                return HookDecision(
                    action="deny",
                    reason="trend_analysis.window must be between 3 and 60",
                    diagnostics={"window": window},
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
            if evidence_count == 0 and tool_name in {"query_news", "trend_analysis"}:
                return HookDecision(
                    action="warn",
                    reason="no_evidence_urls_in_skill_output",
                    diagnostics={"tool": tool_name, "status": output.status},
                )

        return HookDecision(action="allow")
