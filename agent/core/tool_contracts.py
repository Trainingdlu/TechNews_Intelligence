"""tool contracts and envelope schema for structured tool execution."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["ok", "empty", "error"]


class ToolEvidence(BaseModel):
    """Evidence metadata attached to structured tool outputs."""

    url: str
    title: str | None = None
    source: str | None = None
    created_at: str | None = None
    score: float | None = None
    rank: int | None = None
    snippet: str | None = None
    match_score: float | None = None
    score_components: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolEnvelope(BaseModel):
    """Unified output envelope for all tools."""

    tool: str
    status: ToolStatus
    request: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] | list[Any] | None = None
    evidence: list[ToolEvidence] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def build_tool_error_envelope(
    tool: str,
    request: dict[str, Any],
    error: str,
    diagnostics: dict[str, Any] | None = None,
    *,
    error_code: str | None = None,
    data: dict[str, Any] | list[Any] | None = None,
) -> ToolEnvelope:
    """Build a standardized error envelope."""

    normalized_code = _normalize_error_code(error_code or error)
    merged_diagnostics = dict(diagnostics or {})
    merged_diagnostics.setdefault("error_code", normalized_code)
    return ToolEnvelope(
        tool=tool,
        status="error",
        request=request,
        data=data,
        error_code=normalized_code,
        error=error,
        diagnostics=merged_diagnostics,
    )


def build_tool_empty_envelope(
    tool: str,
    request: dict[str, Any],
    empty_reason: str,
    data: dict[str, Any] | list[Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> ToolEnvelope:
    """Build a standardized empty-result envelope."""

    merged_diagnostics = dict(diagnostics or {})
    merged_diagnostics.setdefault("empty_reason", empty_reason)
    merged_diagnostics.setdefault("evidence_count", 0)
    return ToolEnvelope(
        tool=tool,
        status="empty",
        request=request,
        data=data,
        evidence=[],
        diagnostics=merged_diagnostics,
    )


def _normalize_error_code(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "tool_error"
    chars = []
    for char in raw:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    code = "_".join(part for part in "".join(chars).split("_") if part)
    if not code:
        return "tool_error"
    if len(code) > 80:
        return "tool_error"
    return code

