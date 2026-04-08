"""Skill contracts and envelope schema for structured tool execution."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

SKILL_SCHEMA_VERSION = "1.0"
SkillStatus = Literal["ok", "empty", "error"]


class SkillEvidence(BaseModel):
    """Evidence metadata attached to structured skill outputs."""

    url: str
    title: str | None = None
    source: str | None = None
    created_at: str | None = None
    score: float | None = None


class SkillEnvelope(BaseModel):
    """Unified output envelope for all skills."""

    schema_version: str = SKILL_SCHEMA_VERSION
    tool: str
    status: SkillStatus
    request: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] | list[Any] | None = None
    evidence: list[SkillEvidence] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def build_error_envelope(
    tool: str,
    request: dict[str, Any],
    error: str,
    diagnostics: dict[str, Any] | None = None,
) -> SkillEnvelope:
    """Build a standardized error envelope."""

    return SkillEnvelope(
        tool=tool,
        status="error",
        request=request,
        error=error,
        diagnostics=diagnostics or {},
    )
