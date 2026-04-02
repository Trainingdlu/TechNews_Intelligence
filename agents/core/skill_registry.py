"""Skill registry with typed input validation and unified dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .skill_contracts import SkillEnvelope, build_error_envelope

SkillHandler = Callable[[BaseModel], SkillEnvelope | dict[str, Any]]


@dataclass(frozen=True)
class SkillSpec:
    """Registered skill metadata."""

    name: str
    input_model: type[BaseModel]
    handler: SkillHandler
    description: str = ""


class SkillRegistry:
    """In-process registry for skill discovery and execution."""

    def __init__(self) -> None:
        self._specs: dict[str, SkillSpec] = {}

    def register(
        self,
        name: str,
        input_model: type[BaseModel],
        handler: SkillHandler,
        description: str = "",
    ) -> None:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("Skill name must not be empty")
        if normalized_name in self._specs:
            raise ValueError(f"Skill '{normalized_name}' is already registered")
        self._specs[normalized_name] = SkillSpec(
            name=normalized_name,
            input_model=input_model,
            handler=handler,
            description=description,
        )

    def has(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> SkillSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown skill: {name}")
        return self._specs[name]

    def list_skills(self) -> list[str]:
        return sorted(self._specs.keys())

    def input_schema(self, name: str) -> dict[str, Any]:
        return self.get(name).input_model.model_json_schema()

    def execute(self, name: str, payload: dict[str, Any] | None = None) -> SkillEnvelope:
        request_payload = payload or {}
        if name not in self._specs:
            return build_error_envelope(
                tool=name,
                request=request_payload,
                error="unknown_skill",
                diagnostics={"available_skills": self.list_skills()},
            )

        spec = self._specs[name]

        try:
            parsed_input = spec.input_model.model_validate(request_payload)
        except ValidationError as exc:
            return build_error_envelope(
                tool=name,
                request=request_payload,
                error="input_validation_failed",
                diagnostics={"validation_errors": exc.errors()},
            )

        try:
            raw_output = spec.handler(parsed_input)
        except Exception as exc:  # noqa: BLE001
            return build_error_envelope(
                tool=name,
                request=parsed_input.model_dump(mode="python"),
                error="skill_execution_failed",
                diagnostics={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )

        try:
            if isinstance(raw_output, SkillEnvelope):
                envelope = raw_output
            else:
                envelope = SkillEnvelope.model_validate(raw_output)
        except ValidationError as exc:
            return build_error_envelope(
                tool=name,
                request=parsed_input.model_dump(mode="python"),
                error="output_validation_failed",
                diagnostics={"validation_errors": exc.errors()},
            )

        if envelope.tool != name:
            envelope.tool = name
        if not envelope.request:
            envelope.request = parsed_input.model_dump(mode="python")
        return envelope
