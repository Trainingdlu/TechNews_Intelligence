"""tool registry with typed input validation and unified dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .tool_contracts import ToolEnvelope, ToolHandler, build_tool_error_envelope


@dataclass(frozen=True)
class ToolSpec:
    """Registered tool metadata."""

    name: str
    input_model: type[BaseModel]
    handler: ToolHandler
    description: str = ""


class ToolRegistry:
    """In-process registry for tool discovery and execution."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip()

    def register(
        self,
        name: str,
        input_model: type[BaseModel],
        handler: ToolHandler,
        description: str = "",
    ) -> None:
        normalized_name = self._normalize_name(name)
        if not normalized_name:
            raise ValueError("tool name must not be empty")
        if normalized_name in self._specs:
            raise ValueError(f"tool '{normalized_name}' is already registered")
        self._specs[normalized_name] = ToolSpec(
            name=normalized_name,
            input_model=input_model,
            handler=handler,
            description=description,
        )

    def has(self, name: str) -> bool:
        normalized_name = self._normalize_name(name)
        if not normalized_name:
            return False
        return normalized_name in self._specs

    def get(self, name: str) -> ToolSpec:
        normalized_name = self._normalize_name(name)
        if normalized_name not in self._specs:
            raise KeyError(f"Unknown tool: {name}")
        return self._specs[normalized_name]

    def list_tools(self) -> list[str]:
        return sorted(self._specs.keys())

    def input_schema(self, name: str) -> dict[str, Any]:
        return self.get(name).input_model.model_json_schema()

    def execute(self, name: str, payload: dict[str, Any] | None = None) -> ToolEnvelope:
        request_payload = payload or {}
        normalized_name = self._normalize_name(name)
        if normalized_name not in self._specs:
            return build_tool_error_envelope(
                tool=normalized_name or str(name),
                request=request_payload,
                error="unknown_tool",
                diagnostics={"available_tools": self.list_tools()},
            )

        spec = self._specs[normalized_name]

        try:
            parsed_input = spec.input_model.model_validate(request_payload)
        except ValidationError as exc:
            return build_tool_error_envelope(
                tool=normalized_name,
                request=request_payload,
                error="input_validation_failed",
                diagnostics={"validation_errors": exc.errors()},
            )

        try:
            raw_output = spec.handler(parsed_input)
        except Exception as exc:  # noqa: BLE001
            return build_tool_error_envelope(
                tool=normalized_name,
                request=parsed_input.model_dump(mode="python"),
                error="tool_execution_failed",
                diagnostics={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )

        try:
            if isinstance(raw_output, ToolEnvelope):
                envelope = raw_output
            else:
                envelope = ToolEnvelope.model_validate(raw_output)
        except ValidationError as exc:
            return build_tool_error_envelope(
                tool=normalized_name,
                request=parsed_input.model_dump(mode="python"),
                error="output_validation_failed",
                diagnostics={"validation_errors": exc.errors()},
            )

        if envelope.tool != normalized_name:
            envelope.tool = normalized_name
        if not envelope.request:
            envelope.request = parsed_input.model_dump(mode="python")
        return envelope


