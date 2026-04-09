"""In-process MCP server abstraction for NewsDB tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from ..core.skill_catalog import SkillDefinition, iter_skill_definitions
from ..core.skill_contracts import SkillEnvelope, build_error_envelope

MCPToolHandler = Callable[[BaseModel], SkillEnvelope]


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    input_model: type[BaseModel]
    handler: MCPToolHandler
    description: str = ""


class InProcessMCPServer:
    """Minimal MCP-compatible server abstraction for local adapter phase."""

    def __init__(self, server_name: str):
        self.server_name = str(server_name).strip()
        if not self.server_name:
            raise ValueError("server_name is required")
        self._tools: dict[str, MCPToolSpec] = {}

    def register_tool(
        self,
        name: str,
        input_model: type[BaseModel],
        handler: MCPToolHandler,
        description: str = "",
    ) -> None:
        tool_name = str(name).strip()
        if not tool_name:
            raise ValueError("tool name is required")
        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered on server '{self.server_name}'")
        self._tools[tool_name] = MCPToolSpec(
            name=tool_name,
            input_model=input_model,
            handler=handler,
            description=description,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for tool_name in sorted(self._tools.keys()):
            spec = self._tools[tool_name]
            rows.append(
                {
                    "server_name": self.server_name,
                    "name": spec.name,
                    "qualified_name": f"mcp__{self.server_name}__{spec.name}",
                    "description": spec.description,
                    "input_schema": spec.input_model.model_json_schema(),
                }
            )
        return rows

    def call_tool(self, tool_name: str, payload: dict[str, Any] | None = None) -> SkillEnvelope:
        request_payload = payload or {}
        spec = self._tools.get(str(tool_name).strip())
        if spec is None:
            return build_error_envelope(
                tool=str(tool_name),
                request=request_payload,
                error="mcp_unknown_tool",
                diagnostics={"server_name": self.server_name, "available_tools": sorted(self._tools.keys())},
            )

        try:
            parsed_input = spec.input_model.model_validate(request_payload)
        except ValidationError as exc:
            return build_error_envelope(
                tool=spec.name,
                request=request_payload,
                error="mcp_input_validation_failed",
                diagnostics={"validation_errors": exc.errors(), "server_name": self.server_name},
            )

        try:
            envelope = spec.handler(parsed_input)
        except Exception as exc:  # noqa: BLE001
            return build_error_envelope(
                tool=spec.name,
                request=parsed_input.model_dump(mode="python"),
                error="mcp_tool_execution_failed",
                diagnostics={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "server_name": self.server_name,
                },
            )

        envelope.tool = spec.name
        if not envelope.request:
            envelope.request = parsed_input.model_dump(mode="python")
        envelope.diagnostics.update(
            {
                "mcp_server": self.server_name,
                "mcp_tool": spec.name,
            }
        )
        return envelope


def _build_delegated_handler(definition: SkillDefinition) -> MCPToolHandler:
    def _handler(payload: BaseModel) -> SkillEnvelope:
        envelope = definition.handler(payload)
        envelope.diagnostics["delegated_skill"] = definition.name
        return envelope

    return _handler


def build_newsdb_server(server_name: str = "newsdb") -> InProcessMCPServer:
    """Build local NewsDB MCP server with namespaced tool contracts."""

    server = InProcessMCPServer(server_name=server_name)
    for definition in iter_skill_definitions():
        if definition.expose_in_mcp:
            server.register_tool(
                name=definition.mcp_name or definition.name,
                input_model=definition.input_model,
                handler=_build_delegated_handler(definition),
                description=definition.description,
            )
    return server
