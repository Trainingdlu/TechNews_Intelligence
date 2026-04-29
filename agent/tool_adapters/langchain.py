"""LangChain-facing adapter for the framework-independent ToolRuntime."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from ..core.runtime_factories import build_default_tool_runtime
from ..core.tool_catalog import ToolDefinition, iter_tool_definitions
from ..core.tool_runtime import format_tool_envelope_for_model


def _make_langchain_tool(definition: ToolDefinition) -> StructuredTool:
    def _run(**kwargs: Any) -> str:
        runtime = build_default_tool_runtime()
        envelope = runtime.execute(definition.name, kwargs)
        return format_tool_envelope_for_model(envelope)

    _run.__name__ = f"{definition.name}_tool_adapter"
    _run.__doc__ = definition.description
    return StructuredTool.from_function(
        func=_run,
        name=definition.name,
        description=definition.description,
        args_schema=definition.input_model,
    )


def build_langchain_tools() -> list[StructuredTool]:
    """Build LangChain wrappers from ToolCatalog definitions."""

    return [_make_langchain_tool(definition) for definition in iter_tool_definitions()]


LANGCHAIN_TOOLS = build_langchain_tools()
