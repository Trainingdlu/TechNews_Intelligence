"""Shared runtime factories for tool registry and tool hooks.

This module is the canonical home of the runtime component factories used by
the unified ReAct agent path.
"""

from __future__ import annotations

from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntime
from .tool_runtime_hooks import ToolRuntimeHooks
from .tool_catalog import iter_tool_definitions


_DEFAULT_REGISTRY: ToolRegistry | None = None
_DEFAULT_TOOL_RUNTIME_HOOKS: ToolRuntimeHooks | None = None
_DEFAULT_TOOL_RUNTIME: ToolRuntime | None = None


def build_default_registry() -> ToolRegistry:
    """Build the default in-process tool registry."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    registry = ToolRegistry()
    for definition in iter_tool_definitions():
        registry.register(
            name=definition.name,
            input_model=definition.input_model,
            handler=definition.handler,
            description=definition.description,
        )
    _DEFAULT_REGISTRY = registry
    return registry


def build_default_tool_runtime_hooks() -> ToolRuntimeHooks:
    """Build the default shared hook runner."""

    global _DEFAULT_TOOL_RUNTIME_HOOKS
    if _DEFAULT_TOOL_RUNTIME_HOOKS is not None:
        return _DEFAULT_TOOL_RUNTIME_HOOKS
    _DEFAULT_TOOL_RUNTIME_HOOKS = ToolRuntimeHooks()
    return _DEFAULT_TOOL_RUNTIME_HOOKS


def build_default_tool_runtime() -> ToolRuntime:
    """Build the default framework-independent tool runtime."""

    global _DEFAULT_TOOL_RUNTIME
    if _DEFAULT_TOOL_RUNTIME is not None:
        return _DEFAULT_TOOL_RUNTIME
    _DEFAULT_TOOL_RUNTIME = ToolRuntime(
        registry=build_default_registry(),
        hooks=build_default_tool_runtime_hooks(),
    )
    return _DEFAULT_TOOL_RUNTIME

